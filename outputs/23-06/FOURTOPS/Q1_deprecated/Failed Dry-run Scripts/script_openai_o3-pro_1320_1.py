
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: Python 3.12, PyTorch 2.6.0, Torch_Geometric 2.6.1, NumPy 2.2.3, SciPy v1.15.2, SciKit-Learn 1.6.1
import os, sys, pickle, torch, torch_geometric, gc, json, importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
                        
DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv(DATASET["X_train"], dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv(DATASET["Y_train"], dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv(DATASET["X_val"], dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv(DATASET['Y_val'], dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class PairDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
    
        if isinstance(self.x, (tuple, list)) and all(torch.is_tensor(t) for t in self.x):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])

def _make_dataset(x, y):
    custom = globals().get("make_dataset", None)
    if callable(custom):
        ds = custom(x, y)
        if ds is not None:
            return ds
    return PairDataset(x, y)

def make_loaders(X_train, Y_train, X_val, Y_val, *, batch=512, collate_fn=None, loader_cls=None):
    train_ds = _make_dataset(X_train, Y_train)
    val_ds   = _make_dataset(X_val , Y_val)

    if loader_cls is None: 
        loader_cls = DataLoader

    train_ld = loader_cls(train_ds, batch_size=batch, shuffle=True, num_workers=0, 
                        collate_fn=collate_fn)
    val_ld   = loader_cls(val_ds, batch_size=batch, shuffle=False, num_workers=0,
                        collate_fn=collate_fn)

    return train_ld, val_ld

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

# <start code template>
# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
import math
from typing import Optional, List

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    """
    Simple feature-engineering + standardisation pre-processor.

    Added features (per event):
        1. n_obj          – number of reconstructed objects (Energy > 0)
        2. sum_E          – scalar sum of objects' energies
        3. sum_pT         – scalar sum of transverse momenta
        4. mean_abs_eta   – mean absolute pseudo-rapidity of present objects
    Final feature size  = 92 + 4 = 96
    """

    def __init__(self):
        # Fitted statistics
        self.mean: Optional[torch.Tensor] = None  # (96,)
        self.std:  Optional[torch.Tensor] = None  # (96,)

    # ---------- internal helpers ----------
    @staticmethod
    def _add_features(X: torch.Tensor) -> torch.Tensor:
        """
        X: (N,92)  FloatTensor
        returns (N,96)
        """
        # ensure float32
        X = X.float()

        # Slice views ------------------------------------------------------------------
        energies = X[:, 3::5]                                  # (N,18)  energies
        pTs      = X[:, 4::5]                                  # (N,18)  pT
        etas     = X[:, 5::5]                                  # (N,18)  eta

        mask      = (energies > 0).float()                     # (N,18)  1 if object present
        n_obj     = mask.sum(dim=1, keepdim=True)              # (N,1)
        sum_E     = energies.sum(dim=1, keepdim=True)          # (N,1)
        sum_pT    = pTs.sum(dim=1, keepdim=True)               # (N,1)
        mean_eta  = (etas.abs() * mask).sum(dim=1, keepdim=True) / (n_obj + 1e-6)  # (N,1)

        derived = torch.cat([n_obj, sum_E, sum_pT, mean_eta], dim=1)  # (N,4)
        return torch.cat([X, derived], dim=1)                          # (N,96)

    # ---------- API required by harness ----------
    @staticmethod
    def _collate_fn(batch: list):
        # fall back to default collate
        return None

    def make_loader_cfg(self):
        # Slightly larger batch – helps GPU throughput but still memory-safe on CPU.
        return {"batch_size": 1024}

    def fit(self, X, y=None):
        """
        Compute per-feature mean/std on the training set.
        """
        with torch.no_grad():
            X_full = self._add_features(X)                     # (N,96)
            self.mean = X_full.mean(dim=0)                     # (96,)
            self.std  = X_full.std(dim=0)                      # (96,)
            self.std[self.std < 1e-6] = 1.0                    # avoid division by zero
        return self

    def transform(self, X):
        """
        Standardise using statistics from fit().
        """
        X_full = self._add_features(X)                         # (N,96)
        X_norm = (X_full - self.mean) / self.std               # (N,96)
        return X_norm                                          

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object: torch.Tensor):
        """
        sample_object: Tensor of shape (B, F) or (F,)
        """
        super().__init__()
        input_dim = sample_object.shape[-1]  # F = 96

        self.net = nn.Sequential(                       # Output shapes in comments (batch, *)
            nn.Linear(input_dim, 128),                  # -> (B,128)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),                         # -> (B,64)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(64, 32),                          # -> (B,32)
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1)                            # -> (B,1)
        )

    def forward(self, x: torch.Tensor):
        logits = self.net(x)                            # (B,1)
        return logits.squeeze(-1)                       # (B,)

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 12   # Feel-good default – early stopping will cap if not useful.

def _epoch_run(model, loader, criterion, device, train: bool, optimizer=None):
    """
    Utility to run one full epoch in train or eval mode.
    Returns avg_loss, avg_acc
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb = xb.to(device).float()                  # (B,F)
            yb = yb.to(device).float()                 # (B,)

            logits = model(xb)                          # (B,)
            loss   = criterion(logits, yb)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss     += loss.item() * yb.size(0)
            preds           = (torch.sigmoid(logits) > 0.5).long()
            total_correct  += (preds == yb.long()).sum().item()
            total_samples  += yb.size(0)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc  = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    """
    Standard training loop with early stopping on validation loss.
    Returns:
        trained_model, train_loss_history, val_loss_history, train_acc_history, val_acc_history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion  = nn.BCEWithLogitsLoss()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=2, verbose=False)

    patience = 4
    best_val_loss = math.inf
    epochs_no_improve = 0

    # Histories
    tr_loss_hist: List[float] = []
    va_loss_hist: List[float] = []
    tr_acc_hist : List[float] = []
    va_acc_hist : List[float] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _epoch_run(model, train_loader, criterion, device,
                                           train=True, optimizer=optimizer)
        val_loss,   val_acc   = _epoch_run(model, val_loader,   criterion, device,
                                           train=False)

        scheduler.step(val_loss)

        tr_loss_hist.append(train_loss)
        va_loss_hist.append(val_loss)
        tr_acc_hist.append(train_acc)
        va_acc_hist.append(val_acc)

        # Early stopping
        if val_loss + 1e-5 < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break  # Stop training

    return model, tr_loss_hist, va_loss_hist, tr_acc_hist, va_acc_hist

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
# <end code template>

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

def _import_dotted(path: str):
    mod, name = path.rsplit(".", 1)
    module = importlib.import_module(mod)
    return getattr(module, name)

def _plot(series_train, series_val, name, out_path):
    plt.figure()
    epochs = range(1, len(series_train) + 1)
    plt.plot(epochs, series_train, label=f"Train {name}")
    plt.plot(epochs, series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        X_train, Y_train, X_val, Y_val = X_train[:200], Y_train[:200], X_val[:20], Y_val[:20]
    pre     = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train)
    X_val   = pre.transform(X_val)

    collate = getattr(pre, "_collate_fn", None)
    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val, 
                                            batch      = cfg.get("batch_size", 512), 
                                            collate_fn = collate,
                                            loader_cls = loader_cls)

    # 2. Build model
    first_batch    = next(iter(train_loader))
    example_sample = first_batch[0]
    model          = make_model(example_sample)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. Dry-run safety check
    if dryrun:
        sample, _ = first_batch
        try:
            _ = trained_model(*sample) if isinstance(sample, (tuple, list)) else trained_model(sample)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

        pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
        pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
        pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

        torch.save(trained_model.state_dict(), pth_state)
        with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
        with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

        # 6. Save plots
        _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
            "train_loss": tr_loss   if tr_loss else None,
            "val_loss":   va_loss   if va_loss else None,
            "train_acc":  tr_acc    if tr_acc else None,
            "val_acc":    va_acc    if va_acc else None,
        }
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

