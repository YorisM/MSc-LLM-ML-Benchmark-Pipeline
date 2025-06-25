
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

# 0. ---------- IMPORTS ----------
import math
from typing import Tuple, List
from copy import deepcopy

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    """
    1. Computes a small set of high-level aggregate physics features
       (object multiplicity, ΣpT, ΣE, max pT, mean η, std η)
       and concatenates them to the original 92 flat features
       resulting in 98 total features.
    2. Standardises each feature (z-score) w.r.t. the training set.
    All operations are vectorised in torch and are fully picklable.
    """

    def __init__(self):
        self.mean: torch.Tensor = None   # [98]
        self.std:  torch.Tensor = None   # [98]

    # ---------- internal helpers ----------
    @staticmethod
    def _derive_features(x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, 92)  raw flat tensor
        returns (N, 98) tensor with extra 6 aggregated features appended
        """
        N = x.shape[0]
        # Split global / per-object parts
        global_feats = x[:, :2]                    # (N,2)
        obj_flat     = x[:, 2:]                    # (N,90)
        objs         = obj_flat.view(N, 18, 5)     # (N,18,5)  -> features: id,E,pT,eta,phi
        ids    = objs[..., 0]                      # (N,18)
        energies = objs[..., 1]
        pTs     = objs[..., 2]
        etas    = objs[..., 3]

        mask = ids > 0                             # (N,18)   True for real objects

        n_obj   = mask.sum(dim=1).float()                              # (N)
        sum_pT  = (pTs * mask).sum(dim=1)                              # (N)
        sum_E   = (energies * mask).sum(dim=1)                         # (N)
        # max pT – handle no-object case by replacing -inf with 0
        max_pT  = pTs.masked_fill(~mask, float("-inf")).max(dim=1).values
        max_pT  = torch.where(max_pT == float("-inf"), torch.zeros_like(max_pT), max_pT)
        mean_eta = (etas * mask).sum(dim=1) / torch.clamp(n_obj, min=1.0)
        var_eta  = (mask * (etas - mean_eta.unsqueeze(1))**2).sum(dim=1) / torch.clamp(n_obj, min=1.0)
        std_eta  = torch.sqrt(var_eta + 1e-6)

        agg = torch.stack([n_obj, sum_pT, sum_E, max_pT, mean_eta, std_eta], dim=1)  # (N,6)
        return torch.cat([x, agg], dim=1)    # (N, 92+6) = (N,98)

    # ---------- public API ----------
    def fit(self, X: torch.Tensor, y=None):
        with torch.no_grad():
            feats = self._derive_features(X)      # (N,98)
            self.mean = feats.mean(dim=0)         # (98)
            self.std  = feats.std(dim=0).clamp(min=1e-6)  # avoid division by 0
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self._derive_features(X)      # (N,98)
            feats = (feats - self.mean) / self.std
        return feats

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    """
    Hybrid architecture:
      • Shared MLP over each object (5-vector) -> 32-dim embedding
      • Mean-pool over objects
      • Concatenate pooled embedding + 2 global + 6 aggregated features
      • Final MLP for classification
    """
    def __init__(self, sample_object: torch.Tensor):
        super().__init__()
        in_features = sample_object.shape[-1]   # expected 98
        assert in_features == 98, "Pre-processor must output 98 features"

        self.obj_mlp = nn.Sequential(           # 5 -> 32
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Dimensions: 2 global + 6 agg + 32 pooled = 40
        self.classifier = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor):
        # x : (B,98)
        global_feats = x[:, :2]          # (B,2)
        obj_flat     = x[:, 2:92]        # (B,90)
        agg_feats    = x[:, 92:]         # (B,6)

        B = x.size(0)
        objs = obj_flat.view(B, 18, 5)   # (B,18,5)

        obj_emb = self.obj_mlp(objs)     # (B,18,32)
        pooled  = obj_emb.mean(dim=1)    # (B,32)

        event_vec = torch.cat([global_feats, agg_feats, pooled], dim=1)  # (B,40)
        logits = self.classifier(event_vec).squeeze(-1)                  # (B)
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20
def _batch_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = (logits.sigmoid() >= 0.5).long()
    return (preds == y).float().mean().item()

def train_model(model: nn.Module, train_loader, val_loader, epochs: int = EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=2)

    train_loss_hist: List[float] = []
    val_loss_hist:   List[float] = []
    train_acc_hist:  List[float] = []
    val_acc_hist:    List[float] = []

    best_val_loss = math.inf
    patience, patience_cnt = 5, 0

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        running_loss, running_acc, n_batches = 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc  += _batch_accuracy(logits.detach(), yb.detach().long())
            n_batches += 1

        train_loss = running_loss / n_batches
        train_acc  = running_acc  / n_batches

        # ---- Validation ----
        model.eval()
        val_loss_total, val_acc_total, val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.float().to(device)
                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss_total += loss.item()
                val_acc_total  += _batch_accuracy(logits, yb.long())
                val_batches += 1

        val_loss = val_loss_total / val_batches
        val_acc  = val_acc_total  / val_batches

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        scheduler.step(val_loss)

        # ---- Early Stopping ----
        if val_loss + 1e-4 < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    # Restore best model state
    model.load_state_dict(best_state)
    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

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

