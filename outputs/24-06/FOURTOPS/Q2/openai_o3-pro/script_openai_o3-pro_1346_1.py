
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
from itertools import chain
from typing import Optional, List

from sklearn.metrics import roc_auc_score        # allowed (sklearn already imported in env)

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    """
    Build physics-inspired features + standardisation.

    Output feature set:
        • 73 “raw” non-angular quantities (includes obj-id, E, pT, η,  E_T^miss)
        • 38 sin / cos of every φ in the event   (19 φ  →  38)
        •  1 object multiplicity                 (# of objects with id != 0)
        •  1 Σ pT  over all objects
        •  1 Σ E   over all objects
      ---------------------------------------------------
        → 114 features / event (float32)
    """

    def __init__(self):
        self.mu:  Optional[torch.Tensor] = None   # (1, 114)
        self.sig: Optional[torch.Tensor] = None   # (1, 114)

        # index caches (filled during fit)
        self._phi_idx : List[int] = []
        self._other_idx : List[int] = []
        self._id_idx  : List[int] = []
        self._pt_idx  : List[int] = []
        self._E_idx   : List[int] = []

    # ---------- helpers ----------
    def _init_indices(self):
        """Populate cached index lists for quick reuse"""
        # φ columns
        self._phi_idx = [1]                          # φ_Et^miss
        self._phi_idx += [2 + i*5 + 4 for i in range(18)]   # every obj φ

        full = set(range(92))
        self._other_idx = sorted(full.difference(self._phi_idx))

        # per-object specific columns
        for i in range(18):
            base = 2 + i*5
            self._id_idx.append(base)        # obj id
            self._E_idx.append(base + 1)     # Energy
            self._pt_idx.append(base + 2)    # pT

    def _build_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.FloatTensor, shape (N, 92)

        Returns
        -------
        torch.FloatTensor, shape (N, 114)
        """
        # slice groups
        others = X[:, self._other_idx]                            # (N, 73)
        phi    = X[:, self._phi_idx]                              # (N, 19)

        sin_phi = torch.sin(phi)                                  # (N, 19)
        cos_phi = torch.cos(phi)                                  # (N, 19)

        # object-wise aggregations
        id_vals  = X[:, self._id_idx]                             # (N, 18)
        mask     = id_vals.ne(0).float()                          # (N, 18)  1 if present

        obj_mul  = mask.sum(dim=1, keepdim=True)                  # (N, 1)

        pt_vals  = X[:, self._pt_idx] * mask                      # (N, 18)
        sum_pt   = pt_vals.sum(dim=1, keepdim=True)               # (N, 1)

        E_vals   = X[:, self._E_idx]  * mask                      # (N, 18)
        sum_E    = E_vals.sum(dim=1, keepdim=True)                # (N, 1)

        # concatenate in deterministic order
        feats = torch.cat([others, sin_phi, cos_phi,
                           obj_mul, sum_pt, sum_E], dim=1)        # (N, 114)
        return feats

    # ---------- scikit-style API ----------
    def fit(self, X, y=None):
        """
        Compute mean / std over training data AFTER feature construction.
        """
        if not self._phi_idx:          # build index caches once
            self._init_indices()

        with torch.no_grad():
            X_feat = self._build_features(X.float())
            self.mu  = X_feat.mean(0, keepdim=True)
            self.sig = X_feat.std(0,  unbiased=False, keepdim=True).clamp_min(1e-6)
        return self

    def transform(self, X):
        with torch.no_grad():
            X_feat = self._build_features(X.float())
            X_std  = (X_feat - self.mu) / self.sig
        return X_std

    # optional helpers for loader cfg
    def make_loader_cfg(self):
        # larger batch improves AUC convergence and keeps training time reasonable on CPU
        return {"batch_size": 2048}

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    """
    Simple but expressive fully-connected network with
    BatchNorm + Dropout regularisation.
    """
    def __init__(self, sample_object):
        """
        sample_object : torch.Tensor, shape (B, F)
        """
        super().__init__()
        in_features = sample_object.shape[-1]

        hidden = [256, 128, 64]
        layers = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h),
                       nn.BatchNorm1d(h),
                       nn.ReLU(inplace=True),
                       nn.Dropout(p=0.25)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x : torch.FloatTensor, shape (B, F)
        returns logits (B, 1)
        """
        return self.net(x).squeeze(1)          # (B,)

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 12
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, threshold=1e-3)

    # containers for history
    tr_loss_hist, va_loss_hist = [], []
    tr_acc_hist , va_acc_hist  = [], []

    best_val_auc = -float("inf")
    patience, patience_cnt = 3, 0

    for epoch in range(1, epochs + 1):
        # -------- training --------
        model.train()
        running_loss, correct, n_samples = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)                 # (B,)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5)
            correct += preds.eq(yb.bool()).sum().item()
            n_samples += yb.size(0)

        train_loss = running_loss / n_samples
        train_acc  = correct / n_samples
        tr_loss_hist.append(train_loss)
        tr_acc_hist .append(train_acc)

        # -------- validation --------
        model.eval()
        v_loss, v_correct, v_samples = 0.0, 0, 0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                logits = model(xb)
                loss   = criterion(logits, yb)

                v_loss += loss.item() * yb.size(0)
                preds  = (torch.sigmoid(logits) >= 0.5)
                v_correct += preds.eq(yb.bool()).sum().item()
                v_samples += yb.size(0)

                all_logits.append(torch.sigmoid(logits).cpu())
                all_labels.append(yb.cpu())

        val_loss = v_loss / v_samples
        val_acc  = v_correct / v_samples
        va_loss_hist.append(val_loss)
        va_acc_hist .append(val_acc)

        # Compute AUC for early stopping / scheduler
        y_true = torch.cat(all_labels).numpy()
        y_prob = torch.cat(all_logits).numpy()
        val_auc = roc_auc_score(y_true, y_prob)
        scheduler.step(val_loss)

        # Early stopping logic
        if val_auc > best_val_auc + 1e-4:   # significant improvement
            best_val_auc = val_auc
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping on epoch {epoch} (best AUC={best_val_auc:.4f})")
                break

    # restore best weights
    model.load_state_dict(best_state)
    return model, tr_loss_hist, va_loss_hist, tr_acc_hist, va_acc_hist

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

