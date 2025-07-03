
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
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import roc_auc_score

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    """
    Identity pre-processor – keeps the raw physics units untouched because
    the model computes Lorentz/ΔR based features that require the original
    scales.
    """
    def __init__(self):
        pass

    def _raw_reshape(self, X):
        return X

    def make_loader_cfg(self):
        # Slightly smaller batch than default to leave room for the
        # pair-wise broadcasting done inside the network.
        return {"batch_size": 256}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    """
    Builds higher-level physics features on-the-fly and processes the full
    per-event feature vector through a small fully-connected network.

    Extra features per event (computed in forward):
        0: mean ΔR of valid object pairs
        1: min  ΔR
        2: max  ΔR
        3: mean invariant mass of valid object pairs
        4: std  invariant mass
        5: max  invariant mass
    Final feature length = 92 (original) + 6 (engineered) = 98
    """

    def __init__(self, sample_object):
        super().__init__()
        base_dim = sample_object.shape[-1]   # 92
        extra_dim = 6
        in_dim = base_dim + extra_dim       # 98

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)                # logits
        )

    # ------------ Utility functions ------------
    @staticmethod
    def _build_pair_mask(pt: torch.Tensor) -> torch.Tensor:
        # pt: (B, 18)
        mask_i = (pt > 0).unsqueeze(2)       # (B,18,1)
        mask_j = mask_i.transpose(1, 2)      # (B,1,18)
        pair_mask = (mask_i & mask_j).float()  # (B,18,18)
        eye = torch.eye(18, device=pt.device).unsqueeze(0)  # (1,18,18)
        return pair_mask * (1.0 - eye)       # drop diagonal

    # ------------ Forward pass ------------
    def forward(self, x):                    # x: (B, 92)
        B = x.shape[0]

        # ---------- Slice raw tensors ----------
        # Global missing ET quantities are kept as part of `x`
        obj = x[:, 2:].view(B, 18, 5)        # (B,18,5)
        E   = obj[:, :, 1]                   # (B,18)
        pt  = obj[:, :, 2]                   # (B,18)
        eta = obj[:, :, 3]                   # (B,18)
        phi = obj[:, :, 4]                   # (B,18)

        # ---------- Cartesian momenta ----------
        px  = pt * torch.cos(phi)            # (B,18)
        py  = pt * torch.sin(phi)            # (B,18)
        pz  = pt * torch.sinh(eta)           # (B,18)

        # ---------- Pairwise masks ----------
        pair_mask = self._build_pair_mask(pt)     # (B,18,18)
        sum_pairs = pair_mask.sum(dim=(1, 2)).clamp(min=1.0)  # (B,)

        # ---------- ΔR features ----------
        d_eta  = eta.unsqueeze(2) - eta.unsqueeze(1)          # (B,18,18)
        d_phi  = phi.unsqueeze(2) - phi.unsqueeze(1)
        d_phi  = (d_phi + math.pi) % (2 * math.pi) - math.pi
        deltaR = torch.sqrt(d_eta**2 + d_phi**2 + 1e-9)       # (B,18,18)

        deltaR_masked = deltaR * pair_mask
        mean_dR = deltaR_masked.sum(dim=(1, 2)) / sum_pairs    # (B,)

        fill_large = deltaR.clone()
        fill_large[pair_mask == 0] = 1e6
        min_dR = fill_large.view(B, -1).min(dim=1).values      # (B,)
        max_dR = deltaR_masked.view(B, -1).max(dim=1).values   # (B,)

        # ---------- Invariant mass features ----------
        Ei  = E.unsqueeze(2)
        Ej  = E.unsqueeze(1)
        pxi = px.unsqueeze(2)
        pxj = px.unsqueeze(1)
        pyi = py.unsqueeze(2)
        pyj = py.unsqueeze(1)
        pzi = pz.unsqueeze(2)
        pzj = pz.unsqueeze(1)

        E_sum  = Ei + Ej
        px_sum = pxi + pxj
        py_sum = pyi + pyj
        pz_sum = pzi + pzj

        m2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
        m2 = torch.clamp(m2, min=0.0)
        inv_mass = torch.sqrt(m2 + 1e-9)                      # (B,18,18)

        inv_mass_masked = inv_mass * pair_mask
        mean_m  = inv_mass_masked.sum(dim=(1, 2)) / sum_pairs  # (B,)

        # std
        diff = (inv_mass_masked - mean_m.view(B, 1, 1))**2
        var_m = diff.sum(dim=(1, 2)) / sum_pairs
        std_m = torch.sqrt(var_m + 1e-9)

        fill_small = inv_mass.clone()
        fill_small[pair_mask == 0] = -1e6
        max_m = fill_small.view(B, -1).max(dim=1).values       # (B,)

        # ---------- Concatenate engineered features ----------
        extra = torch.stack([mean_dR, min_dR, max_dR,
                             mean_m,  std_m,  max_m], dim=1)   # (B,6)
        x_full = torch.cat([x, extra], dim=1)                  # (B,98)

        logits = self.net(x_full).squeeze(1)                   # (B,)
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ----- class imbalance -----
    ys = train_loader.dataset.y
    pos = (ys == 1).sum().item()
    neg = (ys == 0).sum().item()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    train_loss, val_loss = [], []
    train_acc,  val_acc  = [], []

    best_val_auc = 0.0
    patience, patience_limit = 0, 3
    best_state = None

    for epoch in range(1, epochs + 1):
        # ----------- Training -----------
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total   += yb.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc  = correct / total
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # ----------- Validation -----------
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.float().to(device)

                logits = model(xb)
                loss   = criterion(logits, yb)

                running_loss += loss.item() * yb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds == yb.long()).sum().item()
                total   += yb.size(0)

                all_logits.append(logits.cpu())
                all_labels.append(yb.cpu())

        epoch_val_loss = running_loss / total
        epoch_val_acc  = correct / total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        # AUC for early stopping
        logits_cat = torch.cat(all_logits).numpy()
        labels_cat = torch.cat(all_labels).numpy()
        val_auc = roc_auc_score(labels_cat, logits_cat)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                break

        scheduler.step()

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_loss, val_loss, train_acc, val_acc

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

