
import os, sys, json, pickle, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}

EPOCHS = 10
                        
def load_data():
    to_np = lambda path: pd.read_csv(path).values
    X_train = to_np(DATASET["X_train"])
    Y_train = to_np(DATASET["Y_train"]).ravel()
    X_val   = to_np(DATASET["X_val"])
    Y_val   = to_np(DATASET["Y_val"]).ravel()
    return X_train, Y_train, X_val, Y_val

def make_loaders(X_train, Y_train, X_val, Y_val, batch=1024):
    from torch.utils.data import TensorDataset, DataLoader
    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train))
    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val))
    return (DataLoader(train, batch_size=batch, shuffle=True),
            DataLoader(val, batch_size=batch))
                        
# ----------------  START OF LLM BLOCK  ----------------
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin

# Custom preprocessor: Handles variable-length object section, scaling and masking zeros
class PhysicsPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Will hold means and stds for later scaling
        self.f_scales = None  # np.ndarray, shape (n_features, 2)
        self.epsilon = 1e-6
        # Feature indices
        self.n_objects = 18
        self.obj_stride = 5
        # Indices for first-level features (missing ET)
        self.missing_et_indices = [0,1]
        # Per-object indices
        self.obj_start = 2
        self.total_features = 92
        # Useful object-masked features
        self.valid_obj_mask = None
        self.final_dim = None

    def fit(self, X, y=None):
        '''Compute scale statistics (mean, std) only on valid features.'''
        # X: numpy array, shape (N, 92)
        # Mask for zero-padded objects: obj_id == 0 means padded
        obj_id_slice = slice(self.obj_start, self.total_features, self.obj_stride)
        obj_ids = X[:, obj_id_slice]                       # (N, 18)
        valid_obj_mask = (obj_ids > 0)                     # (N, 18)
        self.valid_obj_mask = valid_obj_mask
        # Compute mean/std for missing ET (not padded)
        met = X[:, self.missing_et_indices]                # (N, 2)
        # For object features, stack only valid objects across events
        obj_feat = []
        for i in range(self.n_objects):
            o_ofs = self.obj_start + i*self.obj_stride
            feats = X[:, o_ofs:o_ofs+self.obj_stride]      # (N,5)
            mask = (obj_ids[:,i]>0)
            obj_feat.append(feats[mask])  # stack all valid
        obj_feat = np.concatenate(obj_feat, axis=0)        # (~N_valid,total 5)
        # Aggregate statistics for all features: [MET], [E,pT,eta,phi] all valid objects
        all_feats = np.concatenate([
            met,                         # [N,2]
            obj_feat                     # [N_valid_obj,5]
        ], axis=0)
        means, stds = all_feats.mean(axis=0), all_feats.std(axis=0) + self.epsilon
        # Save scaler for all 7 features (will apply for every valid, padded gets 0)
        self.f_scales = (means, stds)
        # We will assemble final vector with [MET, per-obj: (E,pT,eta,phi)], zero padded
        self.final_dim = 2 + self.n_objects * 4
        return self

    def transform(self, X):
        '''Return [MET features (2), per-object (E,pT,eta,phi: 4*18)]'''
        N = X.shape[0]
        means, stds = self.f_scales
        arr = np.zeros((N, self.final_dim), dtype=np.float32)
        # MET: indices 0,1
        met = X[:, self.missing_et_indices]                # (N,2)
        arr[:,0:2] = (met - means[:2]) / stds[:2]
        # Per-object:
        idx = 2
        for i in range(self.n_objects):
            o_ofs = self.obj_start + i*self.obj_stride
            obj_ids = X[:, o_ofs]
            # Mask for valid (not padded)
            valid = (obj_ids > 0)
            # Object features: [E, pT, eta, phi], cols 1,2,3,4 in that slice
            obj_feats = X[:, o_ofs+1:o_ofs+5]                   # (N,4)
            # Standardize (broadcast)
            normed_feats = (obj_feats - means[2:6]) / stds[2:6] # (N,4)
            # Now zero out padded
            normed_feats[~valid] = 0
            # Write out
            arr[:,idx:idx+4] = normed_feats
            idx += 4
        return arr

def make_preprocessor():
    return PhysicsPreprocessor()

# Model definition: Use a simple but expressive architecture, e.g. MLP with batchnorm and dropout.
class FourTopClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Will fit in < 50MB even with moderate width
        # Empirically, 2-3 hidden layers with batchnorm and dropout work well for tabular data
        hidden_dim = 192
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        out = self.backbone(x)
        return out.squeeze(-1)   # [N], logits

def make_model(input_dim):
    return FourTopClassifier(input_dim)

EPOCHS = 12  # Good balance for ~250k samples and model size

# Training loop: Use BCEWithLogitsLoss for numerically stable sigmoid output
# Since the original problem is to maximize AUC, we will monitor ROC-AUC in addition to loss and accuracy
from sklearn.metrics import roc_auc_score

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    model.train()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_logits = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*xb.size(0)
            preds = torch.sigmoid(logits)>0.5
            correct += (preds==yb.bool()).sum().item()
            total += xb.size(0)
            # ROC-AUC storage
            all_targets.append(yb.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
        train_loss.append(running_loss/total)
        train_acc.append(correct/total)
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_targets = []
        val_logits = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).float()
                logits = model(xb)
                vloss = criterion(logits, yb)
                val_running_loss += vloss.item()*xb.size(0)
                preds = torch.sigmoid(logits)>0.5
                val_correct += (preds==yb.bool()).sum().item()
                val_total += xb.size(0)
                val_targets.append(yb.detach().cpu().numpy())
                val_logits.append(logits.detach().cpu().numpy())
        val_loss.append(val_running_loss/val_total)
        val_acc.append(val_correct/val_total)
        # Optionally, track AUC for diagnostic
        try:
            y_true  = np.concatenate(all_targets)
            y_pred  = np.concatenate(all_logits)
            roc_auc = roc_auc_score(y_true, y_pred)
        except Exception:
            roc_auc = -1
        try:
            val_y_true = np.concatenate(val_targets)
            val_y_pred = np.concatenate(val_logits)
            val_roc_auc = roc_auc_score(val_y_true, val_y_pred)
        except Exception:
            val_roc_auc = -1
        print(f"Epoch {epoch+1}/{epochs} - Train: loss {train_loss[-1]:.4f}, acc {train_acc[-1]:.4f}, auc {roc_auc:.4f} | Val: loss {val_loss[-1]:.4f}, acc {val_acc[-1]:.4f}, auc {val_roc_auc:.4f}")
    return model, train_loss, val_loss, train_acc, val_acc
# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_tr, y_tr, X_va, y_va = load_data()
    pre = make_preprocessor();  pre.fit(X_tr, y_tr)
    X_tr = pre.transform(X_tr); X_va = pre.transform(X_va)
    tr_loader, va_loader = make_loaders(X_tr, y_tr, X_va, y_va)

    # 2. Build model
    model = make_model(input_dim=X_tr.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    trained, tr_loss, va_loss, tr_acc, va_acc = train_model(
        model, tr_loader, va_loader, epochs=n_epochs
    )

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy = torch.zeros(8, X_tr.shape[1])      # 8 fake events
        try:
            _ = trained(pre.transform(toy))
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return  # no files in dry-run

    # 4. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
    torch.save(trained.state_dict(), f"{base}_state.pt")
    torch.jit.script(trained).save(f"{base}_scripted.pt")
    torch.jit.script(pre).save(f"{base}_preproc.pt")
    with open(f"{base}_pre.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

