
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
EPOCHS = 10   # <LLM: may overwrite this constant>
                        
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
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# ---- Preprocessor ---- #
class ParticlePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # For scaling continuous features
        self.scaler = None
        self.concat_idx = None
        self.n_objects = 18  # Max per event in 92 features
        self.n_obj_feats = 5  # [obj_id, E, pT, eta, phi]
        # We'll create new per-event features and use rescaled per-object features
    def fit(self, X, y=None):
        X = np.asarray(X)
        # Extract per-object features: shape (N, n_objects, 5)
        objs = X[:, 2:].reshape(-1, self.n_objects, self.n_obj_feats)
        # Mask padded objects: obj_id == 0
        obj_mask = objs[..., 0] != 0
        # We'll scale only E, pT, eta (avoid obj_id), phi is angle so we'll sin/cos
        # Use only REAL objects for scaler fit
        obj_feats_to_scale = []
        for k in range(self.n_objects):
            k_mask = obj_mask[:, k]
            if np.any(k_mask):
                f = objs[k_mask, k, 1:4]  # E, pT, eta
                obj_feats_to_scale.append(f)
        if obj_feats_to_scale:
            X_cont = np.vstack(obj_feats_to_scale)
            self.scaler = StandardScaler().fit(X_cont)
        else:
            self.scaler = None
        return self

    def _phi_to_cart(self, phi):
        return np.stack([np.cos(phi), np.sin(phi)], axis=-1)  # shape (..., 2)

    def transform(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        # Per-event features
        miss_ET = X[:, 0]  # shape (n,)
        miss_phi = X[:, 1]
        miss_ET_xy = self._phi_to_cart(miss_phi) * miss_ET[:, None]  # shape (n,2)
        # Per-object features
        objs = X[:, 2:].reshape(-1, self.n_objects, self.n_obj_feats)
        obj_id = objs[..., 0]
        E = objs[..., 1]
        pT = objs[..., 2]
        eta = objs[..., 3]
        phi = objs[..., 4]
        mask_real = (obj_id != 0)
        # Prepare scaled features
        E_scaled = E.copy()
        pT_scaled = pT.copy()
        eta_scaled = eta.copy()
        if self.scaler is not None:
            # Only scale real-object entries
            flat_real = mask_real.flatten()
            stacked = np.stack([E, pT, eta], axis=-1).reshape(-1, 3)
            stacked_real = stacked[flat_real]
            stacked_real_scaled = self.scaler.transform(stacked_real)
            scaled = stacked.copy()
            scaled[flat_real] = stacked_real_scaled
            # Reshape back
            E_scaled = scaled[:, 0].reshape(E.shape)
            pT_scaled = scaled[:, 1].reshape(pT.shape)
            eta_scaled = scaled[:, 2].reshape(eta.shape)
        # Phi as sin/cos
        phi_cart = np.zeros((n_samples, self.n_objects, 2))
        phi_cart[mask_real] = self._phi_to_cart(phi[mask_real])
        # For padded objects, leave all zeros
        # Collate per-object features: [obj_id, E_scaled, pT_scaled, eta_scaled, phi_x, phi_y]
        per_obj = np.stack([
            obj_id, E_scaled, pT_scaled, eta_scaled, phi_cart[..., 0], phi_cart[..., 1]
        ], axis=-1)  # (N, n_objects, 6)
        # Optional: sort objects in decreasing pT (for permutation invariance)
        sort_idx = np.argsort(-pT, axis=1)
        arange = np.arange(n_samples)[:, None]
        per_obj = per_obj[arange, sort_idx]
        mask_real = mask_real[arange, sort_idx]
        # Flatten per-object (N, n_objects*6)
        per_obj = per_obj.reshape(n_samples, -1)
        # Compute per-event aggregate features
        n_real_objs = np.sum(mask_real, axis=1, keepdims=True)
        sum_pT = np.sum(pT * mask_real, axis=1, keepdims=True)
        sum_E = np.sum(E * mask_real, axis=1, keepdims=True)
        mean_pT = np.where(n_real_objs > 0, sum_pT / n_real_objs, 0.0)
        n_lep = np.sum(
            (np.isin(obj_id, [11, 13, 15])) & mask_real, axis=1, keepdims=True)
        n_bjets = np.sum(
            (np.isin(obj_id, [5])) & mask_real, axis=1, keepdims=True)
        # Concatenate all features
        features = np.concatenate([
            miss_ET[:, None],
            miss_ET_xy,  # (n,2)
            n_real_objs, sum_pT, mean_pT, sum_E,
            n_lep, n_bjets,
            per_obj  # (n, n_obj*6)
        ], axis=1)
        return features

def make_preprocessor():
    return ParticlePreprocessor()

# ---- Model ---- #
class ParticleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Simple feedforward NN for tabular+physics features
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        x = self.backbone(x)
        return torch.sigmoid(x).squeeze(-1)

def make_model(input_dim):
    return ParticleNet(input_dim)

EPOCHS = 13  # Small value to avoid overfitting and allow for quick turnaround

# ---- Training ---- #
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2, verbose=False, min_lr=2e-6)
    criterion = nn.BCELoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    for epoch in range(epochs):
        # --- Training --- #
        model.train()
        running_loss, correct, samples = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            preds = (out > 0.5).long()
            correct += (preds == yb.long()).sum().item()
            samples += xb.size(0)
        train_loss.append(running_loss / samples)
        train_acc.append(correct / samples)
        # --- Validation --- #
        model.eval()
        running_loss, correct, samples = 0.0, 0, 0
        all_out, all_yb = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                yb = yb.to(device).float()
                out = model(xb)
                loss = criterion(out, yb)
                running_loss += loss.item() * xb.size(0)
                preds = (out > 0.5).long()
                correct += (preds == yb.long()).sum().item()
                samples += xb.size(0)
                all_out.append(out.cpu().numpy())
                all_yb.append(yb.cpu().numpy())
        val_loss.append(running_loss / samples)
        val_acc.append(correct / samples)
        # Scheduler step uses AUC -- we'll approximate by accuracy for speed
        scheduler.step(val_acc[-1])
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

