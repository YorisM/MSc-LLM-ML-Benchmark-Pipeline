
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from sklearn.preprocessing import StandardScaler

##################################
# Preprocessor
##################################

class FourTopPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.obj_mask = None # will store mask for valid objects
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        # --- Object structure ---
        # [ETmiss, phi_ETmiss, obj_1, E_1, pt_1, eta_1, phi_1, ...]
        N, D = X.shape
        # Determine which per-object slots are zero (padded)
        objects = X[:, 2:].reshape(N, 18, 5)
        obj_valid = (objects[:, :, 0] != 0) # true if obj_i exists
        self.obj_mask = obj_valid # (N, 18) for potential stats
        # Compute valid mask for scalar features for scaling
        # Scalar features: [E_T_miss, phi_ET_miss] + 18*4 dims per obj [E,pT,eta,phi]
        scalars = np.zeros((N, 2 + 18*4), dtype=X.dtype)
        scalars[:,0] = X[:,0] # E_T_miss
        scalars[:,1] = X[:,1] # phi_Et_miss
        for i_obj in range(18):
            start = 2 + i_obj*5
            scalars[:, 2+i_obj*4 : 2+(i_obj+1)*4] = X[:, start+1 : start+5]
        valid = np.ones_like(scalars, dtype=bool)
        for i_obj in range(18):
            valid[:,2+i_obj*4 : 2+(i_obj+1)*4] = (objects[:,i_obj,0] != 0)[:,None]
        self.scaler.fit(scalars[valid].reshape(-1,1))
        self._fitted = True
        return self

    def transform(self, X: np.ndarray):
        assert self._fitted, "Call fit() first!"
        N, D = X.shape
        objects = X[:,2:].reshape(N,18,5)
        # --- derive per-object features in polar coordinates ---
        features = []
        # 2 global features: E_T_miss, phi_{E_T}_miss
        features.append(X[:,0:1]) # (N,1) E_Tmiss
        features.append(X[:,1:2]) # (N,1) phi_ETmiss
        # Per-object derived features
        for i in range(18):
            obj = objects[:,i,:] # (N,5): [obj_id, E, pT, eta, phi]
            obj_id = obj[:,0:1]
            mask = (obj[:,0:1] != 0).astype(X.dtype)
            E = obj[:,1:2]
            pT = obj[:,2:3]
            eta = obj[:,3:4]
            phi = obj[:,4:5]
            # replace zeros with median plausible values for denominators
            abs_eta = np.abs(eta)
            log_E = np.log1p(np.clip(E,0,None)) * mask
            log_pT = np.log1p(np.clip(pT,0,None)) * mask
            # sin/cos phi, sin/cos eta to help periodicity
            features.append(obj_id * mask)
            features.append(log_E)
            features.append(log_pT)
            features.append(abs_eta)
            features.append(np.sin(phi)*mask)
            features.append(np.cos(phi)*mask)
            features.append(mask)  # Whether this object is present
        # Concatenate the derived features
        F = np.concatenate(features, axis=1) # (N, n_features)
        # Now scale only the physically relevant features, i.e. not masks or one-hots
        # Build back the scalar vector
        scalars = np.zeros((N, 2 + 18*4), dtype=X.dtype)
        scalars[:,0] = X[:,0]
        scalars[:,1] = X[:,1]
        for i_obj in range(18):
            start = 2 + i_obj*5
            scalars[:,2+i_obj*4:2+(i_obj+1)*4] = X[:,start+1:start+5]
        scalars_flat = scalars.reshape(N,-1)
        scaled_flat = self.scaler.transform(scalars_flat.reshape(-1,1)).reshape(N,-1)
        # Copy scaled-in variants into features: E_T_miss, phi_ETmiss, E, pT, eta, phi
        out = [scaled_flat[:,:2]] # E_Tmiss, phi_{E_t}_miss
        for i in range(18):
            out.append(features[2+i*7]) # object id
            # scaled E, pT, eta, phi (use as standardization, even though derived)
            out.append(scaled_flat[:,2+i*4:2+(i+1)*4]) # E, pT, eta, phi scaled
            # sin/cos phi, mask
            out.append(features[2+i*7+3]) # sin(phi)
            out.append(features[2+i*7+4]) # cos(phi)
            out.append(features[2+i*7+5]) # present mask
        # Final concatenation
        X_proc = np.concatenate(out, axis=1)
        return X_proc.astype(np.float32)

def make_preprocessor():
    return FourTopPreprocessor()

##################################
# Model Architecture
##################################
class ParticleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Model: robust and light, focus on per-object processing + global sum
        self.input_dim = input_dim
        # Assuming feature order: [ETmiss, phi], + 18 x (id, [E,pT,eta,phi], sinphi, cosphi, mask)
        self.n_objects = 18
        object_dim = 1+4+2+1 # id, 4 kin, sin/cos phi, mask = 8
        # Indices
        self.global_dim = 2 # E_Tmiss & phi
        self.per_obj_dim = 8
        # Embed per-object: (N, 18, 8) -> (N, 18, 24)
        self.obj_embed = nn.Sequential(
            nn.Linear(self.per_obj_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 32),
            nn.ReLU()
        )
        self.global_embed = nn.Sequential(
            nn.Linear(self.global_dim, 16),
            nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(16+32, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (N, D)
        N = x.size(0)
        global_feats = x[:, :2] # (N,2)
        # Unpack per-object: next 18x8
        obj_feats = x[:, 2:].reshape(N, self.n_objects, self.per_obj_dim)  # (N, 18, 8)
        obj_embed = self.obj_embed(obj_feats)  # (N, 18, 32)
        # Mask by presence
        mask = obj_feats[:,:, -1:] # (N, 18, 1)
        obj_embed = obj_embed * mask
        # Pool across per-object
        obj_sum = obj_embed.sum(dim=1)  # (N, 32)
        global_embed = self.global_embed(global_feats) # (N,16)
        concat = torch.cat([global_embed, obj_sum], dim=-1)  # (N, 48)
        out = self.fusion(concat)  # (N, 1)
        return out.squeeze(-1)

def make_model(input_dim: int):
    return ParticleNet(input_dim)

##################################
# Training Hyperparameters
##################################
EPOCHS = 18

##################################
# Training Loop
##################################
def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.6, patience=3, min_lr=4e-5, verbose=True)
    loss_fn = nn.BCEWithLogitsLoss()
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    # For AUC computation:
    from sklearn.metrics import roc_auc_score
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        n_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            # Metrics
            running_loss += float(loss.item()) * xb.shape[0]
            preds = torch.sigmoid(logits) >= 0.5
            correct += (preds == yb.bool()).sum().item()
            n_samples += xb.size(0)
        train_loss.append(running_loss / n_samples)
        train_acc.append(correct / n_samples)
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_samples = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_running_loss += float(loss.item()) * xb.shape[0]
                preds = torch.sigmoid(logits) >= 0.5
                val_correct += (preds == yb.bool()).sum().item()
                val_samples += xb.size(0)
                y_true.append(yb.cpu().numpy())
                y_pred.append(torch.sigmoid(logits).cpu().numpy())
            val_loss.append(val_running_loss / val_samples)
            val_acc.append(val_correct / val_samples)
        # Step scheduler on loss
        scheduler.step(val_loss[-1])
        # Compute and print AUC for interpretability
        y_true_np = np.concatenate(y_true)
        y_pred_np = np.concatenate(y_pred)
        try:
            auc = roc_auc_score(y_true_np, y_pred_np)
            if epoch % 3 == 0 or epoch == epochs-1:
                print(f"Epoch {epoch:2d}: train_loss={train_loss[-1]:.4f} val_loss={val_loss[-1]:.4f} val_acc={val_acc[-1]:.4f} val_auc={auc:.4f}")
        except Exception as ex:
            pass # e.g. if only one class present in batch
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

