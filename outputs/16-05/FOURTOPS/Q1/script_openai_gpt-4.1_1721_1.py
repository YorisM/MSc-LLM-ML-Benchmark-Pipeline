
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

class PhysicsPreprocessor:
    """
    Preprocessing event features for the LHC 4-top challenge. Implements:
      - Masking zero-padded (empty) objects
      - Scaling numerical features to 0 mean & unit variance
      - Aggregating physics-inspired event-level features
    """
    def __init__(self):
        self.scaler_kin = None     # For standardizing kinematic features
        self.object_mask = None    # 1 if object, 0 if zero-padded
        # Store indices for easier reference
        self.obj_start = 2
        self.nobj = 18
        self.per_obj = 5
        # Indices for each object's info
        self.obj_idx = [self.obj_start + i * self.per_obj for i in range(self.nobj)]
        self.E_idx = [o+1 for o in self.obj_idx]
        self.pt_idx = [o+2 for o in self.obj_idx]
        self.eta_idx = [o+3 for o in self.obj_idx]
        self.phi_idx = [o+4 for o in self.obj_idx]
        # Indices of event-level features (E_Tmiss, phi_ETmiss)
        self.event_feats_idx = [0,1]
        # Total features
        self.orig_dim = 92

    def fit(self, X: np.ndarray, y=None):
        # Identify real objects (exclude zero-padded)
        obj_indicator = X[:, self.obj_idx]  # (N, 18) array
        self.object_mask = (obj_indicator > 0).astype(float)
        # Gather all real object kinematic values for scaler fit
        kin_feats = []
        for fi in [self.E_idx, self.pt_idx, self.eta_idx, self.phi_idx]:
            values = [X[:, idx][self.object_mask[:,i]==1] for i, idx in enumerate(fi)]
            kin_feats.append(np.concatenate(values))
        kin_feats = np.stack(kin_feats, axis=1)  # (all_objects, 4)
        # Fit a scaler on E, pT, eta, phi for all real objects
        self.scaler_kin = StandardScaler().fit(kin_feats)
        # Also get global event-level E_Tmiss, phi_ETmiss for normalization
        return self

    def transform(self, X: np.ndarray):
        N = X.shape[0]
        # 18 objects, 4 per-object fields: E, pT, eta, phi
        features = []
        for i in range(self.nobj):
            obj_col = X[:, self.obj_idx[i]]
            is_valid = (obj_col > 0)
            # Get kinematic info and mask those not present
            E = X[:, self.E_idx[i]]
            pT = X[:, self.pt_idx[i]]
            eta = X[:, self.eta_idx[i]]
            phi = X[:, self.phi_idx[i]]
            # stack per object
            arr = np.stack([E,pT,eta,phi], axis=1)  # (N,4)
            arr_std = np.copy(arr)
            # Only scale real objects
            real_mask = is_valid[:,None]
            arr_std[real_mask[:,0]] = self.scaler_kin.transform(arr[real_mask[:,0]])
            arr_std[~real_mask[:,0]] = 0.0  # set padded objects to 0
            features.append(arr_std)
        # Shape: (nobj, N, 4) --> (N, nobj*4)
        feat_flat = np.concatenate(features, axis=1)
        # Event-level features: E_Tmiss (scale), phi_ETmiss (leave as is)
        E_Tmiss = X[:,0]
        phi_met = X[:,1]
        # Scale E_Tmiss with log to reduce dynamic range, keep phi in [-pi,pi]
        E_Tmiss_trans = np.log1p(np.abs(E_Tmiss)) * np.sign(E_Tmiss)
        phi_met_trans = phi_met / np.pi

        # Physics aggregates
        n_objects = np.sum(self.object_mask, axis=1, keepdims=True) / self.nobj
        # Sum of pT of objects
        sum_pt = np.sum(np.reshape(feat_flat, (N,self.nobj,4))[:,:,1], axis=1, keepdims=True)
        max_pt = np.max(np.reshape(feat_flat, (N,self.nobj,4))[:,:,1], axis=1, keepdims=True)
        # For missing energy over visible sum pT
        met_div_sumpt = (E_Tmiss_trans.reshape(-1,1)+1e-5)/(sum_pt+1e-5)

        # Concatenate all features
        X_out = np.concatenate([
            feat_flat,
            E_Tmiss_trans.reshape(-1,1),
            phi_met_trans.reshape(-1,1),
            n_objects,
            sum_pt,
            max_pt,
            met_div_sumpt
        ], axis=1)
        return X_out.astype(np.float32)

def make_preprocessor():
    return PhysicsPreprocessor()

class TopNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Compact but expressive MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.13),
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.19),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1)
        )
    def forward(self, x):
        return self.net(x)

def make_model(input_dim):
    return TopNet(input_dim)

EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4, weight_decay=2e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_auc = 0
    for ep in range(epochs):
        model.train()
        t_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.float().to(device).unsqueeze(1)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * xb.size(0)
            with torch.no_grad():
                preds = torch.sigmoid(out) > 0.5
                correct += (preds == yb.bool()).sum().item()
                total += xb.size(0)
        train_loss.append(t_loss / total)
        train_acc.append(correct / total)
        # Validation
        model.eval()
        v_loss = 0.0; v_correct = 0; v_total = 0
        all_logits = []
        all_ytrue = []
        with torch.no_grad():
            for xbv, ybv in val_loader:
                xbv = xbv.to(device, non_blocking=True)
                ybv = ybv.float().to(device).unsqueeze(1)
                logits = model(xbv)
                loss = criterion(logits, ybv)
                v_loss += loss.item() * xbv.size(0)
                preds = torch.sigmoid(logits) > 0.5
                v_correct += (preds == ybv.bool()).sum().item()
                v_total += xbv.size(0)
                all_logits.append(logits.cpu().numpy())
                all_ytrue.append(ybv.cpu().numpy())
        val_loss.append(v_loss / v_total)
        val_acc.append(v_correct / v_total)
        # (Optional: print AUC now for monitoring)
        # all_logits and all_ytrue are lists of arrays
        val_pred = np.vstack(all_logits).flatten()
        val_true = np.vstack(all_ytrue).flatten()
        val_auc = roc_auc_score(val_true, val_pred)
        if val_auc > best_auc:
            best_auc = val_auc
            # Save model params, etc., if desired
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

