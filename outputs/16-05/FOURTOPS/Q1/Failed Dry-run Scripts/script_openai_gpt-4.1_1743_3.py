
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class PhysicsPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = None
    def fit(self, X, y=None):
        # Reshape for scaler: (num_events, num_features)
        # Compute mask for valid objects (obj id != 0), will assist in feature engineering
        X = np.asarray(X)
        feat = self._feature_engineering(X)
        self.scaler = StandardScaler()
        self.scaler.fit(feat)
        return self
    def transform(self, X):
        X = np.asarray(X)
        feat = self._feature_engineering(X)
        feat = self.scaler.transform(feat)
        feat = feat.astype(np.float32)
        return feat
    def _feature_engineering(self, X):
        X = np.asarray(X)
        n_ev = X.shape[0]
        max_objects = 18
        obj_stride = 5
        obj_slice = X[:,2:].reshape(n_ev, max_objects, obj_stride)
        obj_id = obj_slice[:,:,0]
        mask = (obj_id != 0).astype(np.float32) # shape (n_ev, 18)

        # Basic Event Features
        etmiss = X[:,0:1]         # (n_ev,1)
        phimiss = X[:,1:2]        # (n_ev,1)
        n_objects = mask.sum(axis=1, keepdims=True) # (n_ev,1)
        # Basic Per-Object Features
        E = obj_slice[:,:,1]  * mask
        pt = obj_slice[:,:,2] * mask
        eta = obj_slice[:,:,3] * mask
        phi = obj_slice[:,:,4] * mask
        # Per-event summary stats (mean/std/max/sum) for E, pt, eta, phi on valid objects
        # (set zero if no objects)
        def safe_stat(arr, mask, fn, fill=0.0):
            # arr: (N, #obj)
            s = fn(np.where(mask, arr, np.nan), axis=1)
            s = np.where(np.isnan(s), fill, s)
            return s.reshape(-1,1)
        e_sum  = safe_stat(E, mask, np.nansum)
        e_max  = safe_stat(E, mask, np.nanmax)
        e_std  = safe_stat(E, mask, np.nanstd)
        e_mean = safe_stat(E, mask, np.nanmean)
        pt_mean = safe_stat(pt, mask, np.nanmean)
        pt_max = safe_stat(pt, mask, np.nanmax)
        pt_sum = safe_stat(pt, mask, np.nansum)
        pt_std = safe_stat(pt, mask, np.nanstd)
        eta_mean = safe_stat(eta, mask, np.nanmean)
        eta_std = safe_stat(eta, mask, np.nanstd)
        phi_mean = safe_stat(phi, mask, np.nanmean)
        phi_std = safe_stat(phi, mask, np.nanstd)
        # Leading (highest-pT) object for features
        lead_pt_idx = pt.argmax(axis=1)
        # Gather features of leading pt object
        lead_obj = np.stack([
            E[np.arange(n_ev), lead_pt_idx],
            pt[np.arange(n_ev), lead_pt_idx],
            eta[np.arange(n_ev), lead_pt_idx],
            phi[np.arange(n_ev), lead_pt_idx]
        ], axis=1)
        # If no objects, will be zeros already
        lead_obj = np.where(n_objects>0, lead_obj, np.zeros_like(lead_obj))
        # Second leading object
        pt_cp = pt.copy()
        pt_cp[np.arange(n_ev), lead_pt_idx] = -np.inf
        sec_lead_pt_idx = pt_cp.argmax(axis=1)
        sec_lead_obj = np.stack([
            E[np.arange(n_ev), sec_lead_pt_idx],
            pt[np.arange(n_ev), sec_lead_pt_idx],
            eta[np.arange(n_ev), sec_lead_pt_idx],
            phi[np.arange(n_ev), sec_lead_pt_idx]
        ], axis=1)
        # If no second obj, zero
        sec_lead_obj = np.where(n_objects>1, sec_lead_obj, np.zeros_like(sec_lead_obj))
        # Dijet (leading+second leading) system: invariant mass (approximate), dphi, deltaR
        def delta_phi(phi1, phi2):
            dphi = phi1-phi2
            dphi = (dphi+np.pi)%(2*np.pi)-np.pi
            return dphi
        def delta_r(eta1, phi1, eta2, phi2):
            dphi = delta_phi(phi1, phi2)
            deta = eta1 - eta2
            return np.sqrt(deta**2 + dphi**2)
        dphi_12 = delta_phi(lead_obj[:,3], sec_lead_obj[:,3]).reshape(-1,1)
        dr_12 = delta_r(lead_obj[:,2], lead_obj[:,3], sec_lead_obj[:,2], sec_lead_obj[:,3]).reshape(-1,1)
        # Dijet mass: (E1+E2)**2 - |pvec1+pvec2|^2, approximate using pt/eta/phi (no mass/hardcode massless)
        def object_pvec(pt, eta, phi):
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            pz = pt * np.sinh(eta)
            return np.stack([px, py, pz], axis=-1)
        p1 = object_pvec(lead_obj[:,1], lead_obj[:,2], lead_obj[:,3])
        p2 = object_pvec(sec_lead_obj[:,1], sec_lead_obj[:,2], sec_lead_obj[:,3])
        E12 = lead_obj[:,0] + sec_lead_obj[:,0]
        pvec12 = p1 + p2
        p2_12 = (pvec12**2).sum(axis=1)
        dijet_mass = np.sqrt(np.maximum(E12**2 - p2_12, 0)).reshape(-1,1)
        # MET - leading obj deltaR
        met_px = etmiss[:,0] * np.cos(phimiss[:,0])
        met_py = etmiss[:,0] * np.sin(phimiss[:,0])
        lead_px = lead_obj[:,1]*np.cos(lead_obj[:,3])
        lead_py = lead_obj[:,1]*np.sin(lead_obj[:,3])
        dphi_met_lead = delta_phi(phimiss[:,0], lead_obj[:,3]).reshape(-1,1)
        # Stack all features
        feats = [
            etmiss, phimiss, n_objects, 
            e_sum, e_mean, e_max, e_std, 
            pt_sum, pt_mean, pt_max, pt_std,
            eta_mean, eta_std, phi_mean, phi_std,
            lead_obj, sec_lead_obj, dphi_12, dr_12, dijet_mass, dphi_met_lead
        ]
        feats = np.concatenate(feats, axis=1)
        return feats

def make_preprocessor():
    return PhysicsPreprocessor()

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 96)
        self.bn1 = nn.BatchNorm1d(96)
        self.fc2 = nn.Linear(96, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.drop3 = nn.Dropout(0.15)
        self.fc_out = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.drop3(x)
        x = self.fc_out(x)
        return x.squeeze(-1)

def make_model(input_dim: int):
    return SimpleMLP(input_dim)

EPOCHS = 18   # Reasonably train, but not overfit in 2h

def train_model(model, train_loader, val_loader, epochs):
    import torch.optim as optim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1.7e-3, weight_decay=8e-5)
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        tloss = 0.0
        tcorrect = 0
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tloss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits)>0.5).long()
            tcorrect += (preds == yb.long()).sum().item()
            total += xb.size(0)
        train_loss.append(tloss/total)
        train_acc.append(tcorrect/total)
        # --- Validation ---
        model.eval()
        vloss = 0.0
        vcorrect = 0
        vtotal = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                logits = model(xb)
                loss = criterion(logits, yb)
                vloss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits)>0.5).long()
                vcorrect += (preds == yb.long()).sum().item()
                vtotal += xb.size(0)
        val_loss.append(vloss/vtotal)
        val_acc.append(vcorrect/vtotal)
        # Optionally shuffle optimizer/anneal here if val loss stagnant
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

