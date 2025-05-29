
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
from sklearn.impute import SimpleImputer

# -------------- PREPROCESSOR ---------------- #
class FourTopPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom preprocessor for four-top dataset.
    Implements per-feature scaling, masking padded objects, extraction of physics-inspired global features.
    """
    def __init__(self):
        # Will store scalar and object scalers, and indices.
        self.scalar_scaler = StandardScaler()
        self.perobj_scaler = StandardScaler()
        self.n_objects = 18 # known from format
        self.feat_per_obj = 5
        self.total_feat = 92
        self.scalar_idx = [0, 1] # ETmiss mag, phi
        # Each object starts at 2 + 5*n
        self.obj_slices = [slice(2+5*i,2+5*(i+1)) for i in range(self.n_objects)]

    def fit(self, X, y=None):
        # ETmiss and phi are always present (idx 0,1)
        self.scalar_scaler.fit(X[:, :2])
        # Stack all objects from all events into one big array for scaler fitting
        allobj = []
        for sl in self.obj_slices:
            # Identify non-padded (obj_X > 0)
            mask = X[:, sl.start] > 0
            if mask.sum() > 0:
                allobj.append(X[mask, sl])
        if len(allobj):
            catobj = np.concatenate(allobj, axis=0)
            # Fit only to E, pT, eta, phi (ignore id)
            self.perobj_scaler.fit(catobj[:,1:])
        else:
            self.perobj_scaler.mean_ = np.zeros(4)
            self.perobj_scaler.scale_ = np.ones(4)
        return self

    def transform(self, X):
        X = X.copy()
        N = X.shape[0]
        features = []
        # 1. Per-event scalars
        features.append(self.scalar_scaler.transform(X[:, :2]))
        # 2. Per-object features: For each object, if exists (id>0), scale feat; else zeros
        obj_feats = np.zeros((N, self.n_objects, 4), dtype=np.float32) # E, pT, eta, phi
        presence_mask = np.zeros((N, self.n_objects), dtype=np.float32)
        for i, sl in enumerate(self.obj_slices):
            ids = X[:, sl.start]
            is_present = (ids > 0)
            # For present objects, scale E, pT, eta, phi
            vals = np.zeros((N,4), dtype=np.float32)
            vals[is_present] = self.perobj_scaler.transform(X[is_present, sl][..., 1:])
            obj_feats[:,i] = vals
            presence_mask[:,i] = is_present.astype(np.float32)
        # Flatten these (n_objects x 4)
        features.append(obj_feats.reshape(N, self.n_objects*4))
        # 3. Add presence mask for each object (18 dims)
        features.append(presence_mask)
        # 4. Add derived event-level features:
        # - Number of objects
        n_obj = np.sum(presence_mask, axis=1, keepdims=True)
        features.append(n_obj / self.n_objects) # scale to [0,1]
        # - Sum E, sum pT (for present objects)
        # Recover original, non-scaled E and pT for sum:
        sumE = np.zeros((N,1))
        sumpT = np.zeros((N,1))
        for i, sl in enumerate(self.obj_slices):
            is_present = (X[:, sl.start] > 0)
            sumE[:,0] += X[:,sl.start+1]*is_present
            sumpT[:,0] += X[:,sl.start+2]*is_present
        # Log transform to avoid scale outliers
        features.append(np.log1p(sumE))
        features.append(np.log1p(sumpT))
        # - Average eta of present objects, zero if none
        eta_vals = []
        for i, sl in enumerate(self.obj_slices):
            is_present = (X[:,sl.start]>0)
            eta_this = X[:,sl.start+3].copy()
            eta_this[~is_present] = np.nan
            eta_vals.append(eta_this[:,None])
        eta_cat = np.concatenate(eta_vals, axis=1)
        # avg ignoring NaNs
        eta_avg = np.nanmean(eta_cat, axis=1, keepdims=True)
        eta_avg = np.nan_to_num(eta_avg)
        features.append(eta_avg)
        # - Number of b-tag candidates (assuming object id for b-jet is 5: not available; skip)
        # - Min deltaR between any pair of objects (if at least 2)
        deltaRmin = np.zeros((N,1))
        for ix in range(N):
            phis = []
            etas = []
            for i, sl in enumerate(self.obj_slices):
                if X[ix, sl.start]>0:
                    etas.append(X[ix, sl.start+3])
                    phis.append(X[ix, sl.start+4])
            if len(etas)>=2:
                arr_eta = np.array(etas)
                arr_phi = np.array(phis)
                dR = []
                for i in range(len(etas)):
                    for j in range(i):
                        deta = arr_eta[i]-arr_eta[j]
                        dphi = np.abs(arr_phi[i]-arr_phi[j])
                        dphi = np.where(dphi>np.pi, 2*np.pi-dphi, dphi)
                        dR.append(np.sqrt(deta**2 + dphi**2))
                deltaRmin[ix,0]= np.min(dR) if dR else 0
            else:
                deltaRmin[ix,0]=0
        features.append(deltaRmin)
        # Final concat
        outX = np.concatenate(features, axis=1)
        return outX

def make_preprocessor():
    return FourTopPreprocessor()

# ------------- MODEL ---------------------- #
class FourTopNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Robust but not too large; Room for dropout; ReLU/SELU
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.Dropout(0.16),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.14),
            nn.Linear(96,48),
            nn.ReLU(),
            nn.Linear(48,16),
            nn.ReLU(),
            nn.Linear(16,1),
        )
    def forward(self, x):
        x = self.layers(x)
        return x.squeeze(-1)

def make_model(input_dim: int):
    return FourTopNet(input_dim)

# ----------- TRAINING LOOP ----------------- #
EPOCHS = 18

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=3e-5)
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        t_acc = 0
        t_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * xb.size(0)
            prob = torch.sigmoid(out)
            preds = (prob>0.5).float()
            t_acc += (preds==yb).sum().item()
            t_n += xb.size(0)
        train_loss.append(t_loss/t_n)
        train_acc.append(t_acc/t_n)
        # Validation
        model.eval()
        v_loss = 0
        v_acc = 0
        v_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                yb = yb.to(device).float()
                out = model(xb)
                loss = loss_fn(out, yb)
                v_loss += loss.item()*xb.size(0)
                prob = torch.sigmoid(out)
                preds = (prob>0.5).float()
                v_acc  += (preds==yb).sum().item()
                v_n += xb.size(0)
        val_loss.append(v_loss/v_n)
        val_acc.append(v_acc/v_n)
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

