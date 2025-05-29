
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

class EventPhysicsPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessing for variable-length, padded LHC events with dense kinematic features, missing ET, and object properties.
    - Identify which columns are always present (missing ET, phi) and which are per-object.
    - Aggregate per-object features for permutation invariance (mean, sum, std, max, counts by object type, etc).
    - Scale all features robustly.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.n_obj_max = 26  # (because (105 - 2)/4 = 25.75 objects, so 26 max)        
        self.obj_offsets = 2 + np.arange(self.n_obj_max) * 4
        self.E_col = 2 + np.arange(self.n_obj_max) * 4 + 1
        self.pt_col = 2 + np.arange(self.n_obj_max) * 4 + 2
        self.eta_col = 2 + np.arange(self.n_obj_max) * 4 + 3
        self.phi_col = 2 + np.arange(self.n_obj_max) * 4 + 4
        self.obj_id_col = 2 + np.arange(self.n_obj_max) * 4
        self.feature_names = None
    
    def fit(self, X, y=None):
        Xfeats = self.transform(X)
        self.scaler.fit(Xfeats)
        return self

    def _extract_features(self, X):
        # X = (N, 105)
        N = X.shape[0]
        n_obj = self.n_obj_max
        feats = []
        # Always available features
        E_T_miss = X[:,0:1] # shape (N,1)
        phi_E_T_miss = X[:,1:2]
        feats += [E_T_miss, phi_E_T_miss]
        # Reshape per-object arrays: shape (N, n_obj)
        obj_id = X[:, self.obj_id_col]
        E      = X[:, self.E_col]
        pt     = X[:, self.pt_col]
        eta    = X[:, self.eta_col]
        phi    = X[:, self.phi_col]
        # Mask for valid objects (object id != 0)
        valid = (obj_id != 0)
        obj_id_masked = np.where(valid, obj_id, np.nan) # 0 stands for pad
        
        # Per-event counts by object type (assume discrete ID in obj_id)
        uniq_obj_ids = [1,2,3,4,5,6] # guess (e, mu, tau, photon, jet, B-jet)? Not specified, use top-6 id's
        for oid in uniq_obj_ids:
            feats.append(np.sum(obj_id==oid, axis=1, keepdims=True))
        feats.append(np.sum(valid, axis=1, keepdims=True)) # n_obj
        # Kinematic per-object stats (ignore padded objects)
        def masked_stat(arr, func):
            return func(np.where(valid, arr, np.nan), axis=1, keepdims=True)
        # Means
        for arr in [E, pt, eta, phi]:
            feats.append(masked_stat(arr, np.nanmean))
        # std
        for arr in [E, pt, eta, phi]:
            feats.append(masked_stat(arr, np.nanstd))
        # Max
        for arr in [E, pt, eta, phi]:
            feats.append(masked_stat(arr, np.nanmax))
        # Min
        for arr in [E, pt, eta, phi]:
            feats.append(masked_stat(arr, np.nanmin))
        # pT sum
        feats.append(masked_stat(pt, np.nansum))
        # ET sum
        feats.append(masked_stat(E, np.nansum))
        # ET/pT ratio mean
        ratio = np.where((pt>0) & valid, E/pt, np.nan)
        feats.append(masked_stat(ratio, np.nanmean))
        # per-event delta-R max/min/mean b/w any two objects
        from scipy.spatial.distance import pdist
        deltaR_means = np.zeros((N,1))
        deltaR_maxs = np.zeros((N,1))
        deltaR_mins = np.zeros((N,1))
        for i in range(N):
            v = valid[i]
            if np.sum(v)<2:
                deltaR_means[i,0] = 0
                deltaR_maxs[i,0] = 0
                deltaR_mins[i,0] = 0
                continue
            this_eta = eta[i,v]
            this_phi = phi[i,v]
            # handle wrap for phi
            deltas = []
            for k1 in range(len(this_eta)):
                for k2 in range(k1+1, len(this_eta)):
                    dphi = np.abs(this_phi[k1] - this_phi[k2])
                    dphi = np.minimum(dphi, 2*np.pi - dphi)
                    deta = this_eta[k1] - this_eta[k2]
                    deltaR = np.sqrt(deta**2 + dphi**2)
                    deltas.append(deltaR)
            if len(deltas)>0:
                arr = np.array(deltas)
                deltaR_means[i,0] = np.nanmean(arr)
                deltaR_maxs[i,0] = np.nanmax(arr)
                deltaR_mins[i,0] = np.nanmin(arr)
            else:
                deltaR_means[i,0] = 0
                deltaR_maxs[i,0] = 0
                deltaR_mins[i,0] = 0
        feats += [deltaR_means, deltaR_maxs, deltaR_mins]
        # Concatenate all features
        feats = [np.nan_to_num(f, nan=0.0) for f in feats]
        Xfeats = np.concatenate(feats, axis=1)
        return Xfeats

    def transform(self, X):
        Xfeats = self._extract_features(np.asarray(X))
        Xfeats = self.scaler.transform(Xfeats)
        return Xfeats

    def fit_transform(self, X, y=None):
        Xfeats = self._extract_features(np.asarray(X))
        self.scaler.fit(Xfeats)
        Xfeats = self.scaler.transform(Xfeats)
        return Xfeats

def make_preprocessor():
    return EventPhysicsPreprocessor()

class DeepSetsClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # A robust but compact MLP with dropout/batchnorm.
        self.dense = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(96, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.dense(x).squeeze(-1)   # logits, (N,)

def make_model(input_dim: int):
    return DeepSetsClassifier(input_dim)

# A bit longer training - try to avoid overfit but do good enough
epochs = 26

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int):
    import torch.optim as optim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    for e in range(epochs):
        model.train()
        tr_loss, tr_acc, n = 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.float().to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(xb)
            preds = torch.sigmoid(logits) > 0.5
            tr_acc  += (preds == yb.bool()).float().sum().item()
            n += len(xb)
        train_loss.append(tr_loss/n)
        train_acc.append(tr_acc/n)
        # Validation
        model.eval()
        val_loss_epoch, val_acc_epoch, nval = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.float().to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_epoch += loss.item() * len(xb)
                preds = torch.sigmoid(logits) > 0.5
                val_acc_epoch += (preds == yb.bool()).float().sum().item()
                nval += len(xb)
        val_loss.append(val_loss_epoch / nval)
        val_acc.append(val_acc_epoch / nval)
    return model, train_loss, val_loss, train_acc, val_acc
# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(Y_train, Y_val, name, out):
    plt.figure(); plt.plot(Y_train, label=f'Train {name}')
    plt.plot(Y_val, label=f'Validation {name}')
    plt.legend(); plt.title(name); plt.xlabel('epoch')
    plt.savefig(out); plt.close()        

def _run(dryrun=False):
    X_train, Y_train, X_val, Y_val = load_data()
    pre = make_preprocessor()
    pre.fit(X_train, Y_train)
    X_train = pre.transform(X_train);  X_val = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    model = make_model(input_dim=X_train.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    hist     = train_model(model, train_loader, val_loader, epochs=n_epochs)

    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        torch.save(model.state_dict(), f"{base}_state.pt")
        with open(f"{base}_pre.pkl", "wb") as f: pickle.dump(pre, f)
        _plot(hist['loss'], hist['val_loss'], 'Loss',     f"{base}_loss.png")
        _plot(hist['acc'],  hist['val_acc'],  'Accuracy', f"{base}_acc.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)
