
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
from sklearn.impute import SimpleImputer

# --- PREPROCESSOR ---
def make_preprocessor():
    class PhysicsPreprocessor(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.non_object_feats = 2  # E_T_miss, phi_{E_t}_miss
            self.n_obj = (105-2)//5
            self.scaler = StandardScaler()
            self.imputer = SimpleImputer(strategy='constant', fill_value=0.)
        
        def fit(self, X, y=None):
            # Feature engineering: event-level physics quantities
            base = self._physics_features(X)
            # Fit on non-object features + physics features
            feats = np.concatenate([
                base,
                X[:, :self.non_object_feats],
            ], axis=1)
            feats = self.imputer.fit_transform(feats)
            self.scaler.fit(feats)
            return self
        
        def transform(self, X):
            base = self._physics_features(X)
            feats = np.concatenate([
                base,
                X[:, :self.non_object_feats],
            ], axis=1)
            feats = self.imputer.transform(feats)
            feats = self.scaler.transform(feats)
            return feats.astype(np.float32)

        def _physics_features(self, X):
            # Assumes input X: (N, 105)
            N = X.shape[0]
            objects = []
            start = self.non_object_feats
            obj_feats = []
            for i in range(self.n_obj):
                ofs = start + i*5
                ptslice = slice(ofs, ofs+5)
                obj_feats.append(X[:, ptslice])
            objs = np.stack(obj_feats, axis=1)  # (N, n_obj, 5)

            # Valid mask: at least pT>0?
            valid = objs[:,:,1] > 0.   # (N, n_obj)
            # Sum number of jets, leptons, etc.
            nobj = valid.astype(np.float32).sum(axis=1, keepdims=True)  # (N, 1)
            # Sum of pT, mean eta, et al.
            pT_sum   = (objs[:,:,1]*valid).sum(axis=1, keepdims=True)
            pT_mean  = np.where(valid, objs[:,:,1], 0.).sum(axis=1, keepdims=True) / (nobj+1e-6)
            eta_mean = np.where(valid, objs[:,:,2], 0.).sum(axis=1, keepdims=True) / (nobj+1e-6)
            phi_mean = np.where(valid, objs[:,:,3], 0.).sum(axis=1, keepdims=True) / (nobj+1e-6)

            # - Max pT object
            max_pT = objs[:,:,1].max(axis=1, keepdims=True)
            # - Min pT object
            min_pT = np.where(valid, objs[:,:,1], 1e9).min(axis=1, keepdims=True)

            # - H_T: Scalar sum of all objects' pT
            H_T = (objs[:,:,1]*valid).sum(axis=1, keepdims=True)
            # - Number of objects w/ E>100 GeV
            n_E100 = (objs[:,:,0] > 1e5).sum(axis=1, keepdims=True)

            # - Add missing E_T (already present)
            # Out: [nobj, pT_sum, pT_mean, eta_mean, phi_mean, max_pT, min_pT, H_T, n_E100]
            return np.concatenate([
                nobj, pT_sum, pT_mean, eta_mean, phi_mean, max_pT, min_pT, H_T, n_E100
            ], axis=1)

    return PhysicsPreprocessor()

# --- MODEL ---
def make_model(input_dim: int):
    class FourTopMLP(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            # AUC-oriented: use batchnorm, dropout, deeper network
            self.main = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.main(x).squeeze(-1)
    return FourTopMLP(input_dim)

epochs = 18  # empirically chosen for performance/overfit tradeoff

# --- TRAINING ---
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    import torch.optim as optim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    for ep in range(epochs):
        model.train()
        tl, ta, n_examples = 0., 0., 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.float().to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tl += loss.item()*xb.size(0)
            preds = (torch.sigmoid(out)>0.5).long()
            ta += (preds==yb.long()).sum().item()
            n_examples += xb.size(0)
        train_loss.append(tl/n_examples)
        train_acc.append(ta/n_examples)
        # Validation
        model.eval()
        vl, va, n_val = 0., 0., 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.float().to(device)
                out = model(xb)
                loss = criterion(out, yb)
                vl += loss.item()*xb.size(0)
                preds = (torch.sigmoid(out)>0.5).long()
                va += (preds==yb.long()).sum().item()
                n_val += xb.size(0)
        val_loss.append(vl/n_val)
        val_acc.append(va/n_val)
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
