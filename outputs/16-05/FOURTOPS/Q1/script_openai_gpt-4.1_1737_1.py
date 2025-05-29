
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
class PhysicsPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom preprocessor for LHC event data.
    Implements:
     - Mask out padded objects
     - Concatenate per-object physics features (invariant mass, sum pT, etc)
     - Standardize features
    """
    def __init__(self):
        self.scaler = None
        self.n_objects = 18
        self.perobj = 5
        self.input_dim = 92
        self.valid_mask = None  # For fit-transform only
    def _make_mask(self, X):
        mask = np.zeros((X.shape[0], self.n_objects), dtype=bool)
        # The 'obj_n' field (object id) is at 2,7,12... (start at 2, every 5)
        for i in range(self.n_objects):
            idx = 2 + i*self.perobj
            mask[:,i] = X[:, idx] != 0
        return mask
    def fit(self, X, y=None):
        feats = self._transform_features(X)
        # Standardize features
        self.scaler = StandardScaler().fit(feats)
        return self
    def transform(self, X):
        feats = self._transform_features(X)
        feats = self.scaler.transform(feats)
        return feats
    def _transform_features(self, X):
        """
        Feature engineering:
         - Global event features: E_T_miss (0), phi_Et_miss (1)
         - For each object: E, pT, eta, phi if present (pad zeros)
        Adds:
         - Sum pT, sum E, mean eta, n_jets (count with |eta|<2.5), n_objects
         - Invariant mass for two highest-pT objects (if present), else 0
        """
        feats = []
        mask = self._make_mask(X)
        for i in range(X.shape[0]):
            row = X[i]
            obj_feats = []
            pt_list = []
            E_list = []
            eta_list = []
            phi_list = []
            # Per obj
            for j in range(self.n_objects):
                idx = 2 + j*self.perobj
                present = mask[i,j]
                if present:
                    E = row[idx+1]
                    pt = row[idx+2]
                    eta = row[idx+3]
                    phi = row[idx+4]
                    obj_feats.extend([E, pt, eta, phi])
                    pt_list.append(pt)
                    E_list.append(E)
                    eta_list.append(eta)
                    phi_list.append(phi)
                else:
                    obj_feats.extend([0.0, 0.0, 0.0, 0.0])
            # Global features
            eTmiss = row[0]
            phiE = row[1]
            n_objs = int(mask[i].sum())
            sumpt = np.sum(pt_list) if pt_list else 0.0
            sume = np.sum(E_list) if E_list else 0.0
            meaneta = np.mean(eta_list) if eta_list else 0.0
            njets = np.sum(np.abs(eta_list)<2.5) if eta_list else 0
            # Invariant mass: pick 2 highest pT objects
            invmass = 0.0
            if n_objs >= 2:
                pt_np = np.array(pt_list)
                idxs = np.argsort(-pt_np)[:2]
                # Rebuild Px, Py, Pz, E for both
                for i1 in idxs[:2]:
                    pt1, eta1, phi1, E1 = pt_list[i1], eta_list[i1], phi_list[i1], E_list[i1]
                    px1 = pt1*np.cos(phi1)
                    py1 = pt1*np.sin(phi1)
                    pz1 = pt1*np.sinh(eta1)
                    if 'vec1' in locals():
                        vec2 = (E1, px1, py1, pz1)
                    else:
                        vec1 = (E1, px1, py1, pz1)
                # Invariant mass
                E_tot = vec1[0]+vec2[0]
                px_tot = vec1[1]+vec2[1]
                py_tot = vec1[2]+vec2[2]
                pz_tot = vec1[3]+vec2[3]
                mass2 = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
                invmass = np.sqrt(mass2) if mass2>0 else 0.0
                del vec1, vec2
            # Final feature vector
            rowfeats = [eTmiss, phiE, n_objs, sumpt, sume, meaneta, njets, invmass]
            rowfeats.extend(obj_feats)  # flatten
            feats.append(rowfeats)
        feats = np.array(feats, dtype=np.float32)
        return feats

def make_preprocessor():
    return PhysicsPreprocessor()

class ClassifierNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.18),
            nn.Linear(96, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(48, 18),
            nn.ReLU(),
            nn.Linear(18, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def make_model(input_dim):
    return ClassifierNN(input_dim)

EPOCHS = 16

def train_model(model,
                train_loader,
                val_loader,
                epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    # For monitoring
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(epochs):
        model.train()
        tloss, tcorrect, ttotal = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.float().to(device)
            yb = yb.float().to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tloss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits)>0.5).long()
            tcorrect += (preds == yb.long()).sum().item()
            ttotal += xb.size(0)
        train_loss.append(tloss/ttotal)
        train_acc.append(tcorrect/ttotal)
        # Validation
        model.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.float().to(device)
                yb = yb.float().to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                vloss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits)>0.5).long()
                vcorrect += (preds == yb.long()).sum().item()
                vtotal += xb.size(0)
        val_loss.append(vloss/vtotal)
        val_acc.append(vcorrect/vtotal)
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

