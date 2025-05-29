
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class PhysicsObjectPreprocessor(BaseEstimator, TransformerMixin):
    """
    A physics-informed preprocessor for LHC event data. Handles variable object count
    and zero-padding, extracting physically meaningful features in addition to normalizing
    kinematic variables.
    """
    def __init__(self):
        # Will learn the mask and normalization params
        self.E_scaler = None
        self.pT_scaler = None
        self.eta_scaler = None
        self.phi_scaler = None
        self.etmiss_scaler = None
        self.objtype_scaler = None
        self.n_objects = 18  # Maximum number of objects per event
        self.object_start = 2
        self.object_stride = 5
    
    def fit(self, X, y=None):
        # Parse objects, ignore padded zeros
        objs = X[:, self.object_start:self.object_start+self.n_objects*self.object_stride]
        objs = objs.reshape(-1, self.n_objects, self.object_stride)
        mask = (objs[...,0] != 0)
        E = objs[...,1][mask].reshape(-1,1)
        pT = objs[...,2][mask].reshape(-1,1)
        eta = objs[...,3][mask].reshape(-1,1)
        phi = objs[...,4][mask].reshape(-1,1)
        objtype = objs[...,0][mask].reshape(-1,1)
        # Fit scalers to nonzero objects
        self.E_scaler = StandardScaler().fit(E)
        self.pT_scaler = StandardScaler().fit(pT)
        self.eta_scaler = StandardScaler().fit(eta)
        self.phi_scaler = MinMaxScaler(feature_range=(-1,1)).fit(phi)
        self.objtype_scaler = MinMaxScaler().fit(objtype)
        # For E_T_miss
        self.etmiss_scaler = StandardScaler().fit(X[:,[0]])
        self.etmissphi_scaler = MinMaxScaler(feature_range=(-1,1)).fit(X[:,[1]])
        return self
        
    def _object_features(self, event_objects):
        # event_objects: [n_objects, 5]: objtype, E, pT, eta, phi
        # Mask for present objects (objtype != 0)
        mask = (event_objects[:,0] != 0)
        obj = event_objects[mask]
        n = len(obj)
        # 1. Per-object basic kinematic variables
        if n>0:
            E = self.E_scaler.transform(obj[:,1].reshape(-1,1)).flatten()
            pT = self.pT_scaler.transform(obj[:,2].reshape(-1,1)).flatten()
            eta = self.eta_scaler.transform(obj[:,3].reshape(-1,1)).flatten()
            phi = self.phi_scaler.transform(obj[:,4].reshape(-1,1)).flatten()
            objtype = self.objtype_scaler.transform(obj[:,0].reshape(-1,1)).flatten()
        else:
            E = np.array([])
            pT = np.array([])
            eta = np.array([])
            phi = np.array([])
            objtype = np.array([])
        # 2. Aggregate features: number of objects, energy sums, pT sums
        E_sum = np.sum(E) if len(E) else 0.0
        pT_sum = np.sum(pT) if len(pT) else 0.0
        max_pT = np.max(pT) if len(pT) else 0.0
        mean_eta = np.mean(eta) if len(eta) else 0.0
        std_eta = np.std(eta) if len(eta) else 0.0
        # 3. Leading object features (up to 6 leading, pad with zeros)
        k = 6
        lead_E = np.pad(E[:k], (0,max(0,k-len(E))))
        lead_pT = np.pad(pT[:k], (0,max(0,k-len(pT))))
        lead_eta = np.pad(eta[:k], (0,max(0,k-len(eta))))
        lead_phi = np.pad(phi[:k], (0,max(0,k-len(phi))))
        lead_type = np.pad(objtype[:k], (0,max(0,k-len(objtype))))
        # 4. Pairwise feature (mean delta R of leading objects)
        delta_Rs = []
        for i in range(min(n,k)):
            for j in range(i+1,min(n,k)):
                d_eta = eta[i] - eta[j]
                d_phi = phi[i] - phi[j]
                # wrap-around for phi
                if d_phi > 1: d_phi -= 2
                if d_phi < -1: d_phi += 2
                dist = np.sqrt(d_eta**2 + d_phi**2)
                delta_Rs.append(dist)
        if delta_Rs:
            mean_delta_R = float(np.mean(delta_Rs))
            min_delta_R = float(np.min(delta_Rs))
        else:
            mean_delta_R = 0.0
            min_delta_R = 0.0
        
        return np.concatenate([
            [n, E_sum, pT_sum, max_pT, mean_eta, std_eta, mean_delta_R, min_delta_R],
            lead_E, lead_pT, lead_eta, lead_phi, lead_type
        ])
    
    def transform(self, X):
        res = []
        for i in range(X.shape[0]):
            feats = []
            # Global MET features
            met = self.etmiss_scaler.transform(X[i:i+1,[0]])[0,0]
            metphi = self.etmissphi_scaler.transform(X[i:i+1,[1]])[0,0]
            feats.extend([met, metphi])
            # Physics object features
            objs = X[i,self.object_start:self.object_start+self.n_objects*self.object_stride]
            objs = objs.reshape(self.n_objects, self.object_stride)
            feats.extend(self._object_features(objs))
            res.append(feats)
        return np.array(res, dtype=np.float32)

def make_preprocessor():
    return PhysicsObjectPreprocessor()

class FourTopClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Small but expressive fully connected net
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(p=0.15)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = self.bn0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(self.fc2(x)))
        x = self.fc4(x)
        return x.squeeze(1)

def make_model(input_dim: int):
    model = FourTopClassifier(input_dim)
    return model

EPOCHS = 20

from sklearn.metrics import roc_auc_score

def train_model(model,
                train_loader,
                val_loader,
                epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_auc = 0.0
    for ep in range(epochs):
        model.train()
        t_loss, correct, N = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device).float()
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(Xb)
            preds = (torch.sigmoid(logits) > 0.5).to(torch.int64)
            correct += (preds == yb.to(torch.int64)).sum().item()
            N += len(Xb)
        train_loss.append(t_loss/N)
        train_acc.append(correct/float(N))
        # Validation
        model.eval()
        v_loss, v_correct, v_N = 0.0, 0, 0
        y_true, y_score = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device).float()
                logits = model(Xb)
                loss = criterion(logits, yb)
                v_loss += loss.item() * len(Xb)
                preds = (torch.sigmoid(logits) > 0.5).to(torch.int64)
                v_correct += (preds == yb.to(torch.int64)).sum().item()
                v_N += len(Xb)
                y_true.extend(yb.cpu().numpy().tolist())
                y_score.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        val_loss.append(v_loss/v_N)
        val_acc.append(v_correct/float(v_N))
        val_auc = roc_auc_score(y_true, y_score)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
        print(f"Epoch {ep+1:02d} ... train_loss: {train_loss[-1]:.4f}, val_loss: {val_loss[-1]:.4f}  train_acc: {train_acc[-1]:.4f} val_acc: {val_acc[-1]:.4f} val_auc: {val_auc:.4f}")
        # Early Stopping: if val_auc degrades for more than 4 epochs, break
        if ep>7 and val_auc < best_auc-0.003:
            break
    # Restore best model
    model.load_state_dict(best_state)
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

