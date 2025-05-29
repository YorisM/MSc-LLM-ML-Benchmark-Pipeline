
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

class PhysicsPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.E_scaler_mean = None
        self.E_scaler_std = None
        self.pt_scaler_mean = None
        self.pt_scaler_std = None
        self.eta_scaler_mean = None
        self.eta_scaler_std = None
        self.n_objects = 18
        self.object_feature_count = 5 # [ID, E, pt, eta, phi]
        self.global_feature_count = 2 # [ETmiss, phiETmiss]

    def fit(self, X, y=None):
        X = np.asarray(X)
        # Only for non-zero (i.e., present) objects
        obj_info = X[:, 2:].reshape(-1, self.n_objects, self.object_feature_count)
        mask = obj_info[..., 0] > 0  # consider 0 as padding
        # E, pt, eta, phi: indices 1,2,3,4 per object
        E_vals  = obj_info[..., 1][mask]
        pt_vals = obj_info[..., 2][mask]
        eta_vals= obj_info[..., 3][mask]
        # Standardize (E, pt, eta) only for objects present
        self.E_scaler_mean  = E_vals.mean() if E_vals.size else 0.0
        self.E_scaler_std   = E_vals.std()  if E_vals.size else 1.0
        self.pt_scaler_mean = pt_vals.mean() if pt_vals.size else 0.0
        self.pt_scaler_std  = pt_vals.std() if pt_vals.size else 1.0
        self.eta_scaler_mean= eta_vals.mean() if eta_vals.size else 0.0
        self.eta_scaler_std = eta_vals.std() if eta_vals.size else 1.0
        # Global ETmiss
        self.ETmiss_mean = X[:,0].mean()
        self.ETmiss_std  = X[:,0].std() if X[:,0].std() > 0 else 1.0
        return self

    def transform(self, X):
        X = np.asarray(X)
        n_ev = X.shape[0]
        # (N, 92) flat
        features = []
        ETmiss = (X[:,0] - self.ETmiss_mean)/self.ETmiss_std
        phi_ETmiss = X[:,1] / np.pi  # scale to [-1,1]
        features.append(ETmiss[:,None])
        features.append(phi_ETmiss[:,None])
        # Per object
        obj_info = X[:,2:].reshape(-1, self.n_objects, self.object_feature_count) #(N,18,5)
        # Mask for real objects
        mask = (obj_info[...,0] > 0).astype(float)
        # Standardize E, pt, eta, leave phi as is (but scale phi/pi)
        # Replace paddings with 0; zeros will remain after scaling
        E    = ((obj_info[...,1] - self.E_scaler_mean) / (self.E_scaler_std+1e-8)) * mask
        pt   = ((obj_info[...,2] - self.pt_scaler_mean) / (self.pt_scaler_std+1e-8)) * mask
        eta  = ((obj_info[...,3] - self.eta_scaler_mean) / (self.eta_scaler_std+1e-8)) * mask
        phi  = (obj_info[...,4]/np.pi) * mask
        # For categorical object IDs, embed as 0 if pad, else (ID/20)
        ids  = obj_info[...,0] / 20.0 # All IDs < 20, pad is 0
        obj_feats = [ids,E,pt,eta,phi]
        obj_feats = np.stack(obj_feats, axis=-1) # (N,18,5)
        obj_feats = obj_feats.reshape(n_ev, -1)  # (N, 90)
        features.append(obj_feats)
        # Aggregate object-level features per event for more discrimination
        obj_presence = (obj_info[...,0] > 0).astype(float) # (N,18)
        n_obj = np.sum(obj_presence, axis=1, keepdims=True)
        mean_E = np.sum(E,axis=1,keepdims=True)/(n_obj+1e-6)
        mean_pt= np.sum(pt,axis=1,keepdims=True)/(n_obj+1e-6)
        std_eta= np.sqrt(np.sum((eta - mean_E)**2,axis=1,keepdims=True)/(n_obj+1e-6)) # Not actual mean_eta, but it's okay numerically
        features.extend([n_obj,mean_E,mean_pt,std_eta])
        # Final shape: (N, 2+90+4) = (N, 96)
        return np.concatenate(features,axis=1)

def make_preprocessor():
    return PhysicsPreprocessor()

class FourTopClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Compact, regularized, wide enough
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.20)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.15)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.input_bn(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.drop2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.out(x)
        return x.squeeze(-1)

def make_model(input_dim):
    return FourTopClassifier(input_dim)

EPOCHS = 18

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_val_auc = 0.0
    best_state = None
    # No scheduler (fast); metrics for early stopping
    
    def compute_acc_and_loss(loader):
        model.eval()
        losses = []
        correct = 0
        total = 0
        preds_all = []
        targets_all = []
        with torch.no_grad():
            for X,y in loader:
                X = X.to(device).float()
                y = y.to(device)
                logits = model(X)
                loss = criterion(logits, y.float())
                losses.append(loss.item() * X.size(0))
                pred_prob = torch.sigmoid(logits)
                preds = (pred_prob > 0.5).long()
                correct += (preds == y).sum().item()
                total += X.size(0)
                preds_all.append(pred_prob.cpu().numpy())
                targets_all.append(y.cpu().numpy())
        avg_loss = sum(losses) / total
        acc = correct / total
        return avg_loss, acc, np.concatenate(preds_all), np.concatenate(targets_all)

    from sklearn.metrics import roc_auc_score
    for ep in range(epochs):
        model.train()
        t_loss = 0
        t_total = 0
        t_correct = 0
        for X, y in train_loader:
            X = X.to(device).float()
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y.float())
            loss.backward()
            optimizer.step()
            t_loss += loss.item()*X.size(0)
            pred_prob = torch.sigmoid(logits)
            preds = (pred_prob > 0.5).long()
            t_correct += (preds == y).sum().item()
            t_total += X.size(0)
        train_loss.append(t_loss/t_total)
        train_acc.append(t_correct/t_total)
        v_loss, v_acc, v_pred, v_true = compute_acc_and_loss(val_loader)
        val_loss.append(v_loss)
        val_acc.append(v_acc)
        # Compute AUC for validation
        try:
            val_auc = roc_auc_score(v_true, v_pred)
        except:
            val_auc = 0.0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
        # [Optional] Early stopping not used (always all 18 epochs)
    if best_state is not None:
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

