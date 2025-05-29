
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

# --- Preprocessor ---
class ParticlePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Will hold scalers/means/stds for features
        self.global_scaler = None  # For per-feature normalization
        self.pt_mean = None
        self.pt_std = None
        self.eta_mean = None
        self.eta_std = None
        self.n_obj_mean = None
        self.n_obj_std = None
    def fit(self, X, y=None):
        # X : (N, 92), as numpy array
        X = np.asarray(X)
        N = X.shape[0]
        # feature engineering: count number of objects per event (non-zero obj id)
        n_obj = np.sum(X[:, 2:92:5] != 0, axis=1).reshape(-1,1)
        # Per-object kinematics: shape (N, 18, 5)
        objs = X[:,2:].reshape((N, 18, 5))
        mask = objs[:,:,0] != 0
        # E, pt, eta, phi
        E = objs[:,:,1]
        pt = objs[:,:,2]
        eta = objs[:,:,3]
        # global sum for available objects per event
        sum_pt = (pt * mask).sum(axis=1, keepdims=True)
        mean_eta = np.where(mask, eta, np.nan)
        mean_eta = np.nanmean(mean_eta, axis=1, keepdims=True)
        # Save means/stds for new features for transform time
        self.pt_mean = np.nanmean(sum_pt)
        self.pt_std = np.nanstd(sum_pt) + 1e-6
        self.eta_mean = np.nanmean(mean_eta)
        self.eta_std = np.nanstd(mean_eta) + 1e-6
        self.n_obj_mean = np.mean(n_obj)
        self.n_obj_std = np.std(n_obj) + 1e-6
        # Compose engineering features for fit: [E_T_miss, phi_ET_miss, n_obj, sum_pt, mean_eta]
        hand_feats = np.hstack([
            X[:,[0,1]], # global E_T_miss, phi_Et_miss
            n_obj, sum_pt, mean_eta
        ]) # shape (N, 5)
        # Per-object kinematics (except object id):
        obj_kinematics = objs[:,:,1:5].reshape(N, -1) # (N, 18*4=72)
        # Final feature combines hand+obj kinematics
        feats = np.hstack([hand_feats, obj_kinematics])
        # Fit global scaler
        self.global_scaler = StandardScaler().fit(feats)
        return self
    def transform(self, X):
        X = np.asarray(X)
        N = X.shape[0]
        # feature engineering: count of objects (obj id != 0)
        n_obj = np.sum(X[:, 2:92:5] != 0, axis=1).reshape(-1,1)
        objs = X[:,2:].reshape((N, 18, 5))
        mask = objs[:,:,0] != 0
        pt = objs[:,:,2]
        eta = objs[:,:,3]
        sum_pt = (pt * mask).sum(axis=1, keepdims=True)
        mean_eta = np.where(mask, eta, np.nan)
        mean_eta = np.nanmean(mean_eta, axis=1, keepdims=True)
        # Compose engineered features
        hand_feats = np.hstack([
            X[:,[0,1]],
            n_obj, sum_pt, mean_eta
        ])
        obj_kinematics = objs[:,:,1:5].reshape(N, -1)
        feats = np.hstack([hand_feats, obj_kinematics])
        # Replace nan with 0 for cases with no objects
        feats = np.nan_to_num(feats, nan=0.0)
        feats = self.global_scaler.transform(feats)
        return feats

def make_preprocessor():
    return ParticlePreprocessor()

# --- Neural Net Model ---
class ParticleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Architecture: RelU, dropout, batchnorm. Wide > deep.
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.drop3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        x = self.fc4(x)
        return x.view(-1)

def make_model(input_dim):
    return ParticleMLP(input_dim)

EPOCHS = 15

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for ep in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.float().to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            pred = (torch.sigmoid(out) > 0.5).long()
            correct += (pred == yb.long()).sum().item()
            total += xb.size(0)
        train_loss.append(running_loss / total)
        train_acc.append(correct / total)
        # validation
        model.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.float().to(device)
                out = model(xb)
                loss = criterion(out, yb)
                vloss += loss.item() * xb.size(0)
                pred = (torch.sigmoid(out) > 0.5).long()
                vcorrect += (pred == yb.long()).sum().item()
                vtotal += xb.size(0)
        val_loss.append(vloss / vtotal)
        val_acc.append(vcorrect / vtotal)
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

