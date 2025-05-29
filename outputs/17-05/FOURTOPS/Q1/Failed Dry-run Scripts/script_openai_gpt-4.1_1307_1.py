
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# Preprocessor Class
class FourTopPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Containers for fit/calculation
        self.scaler_et = StandardScaler()   # missing ET scaler
        self.scaler_objs = StandardScaler() # per-object scaler
        self.obj_mask = None                # binary mask of valid objects
        self.n_objects = 18                 # fixed max pouch size
    def fit(self, X, y=None):
        X = np.asarray(X)
        # Identify payload
        et_feats = X[:, :2]    # (E_T_miss, phi_Et_miss)
        # Mask present objects using obj id
        obj_ids = X[:, 2:92:5]
        self.obj_mask = (obj_ids > 0)      # shape (N, 18)
        # Stack present object kinematics for scaler
        all_obj_feats = []
        for i in range(self.n_objects):
            # Indices of kinematic features for this object
            start = 2 + i*5
            # obj_id, E, pT, eta, phi
            # Only grab features (E, pT, eta, phi) if present
            obj_feat = X[self.obj_mask[:,i], start+1:start+5]
            if obj_feat.shape[0] > 0:
                all_obj_feats.append(obj_feat)
        if len(all_obj_feats):
            objs = np.vstack(all_obj_feats)
            self.scaler_objs.fit(objs)
        self.scaler_et.fit(et_feats)
        return self
    def transform(self, X):
        X = np.asarray(X)
        # Preallocate output: [E_T_miss, phi_Et_miss] + 18*(4) = 74
        N = X.shape[0]
        output = np.zeros((N, 74), dtype=np.float32)
        # Standardize missing ET and angle
        output[:, :2] = self.scaler_et.transform(X[:, :2])
        # For each object slot, transform kinematics or mask
        for i in range(self.n_objects):
            base_idx = 2 + i*5
            # Mask for present object
            obj_present = X[:, base_idx] > 0
            # 4 kinematic features: E, pT, eta, phi
            kins = np.zeros((N, 4), dtype=np.float32)
            if obj_present.sum() > 0:
                kins[obj_present] = self.scaler_objs.transform(X[obj_present, base_idx+1:base_idx+5])
            # Compose output feature range for this object
            out_start = 2+i*4
            out_end = out_start+4
            output[:, out_start:out_end] = kins
        return output

def make_preprocessor():
    return FourTopPreprocessor()

# Model Definition
class FourTopNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Empirically determined capacity for target < 50MB
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc_out = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.15)
    def forward(self, x):
        x = self.bn0(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.fc_out(x)
        return x.squeeze(-1)

def make_model(input_dim):
    return FourTopNet(input_dim)

EPOCHS = 15

def train_model(model, train_loader, val_loader, epochs):
    import torch.optim as optim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == yb.bool()).sum().item()
            total += xb.size(0)
        train_loss.append(running_loss / total)
        train_acc.append(correct / total)
        # Validation
        model.eval()
        v_loss, v_corr, v_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                out = model(xb)
                loss = criterion(out, yb)
                v_loss += loss.item() * xb.size(0)
                preds = torch.sigmoid(out) > 0.5
                v_corr += (preds == yb.bool()).sum().item()
                v_total += xb.size(0)
        val_loss.append(v_loss / v_total)
        val_acc.append(v_corr / v_total)
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

