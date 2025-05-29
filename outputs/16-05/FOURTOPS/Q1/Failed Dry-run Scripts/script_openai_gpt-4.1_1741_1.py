
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

### --- Preprocessing --- ###

class FourTopPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # We'll use StandardScaler for E_T_miss and object kinematics (log for E, pT)
        self.meta_scaler = StandardScaler()                    # [E_T_miss, phi_Et_miss]
        self.kin_scaler = StandardScaler()                     # Will hold log(E), log(pT), eta, phi for all objects
        self.n_objects = 18                                   # Maximum number of objects per event according to padding
        self.object_slice_len = 5                             # [obj_id, E, pT, eta, phi]
    def fit(self, X, y=None):
        X = np.asarray(X)
        # 1. Fit scaler for [E_T_miss, phi_Et_miss]
        self.meta_scaler.fit(X[:, 0:2])
        # 2. For object features, ignore zero-padded objects: mask object indices
        obj_ids = X[:, 2::5]
        not_pad = (obj_ids > 0)  # Padding has obj_id=0
        kins = []
        for evt in range(X.shape[0]):
            evt_obj_mask = not_pad[evt]
            if np.any(evt_obj_mask):
                # Stack log(E), log(pT), eta, phi for each present obj in event
                E = X[evt, 3::5][evt_obj_mask]
                pT = X[evt, 4::5][evt_obj_mask]
                eta = X[evt, 5::5][evt_obj_mask]
                phi = X[evt, 6::5][evt_obj_mask]
                # log1p for E, pT to stabilize
                e_ = np.log1p(np.clip(E, 0, None))
                pt_ = np.log1p(np.clip(pT, 0, None))
                kin_stack = np.stack([e_, pt_, eta, phi], axis=1)
                kins.append(kin_stack)
        if kins:
            kin_concat = np.concatenate(kins, axis=0)
            self.kin_scaler.fit(kin_concat)
        return self
    def transform(self, X):
        X = np.asarray(X)
        out = []
        for evt in range(X.shape[0]):
            # 1. Process event-level features [E_T_miss, phi_Et_miss]
            meta = self.meta_scaler.transform(X[evt, 0:2].reshape(1, -1))[0]
            # 2. For each object, encode features (handle padding, log, scaling)
            obj_ids = X[evt, 2::5]     # length=n_objects
            E = X[evt, 3::5]
            pT = X[evt, 4::5]
            eta = X[evt, 5::5]
            phi = X[evt, 6::5]
            # For each object, build 5 features: obj_id_one-hot(7), scaled log(E), log(pT), eta, phi
            # Assume valid obj_ids: 1..6 (jet, muon, electron, bjet, photon, tau) or similar, else 0=padded
            max_objid = int(obj_ids.max())
            oh_dim = min(max(7, int(np.max(obj_ids)+1)), 7)  # just to be safe
            # Compose one-hot obj type
            obj_oh = np.zeros((self.n_objects, oh_dim), dtype=float)
            present = (obj_ids > 0)  # 0 is padding
            obj_indices = obj_ids[present].astype(int)
            for idx, oo in enumerate(obj_indices):
                if oo < oh_dim:
                    obj_oh[idx, int(oo)] = 1.0
            # Gather kinematics, fill with zeros for padding
            kin = np.zeros((self.n_objects, 4), dtype=float) # [log(E), log(pT), eta, phi]
            if np.any(present):
                e_ = np.log1p(np.clip(E[present], 0, None))
                pt_ = np.log1p(np.clip(pT[present], 0, None))
                eta_ = eta[present]
                phi_ = phi[present]
                kin_stack = np.stack([e_, pt_, eta_, phi_], axis=1)
                kin[present] = self.kin_scaler.transform(kin_stack)
            # Flatten object features per event
            obj_feats = np.concatenate([obj_oh.reshape(-1), kin.reshape(-1)])
            # Final vector: [meta (2)] + [obj features]
            out.append(np.concatenate([meta, obj_feats]))
        return np.stack(out, axis=0)

def make_preprocessor():
    return FourTopPreprocessor()

### --- Model --- ###

# Input shape: [2 + n_objects*(obj_onehot_dim + 4*kin)] ~ 2 + 18*11 = 200
class FourTopNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Shallow but wide MLP with batchnorm and dropout
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.15)
        self.out = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x.squeeze(-1)

def make_model(input_dim):
    return FourTopNet(input_dim)

EPOCHS = 16

### --- Training Loop --- ###

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    for ep in range(epochs):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb.long()).sum().item()
            n += xb.size(0)
        train_loss.append(total_loss / n)
        train_acc.append(correct / n)
        # Validation
        model.eval()
        v_loss, v_n, v_correct = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                logits = model(xb)
                loss = criterion(logits, yb)
                v_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).long()
                v_correct += (preds == yb.long()).sum().item()
                v_n += xb.size(0)
        val_loss.append(v_loss / v_n)
        val_acc.append(v_correct / v_n)
        # Optional: print(f'Epoch {ep+1}/{epochs} train_loss={train_loss[-1]:.4f} val_loss={val_loss[-1]:.4f} train_acc={train_acc[-1]:.4f} val_acc={val_acc[-1]:.4f}')
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

