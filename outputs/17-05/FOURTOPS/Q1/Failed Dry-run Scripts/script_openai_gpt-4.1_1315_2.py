
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

#------------------------
# Preprocessing
#------------------------

class PhysicsPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # State for normalization
        self.means = None
        self.stds = None
        self.n_obj = 18
        self.obj_slice = 5
        self.feature_mask = None
    def fit(self, X, y=None):
        X = np.asarray(X)
        # Compute mask for zero-padded objects (obj id==0 -> padded)
        obj_ids = X[:, 2:2+self.n_obj*self.obj_slice:self.obj_slice]
        self.feature_mask = obj_ids != 0
        # Mask for non-padding over events and objs
        mask = self.feature_mask
        # Scalar features: MET (mag,phi)
        scalars = X[:, :2]
        # Per-object physics features: E, pT, eta (ignore obj id and phi for simplicity)
        n_samples = X.shape[0]
        feats = []
        for i in range(self.n_obj):
            o = 2+i*self.obj_slice
            # If object not present, values are zeros
            E = X[:, o+1]
            pT = X[:, o+2]
            eta = X[:, o+3]
            phi = X[:, o+4]
            feats.append(np.stack([E, pT, eta, phi], axis=1))
        feats = np.stack(feats, axis=1)  # shape (N, n_obj, 4)
        # Compute statistics ignoring padded objects (obj_id==0)
        object_presence = self.feature_mask
        vals = feats[object_presence]
        # We'll keep minimal stats for normalization (E, pT, eta, phi)
        means = np.nanmean(vals, axis=0)
        stds = np.nanstd(vals, axis=0) + 1e-6
        # For MET:
        self.met_mean = np.mean(scalars, axis=0)
        self.met_std = np.std(scalars, axis=0) + 1e-6
        self.obj_means = np.mean(vals, axis=0)
        self.obj_stds = np.std(vals, axis=0) + 1e-6
        return self
    def transform(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_obj = self.n_obj
        slices = []
        # Normalize MET and phi_ET (first two indices)
        met = (X[:, :2] - self.met_mean) / self.met_std
        slices.append(met)
        # For each object, grab (E, pT, eta, phi), set mask for non-present
        obj_features = []
        obj_ids = X[:, 2:2+n_obj*self.obj_slice:self.obj_slice]
        for i in range(n_obj):
            o = 2+i*self.obj_slice
            ids = X[:, o]
            E = X[:, o+1]
            pT = X[:, o+2]
            eta = X[:, o+3]
            phi = X[:, o+4]
            # For padded objects: set features to zero after normalization
            mask = (ids != 0).astype(np.float32)
            E = ((E - self.obj_means[0])/self.obj_stds[0])*mask
            pT = ((pT - self.obj_means[1])/self.obj_stds[1])*mask
            eta = ((eta - self.obj_means[2])/self.obj_stds[2])*mask
            phi = ((phi - self.obj_means[3])/self.obj_stds[3])*mask
            obj_features.append(np.stack([E,pT,eta,phi], axis=1))
        obj_feats = np.stack(obj_features, axis=1)  # (N, n_obj, 4)
        obj_feats_flat = obj_feats.reshape(n_samples, -1)
        slices.append(obj_feats_flat)
        # Add object count as a feature
        obj_count = np.sum(obj_ids != 0, axis=1, keepdims=True)
        slices.append(obj_count / self.n_obj)
        # Aggregate features: sum, mean, max pT, E for objects
        # (use non-padded only)
        full_E = []
        full_pt = []
        for i in range(n_obj):
            o = 2+i*self.obj_slice
            ids = X[:, o]
            E = X[:, o+1]
            pT = X[:, o+2]
            mask = (ids != 0).astype(np.float32)
            full_E.append(E * mask)
            full_pt.append(pT * mask)
        full_E = np.stack(full_E, axis=1)
        full_pt = np.stack(full_pt, axis=1)
        sumE = np.sum(full_E, axis=1, keepdims=True)
        meanE = np.where(obj_count>0, sumE/(obj_count+1e-8), 0)
        maxE = np.max(full_E, axis=1, keepdims=True)
        sumpT = np.sum(full_pt, axis=1, keepdims=True)
        meanpT = np.where(obj_count>0, sumpT/(obj_count+1e-8), 0)
        maxpT = np.max(full_pt, axis=1, keepdims=True)
        slices.extend([sumE, meanE, maxE, sumpT, meanpT, maxpT])
        # Concatenate all features
        Xproc = np.concatenate(slices, axis=1)
        return Xproc.astype(np.float32)

def make_preprocessor():
    return PhysicsPreprocessor()

#------------------------
# Model Definition
#------------------------

class WideDeepNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Wide, shallow NN with dropout and batchnorm, for robustness
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.1)
        self.out = nn.Linear(32, 1)
    def forward(self, x):
        x = self.input_bn(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.out(x)
        return x.squeeze(-1)

def make_model(input_dim):
    return WideDeepNet(input_dim)

#-------------------------
# Training Loop
#-------------------------

EPOCHS = 12

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=False)
    loss_fn = nn.BCEWithLogitsLoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_val = 1e6
    for epoch in range(epochs):
        model.train()
        tr_loss, tr_corr, tr_tot = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()
            optimizer.zero_grad()
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred = (torch.sigmoid(logits)>0.5).long()
                tr_corr += (pred == yb.long()).sum().item()
                tr_tot += len(yb)
            tr_loss += loss.item()*Xb.size(0)
        tr_loss /= tr_tot
        train_loss.append(tr_loss)
        train_acc.append(tr_corr/tr_tot)
        # Validation
        model.eval()
        va_loss, va_corr, va_tot = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float()
                logits = model(Xb)
                loss = loss_fn(logits, yb)
                pred = (torch.sigmoid(logits)>0.5).long()
                va_corr += (pred == yb.long()).sum().item()
                va_tot += len(yb)
                va_loss += loss.item()*Xb.size(0)
        va_loss /= va_tot
        val_loss.append(va_loss)
        val_acc.append(va_corr/va_tot)
        scheduler.step(va_loss)
        # Model selection: Save best weights (not required here, as model returned after last epoch)
    model.cpu()
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

