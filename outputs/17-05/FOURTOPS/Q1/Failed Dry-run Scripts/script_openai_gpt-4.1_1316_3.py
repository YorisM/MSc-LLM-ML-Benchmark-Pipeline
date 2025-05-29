
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
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ---- Preprocessing ---- #
class ParticlePhysicsPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scalar_scaler = None
        self.obj_scaler = None
        self.n_objs = 18
        self.n_obj_feats = 5
        self.obj_pad_val = 0.0

    def fit(self, X, y=None):
        X = np.asarray(X)
        # Global features: E_T_miss (mag), phi_Et_miss
        self.scalar_scaler = StandardScaler().fit(X[:, :2])
        # object features: [obj_id, E, p_T, eta, phi]x18
        obj_feats = X[:, 2:].reshape(-1, self.n_objs, self.n_obj_feats)
        # Mask for non-padded objects (obj_id != 0)
        valid = (obj_feats[...,0] != 0)
        # Only fit to valid object entries
        all_obj_data = obj_feats[valid]
        self.obj_scaler = StandardScaler().fit(all_obj_data)
        return self

    def transform(self, X):
        X = np.asarray(X)
        N = X.shape[0]
        # 1. Scale E_Tmiss and phi_ETmiss
        X_scalar = self.scalar_scaler.transform(X[:, :2])
        # 2. Transform objects
        X_obj = X[:, 2:].reshape(N, self.n_objs, self.n_obj_feats)
        obj_valid = (X_obj[...,0] != 0)
        X_obj_scaled = X_obj.copy()
        # Only scale valid objects
        X_obj_scaled[obj_valid] = self.obj_scaler.transform(X_obj[obj_valid])
        # For padded objects (obj_id==0), set to 0
        X_obj_scaled[~obj_valid] = 0.0
        # Flatten again
        X_obj_scaled = X_obj_scaled.reshape(N, self.n_objs*self.n_obj_feats)
        X_proc = np.concatenate([X_scalar, X_obj_scaled], axis=1)
        return X_proc

def make_preprocessor():
    return ParticlePhysicsPreprocessor()

# ---- Model ---- #
class FourTopClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def make_model(input_dim: int):
    return FourTopClassifier(input_dim)

EPOCHS = 18

# ---- Training ---- #
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=4e-5)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_state = None
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        tot_loss, correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits)>0.5).long()
            correct += (preds==yb.long()).sum().item()
            n += xb.size(0)
        train_loss.append(tot_loss/n)
        train_acc.append(correct/n)

        # Validation
        model.eval()
        v_tot_loss, v_correct, v_n = 0.0, 0, 0
        with torch.no_grad():
            for xvb, yvb in val_loader:
                xvb, yvb = xvb.to(device), yvb.to(device).float()
                logits = model(xvb)
                loss = criterion(logits, yvb)
                v_tot_loss += loss.item() * xvb.size(0)
                preds = (torch.sigmoid(logits)>0.5).long()
                v_correct += (preds==yvb.long()).sum().item()
                v_n += xvb.size(0)
        val_loss.append(v_tot_loss/v_n)
        val_acc.append(v_correct/v_n)
        # Save best
        if val_loss[-1] < best_val_loss:
            best_val_loss = val_loss[-1]
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    if best_state:  # Restore best model
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

