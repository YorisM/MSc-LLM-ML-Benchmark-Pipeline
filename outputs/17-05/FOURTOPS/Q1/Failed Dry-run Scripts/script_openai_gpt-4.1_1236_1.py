
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------
# Preprocessor definition
# ------------------------
class EventPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Will populate in fit()
        self.feature_means_ = None
        self.feature_stds_ = None
        self.final_indices_ = None  # Save which features are not all-zero
        self.n_features_out_ = None
    
    def fit(self, X, y=None):
        X = np.asarray(X)

        # X shape: (N, 92); Layout per event (see template)
        #  - Indices 0: E_T_miss (float)
        #  - Index 1: phi_Et_miss
        #  - Indices 2-92: 18 objects × 5 cols ([id, E, pT, eta, phi]), possibly zero-padded

        # Ignore object ID for now, focus on kinematics (E, pt, eta, phi)
        # Extract per-object features
        objs = []
        for i in range(18):
            base = 2 + i * 5
            # id = X[:, base]   # Int, 0 if no object, else [1, ...]
            E   = X[:, base+1]
            pt  = X[:, base+2]
            eta = X[:, base+3]
            phi = X[:, base+4]
            objs.append(np.stack([E, pt, eta, phi], axis=1))
        objs = np.stack(objs, axis=1)  # (N, 18, 4)

        # Compose symmetric statistics per event (per column: all objects)
        # Features: E_Tmiss, phi_ETmiss, then global statistics per kinematic var
        features = [X[:, 0:2]]
        for j in range(4):
            feats_j = objs[:, :, j]  # (N, 18)

            # Mask valid objects (those with nonzero E)
            mask = objs[:, :, 0] > 0
            is_valid = mask.astype(float)

            # Compute statistics, avoid nan if all zeros
            sum_feat = np.sum(feats_j * is_valid, axis=1)
            count = np.sum(is_valid, axis=1)
            mean_feat = np.where(count > 0, sum_feat/count, 0)
            max_feat = np.where(count > 0, np.max(feats_j * is_valid, axis=1), 0)
            min_feat = np.where(count > 0, np.min(np.where(is_valid, feats_j, 1e10), axis=1), 0)
            std_feat = np.where(
                count > 0,
                np.sqrt(np.sum(((feats_j - mean_feat[:, None])**2)*is_valid, axis=1)/(count)),
                0)
            features.append(mean_feat[:, None])
            features.append(std_feat[:, None])
            features.append(max_feat[:, None])
            features.append(min_feat[:, None])
        # features: E_tmiss, phi, then for E,pt,eta,phi: mean,std,max,min => 4*4=16
        # Total: 2+16=18 features

        out_feats = np.concatenate(features, axis=1)

        # Save per-feature mean/std for subsequent normalization
        self.feature_means_ = out_feats.mean(axis=0)
        self.feature_stds_ = out_feats.std(axis=0) + 1e-6  # avoid divide by zero
        # Remove flat features (unlikely)
        self.final_indices_ = np.where(self.feature_stds_ > 1e-8)[0]
        self.n_features_out_ = len(self.final_indices_)
        return self

    def transform(self, X):
        X = np.asarray(X)
        objs = []
        for i in range(18):
            base = 2 + i * 5
            E   = X[:, base+1]
            pt  = X[:, base+2]
            eta = X[:, base+3]
            phi = X[:, base+4]
            objs.append(np.stack([E, pt, eta, phi], axis=1))
        objs = np.stack(objs, axis=1)
        features = [X[:, 0:2]]
        for j in range(4):
            feats_j = objs[:, :, j]
            mask = objs[:, :, 0] > 0
            is_valid = mask.astype(float)
            sum_feat = np.sum(feats_j * is_valid, axis=1)
            count = np.sum(is_valid, axis=1)
            mean_feat = np.where(count > 0, sum_feat/count, 0)
            max_feat = np.where(count > 0, np.max(feats_j * is_valid, axis=1), 0)
            min_feat = np.where(count > 0, np.min(np.where(is_valid, feats_j, 1e10), axis=1), 0)
            std_feat = np.where(
                count > 0,
                np.sqrt(np.sum(((feats_j - mean_feat[:, None])**2)*is_valid, axis=1)/(count)),
                0)
            features.append(mean_feat[:, None])
            features.append(std_feat[:, None])
            features.append(max_feat[:, None])
            features.append(min_feat[:, None])
        out_feats = np.concatenate(features, axis=1)
        x_norm = (out_feats - self.feature_means_) / self.feature_stds_
        x_final = x_norm[:, self.final_indices_]
        return x_final.astype(np.float32)

def make_preprocessor():
    return EventPreprocessor()

# ------------------------
# Model Definition
# ------------------------
class ParticleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Good for tabular: 3-4 dense layers, dropout, batchnorm, leakyrelu
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        x = self.network(x)
        return x.squeeze(-1)

def make_model(input_dim: int):
    return ParticleNet(input_dim)

# ------------------------
# Training Loop
# ------------------------
EPOCHS = 20  # Limited to reduce overfit+final runtime; can tweak up to 40 in practice

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0., 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.float().to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.shape[0]
            # Accuracy: threshold at 0.5
            preds = (torch.sigmoid(out) > 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total += xb.shape[0]
        tl = total_loss / total
        ta = correct / total
        train_loss.append(tl)
        train_acc.append(ta)
        # Validation
        model.eval()
        with torch.no_grad():
            vloss, vcorrect, vtotal = 0., 0, 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.float().to(device)
                out = model(xb)
                loss = criterion(out, yb)
                vloss += loss.item() * xb.shape[0]
                preds = (torch.sigmoid(out) > 0.5).long()
                vcorrect += (preds == yb.long()).sum().item()
                vtotal += xb.shape[0]
            val_l = vloss / vtotal
            val_a = vcorrect / vtotal
            val_loss.append(val_l)
            val_acc.append(val_a)
        # No printing per instructions
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

