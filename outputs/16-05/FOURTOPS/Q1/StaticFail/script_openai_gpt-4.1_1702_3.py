
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import softmax

# Preprocessor with physics-motivated and data-based features
def make_preprocessor():
    class PhysPreprocessor(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.scaler = None
            self.mask = None  # Mask for nonzero objects
            self.feature_indices = None  # To track selected features and columns

        def fit(self, X, y=None):
            # Identify which object slots are nonzero in the dataset (for zero-padding ignoring)
            # X : [n, 92]
            X = np.asarray(X)
            n_obj = 18
            obj_feats = []
            for i in range(n_obj):
                idx = 2 + i * 5
                obj_type = X[:, idx]
                mask = obj_type != 0  # nonzero => object is present
                obj_feats.append(mask)
            # Make a mask matrix for existing objects per sample
            self.mask = np.stack(obj_feats, axis=1)  # [n_samples,n_obj]
            # We'll build physics-motivated features, plus scaled raw features
            # Build feature indices (positional)
            features = []
            # 1. E_T_miss, phi
            features.append([0, 1])
            # 2. Sum over p_Ts, total objects, b-tag fraction, multiplicities
            # We'll just track all object energy, p_T, eta, phi
            obj_index_block = []
            for i in range(n_obj):
                base = 2 + i * 5
                obj_index_block.extend([base, base+1, base+2, base+3, base+4])
            self.feature_indices = features[0] + obj_index_block
            # For scaling: only on present objects; but will do on all features.
            feats_to_scale = []
            # Missing ET, miss phi
            feats_to_scale.append(X[:, 0])      # E_T_miss
            feats_to_scale.append(X[:, 1])      # phi_Et_miss
            for i in range(n_obj):
                base = 2 + i*5
                feats_to_scale.append(X[:,base])     # obj type
                feats_to_scale.append(X[:,base+1])   # E
                feats_to_scale.append(X[:,base+2])   # p_T
                feats_to_scale.append(X[:,base+3])   # eta
                feats_to_scale.append(X[:,base+4])   # phi
            feats_to_scale = np.stack(feats_to_scale, axis=1)  # [n, feats]
            self.scaler = StandardScaler().fit(feats_to_scale)
            return self

        def transform(self, X):
            X = np.asarray(X)
            n = len(X)
            n_obj = 18
            res = []
            # Add missing E_T and phi
            et_miss = X[:,0]  # [n]
            phi_miss = X[:,1]
            # Rescale all numerical features
            feats_to_scale = []
            for i in range(n_obj):
                base = 2 + i*5
                feats_to_scale.append(X[:,base])     # obj type
                feats_to_scale.append(X[:,base+1])   # E
                feats_to_scale.append(X[:,base+2])   # p_T
                feats_to_scale.append(X[:,base+3])   # eta
                feats_to_scale.append(X[:,base+4])   # phi
            feats_to_scale = np.stack([et_miss, phi_miss]+feats_to_scale, axis=1)
            Xscaled = self.scaler.transform(feats_to_scale)
            # Optionally, construct aggregate physics features
            # For each event: get all object types, p_T, eta, phi, mask out zero objects
            all_obj_types = np.zeros((n, n_obj))
            all_pt = np.zeros((n, n_obj))
            all_eta = np.zeros((n, n_obj))
            all_phi = np.zeros((n, n_obj))
            all_E = np.zeros((n, n_obj))
            for i in range(n_obj):
                base = 2 + i*5
                obj_type = X[:,base]
                E = X[:,base+1]
                pt = X[:,base+2]
                eta = X[:,base+3]
                phi = X[:,base+4]
                # Presence mask: nonzero obj_type
                mask = obj_type != 0
                all_obj_types[:,i] = obj_type
                all_E[:,i] = E
                all_pt[:,i] = pt
                all_eta[:,i] = eta
                all_phi[:,i] = phi
            # Object multiplicity (number of nonzero type objects)
            n_obj_present = (all_obj_types != 0).sum(axis=1, keepdims=True)
            # Sum pTs, sum Es
            sum_pt = (all_pt * (all_obj_types!=0)).sum(axis=1, keepdims=True)
            sum_E = (all_E * (all_obj_types!=0)).sum(axis=1, keepdims=True)
            max_pt = (all_pt * (all_obj_types!=0)).max(axis=1, keepdims=True)
            max_E = (all_E * (all_obj_types!=0)).max(axis=1, keepdims=True)
            # Eta (mean abs)
            mean_eta = np.where((all_obj_types!=0), all_eta, np.nan)
            mean_eta = np.nanmean(np.abs(mean_eta), axis=1, keepdims=True)
            # DeltaR between leading pT objects
            sort_pt_idx = np.flip(np.argsort(all_pt, axis=1),axis=1)  # [n, n_obj]
            leading_idx = sort_pt_idx[:,0]  # index of leading object per event
            subleading_idx = sort_pt_idx[:,1] # 2nd leading
            # Build dEta, dPhi between top two
            dEta = all_eta[np.arange(n),leading_idx] - all_eta[np.arange(n),subleading_idx]
            dPhi = all_phi[np.arange(n),leading_idx] - all_phi[np.arange(n),subleading_idx]
            dPhi = (dPhi+np.pi)%(2*np.pi)-np.pi
            deltaR = np.sqrt(dEta**2 + dPhi**2)
            deltaR = np.expand_dims(deltaR,1)
            # concat
            feat_agg = np.concatenate([n_obj_present, sum_pt, sum_E, max_pt, max_E, mean_eta, deltaR], axis=1)
            # Final features: [Stats] + [Standardized raw]
            out = np.concatenate([feat_agg, Xscaled], axis=1)
            return out.astype(np.float32)

    preproc = PhysPreprocessor()
    return preproc

# Model: Feedforward, batchnorm, dropout, shallow to not overfit
def make_model(input_dim):
    class SmallClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(p=0.17),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.17),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32,1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)  # [N]
    return SmallClassifier(input_dim)

EPOCHS = 22

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=6,gamma=0.5)
    # Use BCEWithLogitsLoss for numerics
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == (yb>0.5)).sum().item()
            total += len(xb)
        train_loss.append(total_loss/total)
        train_acc.append(correct/total)
        # Validation
        model.eval()
        val_total_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True).float()
                yb = yb.to(device, non_blocking=True).float()
                logits = model(xb)
                loss = criterion(logits, yb)
                val_total_loss += loss.item()*len(xb)
                preds = torch.sigmoid(logits) > 0.5
                val_correct += (preds == (yb>0.5)).sum().item()
                val_total += len(xb)
        val_loss.append(val_total_loss/val_total)
        val_acc.append(val_correct/val_total)
        scheduler.step()
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

