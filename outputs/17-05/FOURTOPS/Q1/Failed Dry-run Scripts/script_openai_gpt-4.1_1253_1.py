
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

# --- PART 1: Preprocessor ---
class PhysicsPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Will hold means and stds for each feature
        self.means_ = None
        self.stds_ = None
        self.object_mask_ = None # Indicates which objects are present per event
    
    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        # Event has 18 objects, each with 5 features after the first 2 MET feats
        # Index 0: E_T_miss
        # Index 1: phi_MET
        # 2-92: objects (5 features per object)

        # Determine which objects are present (obj_n != 0)
        obj_ids = X_arr[:, 2:92:5]
        self.object_mask_ = (obj_ids != 0).astype(np.float32) # shape (N,18)
        # Mask out zero-padded objects for statistics
        mask = np.repeat(self.object_mask_, 5, axis=1) # shape (N,90)
        X_feats = X_arr[:,2:] # all obj feats (N,90)
        # For per-feature stats, flatten + mask
        X_obj_feats = X_feats[mask==1].reshape(-1, 5) # (num_obj_with_feat, 5)
        # Now get stats: first 2 entries are MET features (always present)
        feats_for_stats = np.concatenate([
            X_arr[:, [0,1]],
            X_obj_feats.reshape(-1,5)
        ], axis=0)
        self.means_ = np.nanmean(feats_for_stats, axis=0)
        self.stds_ = np.nanstd(feats_for_stats, axis=0) + 1e-6 # avoid 0-std
        return self
    def transform(self, X):
        X = np.asarray(X)
        X_new = np.copy(X)
        # Standardize: MET and all object features
        # Standardize MET
        X_new[:,0] = (X_new[:,0] - self.means_[0]) / self.stds_[0]
        X_new[:,1] = (X_new[:,1] - self.means_[1]) / self.stds_[1]
        # Take care for object features (shape: (N,90)), per-feature std/mean
        obj_means = self.means_[2:] if len(self.means_)>2 else np.zeros(5)
        obj_stds = self.stds_[2:] if len(self.stds_)>2 else np.ones(5)
        for o in range(18):
            i0 = 2 + o*5
            i1 = i0+5
            # mask: 0 if obj ID==0 => zero-pad; don't standardize further
            obj_id = X[:,i0]
            mask = (obj_id != 0)
            for k in range(5):
                idx = i0 + k
                # If std/mean vector too short: robust fallback
                m = obj_means[k] if len(obj_means)>k else 0
                s = obj_stds[k] if len(obj_stds)>k else 1
                X_new[mask,idx] = (X_new[mask,idx] - m)/s
        # Add mask feature per object (1 if present, 0 if padded)
        obj_mask = (X[:,2:92:5]!=0).astype(np.float32) # (N,18)
        X_out = np.concatenate([X_new, obj_mask], axis=1) # shape (N,110)
        return X_out

def make_preprocessor():
    return PhysicsPreprocessor()

# --- PART 2: Model ---
class FourTopClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # A moderately-sized MLP
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.12)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        x = torch.sigmoid(x.squeeze(-1))
        return x

def make_model(input_dim: int):
    return FourTopClassifier(input_dim)

# --- PART 3: Training Loop ---
EPOCHS = 16

def compute_binary_acc(y_pred, y_true):
    yhat = (y_pred >= 0.5).astype(np.int64)
    return np.mean(yhat == y_true)

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-5)
    criterion = nn.BCELoss()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for ep in range(epochs):
        model.train()
        train_loss_batch = []
        train_pred = []
        train_true = []
        for xb, yb in train_loader:
            xb = xb.float().to(device)
            yb = yb.float().to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss_batch.append(loss.item())
            train_pred.append(out.detach().cpu().numpy())
            train_true.append(yb.cpu().numpy())
        train_loss.append(np.mean(train_loss_batch))
        train_pred = np.concatenate(train_pred,0)
        train_true = np.concatenate(train_true,0)
        train_acc.append(compute_binary_acc(train_pred, train_true))
        # Validation
        model.eval()
        val_loss_batch = []
        val_pred = []
        val_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.float().to(device)
                yb = yb.float().to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss_batch.append(loss.item())
                val_pred.append(preds.cpu().numpy())
                val_true.append(yb.cpu().numpy())
        val_loss.append(np.mean(val_loss_batch))
        val_pred = np.concatenate(val_pred,0)
        val_true = np.concatenate(val_true,0)
        val_acc.append(compute_binary_acc(val_pred, val_true))
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

