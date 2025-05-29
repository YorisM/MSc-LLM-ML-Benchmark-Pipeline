
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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# --- Preprocessor ---
class LHCEventPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.mask_value = 0.0
        self.used_cols = None

    def fit(self, X, y=None):
        # Replace nan/inf values
        X = np.array(X)
        # Identify zero-mask (counts of zeros per column across all data)
        zero_mask = (X == 0.0).mean(axis=0) > 0.90  # very sparse columns
        # Always keep first two columns (E_T_miss and phi_ET_miss)
        mask = zero_mask.copy()
        mask[:2] = False
        # We'll drop columns where >90% is zero, excluding first two
        self.used_cols = np.where(~mask)[0]
        X_sel = X[:, self.used_cols]
        # Impute nan/inf to zero (should be rare)
        X_sel = np.where(np.isfinite(X_sel), X_sel, 0.0)
        # Impute any remaining nans/zeros for scaler to work
        self.imputer = SimpleImputer(strategy='mean')
        X_imputed = self.imputer.fit_transform(X_sel)
        self.scaler = StandardScaler()
        self.scaler.fit(X_imputed)
        return self

    def transform(self, X):
        X = np.array(X)
        X_sel = X[:, self.used_cols]
        X_sel = np.where(np.isfinite(X_sel), X_sel, 0.0)
        # Impute missing values to mean
        X_imp = self.imputer.transform(X_sel)
        X_out = self.scaler.transform(X_imp)
        # Return as numpy array
        return X_out

def make_preprocessor():
    return LHCEventPreprocessor()

# --- Model ---
class LHCBinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Larger hidden layers & dropout for regularization
        self.net = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(96, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(48, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def make_model(input_dim):
    return LHCBinaryClassifier(input_dim)

# --- Training Routine ---
epochs = 35

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # focal loss helps AUC/maximization, but BCEWithLogitsLoss is standard
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for e in range(epochs):
        model.train()
        losses, corrects, total = [], 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.float().to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            preds = (torch.sigmoid(logits) > 0.5).long()
            corrects += (preds == yb.long()).sum().item()
            total += yb.size(0)
            losses.append(loss.item())
        tloss = np.mean(losses)
        tacc = corrects / total
        train_loss.append(tloss)
        train_acc.append(tacc)

        model.eval()
        vlosses, vcorrects, vtotal = [], 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.float().to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                vlosses.append(loss.item())
                preds = (torch.sigmoid(logits) > 0.5).long()
                vcorrects += (preds == yb.long()).sum().item()
                vtotal += yb.size(0)
        vloss = np.mean(vlosses)
        vacc = vcorrects / vtotal if vtotal else 0.0
        val_loss.append(vloss)
        val_acc.append(vacc)
        lr_scheduler.step()
    return model, train_loss, val_loss, train_acc, val_acc
# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(Y_train, Y_val, name, out):
    plt.figure(); plt.plot(Y_train, label=f'Train {name}')
    plt.plot(Y_val, label=f'Validation {name}')
    plt.legend(); plt.title(name); plt.xlabel('epoch')
    plt.savefig(out); plt.close()        

def _run(dryrun=False):
    X_train, Y_train, X_val, Y_val = load_data()
    pre = make_preprocessor()
    pre.fit(X_train, Y_train)
    X_train = pre.transform(X_train);  X_val = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    model = make_model(input_dim=X_train.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    hist     = train_model(model, train_loader, val_loader, epochs=n_epochs)

    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        torch.save(model.state_dict(), f"{base}_state.pt")
        with open(f"{base}_pre.pkl", "wb") as f: pickle.dump(pre, f)
        _plot(hist['loss'], hist['val_loss'], 'Loss',     f"{base}_loss.png")
        _plot(hist['acc'],  hist['val_acc'],  'Accuracy', f"{base}_acc.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)
