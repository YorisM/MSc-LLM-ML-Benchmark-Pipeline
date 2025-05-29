
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
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, TransformerMixin

class PhysicsPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.means = None
        self.stds = None
        self.feature_mask = None
        self.use_obj_types = False  # Option: set to True to use encoded obj_n
        
    def fit(self, X, y=None):
        # X: numpy array (N, 92)
        X = X.copy()

        # Remove object type indices (obj_n) for per-object features.
        # They're at indices 2,7,12,... (all 5*i+2: i=0..17), i.e., [2, 7, ... 87]
        obj_idx = np.array([2+5*i for i in range(18)])
        if not self.use_obj_types:
            self.feature_mask = np.ones(92, dtype=bool)
            self.feature_mask[obj_idx] = False
        else:
            # If using obj-n, one-hot encode integers 1-6, replace in mask later
            raise NotImplementedError()

        Xf = X[:, self.feature_mask]

        # Zero padding leaves feature rows of all zero; ignore in mean/std calculation
        nonzero_row_mask = Xf.any(axis=1)
        Xnz = Xf[nonzero_row_mask]
        
        self.means = Xnz.mean(axis=0)
        self.stds = Xnz.std(axis=0)
        self.stds[self.stds==0] = 1. # Avoid div0
        return self

    def transform(self, X):
        # Standard scale and mask out object type indices
        X = X.copy()
        X = X[:, self.feature_mask]
        X = (X - self.means) / self.stds
        # Outlier clipping (3 sigma)
        X = np.clip(X, -3.0, 3.0)
        return X.astype(np.float32)

def make_preprocessor():
    return PhysicsPreprocessor()

class ParticleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # moderately deep MLP with regularization
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.block3 = nn.Sequential(
            nn.Linear(96, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x.squeeze(-1)

def make_model(input_dim: int):
    return ParticleMLP(input_dim)

EPOCHS = 20

# --- TRAIN FUNCTION

def compute_accuracy(logits, targets):
    preds = (torch.sigmoid(logits) >= 0.5).long()
    acc = (preds == targets).float().mean().item()
    return acc

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5, verbose=False)
    criterion = nn.BCEWithLogitsLoss()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.
        running_acc = 0.
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()
            batch_acc = compute_accuracy(logits.detach(), yb)
            running_loss += loss.item()*len(xb)
            running_acc += batch_acc*len(xb)
            total += len(xb)
        train_loss.append(running_loss/total)
        train_acc.append(running_acc/total)
        # Validation
        model.eval()
        v_loss = 0.
        v_acc = 0.
        v_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb.float())
                batch_acc = compute_accuracy(logits, yb)
                v_loss += loss.item()*len(xb)
                v_acc += batch_acc*len(xb)
                v_total += len(xb)
        val_loss.append(v_loss/v_total)
        val_acc.append(v_acc/v_total)
        scheduler.step(val_acc[-1])
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

