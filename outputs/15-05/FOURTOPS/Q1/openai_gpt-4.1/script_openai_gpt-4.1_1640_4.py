
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
from sklearn.impute import SimpleImputer

# ---- Preprocessor ----
def make_preprocessor():
    class EventPreprocessor(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.scaler = None
            self.nonzero_mask = None
            self.imputer = None
        def fit(self, X, y=None):
            X = np.asarray(X)
            # First, mask padded (all zeros in an object)
            # objects start at column 2: (E_T_miss, phi_{E_t}_miss) + per object: [id, E, pT, eta, phi] repeated
            # 2 + 5*21 = 107 cols. There are 105 features, so 21 objects possible (some events have less)
            # Object indices: jump by 5 from index 2 up.
            N_obj = (X.shape[1] - 2)//5
            mask = np.zeros_like(X)
            mask[:, :2] = 1
            for i in range(N_obj):
                id_col = 2 + 5*i
                # If pT feature = 0, treat as padded. (objects w/ pT==0 are padding)
                mask[:, id_col:id_col+5] = (X[:, id_col+2] != 0)[:,None] # pT is at id_col+2
            self.nonzero_mask = mask.astype(bool)[0] # all events same pattern

            # For each feature, compute median for imputation
            self.imputer = SimpleImputer(strategy='median') # Replace zeros w/ median
            X_unpad = np.where(mask, X, np.nan)
            self.imputer.fit(X_unpad)

            # StandardScaler on unpadded entries
            X_imp = self.imputer.transform(X_unpad)
            self.scaler = StandardScaler()
            self.scaler.fit(X_imp)
            return self
        def transform(self, X):
            X = np.asarray(X)
            mask = np.zeros_like(X)
            mask[:, :2] = 1
            N_obj = (X.shape[1] - 2)//5
            for i in range(N_obj):
                id_col = 2 + 5*i
                mask[:, id_col:id_col+5] = (X[:, id_col+2] != 0)[:,None]   # pT==0 => padding
            # Impute
            X_unpad = np.where(mask, X, np.nan)
            X_imp = self.imputer.transform(X_unpad)
            # Standardize
            X_scaled = self.scaler.transform(X_imp)
            # Optionally add number of objects and sum E_T (useful high-level features)
            n_obj = ((X[:,2::5][:,::1]!=0).sum(axis=1)).reshape(-1,1)
            sum_e = np.nansum(X[:,3::5], axis=1, keepdims=True)
            out = np.concatenate([X_scaled, n_obj, sum_e], axis=1)
            return out

    return EventPreprocessor()

# ---- Model ----
def make_model(input_dim: int):
    class FourTopClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.bn0 = nn.BatchNorm1d(input_dim)
            self.fc1 = nn.Linear(input_dim, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, 64)
            self.bn3 = nn.BatchNorm1d(64)
            self.fc4 = nn.Linear(64, 1)  # Output: logit
            self.dropout = nn.Dropout(0.25)
        def forward(self, x):
            x = self.bn0(x)
            x = F.gelu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.gelu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = F.gelu(self.bn3(self.fc3(x)))
            x = self.dropout(x)
            x = self.fc4(x)
            return x.squeeze(-1)
    return FourTopClassifier(input_dim)

# ---- Training Loop ----
epochs = 25

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int):
    import torch.optim as optim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * xb.shape[0]
            y_pred = (torch.sigmoid(logits) > 0.5).long()
            t_correct += (y_pred == yb.long()).sum().item()
            t_total += xb.shape[0]
        train_loss.append(t_loss / t_total)
        train_acc.append(t_correct / t_total)

        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                yb = yb.to(device).float()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                v_loss += loss.item() * xb.shape[0]
                y_pred = (torch.sigmoid(logits) > 0.5).long()
                v_correct += (y_pred == yb.long()).sum().item()
                v_total += xb.shape[0]
        val_loss.append(v_loss / v_total)
        val_acc.append(v_correct / v_total)

    trained_model = model
    return trained_model, train_loss, val_loss, train_acc, val_acc
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
