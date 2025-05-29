
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
from sklearn.preprocessing import StandardScaler

# ------- Preprocessor
class PhysicsPreprocessor:
    def __init__(self):
        self.scaler = None
        self.n_obj = (105 - 2) // 5
    def fit(self, X, y=None):
        mask = self._build_object_mask(X)
        # mask: (N, n_obj), True if the object is present
        # For each 5-feature group, only fit scaler on non-masked data per feature.
        physical_feats = []
        for obj in range(self.n_obj):
            start = 2 + obj * 5
            end = start + 5
            obj_mask = mask[:, obj]
            # shape (n_obj, 5)
            feats = X[obj_mask, start:end]
            if len(feats) > 0:
                physical_feats.append(feats)
        if physical_feats:
            physical_feats = np.vstack(physical_feats)
        else:
            physical_feats = np.empty((0,5))
        # Add global features (E_T_miss, phi_{E_T}_miss)
        X_global = X[:, :2]
        feats_all = np.hstack([
            X_global,
            physical_feats if physical_feats.shape[0]>0 else np.zeros((X.shape[0],5))  # avoids empty input
        ])
        self.scaler = StandardScaler()
        self.scaler.fit(feats_all)
        return self
    def _build_object_mask(self, X):
        n_obj = self.n_obj
        n_events = X.shape[0]
        mask = np.ones((n_events, n_obj), dtype=bool)
        # Assumption: if E_n == 0 and pT_n == 0, object missing
        for obj in range(n_obj):
            start = 2 + obj*5
            E = X[:, start+1]
            pT = X[:, start+2]
            mask[:,obj] = (E!=0) | (pT!=0)
        return mask
    def transform(self, X):
        Xnew = np.zeros((X.shape[0], 2+8*3+4))  # 2 global, 8*3 jets/leptons, 4 global sums: HT, n_objs, n_jets, n_leps
        # 0:1 E_T_miss, phi_et
        Xnew[:,:2] = X[:,:2]
        # object features
        n_obj = self.n_obj
        # aggregate features
        HT = np.zeros(X.shape[0])
        n_jets = np.zeros(X.shape[0])
        n_leps = np.zeros(X.shape[0])
        n_obj_count = np.zeros(X.shape[0])
        lept_mask = [0,1,2]  # example: obj_id for electrons, muons (could change)
        jet_mask = [3,4,5,6,7,8,9,10] # the rest, for this template
        obj_feats_acc = np.zeros((X.shape[0],8,3))
        for obj in range(n_obj):
            start = 2 + obj*5
            obj_id = np.round(X[:, start])
            feats = X[:,start+1:start+4]  # E, pT, eta
            mask = (X[:, start+1]!=0) | (X[:, start+2]!=0)
            idx = obj
            for idx_row in range(X.shape[0]):
                if not mask[idx_row]: continue
                # Up to 8 non-zero objects saved
                if obj < 8:
                    obj_feats_acc[idx_row,obj,:] = feats[idx_row]
                HT[idx_row] += X[idx_row,start+2]  # pT
                n_obj_count[idx_row] += 1
                # obj_id grouping example (adapt for realistic data)
                if obj_id[idx_row] in lept_mask:
                    n_leps[idx_row] += 1
                elif obj_id[idx_row] in jet_mask:
                    n_jets[idx_row] += 1
        # flatten first 8 objects
        Xnew[:,2:2+8*3] = obj_feats_acc.reshape(X.shape[0],-1)
        # aggregate
        Xnew[:,-4] = HT
        Xnew[:,-3] = n_obj_count
        Xnew[:,-2] = n_jets
        Xnew[:,-1] = n_leps
        # Scaling
        Xnew = self.scaler.transform(Xnew)
        return Xnew

def make_preprocessor():
    return PhysicsPreprocessor()

# --- Model: attention over variable-length (use pooling, 1/2 layers)
class ParticleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.dropout1 = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.10)
        self.out = nn.Linear(32, 1)
    def forward(self, x):
        # x: (batch, features)
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x.squeeze(-1)

def make_model(input_dim: int):
    return ParticleNet(input_dim)

# Typical value for tabular, medium-size: adjust higher for more capacity!
epochs = 30


def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for ep in range(epochs):
        model.train()
        running_loss, corr, total = 0, 0, 0
        for xb, yb in train_loader:
            xb = xb.float().to(device)
            yb = yb.float().to(device)
            optimizer.zero_grad()
            yh = model(xb)
            loss = criterion(yh, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            with torch.no_grad():
                preds = (torch.sigmoid(yh)>0.5).long()
                corr += (preds==yb.long()).sum().item()
                total += xb.size(0)
        train_loss.append(running_loss/total)
        train_acc.append(corr/total)
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss_epoch, corr, total = 0, 0, 0
            for xb, yb in val_loader:
                xb = xb.float().to(device)
                yb = yb.float().to(device)
                yh = model(xb)
                loss = criterion(yh, yb)
                val_loss_epoch += loss.item()*xb.size(0)
                preds = (torch.sigmoid(yh)>0.5).long()
                corr += (preds==yb.long()).sum().item()
                total += xb.size(0)
            val_loss.append(val_loss_epoch/total)
            val_acc.append(corr/total)
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
