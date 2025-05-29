
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

# ----- Preprocessor -----
class PhysicsEventPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, max_objs=25):
        # (E_T_miss, phi_E_t_miss), followed by (obj_i, E_i, p_Ti, eta_i, phi_i)*25
        self.max_objs = max_objs
        self.scaler = StandardScaler()
        self.epsilon = 1e-8
        # Indices for easier feature engineering:
        self.ET_miss_idx = 0
        self.phi_ET_miss_idx = 1
        self.offset = 2
        self.obj_feats = 5 # (obj_type, E, p_T, eta, phi)
        # Derived feature storage
        self.obj_type_indices = [self.offset + self.obj_feats*i for i in range(max_objs)]
        self.obj_energy_indices = [self.offset + self.obj_feats*i + 1 for i in range(max_objs)]
        self.obj_pt_indices = [self.offset + self.obj_feats*i + 2 for i in range(max_objs)]
        self.obj_eta_indices = [self.offset + self.obj_feats*i + 3 for i in range(max_objs)]
        self.obj_phi_indices = [self.offset + self.obj_feats*i + 4 for i in range(max_objs)]
    def _extract_objs(self, X):
        # X shape: (N, num_feats)
        objs_type = X[:, self.obj_type_indices]
        objs_E = X[:, self.obj_energy_indices]
        objs_pt = X[:, self.obj_pt_indices]
        objs_eta = X[:, self.obj_eta_indices]
        objs_phi = X[:, self.obj_phi_indices]
        return objs_type, objs_E, objs_pt, objs_eta, objs_phi
    def fit(self, X, y=None):
        features = self._featurize(X)
        self.scaler.fit(features)
        return self
    def transform(self, X):
        features = self._featurize(X)
        return self.scaler.transform(features)
    def _featurize(self, X):
        # Derived physics features for each event
        objs_type, objs_E, objs_pt, objs_eta, objs_phi = self._extract_objs(X)
        # Mask for real objects (assuming padding => obj_type==0)
        mask = (objs_type > 0)
        objs_pt_masked = objs_pt * mask
        objs_eta_masked = objs_eta * mask
        objs_phi_masked = objs_phi * mask
        objs_E_masked = objs_E * mask
        n_objs = mask.sum(axis=1, keepdims=True)
        # Jets/leptons identification (minimal)
        # Types are assumed to map to PIDs (to distinguish jets vs leptons), if available
        # For simplicity, use all objects for now
        sum_pt = objs_pt_masked.sum(axis=1, keepdims=True)
        max_pt = objs_pt_masked.max(axis=1, keepdims=True)
        mean_eta = np.where(mask, objs_eta_masked, 0).sum(axis=1, keepdims=True) / (n_objs+self.epsilon)
        std_eta = np.sqrt((np.square(objs_eta_masked - mean_eta) * mask).sum(axis=1, keepdims=True) / (n_objs+self.epsilon))
        sum_E = objs_E_masked.sum(axis=1, keepdims=True)
        # Leading-subleading PT
        sorted_pt = np.sort(objs_pt_masked, axis=1)[:,::-1] # descending
        leading_pt = sorted_pt[:,0:1]
        subleading_pt = sorted_pt[:,1:2]
        # HT (scalar sum of jet/lepton pT)
        HT = sum_pt
        # MET (E_T_miss)
        MET = X[:, self.ET_miss_idx:self.ET_miss_idx+1]
        # MT (transverse mass, estimate via leading object)
        leading_obj_phi = np.take_along_axis(objs_phi, np.argmax(objs_pt_masked,axis=1)[:,None], 1)
        delta_phi = np.mod(MET - leading_obj_phi, 2*np.pi)
        MT = np.sqrt(2 * MET * leading_pt * (1 - np.cos(delta_phi)))
        # Object multiplicities by type
        # (Assume: type 1 = b-jet, type 2/3/4 = leptons,... Unknown mapping: so use total, unique)
        unique_obj_types = (objs_type * mask)
        n_unique_types = (unique_obj_types > 0).sum(axis=1, keepdims=True)
        features = np.concatenate([
            n_objs, n_unique_types, sum_pt, max_pt, leading_pt, subleading_pt,
            mean_eta, std_eta, sum_E, HT, MET, MT
        ], axis=1)
        # Add global event variables
        global_feats = X[:, :self.offset] # E_T_miss, phi_ET_miss
        features = np.concatenate([features, global_feats], axis=1)
        return features.astype(np.float32)

def make_preprocessor():
    # Use max_objs = (105 - 2) // 5
    return PhysicsEventPreprocessor(max_objs=21)

# ----- Model Architecture -----
class ParticleNet(nn.Module):
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
    return ParticleNet(input_dim)

# ----- Training -----
def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int):
    import torch.optim as optim
    from sklearn.metrics import accuracy_score
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        tr_loss, tr_preds, tr_targets = 0.0, [], []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
            tr_preds.append((torch.sigmoid(out).detach().cpu().numpy() > 0.5).astype(np.int64))
            tr_targets.append(yb.cpu().numpy())
        scheduler.step()
        tr_loss = tr_loss / len(train_loader.dataset)
        tr_preds = np.concatenate(tr_preds)
        tr_targets = np.concatenate(tr_targets)
        tr_acc = accuracy_score(tr_targets, tr_preds)
        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        # Validation
        model.eval()
        val_loss_epoch = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.float().to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss_epoch += loss.item() * xb.size(0)
                val_preds.append((torch.sigmoid(out).cpu().numpy() > 0.5).astype(np.int64))
                val_targets.append(yb.cpu().numpy())
        val_loss_epoch = val_loss_epoch / len(val_loader.dataset)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_acc_epoch = accuracy_score(val_targets, val_preds)
        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)
    return model, train_loss, val_loss, train_acc, val_acc

# Training for 25 epochs (enough for convergence/no overfit well regularized)
epochs = 25
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
