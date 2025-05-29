
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

# ---- Preprocessor ----
class PhysPreprocessor:
    def __init__(self):
        self.scaler = None
        self.max_objs = 25  # (since 105 = 4x25 + 1x2 for MET)
        self.object_dim = 4
    def fit(self, X, y=None):
        # Compose physics-motivated features besides raw kinematics
        feats = self._physics_features(X)
        # Standardize
        self.scaler = StandardScaler().fit(feats)
        return self
    def transform(self, X):
        feats = self._physics_features(X)
        feats = self.scaler.transform(feats)
        return feats
    def _physics_features(self, X):
        # X: [N, 105]
        # Parse objects
        N = X.shape[0]
        # Extract MET
        MET = X[:, 0]  # [N]
        MET_phi = X[:, 1]  # [N]
        objects = X[:, 2:].reshape(N, self.max_objs, self.object_dim)
        obj_mask = (objects[:,:,0] != 0) | (objects[:,:,1] != 0)  # [N,25] mask: at least one nonzero in E, pt
        # Features per object (could augment)
        obj_E = objects[:,:,0]  # [N,25]
        obj_pT = objects[:,:,1]
        obj_eta = objects[:,:,2]
        obj_phi = objects[:,:,3]
        # Count number of objects
        n_objs = obj_mask.sum(axis=1, keepdims=True)  # [N,1]
        # Sum pT, E, calculate mean eta, etc
        sum_pT = np.sum(obj_pT * obj_mask, axis=1, keepdims=True)  # [N,1]
        sum_E = np.sum(obj_E * obj_mask, axis=1, keepdims=True)
        mean_eta = np.nan_to_num(np.sum(obj_eta * obj_mask, axis=1, keepdims=True) / np.maximum(1,n_objs))
        # max pT
        max_pT = np.max(obj_pT * obj_mask, axis=1, keepdims=True)
        # Leading object pT, eta, phi
        leading_idx = np.argmax(obj_pT * obj_mask, axis=1)  # [N]
        leading_pT = obj_pT[np.arange(N), leading_idx].reshape(-1,1)
        leading_eta = obj_eta[np.arange(N), leading_idx].reshape(-1,1)
        leading_phi = obj_phi[np.arange(N), leading_idx].reshape(-1,1)
        # Δφ(MET, leading obj)
        delta_phi = np.abs(MET_phi.reshape(-1,1) - leading_phi)
        delta_phi = np.minimum(delta_phi, 2*np.pi - delta_phi)
        # Compute invariant mass for 2-leading objects
        # NaN for events with <2 objects
        sec_idx = np.argsort(obj_pT * obj_mask, axis=1)[:,-2]  # [N]
        mask_2 = (n_objs[:,0] >= 2)
        E1 = obj_E[np.arange(N), leading_idx]
        pt1 = obj_pT[np.arange(N), leading_idx]
        eta1 = obj_eta[np.arange(N), leading_idx]
        phi1 = obj_phi[np.arange(N), leading_idx]
        E2 = obj_E[np.arange(N), sec_idx]
        pt2 = obj_pT[np.arange(N), sec_idx]
        eta2 = obj_eta[np.arange(N), sec_idx]
        phi2 = obj_phi[np.arange(N), sec_idx]
        # build 4-vectors
        mass12 = np.zeros(N)
        for i in range(N):
            if mask_2[i]:
                px1 = pt1[i] * np.cos(phi1[i])
                py1 = pt1[i] * np.sin(phi1[i])
                pz1 = pt1[i] * np.sinh(eta1[i])
                px2 = pt2[i] * np.cos(phi2[i])
                py2 = pt2[i] * np.sin(phi2[i])
                pz2 = pt2[i] * np.sinh(eta2[i])
                E_tot = E1[i] + E2[i]
                px_tot = px1 + px2
                py_tot = py1 + py2
                pz_tot = pz1 + pz2
                m2 = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
                mass12[i] = np.sqrt(m2) if m2>0 else 0.0
        mass12 = mass12.reshape(-1,1)
        # Return features [N,]
        extra = np.concatenate([
            MET.reshape(-1,1), MET_phi.reshape(-1,1), n_objs, sum_pT, sum_E,
            mean_eta, max_pT, leading_pT, leading_eta, leading_phi, delta_phi, mass12
        ], axis=1)
        # Append raw kinematic (flattened, zero padded)
        raw_flat = objects.reshape(N, -1)  # [N,100]
        feats = np.concatenate([extra, raw_flat], axis=1)
        return feats

def make_preprocessor():
    return PhysPreprocessor()

# --- Model ---
class PhysNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # AUC-optimizing: Wide non-linear network, BN, Dropout, residual
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.d1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.d3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.d4 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        x = self.bn0(x)
        x = F.relu(self.bn1(self.d1(x)))
        x = self.dropout(x)
        h = F.relu(self.bn2(self.d2(x)))
        x = x + h  # residual
        x = self.dropout(x)
        x = F.relu(self.bn3(self.d3(x)))
        x = self.dropout(x)
        x = F.relu(self.d4(x))
        x = self.out(x)
        return x.squeeze(-1)

def make_model(input_dim: int):
    return PhysNet(input_dim)

# --- Training Loop ---
epochs = 35  # Reasonable early stopping, large enough for convergence

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int):
    """
    Returns:
        trained_model (same inst),
        train_loss: list[float]
        val_loss: list[float]
        train_acc: list[float]
        val_acc: list[float]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_auc = -np.inf
    best_state = None
    from sklearn.metrics import roc_auc_score
    for ep in range(epochs):
        model.train()
        tlosses=[]
        tcorrect=0
        ttotal=0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()
            tlosses.append(loss.item()*len(xb))
            preds = torch.sigmoid(logits) > 0.5
            tcorrect += (preds == yb).sum().item()
            ttotal += len(xb)
        train_loss.append(sum(tlosses)/ttotal)
        train_acc.append(tcorrect/ttotal)
        # ---- Validation ----
        model.eval()
        vlosses=[]
        vcorrect=0
        vtotal=0
        v_logits = []
        v_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb.float())
                vlosses.append(loss.item()*len(xb))
                preds = torch.sigmoid(logits) > 0.5
                vcorrect += (preds == yb).sum().item()
                vtotal += len(xb)
                v_logits.append(logits.cpu().numpy())
                v_targets.append(yb.cpu().numpy())
        val_loss.append(sum(vlosses)/vtotal)
        val_acc.append(vcorrect/vtotal)
        # Compute AUC for early stopping / scheduling
        v_logits_np = np.concatenate(v_logits)
        v_targets_np = np.concatenate(v_targets)
        v_probs = 1/(1+np.exp(-v_logits_np))
        auc = roc_auc_score(v_targets_np, v_probs)
        scheduler.step(auc)
        # Store best params
        if auc > best_val_auc:
            best_val_auc = auc
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
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
