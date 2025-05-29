
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
from sklearn.preprocessing import StandardScaler

# =============== Preprocessor ======================
class PhysicsPreprocessor:
    """
    Preprocessor for variable-length physics events.
    Outputs an Nx29 numpy array:
      - Per-object stats for 18 objects (for all nonzero objects; zero for padded):
         For each of up to 18 objects:
           [is_present, E, pT, eta, phi]
      - Global features: E_T_miss, phi_Et_miss
      - Aggregate per-event engineered features:
         - n_objects (number of present objects)
         - sum_pT_objects
         - mean_eta_objects
         - std_eta_objects
         - min_dphi_ETmiss_obj (min dphi between ETmiss and objects)
         - max_pT_object
         - sum_E_objects
         - sum_Eta_objects (for possible weak forward-backward differences)
    Standardizes all features except binary mask and angle features (phi, dphi), which are sin/cos transformed.
    """
    def __init__(self):
        self.scaler = None
        self.obj_slices = [(2 + i*5, 2 + (i+1)*5) for i in range(18)]
        # Indices: 0=E_T_miss, 1=phiEtmiss, 2-6=object1, ...
        self.feat_indices = None    # Will hold slice indices for standardization
    def fit(self, X, y=None):
        # Compute mask for real objects (obj_n != 0 for real objects)
        n_samples = X.shape[0]
        n_obj = 18
        feats = []
        for i in range(n_samples):
            feats.append(self._compute_features(X[i]))
        feats = np.stack(feats)
        # Standardize all but binary and angles (sin/cos) columns (columns 1-4,5,...)
        self.feat_indices = [2,3,4,6,7,8,10,11,12,14,15,16,18,19,20,22,23,24,26,27,28]
        self.scaler = StandardScaler().fit(feats[:, self.feat_indices])
        return self
    def transform(self, X):
        n_samples = X.shape[0]
        feats = []
        for i in range(n_samples):
            feats.append(self._compute_features(X[i]))
        feats = np.stack(feats)
        feats_scaled = feats.copy()
        feats_scaled[:, self.feat_indices] = self.scaler.transform(feats[:, self.feat_indices])
        return feats_scaled.astype(np.float32)
    def _compute_features(self, event):
        # 0: E_T_miss, 1: phi_Et_miss
        E_T_miss = event[0]
        phi_Et_miss = event[1]
        # For each object
        objs = []
        n_obj = 18
        present_mask = []
        Es = []
        pTs = []
        etas = []
        phis = []
        for i in range(n_obj):
            s, e = 2 + i*5, 2 + (i+1)*5
            obj_id = event[s]
            if obj_id == 0.0:
                present_mask.append(0.0)
                Es.append(0.0)
                pTs.append(0.0)
                etas.append(0.0)
                phis.append(0.0)
            else:
                present_mask.append(1.0)
                Es.append(event[s+1])
                pTs.append(event[s+2])
                etas.append(event[s+3])
                phis.append(event[s+4])
        present_mask = np.array(present_mask)
        Es = np.array(Es)
        pTs = np.array(pTs)
        etas = np.array(etas)
        phis = np.array(phis)
        n_present = present_mask.sum()
        # Replace missing phis (where present_mask==0) by zero (safe for sin/cos)
        phis_x = np.cos(phis)
        phis_y = np.sin(phis)
        # For ETmiss phi
        phi_etmiss_x = np.cos(phi_Et_miss)
        phi_etmiss_y = np.sin(phi_Et_miss)
        # Object phi-ETmiss phi differences (for objects only)
        valid_idx = present_mask == 1
        dphis = np.angle(np.exp(1j*(phis[valid_idx] - phi_Et_miss))) if np.any(valid_idx) else np.array([0.])
        # Per-event engineered features
        sum_pT = pTs[valid_idx].sum() if np.any(valid_idx) else 0.
        mean_eta = etas[valid_idx].mean() if np.any(valid_idx) else 0.
        std_eta = etas[valid_idx].std() if np.any(valid_idx) else 0.
        min_dphi = np.abs(dphis).min() if np.any(valid_idx) else 0.
        max_pT = pTs[valid_idx].max() if np.any(valid_idx) else 0.
        sum_E = Es[valid_idx].sum() if np.any(valid_idx) else 0.
        sum_Eta = etas[valid_idx].sum() if np.any(valid_idx) else 0.
        # Feature vector: [E_T_miss, phi_Et_miss_x, phi_Et_miss_y, n_objects, sum_pT_objects, mean_eta_objects, std_eta_objects, min_dphi_ETmiss_obj, max_pT_object, sum_E_objects, sum_Eta_objects]
        features = [
            E_T_miss,                 # 0
            phi_etmiss_x,             # 1
            phi_etmiss_y,             # 2
            n_present,                # 3
            sum_pT,                   # 4
            mean_eta,                 # 5
            std_eta,                  # 6
            min_dphi,                 # 7
            max_pT,                   # 8
            sum_E,                    # 9
            sum_Eta                   #10
        ]
        return np.concatenate([
            features,                 # 11
            present_mask,             # 11:29
            Es,                       # 29:47
            pTs,                      # 47:65
            etas,                     # 65:83
            phis_x,                   # 83:101
            phis_y                    # 101:119
        ])[:29]  # Keep output shape, drop padded zeros if present

def make_preprocessor():
    return PhysicsPreprocessor()

# =============== Model =============================
class PhysicsNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Deep FC with wide start, batchnorm and dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def make_model(input_dim):
    model = PhysicsNet(input_dim)
    return model

# ================= Training ========================
EPOCHS = 15

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=3e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total += len(yb)
        avg_loss = total_loss / total
        train_loss.append(avg_loss)
        train_acc.append(correct/total)
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.float().to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                v_loss += loss.item() * len(yb)
                preds = (torch.sigmoid(logits) > 0.5).long()
                v_correct += (preds == yb.long()).sum().item()
                v_total += len(yb)
        val_loss.append(v_loss/v_total)
        val_acc.append(v_correct/v_total)
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

