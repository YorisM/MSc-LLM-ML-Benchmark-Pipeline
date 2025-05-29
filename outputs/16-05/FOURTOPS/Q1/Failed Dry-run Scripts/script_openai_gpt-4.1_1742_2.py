
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
from sklearn.preprocessing import StandardScaler

# ---- Preprocessing ----
class PhysicsEventPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Scaler for numerical features
        self.scaler = None

    def fit(self, X, y=None):
        # Convert to numpy if tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        features = []
        for row in X:
            event_feats = self.extract_features(row)
            features.append(event_feats)
        features = np.stack(features)
        self.scaler = StandardScaler().fit(features)
        return self

    def transform(self, X):
        X_orig = X
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        features = []
        for row in X:
            event_feats = self.extract_features(row)
            features.append(event_feats)
        features = np.stack(features)
        features = self.scaler.transform(features)
        return features.astype(np.float32)

    def extract_features(self, row):
        # Indices
        ETmiss = row[0]     # missing ET
        phi_ETmiss = row[1] # phi of missing ET
        # For 18 objects (zero-padded)
        obj_feats = []
        n_objects = 0
        ob_ids = []
        etas = []
        pts = []
        Es = []
        phis = []
        for i in range(18):
            base = 2 + i*5
            obj_id = row[base]
            E = row[base+1]
            pT = row[base+2]
            eta = row[base+3]
            phi = row[base+4]
            if (E>0) and (pT>0): # Only physical objects
                n_objects += 1
                ob_ids.append(obj_id)
                Es.append(E)
                pts.append(pT)
                etas.append(eta)
                phis.append(phi)
        # Simple event-level stats
        Es = np.array(Es) if len(Es)>0 else np.zeros(1)
        pts = np.array(pts) if len(pts)>0 else np.zeros(1)
        etas = np.array(etas) if len(etas)>0 else np.zeros(1)
        phis = np.array(phis) if len(phis)>0 else np.zeros(1)
        num_objs = n_objects
        HT = pts.sum()                  # Scalar sum of object pT
        M_ETmiss = ETmiss
        mean_eta = etas.mean() if n_objects else 0.
        std_eta = etas.std() if n_objects else 0.
        mean_phi = phis.mean() if n_objects else 0.
        std_phi = phis.std() if n_objects else 0.
        max_pT = pts.max() if n_objects else 0.
        min_pT = pts.min() if n_objects else 0.
        mean_pT = pts.mean() if n_objects else 0.
        std_pT = pts.std() if n_objects else 0.
        max_E = Es.max() if n_objects else 0.
        min_E = Es.min() if n_objects else 0.
        # Angular distance: calculate min deltaR between objects
        min_dR = 99.
        if n_objects>1:
            for j in range(n_objects):
                for k in range(j+1,n_objects):
                    deta = etas[j]-etas[k]
                    dphi = np.arctan2(np.sin(phis[j]-phis[k]), np.cos(phis[j]-phis[k]))
                    dR = np.sqrt(deta**2 + dphi**2)
                    if dR<min_dR: min_dR = dR
        else:
            min_dR=0.
        # Compose feature vector
        feats = [
            num_objs, # Num objects
            HT,       # Sum of transverse momentum
            M_ETmiss, # Metz
            phi_ETmiss,
            mean_eta,std_eta,
            mean_phi,std_phi,
            max_pT,min_pT,mean_pT,std_pT,
            max_E,min_E,
            min_dR
        ]
        # Add leading 4 objects' kinematics
        for j in range(4):
            if j < len(pts):
                feats += [Es[j],pts[j],etas[j],phis[j]]
            else:
                feats += [0.,0.,0.,0.]
        return np.array(feats, dtype=np.float32)

def make_preprocessor():
    return PhysicsEventPreprocessor()

# ---- Model ----
class ParticleClassifierNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # compact MLP
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    def forward(self, x):
        return self.layers(x).squeeze(-1)

def make_model(input_dim: int):
    return ParticleClassifierNet(input_dim)

# ---- Training ----
EPOCHS = 25

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int):
    import torch.optim as optim
    from sklearn.metrics import roc_auc_score
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-5, verbose=False)
    
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    bce_loss = nn.BCEWithLogitsLoss()
    best_val_auc = 0
    best_state = None

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        train_probs = []
        train_labels = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = bce_loss(out, yb.float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(out)>0.5).long()
            correct_train += (preds==yb).sum().item()
            total_train += xb.shape[0]
            train_probs.append(torch.sigmoid(out).detach().cpu().numpy())
            train_labels.append(yb.detach().cpu().numpy())
        train_loss.append(total_train_loss/total_train)
        train_acc.append(correct_train/total_train)
        # Compute train AUC as reference
        train_probs = np.concatenate(train_probs)
        train_labels = np.concatenate(train_labels)
        try:
            train_auc = roc_auc_score(train_labels, train_probs)
        except Exception:
            train_auc = 0.5
        # --- Validation ---
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = bce_loss(out, yb.float())
                total_val_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(out)>0.5).long()
                correct_val += (preds==yb).sum().item()
                total_val += xb.shape[0]
                val_probs.append(torch.sigmoid(out).cpu().numpy())
                val_labels.append(yb.cpu().numpy())
        val_loss.append(total_val_loss/total_val)
        val_acc.append(correct_val/total_val)
        val_probs = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except Exception:
            val_auc = 0.5
        # Learning rate step on val_auc
        scheduler.step(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
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

