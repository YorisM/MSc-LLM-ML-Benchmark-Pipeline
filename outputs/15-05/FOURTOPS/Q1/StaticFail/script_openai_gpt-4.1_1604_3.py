
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
from sklearn.preprocessing import StandardScaler, RobustScaler

# --- Preprocessor definition ---
def make_preprocessor():
    class HighEnergyPreprocessor(BaseEstimator, TransformerMixin):
        def __init__(self):
            # Holds the scalers for each category of feature
            self.scalar_etmiss = None
            self.scalar_phi_etmiss = None
            self.object_scalers = None  # scaler for E, pt, eta, phi globally
            self.n_objects = (105 - 2) // 5
            self.zero_object_marker = 0.0 # likely in 'obj_n'; where not present, is zero
        
        def fit(self, X, y=None):
            X = np.asarray(X)
            # E_T_miss / phi_{E_T}_miss
            et_col = X[:, 0:1]
            phi_col = X[:, 1:2]
            self.scalar_etmiss = RobustScaler().fit(et_col)
            self.scalar_phi_etmiss = StandardScaler().fit(phi_col)
            # For object features (starts at 2): (obj_n, E, pt, eta, phi) repeated
            object_features = X[:, 2:].reshape(-1, self.n_objects, 5)
            # object kinematic columns (skipping obj_n)
            obj_kinematic = object_features[:, :, 1:]
            active_mask = (object_features[:, :, 0] > 0.1) # only scale valid objects
            
            to_scale = obj_kinematic[active_mask]
            # One scaler per kinematic type (E,pt,eta,phi)
            self.object_scalers = [RobustScaler().fit(to_scale[:,i:i+1]) for i in range(4)]
            return self
        
        def transform(self, X):
            X = np.asarray(X)
            out = np.zeros_like(X)
            # Process E_T_miss and phi
            out[:,0:1] = self.scalar_etmiss.transform(X[:,0:1])
            out[:,1:2] = self.scalar_phi_etmiss.transform(X[:,1:2])
            object_features = X[:, 2:].reshape(-1, self.n_objects, 5)
            obj_nums = object_features[:,:,0]
            obj_kinematic = object_features[:,:,1:]
            # Mask where object is present
            active_mask = (obj_nums > 0.1)
            # Create output for objects
            scaled_obj = np.zeros_like(obj_kinematic)
            for j in range(4):
                # Only scale active objects
                to_scale = obj_kinematic[:,:,j]
                # Flat-pass scaler, set to zero for padded
                scaled = np.zeros_like(to_scale)
                if np.any(active_mask):
                    scaled[active_mask] = self.object_scalers[j].transform(to_scale[active_mask].reshape(-1,1)).reshape(-1)
                scaled_obj[:,:,j] = scaled
            # Reconstruct
            object_out = np.zeros_like(object_features)
            object_out[:,:,0] = obj_nums  # pass obj id through (could one-hot, but keep as is for now)
            object_out[:,:,1:] = scaled_obj
            out[:,2:] = object_out.reshape(-1, self.n_objects*5)
            return out
    return HighEnergyPreprocessor()

# --- Model definition ---
def make_model(input_dim: int):
    class HighEnergyClassifier(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            # Simple, deep, wide layers + layernorm for stabilization in high-dim, dropout
            self.input_ln = nn.LayerNorm(in_dim)
            self.fc1 = nn.Linear(in_dim, 384)
            self.ln1 = nn.LayerNorm(384)
            self.fc2 = nn.Linear(384, 192)
            self.ln2 = nn.LayerNorm(192)
            self.fc3 = nn.Linear(192, 64)
            self.dropout = nn.Dropout(0.20)
            self.fc_out = nn.Linear(64, 1)
        
        def forward(self, x):
            x = self.input_ln(x)
            x = F.relu(self.ln1(self.fc1(x)))
            x = F.relu(self.ln2(self.fc2(x)))
            x = F.relu(self.fc3(x))
            x = self.dropout(x)
            x = self.fc_out(x)
            return x.view(-1)
    return HighEnergyClassifier(input_dim)

# --- Training loop ---

epochs = 18

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int):
    import torch.optim as optim
    from sklearn.metrics import roc_auc_score
    model = model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        y_true_ep = []
        y_score_ep = []
        for Xb, yb in train_loader:
            Xb = Xb.to(device).float()
            yb = yb.to(device).float()
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            preds_proba = torch.sigmoid(logits).detach()
            preds_label = (preds_proba > 0.5).long()
            correct += (preds_label == yb.long()).sum().item()
            total += len(yb)
            y_true_ep.append(yb.cpu().numpy())
            y_score_ep.append(preds_proba.cpu().numpy())
        train_loss.append(total_loss / total)
        train_acc.append(correct / total)

        # Val
        model.eval()
        vloss = 0.0
        vcorrect = 0
        vtotal = 0
        y_true_val = []
        y_score_val = []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device).float()
                yv = yv.to(device).float()
                logits = model(Xv)
                loss = criterion(logits, yv)
                vloss += loss.item() * len(yv)
                preds_proba = torch.sigmoid(logits)
                preds_label = (preds_proba > 0.5).long()
                vcorrect += (preds_label == yv.long()).sum().item()
                vtotal += len(yv)
                y_true_val.append(yv.cpu().numpy())
                y_score_val.append(preds_proba.cpu().numpy())
        val_loss.append(vloss / vtotal)
        val_acc.append(vcorrect / vtotal)
        # Print AUC for monitoring
        y_true = np.concatenate(y_true_val)
        y_score = np.concatenate(y_score_val)
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = float('nan')
        print(f"Epoch {ep+1:3d} | Train loss: {train_loss[-1]:.4f} | Val loss: {val_loss[-1]:.4f} | Train acc: {train_acc[-1]:.4f} | Val acc: {val_acc[-1]:.4f} | Val AUC: {auc:.5f}")
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
