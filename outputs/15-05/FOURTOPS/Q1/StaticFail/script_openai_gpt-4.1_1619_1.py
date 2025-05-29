
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

# ========== Preprocessor ==========
def make_preprocessor():
    class PhysicsPreprocessor(BaseEstimator, TransformerMixin):
        def __init__(self, n_obj=25, obj_stride=4):
            self.n_obj = n_obj           # Max objects from shape, 25 objects (1 E_Tmiss + phi, then 25*4=100, perhaps extra columns for weights, etc)
            self.obj_stride = obj_stride # (E, pT, eta, phi) per obj
            self.num_features = 2 + n_obj * self.obj_stride  # E_Tmiss, phi_ETmiss + objects kinematics
            self.scaler = None
            self.mask_value = 0.0
            self.obj_mask_idx = []
            self.objtype_start = 2  # After E_Tmiss and phi
        
        def fit(self, X, y=None):
            # Find valid mask and mean/std for scaling on non-masked
            mask = (X == 0.0)  # zero padding is the mask
            self.mask_value = 0.0
            # Compute mean/std only on unmasked positions
            valid = ~mask
            means = np.zeros(X.shape[1])
            stds = np.ones(X.shape[1])
            for i in range(X.shape[1]):
                # avoid division by zero
                if np.any(valid[:, i]):
                    means[i] = X[valid[:, i], i].mean()
                    stds[i] = X[valid[:, i], i].std(ddof=0) + 1e-6
            self.scaler = (means, stds)
            return self
        
        def transform(self, X):
            # Standard scaling, masking zero-padded columns
            means, stds = self.scaler
            X_scaled = (X - means[None, :]) / stds[None, :]
            # For object features (E, pT), clip outliers for numerical stability (physics-motivated)
            # High values correspond to rare backgrounds, help NN focus on core population.
            E_start = self.objtype_start
            # E's are every 4th entry for each object after first 2 columns
            for i in range(self.n_obj):
                idx = E_start + i*self.obj_stride
                X_scaled[:, idx] = np.clip(X_scaled[:, idx], -5, 5)
            # Optionally, compute object counts (nonzero objects) as an additional feature
            object_presence = np.sum((X[:, E_start::self.obj_stride] > 0), axis=1, keepdims=True)
            # Add number of objects as a feature
            X_out = np.concatenate([X_scaled, object_presence], axis=1)
            return X_out.astype(np.float32)
    return PhysicsPreprocessor()

# ========== Model ==========
def make_model(input_dim):
    class ParticleNet(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            # Input layer
            self.input_bn = nn.BatchNorm1d(input_dim)
            # Wide initial layer to handle sparse physics patterns and allow input pattern learning
            self.fc1 = nn.Linear(input_dim, 384)
            self.do1 = nn.Dropout(0.20)
            self.fc2 = nn.Linear(384, 192)
            self.do2 = nn.Dropout(0.15)
            self.fc3 = nn.Linear(192, 96)
            self.do3 = nn.Dropout(0.10)
            self.fc4 = nn.Linear(96, 32)
            self.fc5 = nn.Linear(32, 1)
        
        def forward(self, x):
            x = self.input_bn(x)
            x = F.relu(self.fc1(x))
            x = self.do1(x)
            x = F.relu(self.fc2(x))
            x = self.do2(x)
            x = F.relu(self.fc3(x))
            x = self.do3(x)
            x = F.relu(self.fc4(x))
            x = self.fc5(x)
            return x.squeeze(-1)  # outputs logits
    return ParticleNet(input_dim)

# ========== Training ========== 
epochs = 30

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=False)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_auc = 0.0
    # For AUC calculation
    from sklearn.metrics import roc_auc_score
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            with torch.no_grad():
                preds = torch.sigmoid(logits) > 0.5
                correct += (preds == yb.bool()).sum().item()
                total += yb.size(0)
        train_loss.append(running_loss / total)
        train_acc.append(correct / total)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                logits = model(xb)
                loss = criterion(logits, yb)
                val_running_loss += loss.item() * xb.size(0)
                probs = torch.sigmoid(logits)
                preds = probs > 0.5
                val_correct += (preds == yb.bool()).sum().item()
                val_total += yb.size(0)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        val_loss.append(val_running_loss / val_total)
        val_acc.append(val_correct / val_total)

        # Compute AUC after every epoch
        probs = np.concatenate(all_probs, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        auc = roc_auc_score(targets, probs)
        scheduler.step(auc)
        # Save best
        if auc > best_auc:
            best_auc = auc
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
        # (Optionally print or log epoch stats here)
    # Reload best weights
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
