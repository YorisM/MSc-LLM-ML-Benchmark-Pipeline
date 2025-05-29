
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

# ---- Data Preprocessing ----
def make_preprocessor():
    class EventPreprocessor(BaseEstimator, TransformerMixin):
        def __init__(self):
            # Store scalers for continuous features
            self.scalar_indices = None
            self.object_type_indices = None
            self.sc = None
        def fit(self, X, y=None):
            X = np.asarray(X)
            # Scalar quantities: E_T_miss (0), phi_{E_t}_miss (1)
            # Then for each object: [ID, E, pT, eta, phi]: 5 columns, repeated
            n_objects = (X.shape[1] - 2) // 5
            self.scalar_indices = [0] # E_T_miss only (phi is angle, leave unscaled)
            # Indices of ID fields for each object
            self.object_type_indices = [2 + 5*i for i in range(n_objects)]
            # Indices of kinematic features for each object (E, pT, eta, phi)
            kin_indices = []
            for i in range(n_objects):
                base = 2 + 5*i
                kin_indices.extend([base+1, base+2, base+3, base+4]) # E, pT, eta, phi
            # We will standardize all non-ID, non-zero-padded (not obj==0) numerical features,
            # and treat IDs as is (or one-hot/embedding in model later)
            # For stability: only scale where ID != 0 (padded)
            mask = np.ones_like(X, dtype=bool)
            # Set mask False where IDs are zero and for their four kin values
            for i, idx in enumerate(self.object_type_indices):
                # For all rows where obj==0, mask the 4 kin values
                nonzero = X[:, idx] != 0
                for j in range(1,5):
                    mask[:, idx+j] = np.logical_and(mask[:, idx+j], nonzero)
            # Now gather features to scale
            to_scale = self.scalar_indices + kin_indices
            aggregate = []
            for i in to_scale:
                valid = mask[:,i]
                vals = X[valid,i]
                aggregate.append(vals)
            agg_all = np.concatenate([v.reshape(-1,1) for v in aggregate], axis=1)
            self.sc = StandardScaler().fit(agg_all)
            return self
        def transform(self, X):
            X = np.array(X, copy=True)
            n_events, n_features = X.shape
            n_objects = (n_features-2)//5
            # scalar_indices, object_type_indices as above
            # First scale E_T_miss if not zero
            # Collect to scale & flatten for vectorized StandardScaler use
            to_scale = self.scalar_indices.copy()
            kin_indices = []
            for i in range(n_objects):
                base = 2 + 5*i
                kin_indices.extend([base+1, base+2, base+3, base+4])
            to_scale += kin_indices
            # Build 2d array of all these features from all events (vectorized)
            flattened = []
            mask_per_col = []
            for i in to_scale:
                if i >= 2:
                    # For objects, mask those with obj == 0
                    obj_idx = 2 + 5 * ((i-2)//5)
                    rel_idx = (i-2)%5
                    valid = (X[:,obj_idx] != 0)
                    flattened.append(X[valid,i])
                    mask_col = np.zeros(X.shape[0], dtype=bool)
                    mask_col[valid] = True
                    mask_per_col.append(mask_col)
                else:
                    # For E_T_miss
                    flattened.append(X[:,i])
                    mask_per_col.append(np.ones(X.shape[0],dtype=bool))
            # Stack into shape (n_samples_total, n_feats)
            arr = np.stack([v for v in flattened], axis=1)
            # Standardize with mutivar scaler
            arr_scaled = self.sc.transform(arr)
            # Now assign back
            col_counter = 0
            for idx, i in enumerate(to_scale):
                if i >= 2:
                    obj_idx = 2 + 5 * ((i-2)//5)
                    rel_idx = (i-2)%5
                    # valid events for this col
                    mask = (X[:,obj_idx] != 0)
                    X[mask,i] = arr_scaled[mask_per_col[idx], col_counter]
                else:
                    # E_T_miss
                    X[:,i] = arr_scaled[:,col_counter]
                col_counter += 1
            # For phi angles, map to sin/cos (periodic var encoding)
            # phi_{E_t}_miss
            phi_miss = X[:,1]
            phi_miss_cos = np.cos(phi_miss)
            phi_miss_sin = np.sin(phi_miss)
            X_aug = [X[:,0], phi_miss_cos, phi_miss_sin]
            # Now for all object phis
            for i in range(n_objects):
                base = 2 + 5*i
                phi = X[:, base+4]
                objid = X[:, base]
                mask = (objid != 0)
                # Only transform if not padded
                phi_cos = np.zeros_like(phi)
                phi_sin = np.zeros_like(phi)
                phi_cos[mask] = np.cos(phi[mask])
                phi_sin[mask] = np.sin(phi[mask])
                # Add to features instead of phi
                X[:, base+4] = 0. # replace phi with 0 (remove the original column)
                X_aug.append(phi_cos)
                X_aug.append(phi_sin)
            # Remaining features: (E_T_miss_scaled), (phi_miss_cos,sin), all object [ID, E_scaled, pT_scaled, eta_scaled, phi_cos, phi_sin]  (phi replaced by cosine/sine) per object
            # Need to remove original phi columns and repack features accordingly
            # We'll organize as [E_T_miss_scaled, phi_{E_t}_miss_cos, phi_{E_t}_miss_sin, [obj_id, E, pT, eta, obj_phi_cos, obj_phi_sin]*N]
            features = [X_aug[0], X_aug[1], X_aug[2]]
            for i in range(n_objects):
                base = 2 + 5*i
                # ID (int), E, pT, eta, phi_cos, phi_sin
                features.append(X[:, base])      # ID
                features.append(X[:, base+1])    # E
                features.append(X[:, base+2])    # pT
                features.append(X[:, base+3])    # eta
                features.append(X_aug[3+2*i])    # phi_cos
                features.append(X_aug[3+2*i+1])  # phi_sin
            X_new = np.stack(features, axis=1).T # shape (n_events, n_feats_packed)
            X_new = np.array(X_new).T
            return X_new
    preproc = EventPreprocessor()
    return preproc

# ---- Model: Deep Set/Attention over Objects ----
def make_model(input_dim: int):
    # Parameters: Max number of objects = (input_dim - 3) // 6
    # Each object: [id, E, pT, eta, phi_cos, phi_sin]
    class ParticleNet(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            nobj = (input_dim - 3)//6
            self.nobj = nobj
            self.obj_fdim = 6
            # 0: E_T_miss, 1: phi_miss_cos, 2: phi_miss_sin
            # rest: objs
            self.obj_start = 3
            # Embedding for object IDs (ID goes from 0 (padding) up to J? Pick 10 as default, can tune later)
            self.emb = nn.Embedding(11, 4, padding_idx=0) 
            # Each object: [emb 4, E, pT, eta, phi_cos, phi_sin]=8
            self.obj_encoder = nn.Sequential(
                nn.Linear(8, 32),
                nn.ReLU(),
                nn.Linear(32, 24),
                nn.ReLU(),
                nn.Linear(24, 16)
            )
            # Attention pooling across objects
            self.attn = nn.Linear(16, 1)
            # Global (pooled) + E_T_miss + phi_{miss}
            self.final = nn.Sequential(
                nn.Linear(16+3, 32),
                nn.ReLU(),
                nn.Linear(32, 12),
                nn.ReLU(),
                nn.Linear(12, 1)
            )
        def forward(self, x):
            # x shape: (N, input_dim)
            batch = x.shape[0]
            dev = x.device
            # Scalar: E_T_miss + phi (cos/sin)
            etmiss = x[:,0:1]
            phimiss = x[:,1:3] # cos, sin
            # objects: shape (N, nobj, 6)
            obj = x[:,self.obj_start:].reshape(batch, self.nobj, self.obj_fdim)
            # ID is int, the rest floats
            ids = obj[:,:,0].long().clamp(0, 10) # max ID
            features = obj[:,:,1:] # (N, nobj, 5)
            emb = self.emb(ids) # (N,nobj,4)
            obj_in = torch.cat([emb, features], dim=-1) # (N, nobj, 9)
            # Encode each object
            obj_encoded = self.obj_encoder(obj_in) # (N,nobj,16)
            # Mask padding (IDs==0)
            mask = (ids != 0)
            # For stability, set pad-rows to large negative before softmax
            attn_score = self.attn(obj_encoded).squeeze(-1) # (N, nobj)
            attn_score[~mask] = -1e9
            alpha = F.softmax(attn_score, dim=-1).unsqueeze(-1) # (N, nobj,1)
            pool = torch.sum(alpha * obj_encoded, dim=1) # (N, 16)
            outcat = torch.cat([pool, etmiss, phimiss], dim=-1) # (N, 16+3)
            out = self.final(outcat)
            return out.squeeze(-1)
    return ParticleNet(input_dim)

# ---- Training ----
epochs = 22

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int):
    import torch.optim as optim
    from sklearn.metrics import roc_auc_score
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(epochs):
        # Training step
        model.train()
        total_loss = 0
        correct = 0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.shape[0]
            outputs = torch.sigmoid(logits) > 0.5
            correct += (outputs == yb.bool()).sum().item()
            n += xb.shape[0]
        train_loss.append(total_loss/n)
        train_acc.append(correct/n)
        # Validation step
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            correct_val = 0
            nval = 0
            all_logits = []
            all_yval = []
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float()
                logits = model(xb)
                val_l = criterion(logits, yb)
                total_val_loss += val_l.item() * xb.shape[0]
                outputs = torch.sigmoid(logits) > 0.5
                correct_val += (outputs == yb.bool()).sum().item()
                nval += xb.shape[0]
                all_logits.append(logits.detach().cpu().numpy())
                all_yval.append(yb.detach().cpu().numpy())
            val_loss.append(total_val_loss/nval)
            val_acc.append(correct_val/nval)
            # Optionally, print AUC at each epoch
            pred = np.concatenate(all_logits)
            true = np.concatenate(all_yval)
            try:
                auc = roc_auc_score(true, pred)
            except Exception:
                auc = -1
            print(f"Epoch {epoch+1}/{epochs} Train Loss: {train_loss[-1]:.5f} Acc: {train_acc[-1]:.4f} | Val Loss: {val_loss[-1]:.5f} Acc: {val_acc[-1]:.4f} | Val AUC: {auc:.5f}")
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
