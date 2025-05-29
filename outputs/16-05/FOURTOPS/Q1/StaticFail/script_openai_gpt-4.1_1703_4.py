
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
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import softmax
from sklearn.metrics import roc_auc_score

class HEPPreprocessor:
    """
    Preprocesses LHC event data. Handles variable-object features; standardizes objectwise features and computes per-event summary stats to expose physics-relevant information while keeping differentiable mapping for neural nets.
    """
    def __init__(self):
        self.obj_scaler = None  # Per-object physical quantities scaler.
        self.missing_et_scaler = None # For missing et, phi
        self.max_objects = 18
        self.n_obj_features = 5  # [id, E, pt, eta, phi]
        self.obj_feat_idx = [1,2,3,4]  # Positions within each object for [E, pt, eta, phi].
        self.eps = 1e-8
        # For storing indices for later slicing
        self.OBJ_SLICE = (2, 2+5*18)
    def fit(self, X, y=None):
        # Split out missing ET
        X = np.asarray(X)
        # Object data: [n_samples, 90] (indices 2:92)
        X_obj = X[:,2:].reshape((-1, self.max_objects, self.n_obj_features))
        # Only nonzero objects (non-zero in id or E)
        mask = (np.abs(X_obj[:,:,1]) > 0)
        X_obj_real = X_obj[mask]
        # Fit per-object feature scaler (exclude id, only on E/pt/eta/phi)
        self.obj_scaler = StandardScaler()
        self.obj_scaler.fit(X_obj_real[:,1:])
        # Fit missing ET scaler
        self.missing_et_scaler = StandardScaler()
        self.missing_et_scaler.fit(X[:,[0,1]])
        return self
    def transform(self, X):
        X = np.asarray(X)
        N = X.shape[0]
        out = []
        # Missing ET features
        missET = self.missing_et_scaler.transform(X[:,[0,1]])  # shape (N,2)
        # Per-object features
        X_obj = X[:,2:].reshape((-1, self.max_objects, self.n_obj_features))
        # Mask for real objects
        mask = (np.abs(X_obj[:,:,1]) > self.eps)  # [N,18]
        # Prepare object feature array for scaling E, pT, eta, phi
        obj_feats = np.zeros((N, self.max_objects, 4))
        for i in range(self.max_objects):
            # For id, E, pT, eta, phi: slice [id,E,pT,eta,phi]
            colE = X_obj[:,i,1]
            colpt = X_obj[:,i,2]
            coleta = X_obj[:,i,3]
            colphi = X_obj[:,i,4]
            obj_feats[:,i,0] = colE
            obj_feats[:,i,1] = colpt
            obj_feats[:,i,2] = coleta
            obj_feats[:,i,3] = colphi
        # Flatten to scale (mask-objects only)
        flat_mask = mask.ravel()
        obj_feats_flat = obj_feats.reshape(-1,4)
        obj_feats_scaled = np.zeros_like(obj_feats_flat)
        if np.sum(flat_mask)>0:
            obj_feats_scaled[flat_mask] = self.obj_scaler.transform(obj_feats_flat[flat_mask])
        obj_feats_scaled = obj_feats_scaled.reshape(N, self.max_objects, 4)
        # Compute per-event summary statistics
        n_obj = mask.sum(axis=1, keepdims=True)  # number of nonzero-objects per event, shape (N,1)
        # Per-object max pT, mean E, total pT, leading |eta|, mean abs(eta)
        max_pt = np.max(obj_feats[:,:,1]*mask, axis=1, keepdims=True)
        mean_E = np.where(n_obj>0, (obj_feats[:,:,0]*mask).sum(axis=1,keepdims=True)/n_obj, 0.)
        sum_pt = (obj_feats[:,:,1]*mask).sum(axis=1, keepdims=True)
        abs_eta = np.abs(obj_feats[:,:,2])
        max_abseta = np.max(abs_eta*mask, axis=1, keepdims=True)
        mean_abseta = np.where(n_obj>0, (abs_eta*mask).sum(axis=1,keepdims=True)/n_obj, 0.)
        # Count of identified lepton-like (id==11,13) or b-jets (id==5) per event
        id_mat = X_obj[:,:,0]
        n_elec = ((id_mat==11)&mask).sum(axis=1,keepdims=True)
        n_muon = ((id_mat==13)&mask).sum(axis=1,keepdims=True)
        n_bjet = ((id_mat==5)&mask).sum(axis=1,keepdims=True)
        # Flatten per-object features for the NN
        obj_flat = obj_feats_scaled.reshape(N, self.max_objects*4)
        # Feature set: [missET(2), n_obj, max_pt, mean_E, sum_pt, max_abseta, mean_abseta, n_elec, n_muon, n_bjet, obj_flat(:)]
        features = [missET, n_obj, max_pt, mean_E, sum_pt, max_abseta, mean_abseta, n_elec, n_muon, n_bjet, obj_flat]
        X_final = np.concatenate(features, axis=1)
        return X_final

def make_preprocessor():
    return HEPPreprocessor()

class HEPMultiDenseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.d1 = nn.Linear(input_dim, 192)
        self.bn1 = nn.BatchNorm1d(192)
        self.d2 = nn.Linear(192, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.d3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.d4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.11)
    def forward(self, x):
        x = F.relu(self.bn1(self.d1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.d2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.d3(x)))
        x = self.dropout(x)
        x = F.relu(self.d4(x))
        x = self.out(x)
        return x.squeeze(-1)

def make_model(input_dim:int):
    return HEPMultiDenseNet(input_dim)

EPOCHS = 32

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=max(epochs//3,3), gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_val_auc = 0.0
    for epoch in range(epochs):
        # Training phase
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()*xb.size(0)
            preds = (torch.sigmoid(logits)>0.5).long()
            tr_correct += (preds==yb.long()).sum().item()
            tr_total += xb.size(0)
        train_loss.append(tr_loss/tr_total)
        train_acc.append(tr_correct/tr_total)
        # Validation phase
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).float()
                logits = model(xb)
                loss = criterion(logits, yb)
                v_loss += loss.item()*xb.size(0)
                preds = (torch.sigmoid(logits)>0.5).long()
                v_correct += (preds==yb.long()).sum().item()
                v_total += xb.size(0)
                all_logits.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        val_loss.append(v_loss/v_total)
        val_acc.append(v_correct/v_total)
        # Compute AUC on current validation set
        pred_prob = np.concatenate(all_logits)
        targ = np.concatenate(all_labels)
        try:
            val_auc = roc_auc_score(targ, pred_prob)
        except Exception:
            val_auc = 0.0
        # Early stop if AUC maximum reached (patience=6, safeguard for overfitting)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        scheduler.step()
        # Optionally print training stats
        #print(f"Epoch {epoch+1}/{epochs} tr_loss={train_loss[-1]:.4f} val_loss={val_loss[-1]:.4f} tr_acc={train_acc[-1]:.3f} val_acc={val_acc[-1]:.3f} val_auc={val_auc:.4f}")
        if patience > 6:  # Allow max 6 bad epochs, then break.
            break
    # Load best model
    if best_val_auc>0:
        model.load_state_dict(best_model_state)
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

