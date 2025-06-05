
import os, sys, pickle, torch, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train),
            torch.from_numpy(Y_train),
            torch.from_numpy(X_val),
            torch.from_numpy(Y_val))

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        if isinstance(self.x, (tuple, list)):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])      

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = PairDataset(X_train, Y_train)
    val_ds   = PairDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))
                        
# ----------------  START OF LLM BLOCK  ----------------

# 0. ---------- IMPORTS ----------
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import copy
import math

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    """
    Physics-inspired tabular pre-processing:
      1. log1p-scale energy–like quantities
      2. feature-wise standardisation
      3. two engineered variables per event
           – N_obj : number of objects with non-zero pT
           – HT    : scalar sum of object pT (log1p-scaled)
    Output size: (N, 94) = 92 (original) + 2 (extra)
    """
    def __init__(self):
        # Will hold μ & σ for standardisation (shape (1, F))
        self.mean = None
        self.std  = None

        # Pre-compute flat column indices for object-wise kinematics
        obj_idx = np.arange(18)                # object number 0..17
        self.E_idx  = (3 + 5*obj_idx).tolist() # energy columns
        self.pT_idx = (4 + 5*obj_idx).tolist() # pT columns

    # ---------- helpers ----------
    def _apply_log_scaling(self, X):
        """
        return X' with log1p applied to:
          – E_T^miss (col 0)
          – object energies & pT (self.E_idx, self.pT_idx)
        """
        X = X.clone()
        X[:, 0] = torch.log1p(X[:, 0])               # E_T^miss magnitude
        X[:, self.E_idx]  = torch.log1p(X[:, self.E_idx])
        X[:, self.pT_idx] = torch.log1p(X[:, self.pT_idx])
        return X

    def _engineer_features(self, X_raw):
        """
        Extract high-level variables from *raw* (un-scaled) input.
        Returns tensor (N, 2) : [N_obj , HT(log1p)].
        """
        pT = X_raw[:, self.pT_idx]                   # (N, 18)
        valid = pT > 0
        N_obj = valid.sum(dim=1, keepdim=True)       # (N,1)
        HT    = torch.log1p(pT.sum(dim=1, keepdim=True))  # (N,1)
        return torch.cat([N_obj, HT], dim=1)

    # ---------- public API ----------
    def fit(self, X, y=None):
        """
        Learn μ,σ after all deterministic transforms on training data.
        """
        X = X.float()
        X_scaled = self._apply_log_scaling(X)                 # (N,92)
        extra    = self._engineer_features(X)                 # (N,2)
        full     = torch.cat([X_scaled, extra], dim=1)        # (N,94)

        self.mean = full.mean(dim=0, keepdim=True)
        self.std  = full.std (dim=0, keepdim=True)
        # Guard against division-by-zero
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        """
        Deterministic, stateless transform. Works for 1-D or 2-D tensors.
        """
        single = (X.dim() == 1)
        if single:
            X = X.unsqueeze(0)

        X_raw   = X.float()
        X_scaled = self._apply_log_scaling(X_raw)
        extra    = self._engineer_features(X_raw)
        full     = torch.cat([X_scaled, extra], dim=1)
        full     = (full - self.mean) / self.std
        full     = full.to(torch.float32)

        if single:
            full = full.squeeze(0)
        return full

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class DenseNet(nn.Module):
    """
    Fully-connected baseline for tabular data with BN + Dropout.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 1)                # final logit
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)      # logits (batch,)

def make_model(input_shape, *, use_mask=False):
    # input_shape is everything after the batch dim → (F,)
    input_dim = input_shape[0]
    return DenseNet(input_dim)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50      # upper bound – early-stopping will usually cut this short

def _to_device(t, device):
    if isinstance(t, (list, tuple)):
        return [_to_device(x, device) for x in t]
    return t.to(device)

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)  # on val AUC

    train_loss, val_loss = [], []
    train_acc,  val_acc  = [], []

    best_auc = -np.inf
    patience, counter = 6, 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # ---- training phase ----
        model.train()
        epoch_loss, preds, targets = 0.0, [], []
        for data, label in train_loader:
            data   = _to_device(data,   device)
            label  = label.float().to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss   = criterion(logits, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * label.size(0)
            preds.append(torch.sigmoid(logits.detach()).cpu())
            targets.append(label.cpu())

        preds   = torch.cat(preds)
        targets = torch.cat(targets)
        epoch_train_loss = epoch_loss / len(train_loader.dataset)
        epoch_train_acc  = accuracy_score(targets.numpy(), (preds.numpy() > 0.5).astype(int))

        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # ---- validation phase ----
        model.eval()
        v_loss, v_preds, v_targets = 0.0, [], []
        with torch.no_grad():
            for data, label in val_loader:
                data  = _to_device(data, device)
                label = label.float().to(device)

                logits = model(data)
                loss   = criterion(logits, label)

                v_loss += loss.item() * label.size(0)
                v_preds.append(torch.sigmoid(logits).cpu())
                v_targets.append(label.cpu())

        v_preds   = torch.cat(v_preds)
        v_targets = torch.cat(v_targets)
        epoch_val_loss = v_loss / len(val_loader.dataset)
        epoch_val_acc  = accuracy_score(v_targets.numpy(), (v_preds.numpy() > 0.5).astype(int))
        val_loss.append(epoch_val_loss)
        val_acc .append(epoch_val_acc)

        # primary metric – AUC
        try:
            val_auc = roc_auc_score(v_targets.numpy(), v_preds.numpy())
        except ValueError:
            val_auc = 0.5  # fallback if only one class present

        scheduler.step(val_auc)

        # ------- early stopping on AUC -------
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break  # no improvement → stop

    model.load_state_dict(best_state)   # restore best model
    trained_model = model
    return trained_model, train_loss, val_loss, train_acc, val_acc

# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    pre = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train) # may be Tensor or Tuple
    X_val   = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    if isinstance(X_train, torch.Tensor):               # single-tensor case
        temp_ref    = X_train
        input_shape = temp_ref.shape[1:]                # e.g. (F,)
        use_mask    = False
    else:                                               # tuple => (data, mask)
        temp_ref    = X_train
        input_shape = temp_ref[0].shape[1:]             # e.g. (L, F)
        use_mask    = True                              
    model = make_model(input_shape, use_mask=use_mask)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy_data = torch.zeros(8, *input_shape, dtype=torch.float32)
        if use_mask:
            toy_mask = torch.zeros(8, input_shape[0], dtype=torch.bool)
            toy_batch = (toy_data, toy_mask)
        else:
            toy_batch = toy_data

        toy_transformed = pre.transform(toy_batch)
        try:
            _ = trained_model(*toy_transformed) if isinstance(toy_transformed, (tuple, list)) \
                else trained_model(toy_transformed)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 6. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
            "train_loss": tr_loss   if tr_loss else None,
            "val_loss":   va_loss   if va_loss else None,
            "train_acc":  tr_acc    if tr_acc else None,
            "val_acc":    va_acc    if va_acc else None,
        }
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

