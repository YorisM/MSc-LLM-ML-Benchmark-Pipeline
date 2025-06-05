
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
from sklearn.metrics import roc_auc_score
import copy
import math

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    """
    Simple numeric pre-processing:
      1. Z-score standardisation for all continuous features
         (global E_T^miss, phi_Et_miss  and all per-object kinematics).
      2. Leave object-ID columns untouched (they will be embedded later).
    Output:  torch.Tensor  (N, 92)  – batch-first.
    """
    def __init__(self):
        # feature index layout helpers ------------------------
        self.global_idx   = [0, 1]              # 2 global features
        self.id_idx       = list(range(2, 92, 5))
        self.kin_idx_E    = list(range(3, 92, 5))
        self.kin_idx_pT   = list(range(4, 92, 5))
        self.kin_idx_eta  = list(range(5, 92, 5))
        self.kin_idx_phi  = list(range(6, 92, 5))
        self.kin_all_idx  = (self.kin_idx_E + self.kin_idx_pT +
                             self.kin_idx_eta + self.kin_idx_phi)
        self.scale_idx    = self.global_idx + self.kin_all_idx  # features to scale

        # statistics containers -------------------------------
        self.mean = None  # (92,) torch float32
        self.std  = None  # (92,) torch float32

    def fit(self, X, y=None):
        """
        Derive per-feature mean / std  (ignoring zero-padded objects).
        """
        X = X.float()
        device = X.device

        self.mean = torch.zeros(92, dtype=torch.float32, device=device)
        self.std  = torch.ones(92, dtype=torch.float32, device=device)

        # --------- global features ----------
        g = X[:, self.global_idx]
        self.mean[self.global_idx] = g.mean(dim=0)
        self.std[self.global_idx]  = g.std(dim=0).clamp_min(1e-6)

        # --------- kinematic features -------
        # use only valid (non-padded) objects
        obj_id_mat = X[:, self.id_idx]                      # (N, 18)
        valid_mask = obj_id_mat != 0                       # (N, 18)

        for idx_group in [self.kin_idx_E, self.kin_idx_pT,
                          self.kin_idx_eta, self.kin_idx_phi]:
            mat = X[:, idx_group]                          # (N, 18)
            vals = mat[valid_mask]                         # 1-D tensor with valid entries
            vals_mean = vals.mean()
            vals_std  = vals.std().clamp_min(1e-6)
            self.mean[idx_group] = vals_mean
            self.std[idx_group]  = vals_std

        # bring tensors to CPU for picklability
        self.mean = self.mean.cpu()
        self.std  = self.std.cpu()
        return self

    def transform(self, X):
        X = X.float().clone()  # defensive copy
        X = X.cpu()            # ensure same device as statistics

        # z-score scaling for selected indices
        X[:, self.scale_idx] = (X[:, self.scale_idx] - self.mean[self.scale_idx]) / self.std[self.scale_idx]
        return X          # (N, 92)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def make_preprocessor():
    return MyPreprocessor()


# 2. ---------- MODEL DEFINITION ----------
class DeepSetEventClassifier(nn.Module):
    """
    DeepSets-style network with learnable object-type embeddings.
    Forward signature:  logits = model(x)   where
      x : torch.FloatTensor (B, 92)  – pre-processed flat feature vector.
    """
    def __init__(self, input_shape):
        super().__init__()
        seq_len = 18                       # maximum number of objects
        embed_dim = 16
        obj_feature_dim = embed_dim + 4    # embed + (E, pT, eta, phi)
        per_obj_hidden = 64
        per_obj_out    = 32
        event_hidden   = 64

        # -------- object-ID embedding --------
        # Determine vocabulary size from training tensor if available
        vocab_size_default = 50000
        try:
            # X_train is expected to be in the global namespace
            global X_train
            vocab_size = int(X_train[:, 2::5].max().item()) + 1
            vocab_size = max(vocab_size, 2)  # at least 2 to keep padding idx = 0 valid
        except Exception:
            vocab_size = vocab_size_default

        self.id_emb = nn.Embedding(num_embeddings=vocab_size,
                                   embedding_dim=embed_dim,
                                   padding_idx=0)

        # -------- per-object network ---------
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_feature_dim, per_obj_hidden),
            nn.ReLU(),
            nn.Linear(per_obj_hidden, per_obj_out),
            nn.ReLU()
        )

        # -------- event-level network --------
        # aggregated object representation (mean + max)  → 2 * per_obj_out
        self.event_net = nn.Sequential(
            nn.Linear(2 * per_obj_out + 2, event_hidden),  # +2 global features
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(event_hidden, event_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(event_hidden // 2, 1)               # logits
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.FloatTensor, shape (B, 92)
             Pre-processed flat feature vector.

        Returns
        -------
        logits : torch.FloatTensor, shape (B, 1)
        """
        B = x.size(0)
        device = x.device

        # -------- extract pieces ------------
        obj_ids  = x[:, 2::5].long()                                   # (B, 18)
        obj_ids  = obj_ids.clamp_max(self.id_emb.num_embeddings - 1)   # safety
        mask     = obj_ids != 0                                        # (B, 18)  – valid objects

        kin_E    = x[:, 3::5]   # (B, 18)
        kin_pT   = x[:, 4::5]
        kin_eta  = x[:, 5::5]
        kin_phi  = x[:, 6::5]

        kinematics = torch.stack([kin_E, kin_pT, kin_eta, kin_phi], dim=2)  # (B, 18, 4)

        # -------- embed & encode -----------
        emb       = self.id_emb(obj_ids)                                # (B, 18, embed_dim)
        per_obj   = torch.cat([emb, kinematics], dim=2)                 # (B, 18, embed+4)
        per_obj   = self.obj_encoder(per_obj)                           # (B, 18, per_obj_out)

        # zero-out padded objects for mean pooling
        per_obj_masked = per_obj * mask.unsqueeze(-1)                   # (B, 18, per_obj_out)

        # mean pool (avoid divide-by-zero)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        mean_pool = per_obj_masked.sum(dim=1) / valid_counts            # (B, per_obj_out)

        # max pool (use large negative to ignore padded entries)
        minus_inf = torch.full_like(per_obj, -1e9)
        max_pool = torch.max(torch.where(mask.unsqueeze(-1), per_obj, minus_inf), dim=1)[0]  # (B, out)

        # global features (first two columns already normalised)
        global_feats = x[:, :2]                                         # (B, 2)

        event_repr = torch.cat([mean_pool, max_pool, global_feats], dim=1)  # (B, 2*out + 2)
        logits = self.event_net(event_repr).squeeze(1)                  # (B,)
        return logits


def make_model(input_shape, *, use_mask=False):
    # input_shape is (92,)  after preprocessing.
    model = DeepSetEventClassifier(input_shape)
    return model


# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20     # can be tuned – early-stopping will take care of over-training


def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',   # monitor AUC
                                                           factor=0.5,
                                                           patience=2)

    train_loss, val_loss = [], []
    train_acc,  val_acc  = [], []

    best_auc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    early_stop_patience = 6
    no_improve_epochs = 0

    for epoch in range(epochs):
        # -------- TRAIN ----------
        model.train()
        running_loss = 0.0
        y_pred_train, y_true_train = [], []

        for xb, yb in train_loader:               # xb : (B, 92)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            y_pred_train.append(logits.detach().cpu())
            y_true_train.append(yb.detach().cpu())

        y_pred_train = torch.cat(y_pred_train).sigmoid().numpy()
        y_true_train = torch.cat(y_true_train).numpy()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc  = ((y_pred_train >= 0.5) == y_true_train).mean()

        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # -------- VALIDATION ----------
        model.eval()
        running_val_loss = 0.0
        y_pred_val, y_true_val = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float()

                logits = model(xb)
                loss = criterion(logits, yb)
                running_val_loss += loss.item() * xb.size(0)

                y_pred_val.append(logits.cpu())
                y_true_val.append(yb.cpu())

        y_pred_val = torch.cat(y_pred_val).sigmoid().numpy()
        y_true_val = torch.cat(y_true_val).numpy()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc  = ((y_pred_val >= 0.5) == y_true_val).mean()
        epoch_val_auc  = roc_auc_score(y_true_val, y_pred_val)

        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        # scheduler on AUC (higher is better)
        scheduler.step(epoch_val_auc)

        # ------- early stopping --------
        if epoch_val_auc > best_auc + 1e-4:
            best_auc = epoch_val_auc
            best_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                break

    # restore best model
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

