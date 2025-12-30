
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
                        
DATASET = {
    "X_train": "./challenges/FOURTOPS/data/train/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/train/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/train/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/train/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv(DATASET["X_train"], dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv(DATASET["Y_train"], dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv(DATASET["X_val"], dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv(DATASET['Y_val'], dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class FourTopsDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

# -------------------------- START OF LLM BLOCK ------------------------------
# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sklearn.preprocessing
import sklearn.metrics

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# class CustomDataset(Dataset):
#  REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
#    def __init__(self, events, pre, train: bool = True, **kwargs):
#        X, y = events
#        self.X = pre.transform(X) if pre is not None else X
#        self.y = y
#    def __len__(self):
#        return int(self.y.shape[0])
#    def __getitem__(self, idx):
#        return self.X[idx], self.y[idx]

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    # TIPS
    #   - When modifying data features or feature engineering: annotate tensor size as comments after 
    #   - each tensor operation to reduce dimension mismatches.

    # DATA SPECIFICS
    #    Total flat length per event (X_train & X_val): 92
    #    Index  0 :  missing-ET magnitude  (E_T_miss)
    #    Index  1 :  missing-ET azimuth    (phi_Et_miss)
    #    Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    #    Indices  7-11 : object 2  ->  obj_2, E_2 , p_T_2 , eta_2 , phi_2
    #    ...
    #    Indices 87-91 : object 18 ->  obj_18, E_18 , p_T_18 , eta_18 , phi_18
    #    Global features       = 2
    #    Per-object slice size = 5
    #    Max objects encoded   = 18

    def __init__(self):
        # Using standard scalers for continuous variables
        self.scaler_E = sklearn.preprocessing.StandardScaler()
        self.scaler_pt = sklearn.preprocessing.StandardScaler()
        self.scaler_eta = sklearn.preprocessing.StandardScaler()
        self.scaler_obj = sklearn.preprocessing.StandardScaler()
        self.scaler_met = sklearn.preprocessing.StandardScaler()

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,

            # NO custom collate callables allowed. Choose one: 
            "collate": None, # (or "ragged_xy" or "identity" - If loader_class is torch_geometric.loader:DataLoader, set "collate": None.)

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # X shape: (N, 92)
        # We need to compute statistics for normalization. 
        # Since we log-transform energy and pT, we compute stats on log(1+x).

        N = X.shape[0]
        # Reshape particle part: (N, 18, 5)
        # Cols 2:92 correspond to particle features
        X_parts = X[:, 2:].reshape(N, 18, 5)

        # Valid particles have E > some epsilon (since padding is 0)
        E = X_parts[:, :, 1]
        valid_mask = (E > 1e-3)

        # --- Fit MET (Index 0) ---
        raw_met = np.log1p(X[:, 0]).reshape(-1, 1)
        self.scaler_met.fit(raw_met)

        # --- Fit Particles ---
        if valid_mask.any():
            # Extract valid values to fit scalers
            p_obj = X_parts[:, :, 0][valid_mask].reshape(-1, 1)
            p_E   = np.log1p(X_parts[:, :, 1][valid_mask]).reshape(-1, 1)
            p_pt  = np.log1p(X_parts[:, :, 2][valid_mask]).reshape(-1, 1)
            p_eta = X_parts[:, :, 3][valid_mask].reshape(-1, 1)

            self.scaler_obj.fit(p_obj)
            self.scaler_E.fit(p_E)
            self.scaler_pt.fit(p_pt)
            self.scaler_eta.fit(p_eta)

        return self

    def transform(self, X):
        # We restructure the flat vector into a sequence of particles.
        # Sequence Length: 19 (1 global MET token + 18 particles)
        # Feature Dim: 7 -> [logE, logPt, eta, sin_phi, cos_phi, obj_id, validity_mask]

        N = X.shape[0]
        out = np.zeros((N, 19, 7), dtype=np.float32)

        # --- 1. Process MET (Global) and place at Index 0 ---
        # Map MET global features to the same feature slots as particles where applicable
        raw_met = np.log1p(X[:, 0]).reshape(-1, 1)
        sc_met = self.scaler_met.transform(raw_met).flatten()
        met_phi = X[:, 1]

        out[:, 0, 0] = sc_met      # Map MET magnitude to Energy slot
        out[:, 0, 1] = sc_met      # Map MET magnitude to pT slot
        # eta (idx 2) remains 0
        out[:, 0, 3] = np.sin(met_phi)
        out[:, 0, 4] = np.cos(met_phi)
        # obj (idx 5) remains 0
        out[:, 0, 6] = 1.0         # Validity mask is 1 for global token

        # --- 2. Process Particles (Indices 1 to 18) ---
        X_parts = X[:, 2:].reshape(N, 18, 5)
        # Raw columns: 0:obj, 1:E, 2:pt, 3:eta, 4:phi

        p_obj = X_parts[:, :, 0]
        p_E   = X_parts[:, :, 1]
        p_pt  = X_parts[:, :, 2]
        p_eta = X_parts[:, :, 3]
        p_phi = X_parts[:, :, 4]

        mask = (p_E > 1e-3)

        # Log transforms
        log_E = np.log1p(p_E)
        log_pt = np.log1p(p_pt)

        # Apply scalers (using reshape tricks for vectorization)
        flat_sh = (N * 18, 1)
        grid_sh = (N, 18)

        out[:, 1:, 0] = self.scaler_E.transform(log_E.reshape(flat_sh)).reshape(grid_sh)
        out[:, 1:, 1] = self.scaler_pt.transform(log_pt.reshape(flat_sh)).reshape(grid_sh)
        out[:, 1:, 2] = self.scaler_eta.transform(p_eta.reshape(flat_sh)).reshape(grid_sh)
        out[:, 1:, 3] = np.sin(p_phi)
        out[:, 1:, 4] = np.cos(p_phi)
        out[:, 1:, 5] = self.scaler_obj.transform(p_obj.reshape(flat_sh)).reshape(grid_sh)
        out[:, 1:, 6] = mask.astype(np.float32)

        # Ensure padding is pure zero
        mask_exp = mask[:, :, None] # (N, 18, 1)
        out[:, 1:, :] *= mask_exp

        return torch.from_numpy(out) # returns (N, 19, 7) Tensor

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is the input tensor for one batch: (B, 19, 7)
        # We use a Transformer Encoder to process the set of particles

        input_dim = 7
        d_model = 128
        nhead = 4
        num_layers = 3
        dim_feedforward = 256
        dropout = 0.1

        # Input embedding projection
        self.embedding = nn.Linear(input_dim, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        # Transformer Encoder
        # batch_first=True expects (Batch, Seq, Feature)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation='gelu',
                                                   batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x: (B, 19, 7)
        # Feature 6 is the validity mask (1=valid, 0=pad)
        # PyTorch Transformer 'src_key_padding_mask' expects True for positions to IGNORE.
        mask_vals = batch_x[:, :, 6]
        padding_mask = (mask_vals < 0.5) # (B, 19) boolean mask

        x = self.embedding(batch_x)
        x = self.ln_in(x)

        # (B, 19, d_model)
        encoded = self.transformer(x, src_key_padding_mask=padding_mask)

        # Pooling Strategy: 
        # 1. MET token (index 0) naturally aggregates global info via attention
        met_repr = encoded[:, 0, :]

        # 2. Average pooling of particle tokens (indices 1..18), ignoring padding
        part_repr = encoded[:, 1:, :] 
        part_mask = mask_vals[:, 1:].unsqueeze(-1) # (B, 18, 1)

        sum_part = (part_repr * part_mask).sum(dim=1)
        count_part = part_mask.sum(dim=1).clamp(min=1.0)
        mean_part = sum_part / count_part

        # Fusion
        global_repr = met_repr + mean_part

        logits = self.classifier(global_repr) # (B, 1)
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # Setup basics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)

    # OneCycle LR Scheduler for better convergence
    steps = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, epochs=epochs, 
                                              steps_per_epoch=steps, pct_start=0.3)

    best_auc = 0.0
    best_weights = None

    tr_hist_loss, val_hist_loss = [], []
    tr_hist_acc, val_hist_acc = [], []

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        sum_loss = 0.0
        all_y, all_probs = [], []

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            logits = model(xb).view(-1)
            loss = criterion(logits, yb.float())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            sum_loss += loss.item() * yb.size(0)
            with torch.no_grad():
                all_y.append(yb.cpu().numpy())
                all_probs.append(torch.sigmoid(logits).cpu().numpy())

        epoch_loss = sum_loss / len(train_loader.dataset)
        epoch_auc = sklearn.metrics.roc_auc_score(np.concatenate(all_y), np.concatenate(all_probs))

        tr_hist_loss.append(epoch_loss)
        tr_hist_acc.append(epoch_auc)

        # --- VAL ---
        model.eval()
        val_sum_loss = 0.0
        val_y, val_probs = [], []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                logits = model(xb).view(-1)
                loss = criterion(logits, yb.float())

                val_sum_loss += loss.item() * yb.size(0)
                val_y.append(yb.cpu().numpy())
                val_probs.append(torch.sigmoid(logits).cpu().numpy())

        val_loss = val_sum_loss / len(val_loader.dataset)
        val_auc = sklearn.metrics.roc_auc_score(np.concatenate(val_y), np.concatenate(val_probs))

        val_hist_loss.append(val_loss)
        val_hist_acc.append(val_auc)

        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f} AUC={epoch_auc:.4f} | Val Loss={val_loss:.4f} AUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_weights = model.state_dict()

    # Restore best model
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, tr_hist_loss, val_hist_loss, tr_hist_acc, val_hist_acc

# <end code template>
# ---------------------------  END OF LLM-CODE BLOCK  ---------------------------

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:20]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre     = make_preprocessor().fit(X_train, Y_train)
    
    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec, require_torch_collate=False)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, (X_train, Y_train), pre, train=True)
    val_ds       = build_dataset(spec, (X_val,   Y_val),   pre, train=False)
    train_loader = build_dataloader(spec, train_ds, is_eval=False)
    val_loader   = build_dataloader(spec, val_ds,   is_eval=True)

    # Build model
    first_batch = next(iter(train_loader))
    view        = normalise_batch(first_batch, device=device)
    model       = make_model(view.batch_x).to(device)

    # Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        try:
            with torch.no_grad():
                view = normalise_batch(first_batch, device=device)
                out  = trained_model(view.batch_x)
                scores, kind = assert_binary_output(view, out)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e

    if not dryrun:
        # Persist artefacts
        base = base_from_argv0()
        persist_artefacts(base, SCRIPT_DIR, trained_model, pre, spec)

        # Save plots
        plot_train_val(tr_loss, va_loss, f"{base} Loss", os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        plot_train_val(tr_acc, va_acc, f"{base} Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))
        
        # Write JSON Summary
        summary = {
            "epochs": n_epochs      if n_epochs else None,
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

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

