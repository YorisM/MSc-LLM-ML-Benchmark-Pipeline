
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
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

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# class CustomDataset(Dataset):
#  REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataSet"
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
    # Must implement:
    #   - fit() 
    #   - transform()

    # DATA SPECIFICS
    # Total flat length per event (X_train & X_val): 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    # Indices  7-11 : object 2  ->  obj_2, E_2 , p_T_2 , eta_2 , phi_2
    # ...
    # Indices 87-91 : object 18 ->  obj_18, E_18 , p_T_18 , eta_18 , phi_18
    # Global features       = 2
    # Per-object slice size = 5
    # Max objects encoded   = 18

    # TIPS
    # When modifying data features or feature engineering: annotate tensor size as comments after 
    # each tensor operation to reduce dimension mismatches.

    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module.
    # May allocate NumPy arrays or Torch tensors internally, but:
    # transform() must be deterministic.
    # Store only derived parameters needed for transform i.e. do not store the raw data
    # itself in the preprocessor object.

    def __init__(self):
        self.stats = {}

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 1024,
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
        # <LLM: Extract statistics for transform>
        # Input X: (N, 92) NumPy array or Tensor
        X_t = torch.from_numpy(X) if not torch.is_tensor(X) else X
        N = X_t.shape[0]

        # Structure analysis:
        # MET: X[:, 0:2]
        # Particles: X[:, 2:92] -> reshapes to (N, 18, 5)

        objs = X_t[:, 2:].reshape(N, 18, 5)
        # Features: 0:id, 1:E, 2:pt, 3:eta, 4:phi

        # Identify valid particles
        ids = objs[:, :, 0]
        mask = (ids != 0)

        # Extract data for normalization
        E = objs[:, :, 1][mask]
        pt = objs[:, :, 2][mask]
        eta = objs[:, :, 3][mask]
        phi = objs[:, :, 4][mask]

        met = X_t[:, 0]

        # Calculate stats (Log transform for E/Pt due to heavy tails)
        self.stats['logE_mean'] = torch.log1p(E).mean().item()
        self.stats['logE_std'] = torch.log1p(E).std().item()
        self.stats['logpt_mean'] = torch.log1p(pt).mean().item()
        self.stats['logpt_std'] = torch.log1p(pt).std().item()
        self.stats['eta_mean'] = eta.mean().item()
        self.stats['eta_std'] = eta.std().item()
        self.stats['phi_mean'] = phi.mean().item()
        self.stats['phi_std'] = phi.std().item()

        self.stats['logmet_mean'] = torch.log1p(met).mean().item()
        self.stats['logmet_std'] = torch.log1p(met).std().item()

        return self

    def transform(self, X):
        # <LLM: Apply pre-processing logic>
        # Returns Tensor of shape (N, 19, 8)
        # The flattened 18 particles become a sequence of 19 tokens (including MET).
        # Feature dim 8: [valid_mask, is_met, logE, logpt, eta, phi, cos(phi), sin(phi)]

        X_t = torch.from_numpy(X) if not torch.is_tensor(X) else X
        N = X_t.shape[0]
        device = X_t.device

        # Preallocate output
        # 19 tokens: 1 MET + 18 Particles
        out = torch.zeros((N, 19, 8), dtype=torch.float32, device=device)

        # --- Process MET (Global) ---
        met_val = X_t[:, 0]
        met_phi = X_t[:, 1]

        out[:, 0, 0] = 1.0 # Valid
        out[:, 0, 1] = 1.0 # Is MET type
        out[:, 0, 2] = (torch.log1p(met_val) - self.stats['logmet_mean']) / (self.stats['logmet_std'] + 1e-6) # Energy
        out[:, 0, 3] = out[:, 0, 2] # Pt (same as E for MET approx)
        out[:, 0, 4] = 0.0 # Eta
        out[:, 0, 5] = (met_phi - self.stats['phi_mean']) / (self.stats['phi_std'] + 1e-6) # Phi
        out[:, 0, 6] = torch.cos(met_phi)
        out[:, 0, 7] = torch.sin(met_phi)

        # --- Process Particles ---
        objs = X_t[:, 2:].reshape(N, 18, 5)
        # 0:id, 1:E, 2:pt, 3:eta, 4:phi

        ids = objs[:, :, 0]
        mask = (ids != 0).float()

        E = objs[:, :, 1]
        pt = objs[:, :, 2]
        eta = objs[:, :, 3]
        phi = objs[:, :, 4]

        out[:, 1:, 0] = mask
        out[:, 1:, 1] = 0.0 # Not MET
        out[:, 1:, 2] = (torch.log1p(E) - self.stats['logE_mean']) / (self.stats['logE_std'] + 1e-6) * mask
        out[:, 1:, 3] = (torch.log1p(pt) - self.stats['logpt_mean']) / (self.stats['logpt_std'] + 1e-6) * mask
        out[:, 1:, 4] = (eta - self.stats['eta_mean']) / (self.stats['eta_std'] + 1e-6) * mask
        out[:, 1:, 5] = (phi - self.stats['phi_mean']) / (self.stats['phi_std'] + 1e-6) * mask
        out[:, 1:, 6] = torch.cos(phi) * mask
        out[:, 1:, 7] = torch.sin(phi) * mask

        return out # must return an indexable, picklable object

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object shape: (Batch, 19, 8)
        input_dim = sample_object.shape[-1]
        d_model = 128
        nhead = 4
        num_layers = 3
        dim_feedforward = 256
        dropout = 0.1

        # Embedding projection
        self.embedding = nn.Linear(input_dim, d_model)
        self.norm_in = nn.LayerNorm(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    # <LLM: optionally build extra layers here>

    def forward(self, batch_x):
        # batch_x: (B, 19, 8)
        # Extract validity mask for padding
        mask = batch_x[:, :, 0] # (B, 19) where 1=valid, 0=pad

        # Embed
        x = self.embedding(batch_x) # (B, 19, d_model)
        x = self.norm_in(x)

        # Create key_padding_mask
        # PyTorch MHA expects True for positions to IGNORE
        key_padding_mask = (mask == 0)

        # Transformer Pass
        x = self.transformer(x, src_key_padding_mask=key_padding_mask) # (B, 19, d_model)

        # Global Pooling (Average of VALID tokens)
        mask_expanded = mask.unsqueeze(-1) # (B, 19, 1)
        x_masked = x * mask_expanded
        sum_x = x_masked.sum(dim=1) # (B, d_model)
        counts = mask.sum(dim=1).unsqueeze(-1).clamp(min=1.0) # (B, 1)
        mean_x = sum_x / counts

        # Prediction
        logits = self.head(mean_x) # (B, 1)
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20   # <LLM: adjust if you wish>
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS 
    #   Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #   Must return trained_model, train_loss, val_loss, train_acc, val_acc
    #   Use CUDA - torch.cuda.is_available()
    #   Implement early-stopping.
    #   Forward signature must match.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_model_state = None
    early_stop_patience = 5
    patience_counter = 0

    train_loss_val = 0.0
    val_loss_val = 0.0
    train_auc = 0.0
    val_auc = 0.0

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        sum_loss = 0
        all_preds = []
        all_targets = []

        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().unsqueeze(1)

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item() * X.size(0)

            with torch.no_grad():
                all_preds.append(torch.sigmoid(out).detach().cpu())
                all_targets.append(y.detach().cpu())

        train_loss_val = sum_loss / len(train_loader.dataset)
        all_preds_np = torch.cat(all_preds).numpy()
        all_targets_np = torch.cat(all_targets).numpy()
        try:
            train_auc = roc_auc_score(all_targets_np, all_preds_np)
        except:
            train_auc = 0.5

        # --- VALIDATION ---
        model.eval()
        sum_val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).float().unsqueeze(1)

                out = model(X)
                loss = criterion(out, y)
                sum_val_loss += loss.item() * X.size(0)

                val_preds.append(torch.sigmoid(out).cpu())
                val_targets.append(y.cpu())

        val_loss_val = sum_val_loss / len(val_loader.dataset)
        val_preds_np = torch.cat(val_preds).numpy()
        val_targets_np = torch.cat(val_targets).numpy()
        try:
            val_auc = roc_auc_score(val_targets_np, val_preds_np)
        except:
            val_auc = 0.5

        # --- CHECK EARLY STOPPING ---
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Return AUC as accuracy metric for harness plotting
    return model, train_loss_val, val_loss_val, train_auc, val_auc

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
# <end code template>

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

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


