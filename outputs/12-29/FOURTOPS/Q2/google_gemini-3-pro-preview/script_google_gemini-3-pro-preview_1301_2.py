
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
        x = self.X[idx]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x, self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

# -------------------------- START OF LLM BLOCK ------------------------------
# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# class CustomDataset(Dataset):
#     ... (Using default FourTopsDataset provided in harness)

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    def __init__(self):
        # Store mean and std for feature normalization
        self.stats = {} 
        self.eps = 1e-5

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,

            # NO custom collate callables allowed. Choose one: 
            "collate": None, 

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False, "batch_size": 512},
        }

    def fit(self, X, y=None):
        # Calculate mean and std of the features for normalization
        # X is (N, 92)
        if not torch.is_tensor(X):
            X = torch.from_numpy(X)

        # We need to extract features first to calculate stats over them
        # Do not modify self state during extraction
        with torch.no_grad():
            feats = self._extract_raw(X) # (N, 19, 8)

            # Flatten to consider all particles together for statistics
            flat_feats = feats.view(-1, feats.shape[-1])

            # Use the 'is_valid' flag (last column) to filter padding
            mask = flat_feats[:, -1] > 0.5
            valid_feats = flat_feats[mask]

            # Calculate stats for the first 7 features (exclude is_valid)
            # Features: [log_pt, eta, sin_phi, cos_phi, log_E, obj_id, is_met, is_valid]
            # We normalize everything except the binary flags (is_met, is_valid) ideally,
            # but standardizing everything is also robust for NNs.
            # Let's separate the last column (is_valid) which acts as a mask.

            # Mean and Std of features 0:-1
            feat_subset = valid_feats[:, :-1]
            mean = feat_subset.mean(dim=0)
            std = feat_subset.std(dim=0) + self.eps

            self.stats = {
                'mean': mean,
                'std': std
            }

        return self

    def transform(self, X):
        # Apply pre-processing logic
        if not torch.is_tensor(X):
            X = torch.from_numpy(X)

        # Extract structured features
        feats = self._extract_raw(X) # (N, 19, 8)

        if self.stats:
            device = feats.device
            mean = self.stats['mean'].to(device)
            std = self.stats['std'].to(device)

            # Normalize channels 0:-1
            feats[..., :-1] = (feats[..., :-1] - mean) / std

            # Masking: set padded particles cleanly to 0 (except mask flag)
            # Channel -1 is is_valid (1.0 or 0.0)
            is_valid = feats[..., -1:]

            # Apply zero-masking to the feature vector to keep padding clean
            feats = feats * is_valid

        return feats # Returns (N, 19, 8) tensor

    def _extract_raw(self, X):
        # Internal helper to parse 92-dim vector into (19, 8) particle sequence
        N = X.shape[0]
        device = X.device

        # --- 1. Process MET (Global info) ---
        # Stored at indices 0, 1
        met_val = X[:, 0:1] # E_T^miss
        met_phi = X[:, 1:2] # phi

        # Treat MET as a particle with eta=0, ID=0
        # Log-transform Energy/pT
        met_log_pt = torch.log1p(met_val)

        # Construct MET feature vector: [log_pt, eta, sin, cos, log_E, id, is_met, is_valid]
        met_feats = torch.cat([
            met_log_pt,                              # log_pt
            torch.zeros_like(met_val),               # eta
            torch.sin(met_phi),                      # sin_phi
            torch.cos(met_phi),                      # cos_phi
            met_log_pt,                              # log_E (approx same as pt for MET)
            torch.zeros_like(met_val),               # id
            torch.ones_like(met_val),                # is_met = 1
            torch.ones_like(met_val)                 # is_valid = 1
        ], dim=1).unsqueeze(1) # (N, 1, 8)

        # --- 2. Process Objects ---
        # Indices 2 to 92 (18 objects * 5 features)
        objs = X[:, 2:92].view(N, 18, 5)

        # Unpack: [id, E, pt, eta, phi]
        o_id  = objs[:, :, 0]
        o_E   = objs[:, :, 1]
        o_pt  = objs[:, :, 2]
        o_eta = objs[:, :, 3]
        o_phi = objs[:, :, 4]

        # Mask: pt > 0 indicates a real object
        is_obj_valid = (o_pt > 1e-3).float().unsqueeze(-1)

        # Features
        o_feats = torch.stack([
            torch.log1p(o_pt),
            o_eta,
            torch.sin(o_phi),
            torch.cos(o_phi),
            torch.log1p(o_E),
            o_id,
            torch.zeros_like(o_id), # is_met = 0
        ], dim=2)

        # Append validity flag
        o_feats = torch.cat([o_feats, is_obj_valid], dim=2) # (N, 18, 8)

        # --- 3. Concatenate MET + Objects ---
        # Result shape: (N, 19, 8)
        return torch.cat([met_feats, o_feats], dim=1)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object -> (B, 19, 8)
        input_dim = sample_object.shape[-1]

        # Transformer hyperparameters
        d_model = 128
        n_heads = 4
        n_layers = 4
        d_feedforward = 512
        dropout = 0.1

        # Token embedding
        self.embedding = nn.Linear(input_dim, d_model)
        self.norm_in = nn.LayerNorm(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification Head
        # Concatenate: [MET_token, Mean_Pool, Max_Pool] -> 3 * d_model
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1) # Logit output
        )

    def forward(self, batch_x):
        # batch_x: (N, 19, 8)

        # Create Padding Mask for Transformer (True where padded)
        # Last feature is 'is_valid' (1=valid, 0=pad)
        is_valid = batch_x[:, :, -1] # (N, 19)
        src_key_padding_mask = (is_valid < 0.5) 

        # Embed
        x = self.embedding(batch_x) # (N, 19, d_model)
        x = self.norm_in(x)

        # Transformer Pass
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        # x: (N, 19, d_model)

        # --- Pooling ---
        # Mask out padding tokens to effectively ignore them in pooling
        mask_expanded = is_valid.unsqueeze(-1) # (N, 19, 1)
        x_masked = x * mask_expanded

        # 1. MET Token (Index 0 is always MET, always valid)
        feat_met = x[:, 0, :]

        # 2. Mean Pooling (over valid tokens)
        sum_pooled = x_masked.sum(dim=1)
        count_valid = mask_expanded.sum(dim=1).clamp(min=1.0)
        feat_mean = sum_pooled / count_valid

        # 3. Max Pooling
        # Set padded values to very small number before max
        x_max = x.clone()
        x_max[src_key_padding_mask] = -1e9
        feat_max = x_max.max(dim=1)[0]

        # Combine
        combined = torch.cat([feat_met, feat_mean, feat_max], dim=1)

        # Predict
        return self.head(combined).squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 12 

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Optimization Setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs,
        pct_start=0.3
    )
    criterion = nn.BCEWithLogitsLoss()

    # Tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            yb = yb.float()

            optimizer.zero_grad(set_to_none=True)

            out = model(xb)
            loss = criterion(out, yb)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Metrics
            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            preds = (out > 0.0).float()
            correct += (preds == yb).sum().item()
            total += batch_size

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # --- Validation ---
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                yb = yb.float()

                out = model(xb)
                loss = criterion(out, yb)

                val_running_loss += loss.item() * xb.size(0)
                preds = (out > 0.0).float()
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Optional: Print progress (not required by harness but helpful for debugging if strict rules allowed, staying silent to be safe)

    return model, train_losses, val_losses, train_accs, val_accs
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

