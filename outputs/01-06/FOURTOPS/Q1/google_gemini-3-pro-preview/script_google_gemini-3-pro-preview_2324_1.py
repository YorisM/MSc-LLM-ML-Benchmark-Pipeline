
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops

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
        X2 = pre.transform(X) if pre is not None else X
        if not torch.is_tensor(X2):
            X2 = torch.as_tensor(X2)
        self.X = X2.float()
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        self.y = y.long()
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
import torch.nn.functional as F
import numpy as np
import sklearn.preprocessing
import math

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
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.id_map = {}
        self.met_id = 0
        self.pad_id = 0
        self.num_ids = 0

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this. Configure as you please.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,

            # NO custom collate callables allowed.
            "collate": None,

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False, 
                                "batch_size": 512} # Or whatever you want
        }

    def fit(self, X, y=None):
        # Identify unique particle IDs
        # Objects are at indices: 2, 7, ..., 87
        obj_indices = np.arange(2, 92, 5)
        all_ids = X[:, obj_indices].flatten()

        # Check energy to filter padding
        e_indices = np.arange(3, 92, 5)
        all_es = X[:, e_indices].flatten()

        valid_mask = all_es > 1e-3
        valid_ids = all_ids[valid_mask]
        unique_ids = np.unique(valid_ids)

        # Map: 0 is PAD, 1..N are particles, N+1 is MET
        self.id_map = {int(uid): i + 1 for i, uid in enumerate(unique_ids)}
        self.pad_id = 0
        self.met_id = len(unique_ids) + 1
        self.num_ids = self.met_id + 1

        # Collect kinematics for scaler: log(E), log(pT), eta
        # 1. MET (treated as particle)
        met_e = X[:, 0]
        # MET "eta" is 0
        met_vals = np.stack([np.log1p(met_e), np.log1p(met_e), np.zeros_like(met_e)], axis=1)

        # 2. Objects
        feat_list = [met_vals]
        for k in range(18):
            base = 2 + k*5
            col_e = X[:, base+1]
            col_pt = X[:, base+2]
            col_eta = X[:, base+3]

            mask = col_e > 1e-3
            if mask.sum() > 0:
                subset = np.stack([
                    np.log1p(col_e[mask]),
                    np.log1p(col_pt[mask]),
                    col_eta[mask]
                ], axis=1)
                feat_list.append(subset)

        all_feats = np.concatenate(feat_list, axis=0)
        self.scaler.fit(all_feats)
        return self

    def transform(self, X):
        # Input: (N, 92)
        # We produce a Particle Table of shape (N, 19, 7)
        # However, to satisfy Lane A (Dense) contract Batch=[B, F], we flatten to (N, 19*7)
        # Model will reshape.
        # Features per particle:
        # 0: log(E)_norm
        # 1: log(pT)_norm
        # 2: eta_norm
        # 3: sin(phi)
        # 4: cos(phi)
        # 5: ID (mapped)
        # 6: Valid Mask (1.0 = valid, 0.0 = pad)

        N = X.shape[0]
        out = np.zeros((N, 19, 7), dtype=np.float32)

        # --- MET (Index 0) ---
        met_e = X[:, 0]
        met_phi = X[:, 1]

        met_kin = np.stack([np.log1p(met_e), np.log1p(met_e), np.zeros(N)], axis=1)
        out[:, 0, 0:3] = self.scaler.transform(met_kin)
        out[:, 0, 3] = np.sin(met_phi)
        out[:, 0, 4] = np.cos(met_phi)
        out[:, 0, 5] = self.met_id
        out[:, 0, 6] = 1.0

        # --- Objects (Indices 1-18) ---
        for k in range(18):
            base = 2 + k*5
            oid = X[:, base]
            oe = X[:, base+1]
            opt = X[:, base+2]
            oeta = X[:, base+3]
            ophi = X[:, base+4]

            valid = oe > 1e-3
            if valid.sum() > 0:
                # Kinematics
                kin_raw = np.zeros((valid.sum(), 3), dtype=np.float32)
                kin_raw[:, 0] = np.log1p(oe[valid])
                kin_raw[:, 1] = np.log1p(opt[valid])
                kin_raw[:, 2] = oeta[valid]

                norm_kin = self.scaler.transform(kin_raw)
                out[valid, k+1, 0:3] = norm_kin

                # Phi
                out[valid, k+1, 3] = np.sin(ophi[valid])
                out[valid, k+1, 4] = np.cos(ophi[valid])

                # ID Mapping
                # oid[valid] contains float IDs.
                mapped = np.zeros(valid.sum(), dtype=np.float32)
                for raw_id, map_idx in self.id_map.items():
                    # Check equality with tolerance
                    is_id = np.abs(oid[valid] - raw_id) < 0.1
                    mapped[is_id] = map_idx
                out[valid, k+1, 5] = mapped

                # Mask
                out[valid, k+1, 6] = 1.0

        return out.reshape(N, -1) # Return flat (N, 133)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
# MODEL I/O BATCH CONTRACT (CHOOSE ONE LANE)
# You MUST choose exactly one of the two supported input lanes and keep it consistent:
#
# --- LANE A: Torch dense batch (default) ---
# Loader:
#   - loader_class: "torch.utils.data:DataLoader"
#   - collate: None
# Batch from DataLoader:
#   (Xb, yb) where
#     Xb: FloatTensor[B, F]
#     yb: LongTensor[B] (or [B,1])
# Model forward:
#   out = model(Xb)
#   out must be FloatTensor[B] or FloatTensor[B,1] (logits or probabilities)

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is (B, 133)
        self.num_particles = 19
        self.num_features = 7

        self.d_model = 128
        self.num_heads = 4

        # Input projections
        # Features 0-4 are continuous -> embed to d_model
        self.kin_encoder = nn.Linear(5, self.d_model)

        # Feature 5 is ID -> Embedding
        # We assume max ID < 100 based on standard HEP formats
        self.id_embedding = nn.Embedding(100, self.d_model, padding_idx=0)

        # Transformer
        layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads, 
                                           dim_feedforward=512, dropout=0.1, 
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)

        # Head
        self.head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # Unwrap flattened input
        # batch_x: (B, 133) -> (B, 19, 7)
        B = batch_x.shape[0]
        x = batch_x.view(B, self.num_particles, self.num_features)

        # Slice features
        kin = x[:, :, 0:5]      # logE, logPt, eta, sin, cos
        ids = x[:, :, 5].long() # ID
        mask = x[:, :, 6]       # 1=Valid, 0=Pad

        # Embed
        h = self.kin_encoder(kin) + self.id_embedding(ids)

        # Padding mask for transformer (True = Ignore)
        # mask is 1 for valid, 0 for pad.
        key_padding_mask = (mask == 0)

        # Transformer
        # out: (B, 19, d_model)
        out = self.transformer(h, src_key_padding_mask=key_padding_mask)

        # Average Pooling with mask
        # Expand mask to (B, 19, 1)
        mask_exp = mask.unsqueeze(-1)

        # Sum of valid tokens
        sum_pooled = torch.sum(out * mask_exp, dim=1)
        # Count of valid tokens (avoid div0)
        counts = torch.sum(mask_exp, dim=1)
        avg_pooled = sum_pooled / (counts + 1e-6)

        logits = self.head(avg_pooled)
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 15   # <LLM: adjust if you wish>
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    #   - Must return: trained_model, train_loss, val_loss, train_acc, val_acc
    #   - Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).

    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Scheduler
    steps = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                              epochs=epochs, steps_per_epoch=steps)

    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    for ep in range(epochs):
        model.train()
        sum_loss = 0.0
        sum_acc = 0.0
        total = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Metrics
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == yb).float().mean()

            sum_loss += loss.item() * Xb.size(0)
            sum_acc += acc.item() * Xb.size(0)
            total += Xb.size(0)

        avg_train_loss = sum_loss / total
        avg_train_acc = sum_acc / total
        train_loss_hist.append(avg_train_loss)
        train_acc_hist.append(avg_train_acc)

        # Validation
        model.eval()
        v_loss = 0.0
        v_acc = 0.0
        v_total = 0

        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                yv = yv.to(device).float().unsqueeze(1)

                out = model(Xv)
                loss = criterion(out, yv)

                preds = (torch.sigmoid(out) > 0.5).float()
                acc = (preds == yv).float().mean()

                v_loss += loss.item() * Xv.size(0)
                v_acc += acc.item() * Xv.size(0)
                v_total += Xv.size(0)

        avg_val_loss = v_loss / v_total
        avg_val_acc = v_acc / v_total
        val_loss_hist.append(avg_val_loss)
        val_acc_hist.append(avg_val_acc)

        print(f"Epoch {ep+1}: TrL={avg_train_loss:.4f}, TrA={avg_train_acc:.4f}, VaL={avg_val_loss:.4f}, VaA={avg_val_acc:.4f}")

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

# DO NOT execute the pipeline here – the harness will do that.
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

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane_fourtops(spec, first_batch)
    view = make_view_by_lane_fourtops(mode, first_batch, device)

    # Build model
    model = make_model(view.batch_x).to(device)

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
                mode = detect_and_assert_lane_fourtops(spec, first_batch)
                view = make_view_by_lane_fourtops(mode, first_batch, device)
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

