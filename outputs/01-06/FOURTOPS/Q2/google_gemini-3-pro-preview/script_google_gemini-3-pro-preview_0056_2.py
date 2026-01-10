
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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

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
        # Scalers for physical quantities
        self.scaler_met = StandardScaler()
        self.scaler_E = StandardScaler()
        self.scaler_pt = StandardScaler()
        self.scaler_eta = StandardScaler()
        # Mapping for object IDs
        self.obj_map = {}
        self.max_id = 0

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this. Configure as you please.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,

            # NO custom collate callables allowed.
            "collate": None,

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False, 
                                "batch_size": 1024} # Or whatever you want
        }

    def fit(self, X, y=None):
        if hasattr(X, "cpu"): X = X.cpu().numpy()

        # 1. Fit MET Scaler
        # Apply log1p to MET magnitude to reduce skew
        met_mag = np.log1p(np.abs(X[:, 0:1]))
        self.scaler_met.fit(met_mag)

        # 2. Fit Object Scalers
        # We aggregate all valid objects to fit a common scaler for E, pT, eta
        Es, pts, etas, ids = [], [], [], []
        n_objs = 18

        # Heuristic: objects with E > 1e-3 are considered valid (not padding)
        for i in range(n_objs):
            base_idx = 2 + i * 5
            # Extract E column to check validity
            col_E = X[:, base_idx + 1]
            valid_mask = col_E > 1e-3

            if valid_mask.any():
                Es.append(np.log1p(np.abs(col_E[valid_mask])))
                pts.append(np.log1p(np.abs(X[valid_mask, base_idx + 2])))
                etas.append(X[valid_mask, base_idx + 3])
                ids.append(X[valid_mask, base_idx].astype(np.int64))

        if Es:
            all_E = np.concatenate(Es).reshape(-1, 1)
            all_pt = np.concatenate(pts).reshape(-1, 1)
            all_eta = np.concatenate(etas).reshape(-1, 1)
            all_id = np.concatenate(ids)

            self.scaler_E.fit(all_E)
            self.scaler_pt.fit(all_pt)
            self.scaler_eta.fit(all_eta)

            # Map unique IDs to continuous integers
            unique_ids = np.unique(all_id)
            # Reserve 0 for padding.
            current_id = 1
            for uid in unique_ids:
                if uid != 0:
                    self.obj_map[int(uid)] = current_id
                    current_id += 1
            self.max_id = current_id

        return self

    def transform(self, X):
        if hasattr(X, "cpu"): X = X.cpu().numpy()
        X_out = np.zeros_like(X, dtype=np.float32)

        # 1. Globals
        # MET
        met_log = np.log1p(np.abs(X[:, 0:1]))
        X_out[:, 0:1] = self.scaler_met.transform(met_log)
        X_out[:, 1] = X[:, 1] # Keep MET Phi as is

        # 2. Objects
        n_objs = 18
        # We want to map IDs efficiently. Since IDs are likely sparse or small integers,
        # we iterate over the map.

        for i in range(n_objs):
            base = 2 + i * 5

            # Map Object ID
            raw_ids = X[:, base].astype(np.int64)
            mapped_ids = np.zeros_like(raw_ids)
            for k, v in self.obj_map.items():
                # vectorized assignment
                mapped_ids[raw_ids == k] = v
            X_out[:, base] = mapped_ids

            # Scalers for Kinematics
            # E
            e_vals = np.log1p(np.abs(X[:, base+1])).reshape(-1, 1)
            X_out[:, base+1:base+2] = self.scaler_E.transform(e_vals)

            # pT
            pt_vals = np.log1p(np.abs(X[:, base+2])).reshape(-1, 1)
            X_out[:, base+2:base+3] = self.scaler_pt.transform(pt_vals)

            # Eta
            eta_vals = X[:, base+3].reshape(-1, 1)
            X_out[:, base+3:base+4] = self.scaler_eta.transform(eta_vals)

            # Phi
            X_out[:, base+4] = X[:, base+4]

        return X_out # must return an indexable, picklable object

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
# MODEL I/O BATCH CONTRACT (CHOOSE ONE LANE)
# Lane A: Torch dense batch (default)
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object shape: [B, 92]

        self.d_model = 128
        self.n_heads = 4
        self.n_layers = 4

        # Embeddings
        # ID embedding (0 is padding)
        # Assuming max_id fits in 256. 
        self.id_emb = nn.Embedding(256, self.d_model, padding_idx=0)

        # Continuous feature projection
        # Objects: E, pt, eta, cos(phi), sin(phi) -> 5 features
        self.obj_proj = nn.Sequential(
            nn.Linear(5, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )

        # Global Features: MET, cos(phi), sin(phi) -> 3 features
        self.glob_proj = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.n_heads, 
            dim_feedforward=512, 
            dropout=0.1, 
            batch_first=True, 
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # Output Head
        self.head = nn.Sequential(
            nn.Linear(self.d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, batch_x):
        # batch_x: [B, 92]
        B = batch_x.shape[0]

        # 1. Process Global Features
        # Indices 0, 1 -> Met, Phi
        met = batch_x[:, 0:1]
        met_phi = batch_x[:, 1:2]
        glob_feat = torch.cat([met, torch.cos(met_phi), torch.sin(met_phi)], dim=1) # [B, 3]
        glob_tok = self.glob_proj(glob_feat).unsqueeze(1) # [B, 1, d]

        # 2. Process Object Features
        # Reshape to [B, 18, 5]
        objs = batch_x[:, 2:].reshape(B, 18, 5)

        # Extract components
        ids = objs[:, :, 0].long().clamp(0, 255) # [B, 18]
        kin = objs[:, :, 1:4] # E, pt, eta
        phi = objs[:, :, 4:5]

        # Kinematic vector for projection
        kin_feat = torch.cat([kin, torch.cos(phi), torch.sin(phi)], dim=-1) # [B, 18, 5]

        # Compute Embeddings
        emb_kin = self.obj_proj(kin_feat)
        emb_id = self.id_emb(ids)
        obj_toks = emb_kin + emb_id

        # 3. Transformer
        # Concat global + objects -> [B, 19, d]
        seq = torch.cat([glob_tok, obj_toks], dim=1)

        # Padding Mask
        # Pad ID is 0.
        obj_pad_mask = (ids == 0) # [B, 18]
        # Global token never padded
        glob_pad_mask = torch.zeros((B, 1), dtype=torch.bool, device=batch_x.device)
        full_mask = torch.cat([glob_pad_mask, obj_pad_mask], dim=1) # [B, 19]

        out = self.transformer(seq, src_key_padding_mask=full_mask) # [B, 19, d]

        # 4. Pooling
        # Global token output
        out_g = out[:, 0, :]

        # Mean pool of objects (ignoring pad)
        out_o = out[:, 1:, :] 
        valid_mask = ~obj_pad_mask # True for valid
        # Zero out padding junk
        out_o_masked = out_o * valid_mask.unsqueeze(-1)
        sum_o = out_o_masked.sum(dim=1)
        cnt = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_o = sum_o / cnt

        # Classification
        rep = torch.cat([out_g, mean_o], dim=1) # [B, 2*d]
        logits = self.head(rep)

        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 10   # <LLM: adjust if you wish>
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # Setup
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # OneCycleLR fits well for fixed epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                              steps_per_epoch=len(train_loader), 
                                              epochs=epochs)

    best_auc = 0.0
    # Container for return. We return the final model state, 
    # but could implement state dict saving if needed.

    tr_loss_hist = []
    va_loss_hist = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        all_targets = []
        all_preds = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

            # Metrics (no grad)
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        train_loss = np.mean(epoch_losses)
        train_y = np.concatenate(all_targets)
        train_p = np.concatenate(all_preds)
        try:
            train_auc = roc_auc_score(train_y, train_p)
        except:
            train_auc = 0.5

        # Validation
        model.eval()
        val_losses = []
        val_targets = []
        val_preds = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).float().unsqueeze(1)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        val_loss = np.mean(val_losses)
        val_y = np.concatenate(val_targets)
        val_p = np.concatenate(val_preds)
        try:
            val_auc = roc_auc_score(val_y, val_p)
        except:
            val_auc = 0.5

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc

        # Variables for final return
        final_tr_loss = train_loss
        final_va_loss = val_loss
        final_tr_acc = train_auc # Harness expects a scalar "acc", we provide AUC
        final_va_acc = val_auc

    return model, final_tr_loss, final_va_loss, final_tr_acc, final_va_acc

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

