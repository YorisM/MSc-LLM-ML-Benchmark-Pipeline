
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
# <LLM: Import modules>
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

    # <LLM: Write code to preprocess the data> 

    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        self.stats = {}
        self.epsilon = 1e-7

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this. Configure as you please.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,

            # NO custom collate callables allowed.
            "collate": None,

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False, 
                                "batch_size": 2048} 
        }

    def fit(self, X, y=None):
        # <LLM: Extract statistics for transform>
        # X shape: [N, 92]

        # We calculate statistics for normalization: log(E), log(pT), eta, log(MET)
        with torch.no_grad():
            # 1. Objects Stats
            # Reshape strictly to objects part: Columns 2 to 92 -> 90 columns -> 18 objects * 5 features
            objs = X[:, 2:].reshape(-1, 18, 5) # [N, 18, 5]

            # Slice features: 1=E, 2=pT, 3=eta
            E = objs[:, :, 1]
            pT = objs[:, :, 2]
            eta = objs[:, :, 3]

            # Mask valid objects (using small threshold for pT which is 0 for padding)
            mask = (pT > 1e-3)

            # Select valid values
            valid_E = E[mask]
            valid_pT = pT[mask]
            valid_eta = eta[mask]

            # Log transform
            log_E = torch.log(valid_E + 1.0)
            log_pT = torch.log(valid_pT + 1.0)

            # 2. MET Stats
            # MET is at index 0
            met_val = X[:, 0]
            log_met = torch.log(met_val + 1.0)

            # Store mean and std
            self.stats['log_E_mean'] = log_E.mean().item()
            self.stats['log_E_std'] = log_E.std().item() + self.epsilon

            self.stats['log_pT_mean'] = log_pT.mean().item()
            self.stats['log_pT_std'] = log_pT.std().item() + self.epsilon

            self.stats['eta_mean'] = valid_eta.mean().item()
            self.stats['eta_std'] = valid_eta.std().item() + self.epsilon

            self.stats['log_met_mean'] = log_met.mean().item()
            self.stats['log_met_std'] = log_met.std().item() + self.epsilon

        return self

    def transform(self, X):
        # <LLM: Apply pre-processing logic>
        # Input X: [B, 92] Tensor
        # Output: [B, 19, 8] Tensor

        B = X.shape[0]
        device = X.device

        # ---- Process MET (Token 0) ----
        # MET Features: [log_met_norm, log_met_norm, 0, sin(phi), cos(phi), 0 (id), 1 (is_met), 1 (valid)]
        met_mag = X[:, 0]
        met_phi = X[:, 1]

        log_met = torch.log(met_mag + 1.0)
        n_log_met = (log_met - self.stats['log_met_mean']) / self.stats['log_met_std']

        # Build MET Token tensor
        # We fill shape [B, 1, 8]
        t_met = torch.zeros((B, 1, 8), dtype=torch.float32, device=device)
        t_met[:, 0, 0] = n_log_met       # log_E equivalent
        t_met[:, 0, 1] = n_log_met       # log_pT equivalent
        t_met[:, 0, 2] = 0.0             # eta
        t_met[:, 0, 3] = torch.sin(met_phi)
        t_met[:, 0, 4] = torch.cos(met_phi)
        t_met[:, 0, 5] = 0.0             # ID
        t_met[:, 0, 6] = 1.0             # is_met flag
        t_met[:, 0, 7] = 1.0             # valid flag

        # ---- Process Objects (Tokens 1..18) ----
        # Shape [B, 18, 5]
        objs = X[:, 2:].reshape(B, 18, 5)
        # Features: 0=id, 1=E, 2=pT, 3=eta, 4=phi

        ids = objs[:, :, 0]
        E   = objs[:, :, 1]
        pT  = objs[:, :, 2]
        eta = objs[:, :, 3]
        phi = objs[:, :, 4]

        # Create validity mask
        valid = (pT > 1e-3).float() # [B, 18]
        valid_exp = valid.unsqueeze(-1) # [B, 18, 1]

        # Normalize
        n_log_E = (torch.log(E + 1.0) - self.stats['log_E_mean']) / self.stats['log_E_std']
        n_log_pT = (torch.log(pT + 1.0) - self.stats['log_pT_mean']) / self.stats['log_pT_std']
        n_eta = (eta - self.stats['eta_mean']) / self.stats['eta_std']

        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        # Stack features: [B, 18, 8]
        # [log_E, log_pT, eta, sin, cos, id, is_met(0), valid]
        t_objs = torch.stack([
            n_log_E, 
            n_log_pT, 
            n_eta, 
            sin_phi, 
            cos_phi, 
            ids, 
            torch.zeros_like(ids), 
            valid
        ], dim=-1)

        # Zero out invalid objects (retain 0s where valid=0, except possibly 'valid' flag which is 0)
        t_objs = t_objs * valid_exp

        # ---- Concatenate ----
        # Final shape: [B, 19, 8]
        out = torch.cat([t_met, t_objs], dim=1)

        return out # must return an indexable, picklable object

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
#
# --- LANE B: PyTorch Geometric (PyG) graphs ---
# Loader:
#   - loader_class: "torch_geometric.loader:DataLoader"
#   - collate: None
# Dataset samples MUST be torch_geometric.data.Data with at least:
#   data.x : FloatTensor[N_i, F]
#   data.edge_index : LongTensor[2, E_i]   (or equivalent; your model can build edges too)
#   data.y : LongTensor[1]                (GRAPH-LEVEL label for the event!)
# Batch from DataLoader:
#   G : torch_geometric.data.Batch (has G.x, G.edge_index, G.batch, and G.y)
# Model forward:
#   out = model(G)
#   out must be FloatTensor[num_graphs] or FloatTensor[num_graphs,1] (logits or probabilities)
#
# Any other batch shapes are NOT supported.

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # <LLM: Define and initialize any stateful components here>
        # sample_object shape: [B, 19, 8] or [19, 8] depending on lane check view. 
        # But here we know strict shape is [B, 19, 8] from preprocessor output.

        self.d_model = 128
        n_heads = 4
        n_layers = 3
        dim_feedforward = 256
        dropout = 0.1
        input_dim = 8

        # Input projection
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.norm_in = nn.LayerNorm(self.d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification Head
        # We classify based on the first token (MET/CLS)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    # <LLM: optionally build extra layers here>

    def forward(self, batch_x):
        # IMPORTANT output must be logits/probabilities per event
        # <LLM: Define your model's forward pass here>
        # batch_x: [B, 19, 8]

        # Create padding mask for Transformer
        # feature 7 is 'valid' (1.0 for valid, 0.0 for pad)
        # src_key_padding_mask expects True where we want to IGNORE (pad).
        # So mask = (valid < 0.5)

        valid_flag = batch_x[:, :, 7]
        src_mask = (valid_flag < 0.5) # [B, 19] bool

        # Project inputs
        x = self.input_proj(batch_x) # [B, 19, d_model]
        x = self.norm_in(x)

        # Transformer Pass
        # x shape matches batch_first=True
        out = self.transformer(x, src_key_padding_mask=src_mask) # [B, 19, d_model]

        # Pooling: Use only the first token (MET token) which serves as CLS
        cls_token = out[:, 0, :] # [B, d_model]

        # Head
        logits = self.head(cls_token) # [B, 1]

        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 15   # <LLM: adjust if you wish>
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    #   - Must return: trained_model, train_loss, val_loss, train_acc, val_acc
    #   - Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).

    # <LLM: Write code to define training loop, use the code above>
    # <LLM: Implement early stopping if possible>

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs,
        pct_start=0.3
    )

    best_val_auc = 0.0
    best_model_state = None

    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device).float().unsqueeze(1) # [B, 1]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * X_batch.size(0)

            # Store for AUC
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(Y_batch.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        cat_preds = np.concatenate(all_preds)
        cat_targets = np.concatenate(all_targets)
        try:
            epoch_auc = roc_auc_score(cat_targets, cat_preds)
        except:
            epoch_auc = 0.5

        train_losses.append(epoch_loss)
        train_aucs.append(epoch_auc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device).float().unsqueeze(1)

                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                val_running_loss += loss.item() * X_batch.size(0)

                probs = torch.sigmoid(outputs)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(Y_batch.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        cat_val_preds = np.concatenate(val_preds)
        cat_val_targets = np.concatenate(val_targets)
        try:
            val_auc = roc_auc_score(cat_val_targets, cat_val_preds)
        except:
            val_auc = 0.5

        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} AUC: {epoch_auc:.4f} | Val Loss: {val_loss:.4f} AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_aucs, val_aucs

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

