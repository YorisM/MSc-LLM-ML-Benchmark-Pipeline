
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, to_python
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops, dryrun_finite_check_fourtops

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
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

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
        # Stats for MET (2 features)
        self.met_mean = np.zeros(2, dtype=np.float32)
        self.met_std = np.ones(2, dtype=np.float32)

        # Stats for Objects (5 features: ID, E, pT, eta, phi)
        self.obj_mean = np.zeros(5, dtype=np.float32)
        self.obj_std = np.ones(5, dtype=np.float32)

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this. Configure as you please.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,

            # NO custom collate callables allowed.
            "collate": None,

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False, 
                                "batch_size": 2048} 
        }

    def fit(self, X, y=None):
        # X shape: [N, 92]
        if torch.is_tensor(X):
            X = X.cpu().numpy()

        # 1. Fit MET (Indices 0, 1)
        met = X[:, :2].copy()
        # Log1p MET magnitude
        met[:, 0] = np.log1p(met[:, 0])

        self.met_mean = np.mean(met, axis=0)
        self.met_std = np.std(met, axis=0)
        # Avoid div by zero
        self.met_std[self.met_std < 1e-6] = 1.0

        # 2. Fit Objects
        # Reshape to [N*18, 5]
        objs = X[:, 2:].reshape(-1, 5)

        # Filter valid objects based on pT (index 2) > 0
        valid_mask = objs[:, 2] > 0
        valid_objs = objs[valid_mask].copy()

        if valid_objs.shape[0] > 0:
            # Apply Log1p to E (idx 1) and pT (idx 2)
            valid_objs[:, 1] = np.log1p(valid_objs[:, 1])
            valid_objs[:, 2] = np.log1p(valid_objs[:, 2])

            # Compute stats across all valid objects
            self.obj_mean = np.mean(valid_objs, axis=0)
            self.obj_std = np.std(valid_objs, axis=0)
            self.obj_std[self.obj_std < 1e-6] = 1.0

        return self

    def transform(self, X):
        # Ensure Tensor
        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.float().clone() # [N, 92]

        device = X.device

        # --- Transform MET ---
        X[:, 0] = torch.log1p(X[:, 0])
        met_mean = torch.tensor(self.met_mean, device=device)
        met_std = torch.tensor(self.met_std, device=device)
        X[:, :2] = (X[:, :2] - met_mean) / met_std

        # --- Transform Objects ---
        B = X.shape[0]
        objs = X[:, 2:].view(B, 18, 5)

        # Create mask of valid objects (pT > 0) BEFORE log transform or scaling
        # Original pT is at index 2
        # Use a small epsilon for float comparison just in case
        valid_mask = (objs[:, :, 2] > 1e-3) # [B, 18]

        # Apply Log1p to E and pT
        # We process everything then re-mask to 0 to avoid NaNs or wrong stats on pads
        objs[:, :, 1] = torch.log1p(objs[:, :, 1])
        objs[:, :, 2] = torch.log1p(objs[:, :, 2])

        # Standard Scale
        obj_mean = torch.tensor(self.obj_mean, device=device)
        obj_std = torch.tensor(self.obj_std, device=device)

        # (X - u) / s
        objs = (objs - obj_mean) / obj_std

        # Zero-out padding again (scaling moved 0s to roughly -mean/std)
        # Expand mask to [B, 18, 5]
        mask_expanded = valid_mask.unsqueeze(-1).expand_as(objs)
        objs = objs * mask_expanded.float()

        # Pack back
        X[:, 2:] = objs.view(B, -1)

        return X # must return an indexable, picklable object

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
# Lane A: Torch dense batch (default)
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object: [B, 92]

        self.d_model = 128
        self.nhead = 4
        self.num_layers = 4
        self.dim_feedforward = 256
        self.dropout = 0.1

        # MET Embedding (2 features)
        self.met_msg = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model)
        )

        # Object Embedding (5 features)
        self.obj_msg = nn.Sequential(
            nn.Linear(5, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model)
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout, 
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Final prediction head
        self.head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x: [B, 92]
        B = batch_x.size(0)

        # 1. Extract MET and Project -> [B, 1, d_model]
        met = batch_x[:, :2]
        met_emb = self.met_msg(met).unsqueeze(1)

        # 2. Extract Objects and Project -> [B, 18, d_model]
        objs = batch_x[:, 2:].view(B, 18, 5)
        obj_emb = self.obj_msg(objs)

        # 3. Create Padding Mask
        # Padded objects are zero vectors.
        # Check L1 norm roughly zero.
        # size: [B, 18]
        # src_key_padding_mask requires True for Padded elements
        obj_pad_mask = (objs.abs().sum(dim=-1) < 1e-4) # [B, 18]

        # 4. Construct Sequence
        # [MET, Obj1, ..., Obj18]
        x = torch.cat([met_emb, obj_emb], dim=1) # [B, 19, d_model]

        # MET is never padded (False)
        met_mask = torch.zeros((B, 1), dtype=torch.bool, device=batch_x.device)
        full_mask = torch.cat([met_mask, obj_pad_mask], dim=1) # [B, 19]

        # 5. Transformer
        out = self.transformer(x, src_key_padding_mask=full_mask) # [B, 19, d_model]

        # 6. Pooling
        # Take the MET token (index 0) as it attended to the whole event (like CLS)
        global_repr = out[:, 0, :] # [B, d_model]

        return self.head(global_repr)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # Setup
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                              steps_per_epoch=len(train_loader), 
                                              epochs=epochs)

    # Tracking
    best_auc = 0.0
    best_state = None

    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    print(f"Starting training on {device} for {epochs} epochs.")

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        sum_loss = 0.0
        all_train_preds = []
        all_train_targets = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            bs = X_batch.size(0)
            sum_loss += loss.item() * bs

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                all_train_preds.append(probs.cpu().numpy())
                all_train_targets.append(y_batch.cpu().numpy())

        train_loss = sum_loss / len(train_loader.dataset)

        train_p = np.vstack(all_train_preds)
        train_t = np.vstack(all_train_targets)
        train_acc = accuracy_score(train_t, train_p > 0.5)

        # --- VALIDATION ---
        model.eval()
        sum_val_loss = 0.0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device).float().unsqueeze(1)

                logits = model(X_val)
                loss = criterion(logits, y_val)
                sum_val_loss += loss.item() * X_val.size(0)

                probs = torch.sigmoid(logits)
                all_val_preds.append(probs.cpu().numpy())
                all_val_targets.append(y_val.cpu().numpy())

        val_loss = sum_val_loss / len(val_loader.dataset)

        val_p = np.vstack(all_val_preds)
        val_t = np.vstack(all_val_targets)

        val_acc = accuracy_score(val_t, val_p > 0.5)
        val_auc = roc_auc_score(val_t, val_p)

        # Logging
        print(f"Epoch {epoch+1:02d}: T_Loss={train_loss:.4f} T_Acc={train_acc:.4f} | V_Loss={val_loss:.4f} V_Acc={val_acc:.4f} V_AUC={val_auc:.4f}")

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist
# <end code template>

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    X_fit, Y_fit = X_train, Y_train
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:200]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre = make_preprocessor().fit(X_fit, Y_fit)
    
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
    n_epochs = 10 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        try:
            dryrun_finite_check_fourtops(trained_model, spec, val_loader, device, batches=10)
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
        summary = to_python(summary)
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

