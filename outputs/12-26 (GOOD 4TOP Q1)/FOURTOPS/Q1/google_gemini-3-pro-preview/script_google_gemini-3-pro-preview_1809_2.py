
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
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

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
        self.scaler = StandardScaler()
        # Identify columns to log-transform: MET (0) and Object Energies (offset 1) and Pt (offset 2)
        # Object blocks start at index 2, stride 5.
        # Energies: 2 + 5*i + 1 -> 3, 8, ...
        # Pt:       2 + 5*i + 2 -> 4, 9, ...
        self.log_indices = [0] + [3 + 5*i for i in range(18)] + [4 + 5*i for i in range(18)]

        # Identify Pt columns to determine padding mask (Pt == 0 means padding)
        self.pt_indices = [4 + 5*i for i in range(18)]

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 512,
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
        # Handle input being a tensor (which harness load_data returns)
        if torch.is_tensor(X):
            X_np = X.cpu().numpy()
        else:
            X_np = X

        # Fit scaler on log-transformed data
        # We perform the same transformation as in 'transform' but just for fitting stats
        X_copy = np.copy(X_np)
        # Log1p ensures non-negative results if input >= 0. Clamp negative physics values to 0 just in case.
        X_copy[:, self.log_indices] = np.log1p(np.maximum(X_copy[:, self.log_indices], 0))

        self.scaler.fit(X_copy)
        return self

    def transform(self, X):
        # Input shape: (N, 92)
        is_tensor = False
        if torch.is_tensor(X):
            X_np = X.cpu().numpy()
            is_tensor = True
        else:
            X_np = X

        # 1. Create padding mask based on raw pT values (exactly 0.0 for padding)
        # We output a mask where 1.0 indicates padding (to be consistent with PyTorch Ignore logic later)
        # Pt indices: [4, 9, 14, ..., 89]
        pts = X_np[:, self.pt_indices] # Shape (N, 18)
        # Padding if pT is effectively 0
        padding_mask = (np.abs(pts) < 1e-5).astype(np.float32) # Shape (N, 18)

        # 2. Log transform
        X_copy = np.copy(X_np)
        X_copy[:, self.log_indices] = np.log1p(np.maximum(X_copy[:, self.log_indices], 0))

        # 3. Scale features
        X_scaled = self.scaler.transform(X_copy) # Shape (N, 92)

        # 4. Concatenate Mask to features to allow Model to see what is padding
        # New Shape: (N, 92 + 18) = (N, 110)
        X_out = np.hstack([X_scaled, padding_mask])

        # Return FloatTensor
        if is_tensor:
            return torch.from_numpy(X_out).float()
        return torch.from_numpy(X_out).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object shape: (Batch, 110)
        # Decomposed into: 92 kinematic features (2 Glob + 18*5 Obj) + 18 Mask features

        self.d_model = 128
        self.nhead = 4
        self.num_layers = 4

        # Global Feature Embedding (Inputs: MET, Phi_MET -> 2 features)
        self.global_embed = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model) # (B, 1, d)
        )

        # Object Feature Embedding (Inputs: ID, E, Pt, Eta, Phi -> 5 features)
        self.obj_embed = nn.Sequential(
            nn.Linear(5, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model) # (B, 18, d)
        )

        # Transformer Encoder
        # Set batch_first=True for (Batch, Seq, Feature)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, 
                                                   dim_feedforward=256, dropout=0.1, 
                                                   activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Logit output
        )

        # Feature sizes
        self.n_globals = 2
        self.n_obj = 18
        self.n_obj_feats = 5

    def forward(self, batch_x):
        # batch_x: (Batch, 110)

        # Slicing inputs
        # Features: indices 0 to 91
        # Mask: indices 92 to 109
        features = batch_x[:, :92]
        padding_mask = batch_x[:, 92:] # (B, 18), 1.0=Pad, 0.0=Valid

        # Extract components
        global_feats = features[:, :self.n_globals] # (B, 2)
        obj_feats = features[:, self.n_globals:].reshape(-1, self.n_obj, self.n_obj_feats) # (B, 18, 5)

        # Embed
        g_emb = self.global_embed(global_feats).unsqueeze(1) # (B, 1, d)
        o_emb = self.obj_embed(obj_feats)                    # (B, 18, d)

        # Concatenate tokens: [Global, Obj1, ..., Obj18]
        tokens = torch.cat([g_emb, o_emb], dim=1) # (B, 19, d)

        # Create Attention Mask
        # src_key_padding_mask expected shape: (B, SeqLen)
        # Value True indicates padding (ignore).
        # Global token is never padded (False).
        # Object tokens padded if padding_mask > 0.5.

        batch_size = batch_x.size(0)
        device = batch_x.device

        global_mask = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
        obj_mask = (padding_mask > 0.5) # Convert float mask to bool

        full_mask = torch.cat([global_mask, obj_mask], dim=1) # (B, 19)

        # Transformer Pass
        encoded = self.transformer(tokens, src_key_padding_mask=full_mask) # (B, 19, d)

        # Pool: Use the Global Token (index 0) as it aggregates context from all objects
        cls_token = encoded[:, 0, :] # (B, d)

        # Head
        logits = self.head(cls_token) # (B, 1)

        return logits.squeeze(-1) # (B,)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 12   
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss()
    # Optimizer and Scheduler for better convergence (OneCycleLR)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                              epochs=epochs, steps_per_epoch=steps_per_epoch)

    for epoch in range(epochs):
        model.train()
        train_loss_acc = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb.float())
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            batch_size = xb.size(0)
            train_loss_acc += loss.item() * batch_size

            preds = (torch.sigmoid(logits) > 0.5).long()
            train_correct += (preds == yb).sum().item()
            train_total += batch_size

        train_loss = train_loss_acc / train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss_acc = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                logits = model(xb)
                loss = criterion(logits, yb.float())

                batch_size = xb.size(0)
                val_loss_acc += loss.item() * batch_size

                preds = (torch.sigmoid(logits) > 0.5).long()
                val_correct += (preds == yb).sum().item()
                val_total += batch_size

        val_loss = val_loss_acc / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

    return model, train_loss, val_loss, train_acc, val_acc

# <end code template>

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

