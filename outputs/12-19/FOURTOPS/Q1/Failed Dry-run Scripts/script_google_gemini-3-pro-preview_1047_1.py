
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

# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# def make_dataset(events, pre, train: bool, **kwargs):
#   REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:make_dataset"
#   k = kwargs.get("k", 16)
#   <LLM: Insert custom dataset logic here>
#   return CustomDataset(events, pre, train=train, k=k)

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
        self.g_mean = None
        self.g_std = None
        self.p_mean = None
        self.p_std = None
        self.eps = 1e-6

    def make_loader_cfg(self):
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,

            # collate must be builtin string or None (torch default collate / PyG)
            "collate": None,

            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # X: [N, 92]
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        # 1. Global features [MET, phi]
        g = X[:, :2] # [N, 2]
        # Log MET
        g_trans = np.copy(g)
        g_trans[:, 0] = np.log1p(g_trans[:, 0]) 

        self.g_mean = np.mean(g_trans, axis=0).astype(np.float32)
        self.g_std = np.std(g_trans, axis=0).astype(np.float32) + 1e-5

        # 2. Particle features
        # Reshape to [N*18, 5] to compute stats over all particles
        p = X[:, 2:].reshape(-1, 5)

        # Mask zero-padded particles. Using pT (index 2) > 0 as check
        mask = p[:, 2] > self.eps
        valid_p = p[mask] # [M, 5]

        # Transform E (idx 1) and pT (idx 2) with log1p
        valid_p_trans = np.copy(valid_p)
        valid_p_trans[:, 1] = np.log1p(valid_p_trans[:, 1])
        valid_p_trans[:, 2] = np.log1p(valid_p_trans[:, 2])

        # Compute stats for [obj, E, pT, eta, phi]
        self.p_mean = np.mean(valid_p_trans, axis=0).astype(np.float32)
        self.p_std = np.std(valid_p_trans, axis=0).astype(np.float32) + 1e-5

        return self

    def transform(self, X):
        # Input X: [N, 92] Tensor
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X)

        X_out = torch.zeros_like(X)

        # --- Globals ---
        g = X[:, :2].clone()
        g[:, 0] = torch.log1p(g[:, 0])

        gm = torch.tensor(self.g_mean, device=X.device, dtype=X.dtype)
        gs = torch.tensor(self.g_std, device=X.device, dtype=X.dtype)

        g = (g - gm) / gs
        X_out[:, :2] = g

        # --- Particles ---
        # Shape [N, 18, 5]
        p = X[:, 2:].reshape(-1, 18, 5).clone()

        # Identify padding before transformation (pT > 0)
        # pT is index 2
        mask = p[:, :, 2] > self.eps 

        # Log transform E and pT
        p[:, :, 1] = torch.log1p(p[:, :, 1])
        p[:, :, 2] = torch.log1p(p[:, :, 2])

        pm = torch.tensor(self.p_mean, device=X.device, dtype=X.dtype)
        ps = torch.tensor(self.p_std, device=X.device, dtype=X.dtype)

        # Normalize
        p = (p - pm) / ps

        # Zero out padding explicitly (so model can detect exactly 0 or we use mask)
        # Broadcasting mask [N, 18] -> [N, 18, 5]
        p[~mask] = 0.0

        X_out[:, 2:] = p.reshape(-1, 90)

        return X_out # must return an indexable, picklable object

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object shape: [92]

        self.d_model = 128
        self.n_heads = 4
        self.n_layers = 3
        self.dim_feedforward = 256
        self.dropout_rate = 0.1

        # Embedding for particle features (5 input features)
        self.input_proj = nn.Sequential(
            nn.Linear(5, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model)
        )

        # Transformer Encoder (Set Transformer / Particle Transformer style)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # Global features embedding (2 input features)
        self.global_proj = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32)
        )

        # Final Classifier
        self.head = nn.Sequential(
            nn.Linear(self.d_model + 32, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x: [Batch, 92]

        # 1. Process Globals
        globals_in = batch_x[:, :2] # [Batch, 2]
        g_emb = self.global_proj(globals_in) # [Batch, 32]

        # 2. Process Particles
        particles_in = batch_x[:, 2:].reshape(-1, 18, 5) # [Batch, 18, 5]

        # Create padding mask. 
        # In transform, we set padded entries to exactly 0.
        # We check if the sum of absolute values of features is 0 for a particle.
        # This is safe because normalized features for a real particle are extremely unlikely to be all 0.0simultaneously.
        # src_key_padding_mask: True for padded positions
        padding_mask = torch.abs(particles_in).sum(dim=-1) < 1e-4 # [Batch, 18]

        x = self.input_proj(particles_in) # [Batch, 18, d_model]

        # Transformer
        # Output: [Batch, 18, d_model]
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Attention Pooling (Masked Mean)
        # Invert mask (True for valid)
        valid_mask = (~padding_mask).unsqueeze(-1).float() # [Batch, 18, 1]
        sum_x = (x * valid_mask).sum(dim=1) # [Batch, d_model]
        count = valid_mask.sum(dim=1).clamp(min=1.0) # Avoid div/0
        pooled = sum_x / count

        # 3. Combine
        combined = torch.cat([pooled, g_emb], dim=1)

        # 4. Predict
        logits = self.head(combined)
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 15   # <LLM: adjust if you wish>
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # Requirements: Return trained_model, train_loss, val_loss, train_acc, val_acc

    device = next(model.parameters()).device
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # OneCycle Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-3,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3
    )

    # Metric history
    hist = {"t_loss": [], "v_loss": [], "t_acc": [], "v_acc": []}

    best_val_loss = float('inf')
    early_stop_cnt = 0
    patience = 5
    best_state = None

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        t_loss_sum = 0
        t_correct = 0
        t_total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device).float()

            optimizer.zero_grad()
            logits = model(X).view(-1)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            t_loss_sum += loss.item() * X.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            t_correct += (preds == y).sum().item()
            t_total += X.size(0)

        train_loss = t_loss_sum / t_total
        train_acc = t_correct / t_total

        # --- VAL ---
        model.eval()
        v_loss_sum = 0
        v_correct = 0
        v_total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).float()
                logits = model(X).view(-1)
                loss = criterion(logits, y)

                v_loss_sum += loss.item() * X.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                v_correct += (preds == y).sum().item()
                v_total += X.size(0)

        val_loss = v_loss_sum / v_total
        val_acc = v_correct / v_total

        hist["t_loss"].append(train_loss)
        hist["v_loss"].append(val_loss)
        hist["t_acc"].append(train_acc)
        hist["v_acc"].append(val_acc)

        # print(f"Epoch {epoch+1}/{epochs} - TL: {train_loss:.4f} VL: {val_loss:.4f} TA: {train_acc:.4f} VA: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= patience:
                # print("Early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, hist["t_loss"], hist["v_loss"], hist["t_acc"], hist["v_acc"]

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


