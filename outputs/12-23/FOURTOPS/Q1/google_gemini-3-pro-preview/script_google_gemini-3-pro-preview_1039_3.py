
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
from sklearn.metrics import roc_auc_score
import math

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
    # Index 0-1: MET
    # Index 2-91: 18 objects * 5 features

    def __init__(self):
        self.kin_mean = None
        self.kin_std = None
        self.met_mean = None
        self.met_std = None
        self.id_map = {}
        self.max_mapped_id = 0

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",     
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate": None, 
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # X shape: [N, 92]
        X_np = X.numpy()

        # 1. Analyze IDs
        # IDs are at indices 2, 7, 12, ... 87 based on (obj_n, E, pT, eta, phi)
        # Reshape to (N, 18, 5)
        # Slicing: X[:, 2:] matches 90 elements (18*5)
        obj_data = X_np[:, 2:].reshape(-1, 18, 5)
        ids = obj_data[:, :, 0].astype(int).flatten()

        # Unique IDs map
        unique_ids = np.unique(ids)
        self.id_map = {0: 0} # Reserve 0 for padding
        curr = 1
        for uid in unique_ids:
            if uid == 0: continue
            self.id_map[uid] = curr
            curr += 1
        self.max_mapped_id = curr

        # 2. Kinematic Stats
        # Avoid stats from padding
        objs_flat = obj_data.reshape(-1, 5)
        valid_mask = (objs_flat[:, 0] != 0)
        valid_objs = objs_flat[valid_mask, 1:] # E, pt, eta, phi

        # E(0), pt(1) -> log1p
        e_pt = np.log1p(np.maximum(valid_objs[:, 0:2], 0))
        eta_phi = valid_objs[:, 2:4]

        feats = np.hstack([e_pt, eta_phi])
        self.kin_mean = feats.mean(axis=0)
        self.kin_std = feats.std(axis=0) + 1e-6

        # 3. MET Stats (indices 0, 1)
        met = X_np[:, 0:2]
        met_pt = np.log1p(np.maximum(met[:, 0], 0))
        met_phi = met[:, 1]

        self.met_mean = np.array([met_pt.mean(), met_phi.mean()])
        self.met_std = np.array([met_pt.std(), met_phi.std()]) + 1e-6

        return self

    def transform(self, X):
        X_np = X.numpy()
        N = X_np.shape[0]

        # MET
        met = X_np[:, 0:2].copy()
        met[:, 0] = np.log1p(np.maximum(met[:, 0], 0))
        met = (met - self.met_mean) / self.met_std

        # Objects
        objs = X_np[:, 2:].reshape(N, 18, 5).copy()

        raw_ids = objs[:, :, 0].astype(int)
        raw_kin = objs[:, :, 1:]

        # Log1p E, pt
        raw_kin[:, :, 0:2] = np.log1p(np.maximum(raw_kin[:, :, 0:2], 0))
        # Norm
        raw_kin = (raw_kin - self.kin_mean) / self.kin_std

        # ID Mapping
        flat_ids = raw_ids.flatten()
        # Vectorized map
        mapper = np.vectorize(lambda x: self.id_map.get(x, 0), otypes=[np.float32])
        mapped_ids = mapper(flat_ids).reshape(N, 18)

        # Reconstruct
        # Store Mapped ID in first slot (index 0) of object vector
        objs_out = np.empty_like(objs)
        objs_out[:, :, 0] = mapped_ids
        objs_out[:, :, 1:] = raw_kin

        # Flatten and Cat
        # Shape out: [N, 2 + 90] = [N, 92]
        X_out = np.hstack([met, objs_out.reshape(N, -1)])

        return torch.from_numpy(X_out).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Config
        self.d_model = 128
        self.nhead = 4
        self.num_layers = 4
        self.dim_feedforward = 512
        self.num_emb = 128 # Large enough for mapped IDs space
        self.emb_dim = 32

        # Embedding for Object ID
        self.obj_emb = nn.Embedding(self.num_emb, self.emb_dim, padding_idx=0)

        # Project Kinematics (4 features)
        self.kin_proj = nn.Linear(4, self.d_model - self.emb_dim)

        self.norm_input = nn.LayerNorm(self.d_model)

        # MET Projection
        self.met_proj = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Classifier Head
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x shape: [B, 92]
        B = batch_x.size(0)

        # 1. Parse Input
        # MET: indices 0-1
        met_data = batch_x[:, 0:2] # [B, 2]
        # Objects: indices 2-91 -> [B, 18, 5]
        obj_data = batch_x[:, 2:].view(B, 18, 5)

        ids = obj_data[:, :, 0].long() # [B, 18]
        kin = obj_data[:, :, 1:]       # [B, 18, 4]

        # 2. Embeddings
        # Clamp IDs for safety
        ids = torch.clamp(ids, 0, self.num_emb - 1)

        id_embs = self.obj_emb(ids) # [B, 18, emb_dim]
        kin_feats = self.kin_proj(kin) # [B, 18, d_model - emb_dim]

        obj_tokens = torch.cat([id_embs, kin_feats], dim=2) # [B, 18, d_model]
        obj_tokens = self.norm_input(obj_tokens)

        # MET Token (prepend)
        met_token = self.met_proj(met_data).unsqueeze(1) # [B, 1, d_model]

        # Sequence: [MET, Obj1, ..., Obj18] -> Length 19
        x = torch.cat([met_token, obj_tokens], dim=1) # [B, 19, d_model]

        # 3. Masking
        # Mask padded objects. MET (idx 0) is never padded.
        # ID==0 implies padding (from preprocessing)
        obj_mask = (ids == 0) # [B, 18]
        # Create MET mask (False)
        met_mask = torch.zeros((B, 1), dtype=torch.bool, device=batch_x.device)
        # Combined mask
        src_key_padding_mask = torch.cat([met_mask, obj_mask], dim=1) # [B, 19]

        # 4. Transformer
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 5. Pooling
        # Take MET token (idx 0) as context summary
        cls_token = out[:, 0, :]

        # 6. Head
        return self.head(cls_token).view(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20 
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # OneCycleLR is effective for convergence in fixed epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs,
        pct_start=0.3
    )

    best_val_auc = -1.0
    best_state = None

    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_preds = []
        train_targets = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item() * batch_x.size(0)

            # Detach for metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                train_preds.append(probs.cpu().numpy())
                train_targets.append(batch_y.cpu().numpy())

        # Train Metrics
        train_loss = train_loss_sum / len(train_loader.dataset)
        train_preds = np.concatenate(train_preds)
        train_targets = np.concatenate(train_targets)
        train_auc = roc_auc_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss_sum += loss.item() * batch_x.size(0)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} AUC: {val_auc:.4f}")

        # Checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics calculation for return
    train_acc = ((train_preds > 0.5) == train_targets).mean()
    val_acc = ((val_preds > 0.5) == val_targets).mean()

    return model, train_loss, val_loss, train_acc, val_acc

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


