
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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit() 
    #   - transform()

    def __init__(self):
        # Stats containers
        self.mean_global = None
        self.std_global = None
        self.mean_obj = None
        self.std_obj = None
        self.unique_ids = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "collate": None, 
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # X shape: (N, 92)
        # Global features: 0 (MET), 1 (Phi_MET)
        X_glob = X[:, :2]
        met = np.log1p(X_glob[:, 0])
        met_phi = X_glob[:, 1]

        self.mean_global = np.array([np.mean(met), np.mean(met_phi)], dtype=np.float32)
        self.std_global  = np.array([np.std(met), np.std(met_phi)], dtype=np.float32)
        self.std_global[self.std_global < 1e-6] = 1.0 # Avoid division by zero

        # Object features: 18 blocks of 5 features (ID, E, pT, eta, phi)
        objs = X[:, 2:].reshape(-1, 5)

        # Valid mask: usually pT > 0 implies a real object (zero-padded otherwise)
        mask = objs[:, 2] > 0.001
        valid_objs = objs[mask]

        if valid_objs.shape[0] > 0:
            # Handle IDs: find unique IDs to create a mapping later
            self.unique_ids = np.unique(valid_objs[:, 0])
            # Cap ID vocabulary to reasonable size (e.g. top 250) to prevent explosion
            if len(self.unique_ids) > 250:
                self.unique_ids = self.unique_ids[:250]

            # Continuous features: E, pT (log-transform), eta, phi
            E = np.log1p(valid_objs[:, 1])
            pT = np.log1p(valid_objs[:, 2])
            eta = valid_objs[:, 3]
            phi = valid_objs[:, 4]

            feats = np.stack([E, pT, eta, phi], axis=1)
            self.mean_obj = np.mean(feats, axis=0)
            self.std_obj = np.std(feats, axis=0)
            self.std_obj[self.std_obj < 1e-6] = 1.0
        else:
            # Fallback for safe defaults
            self.unique_ids = np.array([], dtype=np.float32)
            self.mean_obj = np.zeros(4, dtype=np.float32)
            self.std_obj = np.ones(4, dtype=np.float32)

        return self

    def transform(self, X):
        N = X.shape[0]

        # 1. Transform Global
        glob = X[:, :2].copy()
        glob[:, 0] = np.log1p(glob[:, 0]) # MET
        glob = (glob - self.mean_global) / self.std_global

        # 2. Transform Objects
        raw_objs = X[:, 2:].reshape(N, 18, 5)
        # Determine mask (True where object exists)
        mask_obj = raw_objs[:, :, 2] > 0.001

        # ID Mapping
        ids = raw_objs[:, :, 0].flatten()
        idx_map = np.searchsorted(self.unique_ids, ids)
        # Handle unknowns/padding
        idx_map[idx_map >= len(self.unique_ids)] = 0
        # Check exact match to confirm known ID
        is_known = (idx_map < len(self.unique_ids)) & (self.unique_ids[idx_map] == ids)

        # Map IDs to 1..K (0 reserved for padding/unknown)
        mapped_ids = np.zeros_like(ids, dtype=np.float32)
        mapped_ids[is_known] = idx_map[is_known] + 1.0
        mapped_ids = mapped_ids.reshape(N, 18)

        # Continuous feats: E, pT, eta, phi
        feats = raw_objs[:, :, 1:].copy()
        feats[:, :, 0] = np.log1p(feats[:, :, 0]) # E
        feats[:, :, 1] = np.log1p(feats[:, :, 1]) # pT

        # Normalize
        feats = (feats - self.mean_obj) / self.std_obj

        # Apply Zero-Masking to preserve padding structure
        mask_exp = mask_obj[:, :, None]
        feats = feats * mask_exp
        mapped_ids = mapped_ids * mask_obj

        # Concatenate Object Features: ID_mapped (1) + Kinematics (4) = 5
        objs_out = np.concatenate([mapped_ids[:, :, None], feats], axis=2)
        objs_flat = objs_out.reshape(N, -1)

        # Append the boolean mask to the output tensor so the model knows what is padding
        mask_flat = mask_obj.astype(np.float32)

        # Final Layout: Globals (2) + Objects (90) + Mask (18) = 110
        out = np.concatenate([glob, objs_flat, mask_flat], axis=1)
        return torch.from_numpy(out)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object shape: (110,)

        self.d_model = 128
        self.n_heads = 4
        self.n_enc_layers = 3

        # Embedding for IDs (max ID approx 250, +1 for padding)
        self.id_emb = nn.Embedding(256, self.d_model) 

        # Embedding for Continuous Variables (4)
        self.cont_emb = nn.Sequential(
            nn.Linear(4, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )

        # Transformer Encoder
        # batch_first=True requires PyTorch 1.9+ - Safe usually
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, 
                                                   dim_feedforward=self.d_model*2, 
                                                   dropout=0.1, activation='gelu', 
                                                   batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_enc_layers)

        # Global Feature Projection
        self.glob_proj = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.GELU()
        )

        # Prediction Head
        self.head = nn.Sequential(
            nn.Linear(self.d_model * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x: (B, 110)
        # Slicing
        xg = batch_x[:, :2]          # Global
        xp = batch_x[:, 2:92]        # Objects (18*5)
        xm = batch_x[:, 92:]         # Mask (18) -> 1.0 valid, 0.0 pad

        B = batch_x.shape[0]
        xp = xp.view(B, 18, 5)

        # 1. Embeddings
        # ID is feature 0
        ids = xp[:, :, 0].long().clamp(0, 255)
        emb_id = self.id_emb(ids)

        # Continuous feats are 1..4
        cont = xp[:, :, 1:]
        emb_cont = self.cont_emb(cont)

        # Combine
        tokens = emb_id + emb_cont

        # 2. Transformer
        # Create attention mask. src_key_padding_mask: True where padding (ignore).
        # xm is 1.0 for valid, 0.0 for pad. So we want xm == 0.0.
        padding_mask = (xm < 0.5)

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)

        # 3. Pooling (Global Mean Pool of Valid Tokens)
        # Apply mask to zero out padding noise
        mask_float = xm.unsqueeze(-1) # (B, 18, 1)
        encoded = encoded * mask_float

        sum_pooled = encoded.sum(dim=1)
        count = mask_float.sum(dim=1).clamp(min=1.0)
        mean_pooled = sum_pooled / count

        # 4. Concatenate Global and Head
        g_emb = self.glob_proj(xg)
        combined = torch.cat([mean_pooled, g_emb], dim=1)

        logits = self.head(combined)
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                              steps_per_epoch=len(train_loader), 
                                              epochs=epochs, pct_start=0.1)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    patience = 5
    counter = 0
    best_weights = None

    tr_losses, va_losses = [], []
    tr_accs, va_accs = [], []

    for epoch in range(epochs):
        model.train()
        losses = []
        y_true, y_probs = [], []

        for X, y in train_loader:
            X, y = X.to(device), y.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                y_true.append(y.cpu().numpy())
                y_probs.append(probs.cpu().numpy())

        tr_loss = np.mean(losses)
        try:
            tr_auc = roc_auc_score(np.concatenate(y_true), np.concatenate(y_probs))
        except:
            tr_auc = 0.5

        # Validation
        model.eval()
        v_losses = []
        vy_true, vy_probs = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).float().unsqueeze(1)
                logits = model(X)
                v_losses.append(criterion(logits, y).item())
                probs = torch.sigmoid(logits)
                vy_true.append(y.cpu().numpy())
                vy_probs.append(probs.cpu().numpy())

        va_loss = np.mean(v_losses)
        try:
            va_auc = roc_auc_score(np.concatenate(vy_true), np.concatenate(vy_probs))
        except:
            va_auc = 0.5

        tr_losses.append(tr_loss)
        va_losses.append(va_loss)
        tr_accs.append(tr_auc)
        va_accs.append(va_auc)

        print(f"Epoch {epoch+1}/{epochs} - TrL: {tr_loss:.4f} TrAUC: {tr_auc:.4f} | VaL: {va_loss:.4f} VaAUC: {va_auc:.4f}")

        if va_auc > best_val_auc:
            best_val_auc = va_auc
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, tr_losses, va_losses, tr_accs, va_accs

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


