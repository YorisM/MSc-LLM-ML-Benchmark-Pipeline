
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import math
import warnings
from sklearn.metrics import roc_auc_score
import copy

warnings.filterwarnings("ignore")

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# Not using a custom dataset class, leveraging the harness 'FourTopsDataset' with standard DataLoader

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # REQUIREMENTS
    #   - transform() must be deterministic.
    #   - Store derived parameters.

    def __init__(self):
        self.stats = {}
        # Reserved IDs: 0 = Padding, 1 = MET
        self.MET_ID = 1.0
        self.SHIFT_ID = 2.0 
        self.max_obj_id = 10 # Default fallback

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate": None, 
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512},
        }

    def fit(self, X, y=None):
        # Calculate statistics for normalization
        # X shape: (N, 92)
        # Structure: MET(0,1), Particle i (2+5i ... 6+5i)

        # View as particles: (N, 18, 5) to compute stats on valid objects
        particles = X[:, 2:].view(-1, 18, 5)

        # Valid mask: Check E > 0 (index 1 in particle dim)
        E = particles[:, :, 1]
        valid_mask = E > 1e-4

        # Extract features for stats
        # Particle feat indices: 0:obj, 1:E, 2:pT, 3:eta, 4:phi
        valid_E   = particles[:, :, 1][valid_mask]
        valid_pT  = particles[:, :, 2][valid_mask]
        valid_eta = particles[:, :, 3][valid_mask]

        # Global MET stats (indices 0:MET, 1:phi)
        met_val = X[:, 0]

        # Compute Log-space stats for Energy/pT
        self.stats['log_E_mean']  = torch.log1p(valid_E).mean().item()
        self.stats['log_E_std']   = torch.log1p(valid_E).std().item()
        self.stats['log_pt_mean'] = torch.log1p(valid_pT).mean().item()
        self.stats['log_pt_std']  = torch.log1p(valid_pT).std().item()

        self.stats['eta_mean'] = valid_eta.mean().item()
        self.stats['eta_std']  = valid_eta.std().item()

        self.stats['log_met_mean'] = torch.log1p(met_val).mean().item()
        self.stats['log_met_std']  = torch.log1p(met_val).std().item()

        # ID handling
        obj_ids = particles[:, :, 0][valid_mask]
        if len(obj_ids) > 0:
            max_raw = int(obj_ids.max().item())
            # We map RAW_ID -> RAW_ID + 2. So max embedding index needed is max_raw + 2 + 1 (for 0 index)
            self.max_obj_id = max_raw + 3

        return self

    def transform(self, X):
        # Input: (N, 92)
        # goal: produced (N, 19, 6) tensor
        # 19 tokens: 1 MET + 18 Particles
        # 6 features: [obj_id_embed_idx, log_E_norm, log_pT_norm, eta_norm, cos_phi, sin_phi]

        N = X.shape[0]
        device = X.device

        # --- 1. Process MET (Global) ---
        met_mag = X[:, 0]
        met_phi = X[:, 1]

        met_log = (torch.log1p(met_mag) - self.stats['log_met_mean']) / (self.stats['log_met_std'] + 1e-7)
        met_eta = torch.zeros_like(met_mag) # MET has no eta, set 0
        met_cos = torch.cos(met_phi)
        met_sin = torch.sin(met_phi)
        met_ids = torch.full((N,), self.MET_ID, device=device, dtype=torch.float32)

        # Stack MET: (N, 1, 6) -> [id, E, pT, eta, cos, sin] (duplicate MET to E and pT)
        met_token = torch.stack([met_ids, met_log, met_log, met_eta, met_cos, met_sin], dim=1).unsqueeze(1)

        # --- 2. Process Particles ---
        raw_p = X[:, 2:].view(N, 18, 5)
        p_obj, p_E, p_pt, p_eta, p_phi = raw_p.unbind(dim=-1)

        # Identify padding
        is_pad = p_E < 1e-4

        # Normalize
        p_logE  = (torch.log1p(p_E) - self.stats['log_E_mean']) / (self.stats['log_E_std'] + 1e-7)
        p_logpt = (torch.log1p(p_pt)- self.stats['log_pt_mean']) / (self.stats['log_pt_std'] + 1e-7)
        p_eta_n = (p_eta - self.stats['eta_mean']) / (self.stats['eta_std'] + 1e-7)
        p_cos   = torch.cos(p_phi)
        p_sin   = torch.sin(p_phi)

        # ID Logic: Shift raw IDs by self.SHIFT_ID (2) to clear 0(pad) and 1(MET)
        p_id_idx = p_obj + self.SHIFT_ID

        # Apply Zero-Masking for padding
        # Ensure padding tokens have strictly 0 features (including ID=0)
        zeros = torch.zeros_like(p_logE)
        p_logE  = torch.where(is_pad, zeros, p_logE)
        p_logpt = torch.where(is_pad, zeros, p_logpt)
        p_eta_n = torch.where(is_pad, zeros, p_eta_n)
        p_cos   = torch.where(is_pad, zeros, p_cos)
        p_sin   = torch.where(is_pad, zeros, p_sin)
        p_id_idx= torch.where(is_pad, zeros, p_id_idx)

        # Stack particles: (N, 18, 6)
        p_tokens = torch.stack([p_id_idx, p_logE, p_logpt, p_eta_n, p_cos, p_sin], dim=-1)

        # --- 3. Concatenate ---
        # Result: (N, 19, 6)
        out = torch.cat([met_token, p_tokens], dim=1)

        return out

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object shape: (B, 19, 6)
        # Feature 0 is ID (Categorical), 1-5 are Continuous

        dim_model = 128
        n_heads = 4
        n_layers = 3
        dim_feedforward = 512
        dropout = 0.1

        # Embeddings
        # We assume max ID is around 64 based on standard PDG/Jet mappings, but allow flexibility
        self.id_embedding = nn.Embedding(64, dim_model)

        # Continuous projection: 5 features -> dim_model
        self.cont_projection = nn.Sequential(
            nn.Linear(5, dim_model),
            nn.GELU(),
            nn.Linear(dim_model, dim_model)
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Heads
        self.pooler_norm = nn.LayerNorm(dim_model)
        self.classifier = nn.Sequential(
            nn.Linear(dim_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x: (B, 19, 6)
        # Split features
        ids = batch_x[:, :, 0].long()
        conts = batch_x[:, :, 1:]

        # Generate mask: True where ID == 0 (Padding)
        # Note: MET (index 0) is ID 1, never masked.
        padding_mask = (ids == 0)

        # Embed
        x_id = self.id_embedding(ids)
        x_cont = self.cont_projection(conts)
        x = x_id + x_cont

        # Transform
        out = self.transformer(x, src_key_padding_mask=padding_mask)

        # Pooling Strategy
        # 1. Take MET token (index 0) which aggregates global info via attention
        met_out = out[:, 0, :]

        # 2. Max pool over valid particles (indices 1:)
        # Apply large negative to padded positions for max pooling
        mask_expanded = padding_mask.unsqueeze(-1) # (B, 19, 1)
        out_masked = out.masked_fill(mask_expanded, -1e9)
        # Exclude MET from max pool to get purely hadronic signal? 
        # Actually max over everything (including MET) is fine, or just particles.
        # Let's max pool over everything.
        max_pool = out_masked.max(dim=1)[0]

        # Combine
        pooled = self.pooler_norm(met_out + max_pool)

        return self.classifier(pooled).squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 10 
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_auc = 0.0
    best_model_state = None

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        sum_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            yb = yb.float()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            sum_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = sum_loss / total
        train_acc = correct / total
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        # --- VALIDATION ---
        model.eval()
        sum_val_loss = 0.0
        val_correct = 0
        val_total = 0

        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                yb = yb.float()

                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    logits = model(xb)
                    vloss = criterion(logits, yb)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                sum_val_loss += vloss.item() * xb.size(0)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

                all_probs.append(probs.cpu())
                all_targets.append(yb.cpu())

        val_loss = sum_val_loss / val_total
        val_acc = val_correct / val_total
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        # AUC Calc
        cat_probs = torch.cat(all_probs).float().numpy()
        cat_targets = torch.cat(all_targets).float().numpy()
        try:
            val_auc = roc_auc_score(cat_targets, cat_probs)
        except:
            val_auc = 0.5

        print(f"Epoch {epoch+1}: T-Loss={train_loss:.4f} T-Acc={train_acc:.4f} | V-Loss={val_loss:.4f} V-Acc={val_acc:.4f} AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

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

