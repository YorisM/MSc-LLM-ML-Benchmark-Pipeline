
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
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import copy
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

    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.part_mean = None
        self.part_std = None

        # Feature indices
        # Global: 0 (MET), 1 (Phi)
        self.idx_global = [0, 1]

        # Objects: 18 objects, stride 5
        # Indices in flat array (0-91)
        # Format: obj_n, E, pT, eta, phi
        self.idx_obj = list(range(2, 92, 5))
        self.idx_E   = list(range(3, 92, 5))
        self.idx_pT  = list(range(4, 92, 5))
        self.idx_eta = list(range(5, 92, 5))
        self.idx_phi = list(range(6, 92, 5))

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate": None, 
            "extra_loader_kwargs": {"drop_last": True},
            "eval_overrides": {"shuffle": False, "drop_last": False},
        }

    def fit(self, X, y=None):
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        N = X.shape[0]
        epsilon = 1e-5

        # 1. Global Features
        # MET: Log1p
        g_data = X[:, self.idx_global].copy()
        g_data[:, 0] = np.log1p(g_data[:, 0])

        self.global_mean = np.mean(g_data, axis=0).astype(np.float32)
        self.global_std = (np.std(g_data, axis=0) + epsilon).astype(np.float32)

        # 2. Particle Features
        # Compute stats only on non-padded particles (E > 0)
        # Flattened Energy column
        all_E = X[:, self.idx_E].flatten()
        mask = all_E > 0

        # E (log1p)
        vals_E = np.log1p(all_E[mask])

        # pT (log1p)
        all_pT = X[:, self.idx_pT].flatten()
        vals_pT = np.log1p(all_pT[mask])

        # Eta
        all_eta = X[:, self.idx_eta].flatten()
        vals_eta = all_eta[mask]

        # Phi
        all_phi = X[:, self.idx_phi].flatten()
        vals_phi = all_phi[mask]

        self.part_mean = np.array([
            vals_E.mean(), vals_pT.mean(), vals_eta.mean(), vals_phi.mean()
        ], dtype=np.float32)

        self.part_std = np.array([
            vals_E.std(), vals_pT.std(), vals_eta.std(), vals_phi.std()
        ], dtype=np.float32) + epsilon

        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        X_out = X.clone()
        device = X.device

        # 1. Globals
        X_out[:, 0] = torch.log1p(X_out[:, 0])
        gm = torch.tensor(self.global_mean, device=device)
        gs = torch.tensor(self.global_std, device=device)
        X_out[:, 0:2] = (X_out[:, 0:2] - gm) / gs

        # 2. Particles
        pm = torch.tensor(self.part_mean, device=device)
        ps = torch.tensor(self.part_std, device=device)

        # E
        X_out[:, self.idx_E] = torch.log1p(X_out[:, self.idx_E])
        X_out[:, self.idx_E] = (X_out[:, self.idx_E] - pm[0]) / ps[0]

        # pT
        X_out[:, self.idx_pT] = torch.log1p(X_out[:, self.idx_pT])
        X_out[:, self.idx_pT] = (X_out[:, self.idx_pT] - pm[1]) / ps[1]

        # Eta
        X_out[:, self.idx_eta] = (X_out[:, self.idx_eta] - pm[2]) / ps[2]

        # Phi
        X_out[:, self.idx_phi] = (X_out[:, self.idx_phi] - pm[3]) / ps[3]

        # Object IDs (index 2, 7...) are left untouched logic-wise
        # They will be used to create masks in the model

        return X_out

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Architecture: Particle Flow Network / Deep Sets
        # Input shape typically (B, 92)

        # ID Embedding
        self.id_emb_dim = 16
        self.id_buckets = 64 # Hash bucket size for object IDs
        self.id_embedding = nn.Embedding(self.id_buckets, self.id_emb_dim, padding_idx=0)

        # Per-Particle MLP (Phi)
        # Input: Embedding(16) + E(1) + pT(1) + eta(1) + phi(1) = 20
        self.phi = nn.Sequential(
            nn.Linear(20, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
        )

        # Global MLP (Rho)
        # Input: MeanPool(128) + MaxPool(128) + Globals(2) = 258
        self.rho = nn.Sequential(
            nn.Linear(258, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )

    # <LLM: optionally build extra layers here>

    def forward(self, batch_x):
        # batch_x: (B, 92)

        # 1. Globals (indices 0, 1)
        globals_ = batch_x[:, :2]

        # 2. Particles (indices 2 to 91)
        # Reshape to (B, 18, 5)
        particles = batch_x[:, 2:].view(-1, 18, 5)

        # Split ID and Kinematics
        # Obj ID is at index 0 of the last dim in 'particles' tensor
        p_id = particles[:, :, 0].long()
        p_kin = particles[:, :, 1:] # Features: E, pT, eta, phi

        # Create mask for padding
        # Assuming ID=0 indicates padding (standard in zero-padded data where ID is non-zero)
        # or check if all kinematics are effectively transformed zeros.
        # Safest: check original ID. ID was not normalized.
        mask = (p_id != 0).float().unsqueeze(-1) # (B, 18, 1)

        # Embed ID
        # Hash ID to fit in embedding table
        p_id_hashed = torch.abs(p_id) % self.id_buckets
        emb = self.id_embedding(p_id_hashed) # (B, 18, 16)

        # Concat per-particle features
        x_part = torch.cat([emb, p_kin], dim=-1) # (B, 18, 20)

        # Pass through Phi (Particle mapping)
        B, N, D = x_part.shape
        x_flat = x_part.view(B*N, D)
        h = self.phi(x_flat) # (B*N, 128)
        h = h.view(B, N, -1) # (B, 18, 128)

        # Apply mask
        h = h * mask

        # Aggregation (Pooling)
        # Mean
        count = mask.sum(dim=1)
        h_mean = h.sum(dim=1) / (count + 1e-5)

        # Max
        # Masked values are 0. If activations are negative, max might pick 0 incorrectly.
        # However, for classification, max pooling on features often highlights 'presence'.
        h_max, _ = h.max(dim=1)

        # Global representation
        x_global = torch.cat([globals_, h_mean, h_max], dim=1) # (B, 258)

        # Final classification
        logits = self.rho(x_global)
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20   # <LLM: adjust if you wish>
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, 
                                              steps_per_epoch=len(train_loader), 
                                              epochs=epochs, pct_start=0.3)

    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    # State tracking
    track_tr_loss, track_val_loss = 0.0, 0.0
    track_tr_acc, track_val_acc = 0.0, 0.0

    for epoch in range(epochs):
        # TRAIN
        model.train()
        losses = []
        preds_list = []
        targets_list = []

        for X, y in train_loader:
            X, y = X.to(device), y.to(device).float()

            optimizer.zero_grad()
            logits = model(X).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds_list.append(probs.detach().cpu())
                targets_list.append(y.cpu())

        tr_loss = np.mean(losses)
        tr_preds = torch.cat(preds_list).numpy()
        tr_targets = torch.cat(targets_list).numpy()
        tr_acc = ((tr_preds > 0.5) == tr_targets).mean()

        # VALIDATION
        model.eval()
        val_losses = []
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).float()
                logits = model(X).squeeze(1)
                loss = criterion(logits, y)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits)
                val_preds_list.append(probs.cpu())
                val_targets_list.append(y.cpu())

        val_loss = np.mean(val_losses)
        val_preds = torch.cat(val_preds_list).numpy()
        val_targets = torch.cat(val_targets_list).numpy()
        val_acc = ((val_preds > 0.5) == val_targets).mean()

        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            val_auc = 0.5

        print(f"Epoch {epoch+1}/{epochs} | Loss: {tr_loss:.4f}/{val_loss:.4f} | Acc: {tr_acc:.4f}/{val_acc:.4f} | AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())

        track_tr_loss, track_val_loss = tr_loss, val_loss
        track_tr_acc, track_val_acc = tr_acc, val_acc

    # Restore best model for return
    model.load_state_dict(best_model_state)
    return model, track_tr_loss, track_val_loss, track_tr_acc, track_val_acc

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
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


