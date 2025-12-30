
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
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.kinematic_mean = None
        self.kinematic_std = None
        self.obj_id_stats = None
        self.max_abs_eta = None
        self.phi_scaling = math.pi

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "num_workers": 2},
        }

    def fit(self, X, y=None):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X

        # Global features: E_T_miss, phi_Et_miss
        global_features = X_np[:, :2]
        self.global_mean = global_features.mean(axis=0)
        self.global_std = global_features.std(axis=0) + 1e-8

        # Process object features (18 objects × 5 features)
        obj_features = X_np[:, 2:].reshape(-1, 18, 5)

        # Kinematic features: E, pT, eta, phi (skip obj_id)
        kinematic = obj_features[:, :, 1:].reshape(-1, 4)  # [n_events*18, 4]

        # Only use non-zero padded objects for statistics
        mask = (obj_features[:, :, 0] != 0).flatten()  # obj_id != 0
        valid_kinematic = kinematic[mask]

        self.kinematic_mean = valid_kinematic.mean(axis=0)
        self.kinematic_std = valid_kinematic.std(axis=0) + 1e-8

        # For eta scaling
        self.max_abs_eta = np.abs(valid_kinematic[:, 2]).max()

        # Object type statistics
        valid_obj_ids = obj_features[:, :, 0].flatten()[mask]
        unique_ids, counts = np.unique(valid_obj_ids, return_counts=True)
        self.obj_id_stats = {
            'unique_ids': unique_ids,
            'counts': counts,
            'max_id': int(unique_ids.max()) if len(unique_ids) > 0 else 0
        }

        return self

    def transform(self, X):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X

        # Normalize global features
        global_norm = (X_np[:, :2] - self.global_mean) / self.global_std

        # Reshape object features
        obj_features = X_np[:, 2:].reshape(-1, 18, 5)  # [n_events, 18, 5]

        # Create mask for valid objects (obj_id != 0)
        obj_mask = (obj_features[:, :, 0] != 0).astype(np.float32)[:, :, None]  # [n_events, 18, 1]

        # Normalize kinematic features
        kinematic = obj_features[:, :, 1:].copy()  # [n_events, 18, 4]
        kinematic = (kinematic - self.kinematic_mean) / self.kinematic_std

        # Additional feature engineering
        # 1. Trigonometric encoding for phi angles (both global and per-object)
        global_phi = X_np[:, 1:2]  # phi_Et_miss
        global_phi_sin = np.sin(global_phi / self.phi_scaling)
        global_phi_cos = np.cos(global_phi / self.phi_scaling)

        obj_phi = kinematic[:, :, 3:4] * self.kinematic_std[3] + self.kinematic_mean[3]  # Recover original phi
        obj_phi_sin = np.sin(obj_phi / self.phi_scaling)
        obj_phi_cos = np.cos(obj_phi / self.phi_scaling)

        # 2. Replace original phi with sin/cos
        kinematic = np.concatenate([
            kinematic[:, :, :3],  # E, pT, eta
            obj_phi_sin,
            obj_phi_cos
        ], axis=2)  # [n_events, 18, 5] now

        # 3. Add relative eta between consecutive objects (as a simple delta feature)
        # Use circular padding for edge cases
        eta_features = kinematic[:, :, 2:3]  # eta values
        eta_padded = np.pad(eta_features, ((0, 0), (1, 1), (0, 0)), mode='edge')
        eta_diff = eta_padded[:, 2:, :] - eta_padded[:, :-2, :]  # Forward difference
        kinematic = np.concatenate([kinematic, eta_diff], axis=2)  # [n_events, 18, 6]

        # 4. Add pT/E ratio (measure of transverseness)
        E = kinematic[:, :, 0:1] * self.kinematic_std[0] + self.kinematic_mean[0]
        pT = kinematic[:, :, 1:2] * self.kinematic_std[1] + self.kinematic_mean[1]
        pT_over_E = np.where(E != 0, pT / (E + 1e-8), 0)
        kinematic = np.concatenate([kinematic, pT_over_E], axis=2)  # [n_events, 18, 7]

        # Normalize new features
        new_feat_start = 5  # After original 4 kinematic + phi_sin/cos
        new_feats = kinematic[:, :, new_feat_start:].reshape(-1, kinematic.shape[2] - new_feat_start)
        new_feat_mean = new_feats.mean(axis=0)
        new_feat_std = new_feats.std(axis=0) + 1e-8
        kinematic[:, :, new_feat_start:] = (kinematic[:, :, new_feat_start:] - new_feat_mean) / new_feat_std

        # Object type features (one-hot encoded, limited to most frequent types)
        obj_ids = obj_features[:, :, 0:1].astype(np.int32)  # [n_events, 18, 1]

        # One-hot encode top 10 object types (plus 1 for others, 1 for padding)
        n_obj_types = min(12, len(self.obj_id_stats['unique_ids']) + 2)
        obj_onehot = np.zeros((obj_ids.shape[0], obj_ids.shape[1], n_obj_types), dtype=np.float32)

        # Create mapping: 0=padded, 1=unknown, 2+=known types
        if len(self.obj_id_stats['unique_ids']) > 0:
            # Sort by frequency
            sorted_indices = np.argsort(-self.obj_id_stats['counts'])
            top_ids = self.obj_id_stats['unique_ids'][sorted_indices[:n_obj_types-2]]
            id_to_idx = {int(id): idx+2 for idx, id in enumerate(top_ids)}

            for i in range(obj_ids.shape[0]):
                for j in range(obj_ids.shape[1]):
                    id_val = int(obj_ids[i, j, 0])
                    if id_val == 0:
                        obj_onehot[i, j, 0] = 1.0  # padded
                    elif id_val in id_to_idx:
                        obj_onehot[i, j, id_to_idx[id_val]] = 1.0
                    else:
                        obj_onehot[i, j, 1] = 1.0  # unknown

        # Combine all object features
        obj_combined = np.concatenate([
            kinematic,  # [n_events, 18, 7]
            obj_onehot,  # [n_events, 18, n_obj_types]
            obj_mask  # [n_events, 18, 1]
        ], axis=2)  # Total per-object features: 7 + n_obj_types + 1

        # Add enhanced global features
        global_enhanced = np.concatenate([
            global_norm,  # [n_events, 2]
            global_phi_sin,  # [n_events, 1]
            global_phi_cos,  # [n_events, 1]
            np.sum(obj_mask, axis=1),  # count of valid objects [n_events, 1]
            np.sum(kinematic[:, :, 1] * obj_mask[:, :, 0], axis=1, keepdims=True),  # sum pT [n_events, 1]
        ], axis=1)

        # Normalize enhanced global features (skip first 2 already normalized)
        if global_enhanced.shape[1] > 2:
            global_extra = global_enhanced[:, 2:]
            global_extra_mean = global_extra.mean(axis=0)
            global_extra_std = global_extra.std(axis=0) + 1e-8
            global_enhanced[:, 2:] = (global_extra - global_extra_mean) / global_extra_std

        # Return as dictionary for multi-input model
        return {
            'global': global_enhanced.astype(np.float32),
            'objects': obj_combined.astype(np.float32),
            'mask': obj_mask.astype(np.float32)
        }

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, dim]
        batch_size, seq_len, _ = x.shape

        # Self-attention
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        x = self.proj(x)
        x = self.dropout(x)
        x = residual + x

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Extract dimensions from sample
        global_dim = sample_object['global'].shape[1]  # ~6 features
        obj_dim = sample_object['objects'].shape[2]    # ~20 features

        # Object processing branch
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_dim, 128),
            nn.BatchNorm1d(128) if len(sample_object['objects'].shape) == 2 else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Attention-based aggregation
        self.attention = AttentionBlock(256, num_heads=8, dropout=0.1)

        # Global feature processing
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Final classifier
        combined_dim = 256 + 128  # obj features + global features
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, batch_x):
        # Handle dictionary input from preprocessor
        if isinstance(batch_x, dict):
            global_feats = batch_x['global']
            obj_feats = batch_x['objects']
            mask = batch_x['mask']
        else:
            # Fallback for raw input
            global_feats = batch_x[:, :2]
            obj_feats = batch_x[:, 2:].view(batch_x.size(0), 18, 5)
            mask = (obj_feats[:, :, 0] != 0).float().unsqueeze(-1)

        # Process objects
        batch_size, num_objects, feat_dim = obj_feats.shape
        obj_flat = obj_feats.view(-1, feat_dim)  # [batch*18, feat_dim]
        obj_encoded = self.obj_encoder(obj_flat)
        obj_encoded = obj_encoded.view(batch_size, num_objects, -1)  # [batch, 18, 256]

        # Apply attention with mask
        mask_expanded = mask.squeeze(-1)  # [batch, 18]
        attn_mask = mask_expanded.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, 18]
        obj_attended = self.attention(obj_encoded, mask_expanded.unsqueeze(1).unsqueeze(2))

        # Aggregate object features (masked mean)
        masked_obj = obj_attended * mask
        obj_aggregated = masked_obj.sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [batch, 256]

        # Process global features
        global_encoded = self.global_encoder(global_feats)  # [batch, 128]

        # Combine features
        combined = torch.cat([obj_aggregated, global_encoded], dim=1)  # [batch, 384]

        # Final classification
        logits = self.classifier(combined)  # [batch, 1]

        return logits.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 60

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    # Loss function with class weighting (slightly imbalanced)
    pos_weight = torch.tensor([0.95]).to(device)  # Slight adjustment
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_auc = 0.0
    patience_counter = 0
    patience = 15
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb.float())
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * len(yb)
            train_preds.append(torch.sigmoid(logits).detach().cpu())
            train_targets.append(yb.cpu())

        # Calculate training metrics
        train_loss /= len(train_loader.dataset)
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_acc = ((train_preds > 0.5).float() == train_targets).float().mean().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                logits = model(xb)
                loss = criterion(logits, yb.float())

                val_loss += loss.item() * len(yb)
                val_preds.append(torch.sigmoid(logits).cpu())
                val_targets.append(yb.cpu())

        val_loss /= len(val_loader.dataset)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_acc = ((val_preds > 0.5).float() == val_targets).float().mean().item()

        # Calculate AUC
        try:
            val_auc = roc_auc_score(val_targets.numpy(), val_preds.numpy())
            train_auc = roc_auc_score(train_targets.numpy(), train_preds.numpy())
        except:
            val_auc = 0.5
            train_auc = 0.5

        # Update scheduler
        scheduler.step(val_auc)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_auc)  # Store AUC instead of accuracy
        val_accs.append(val_auc)

        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs
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

