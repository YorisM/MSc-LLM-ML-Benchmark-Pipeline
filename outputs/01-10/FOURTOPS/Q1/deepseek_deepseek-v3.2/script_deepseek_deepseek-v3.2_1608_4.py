
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ---------- IMPORTS ----------
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import copy

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
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
        self.global_scaler = StandardScaler()
        self.object_scalers = [StandardScaler() for _ in range(5)]  # For obj_id, E, pT, eta, phi
        self.edge_scalers = [StandardScaler() for _ in range(4)]     # For edge features
        self.num_objects = 18
        self.object_feat_len = 5
        self.global_feat_len = 2

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
            "eval_overrides": {"shuffle": False, "batch_size": 1024, "num_workers": 4, "pin_memory": True}
        }

    def _extract_object_features(self, X):
        """Extract object features from flat tensor."""
        batch_size = X.shape[0]
        # Reshape to [batch_size, num_objects, object_feat_len]
        objects = X[:, 2:].reshape(batch_size, self.num_objects, self.object_feat_len)
        return objects

    def _create_edge_features(self, objects):
        """Create edge features between objects."""
        batch_size, num_objects, feat_dim = objects.shape
        # Use only kinematic features (E, pT, eta, phi) for edge computation
        kinematic = objects[:, :, 1:]  # [batch_size, num_objects, 4]

        # Create delta features between all pairs
        kinematic_a = kinematic.unsqueeze(2)  # [batch_size, num_objects, 1, 4]
        kinematic_b = kinematic.unsqueeze(1)  # [batch_size, 1, num_objects, 4]

        delta = kinematic_a - kinematic_b  # [batch_size, num_objects, num_objects, 4]
        delta_abs = torch.abs(delta)

        # Combine for edge features: [batch_size, num_objects, num_objects, 8]
        edge_features = torch.cat([delta, delta_abs], dim=-1)
        return edge_features

    def fit(self, X, y=None):
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Fit global features
        self.global_scaler.fit(X_np[:, :2])

        # Fit object features (only non-zero objects)
        objects = self._extract_object_features(torch.tensor(X_np)).numpy()
        objects_flat = objects.reshape(-1, self.object_feat_len)
        mask = objects_flat[:, 0] != 0  # obj_id != 0 indicates real object

        for i in range(self.object_feat_len):
            self.object_scalers[i].fit(objects_flat[mask, i:i+1])

        # Fit edge features
        objects_tensor = torch.tensor(X_np[:1000])  # Use subset for efficiency
        edge_features = self._create_edge_features(self._extract_object_features(objects_tensor))
        edge_flat = edge_features.reshape(-1, 8)[:, :4]  # Use only first 4 delta features

        for i in range(4):
            self.edge_scalers[i].fit(edge_flat[:, i:i+1].numpy())

        return self

    def transform(self, X):
        X_np = X.numpy() if torch.is_tensor(X) else X
        batch_size = X_np.shape[0]

        # Transform global features
        global_feats = self.global_scaler.transform(X_np[:, :2])

        # Transform object features
        objects = self._extract_object_features(torch.tensor(X_np)).numpy()
        objects_flat = objects.reshape(-1, self.object_feat_len)
        mask = objects_flat[:, 0] != 0

        for i in range(self.object_feat_len):
            objects_flat[mask, i:i+1] = self.object_scalers[i].transform(objects_flat[mask, i:i+1])

        objects_transformed = objects_flat.reshape(batch_size, self.num_objects, self.object_feat_len)

        # Create enhanced features
        # 1. Object counts by type
        obj_ids = objects_transformed[:, :, 0]
        obj_counts = np.stack([
            np.sum(obj_ids == 1, axis=1),  # type 1
            np.sum(obj_ids == 2, axis=1),  # type 2
            np.sum(obj_ids == 3, axis=1),  # type 3
            np.sum(obj_ids == 4, axis=1),  # type 4
            np.sum(obj_ids > 0, axis=1)    # total real objects
        ], axis=1)

        # 2. Kinematic summaries
        pT_all = objects_transformed[:, :, 2]
        eta_all = objects_transformed[:, :, 3]
        phi_all = objects_transformed[:, :, 4]

        # Mask for real objects
        real_mask = (obj_ids != 0).astype(float)
        pT_masked = pT_all * real_mask
        eta_masked = eta_all * real_mask

        # Summary statistics
        pT_sum = np.sum(pT_masked, axis=1)
        pT_max = np.max(pT_masked, axis=1)
        pT_mean = np.sum(pT_masked, axis=1) / (np.sum(real_mask, axis=1) + 1e-8)
        eta_range = np.max(eta_masked, axis=1) - np.min(eta_masked + (1-real_mask)*100, axis=1)

        # 3. Angular features
        # Calculate delta R between all object pairs (simplified)
        delta_phi = phi_all[:, :, np.newaxis] - phi_all[:, np.newaxis, :]
        delta_eta = eta_all[:, :, np.newaxis] - eta_all[:, np.newaxis, :]
        delta_R = np.sqrt(delta_phi**2 + delta_eta**2)

        # Mask diagonal and zero objects
        mask_diag = np.ones((batch_size, self.num_objects, self.num_objects))
        for i in range(self.num_objects):
            mask_diag[:, i, i] = 0

        real_mask_expanded = real_mask[:, :, np.newaxis] * real_mask[:, np.newaxis, :]
        valid_pairs = mask_diag * real_mask_expanded

        # Find minimum deltaR between real objects
        delta_R_masked = delta_R + (1 - valid_pairs) * 1000
        min_delta_R = np.min(delta_R_masked, axis=(1, 2))

        # Combine all engineered features
        engineered_features = np.stack([
            obj_counts[:, 0], obj_counts[:, 1], obj_counts[:, 2], obj_counts[:, 3], obj_counts[:, 4],
            pT_sum, pT_max, pT_mean, eta_range, min_delta_R,
            global_feats[:, 0], global_feats[:, 1]  # Keep original global features
        ], axis=1)  # [batch_size, 12]

        # Flatten objects for final output
        objects_flat_out = objects_transformed.reshape(batch_size, -1)

        # Combine engineered features with flattened objects
        output = np.concatenate([engineered_features, objects_flat_out], axis=1)  # [batch_size, 12 + 90 = 102]

        return torch.tensor(output, dtype=torch.float32)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[-1]  # Should be 102

        # Enhanced feature processing (first 12 features)
        self.enhanced_net = nn.Sequential(
            nn.Linear(12, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU()
        )

        # Object feature processing (90 features flattened -> 18 objects * 5)
        self.object_proj = nn.Linear(5, 32)
        self.object_norm = nn.LayerNorm(32)

        # Transformer encoder for objects
        self.transformer_layers = nn.ModuleList([
            AttentionBlock(32, num_heads=8, dropout=0.1) for _ in range(4)
        ])

        # Pooling and final layers
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Final classifier with multiple heads
        self.final_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(3)
        ])

        self.attention_weights = nn.Parameter(torch.ones(3))

    def forward(self, batch_x):
        # batch_x: [B, 102]
        batch_size = batch_x.shape[0]

        # Split features
        enhanced_features = batch_x[:, :12]  # [B, 12]
        object_features_flat = batch_x[:, 12:]  # [B, 90]

        # Process enhanced features
        enhanced_encoded = self.enhanced_net(enhanced_features)  # [B, 32]

        # Process object features
        objects = object_features_flat.view(batch_size, 18, 5)  # [B, 18, 5]
        object_mask = (objects[:, :, 0] == 0)  # [B, 18], True for padded objects

        # Project object features
        object_proj = self.object_proj(objects)  # [B, 18, 32]
        object_proj = self.object_norm(object_proj)

        # Apply transformer with masking
        x = object_proj
        for layer in self.transformer_layers:
            x = layer(x, mask=object_mask)

        # Global pooling
        x_pooled = self.pool(x.transpose(1, 2)).squeeze(-1)  # [B, 32]

        # Combine features
        combined = torch.cat([enhanced_encoded, x_pooled], dim=-1)  # [B, 64]

        # Multiple head ensemble
        outputs = []
        for head in self.final_layers:
            outputs.append(head(combined))

        # Weighted ensemble
        weights = F.softmax(self.attention_weights, dim=0)
        final_output = sum(w * out for w, out in zip(weights, outputs))

        return final_output.squeeze(-1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Loss function with label smoothing
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2]).to(device))

    # Optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    # Gradient scaler for mixed precision
    scaler = GradScaler()

    # Early stopping
    best_val_auc = 0
    patience = 10
    patience_counter = 0
    best_model_state = None

    # Training history
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.float().to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = torch.sigmoid(output) > 0.5
            train_correct += (preds == target.bool()).sum().item()
            train_total += target.size(0)

            all_train_preds.extend(output.detach().cpu().numpy())
            all_train_labels.extend(target.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.float().to(device)

                with autocast():
                    output = model(data)
                    loss = criterion(output, target)

                val_loss += loss.item()
                preds = torch.sigmoid(output) > 0.5
                val_correct += (preds == target.bool()).sum().item()
                val_total += target.size(0)

                all_val_preds.extend(output.cpu().numpy())
                all_val_labels.extend(target.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Calculate AUC
        train_auc = roc_auc_score(all_train_labels, all_train_preds)
        val_auc = roc_auc_score(all_val_labels, all_val_preds)

        # Learning rate scheduling based on validation AUC
        scheduler.step(val_auc)

        # Early stopping based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        # Store history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

