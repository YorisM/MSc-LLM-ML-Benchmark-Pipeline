
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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------- IMPORTS ----------
# Additional imports beyond template
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler_global = StandardScaler()
        self.scaler_obj = StandardScaler()
        self.obj_feature_indices = []
        self.global_feature_indices = []
        self.feature_names = []

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {
                "shuffle": False,
                "batch_size": 1024
            }
        }

    def fit(self, X, y=None):
        X_np = X.numpy()
        n_samples = X_np.shape[0]

        # Identify global and object features
        # Global features: E_T_miss, phi_Et_miss
        self.global_feature_indices = [0, 1]

        # Object features: 18 objects × 5 features each
        self.obj_feature_indices = []
        obj_features_list = []

        for obj_idx in range(18):
            start_idx = 2 + obj_idx * 5
            # Only use non-zero padded objects for statistics
            obj_id_col = X_np[:, start_idx]
            mask = obj_id_col != 0
            if np.sum(mask) > 100:  # Only if we have enough samples
                for offset in range(1, 5):  # E, pT, eta, phi (skip obj_id)
                    feat_col = X_np[:, start_idx + offset]
                    obj_features_list.append(feat_col[mask])
                    self.obj_feature_indices.append(start_idx + offset)

        # Fit scalers
        global_features = X_np[:, self.global_feature_indices]
        self.scaler_global.fit(global_features)

        if obj_features_list:
            obj_features = np.column_stack(obj_features_list)
            self.scaler_obj.fit(obj_features)

        return self

    def transform(self, X):
        X_np = X.numpy() if torch.is_tensor(X) else X
        X_transformed = X_np.copy()

        # Transform global features
        if len(self.global_feature_indices) > 0:
            global_features = X_np[:, self.global_feature_indices]
            global_scaled = self.scaler_global.transform(global_features)
            X_transformed[:, self.global_feature_indices] = global_scaled

        # Transform object features
        if len(self.obj_feature_indices) > 0:
            obj_features = X_np[:, self.obj_feature_indices]
            obj_scaled = self.scaler_obj.transform(obj_features)
            X_transformed[:, self.obj_feature_indices] = obj_scaled

        # Additional feature engineering
        # 1. Create object masks (non-zero objects)
        obj_masks = []
        for i in range(18):
            obj_id = X_np[:, 2 + i*5]
            mask = (obj_id != 0).astype(np.float32)[:, np.newaxis]
            obj_masks.append(mask)

        # 2. Create aggregated features
        # Number of objects per event
        n_objects = np.sum([X_np[:, 2 + i*5] != 0 for i in range(18)], axis=0, keepdims=True).T
        n_objects = n_objects / 18.0  # Normalize

        # Sum of transverse momenta
        sum_pt = np.zeros((X_np.shape[0], 1), dtype=np.float32)
        for i in range(18):
            pt_idx = 2 + i*5 + 2  # pT is at offset 2 from object start
            sum_pt[:, 0] += X_np[:, pt_idx]
        sum_pt = sum_pt / 100000.0  # Scale to reasonable range

        # H_T-like feature (scalar sum of pT)
        ht = np.zeros((X_np.shape[0], 1), dtype=np.float32)
        for i in range(18):
            pt_idx = 2 + i*5 + 2
            ht[:, 0] += np.abs(X_np[:, pt_idx])
        ht = ht / 100000.0

        # Combine all engineered features
        engineered_features = np.hstack([
            n_objects,
            sum_pt,
            ht,
            *obj_masks
        ])

        # Combine with original features
        result = np.hstack([X_transformed, engineered_features])

        # Final standardization
        result = (result - result.mean(axis=0, keepdims=True)) / (result.std(axis=0, keepdims=True) + 1e-8)

        # Replace NaN with 0
        result = np.nan_to_num(result, nan=0.0)

        return torch.from_numpy(result).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x, mask=None):
        # x: [batch, seq_len, features]
        attn_weights = self.attention(x)  # [batch, seq_len, 1]
        if mask is not None:
            attn_weights = attn_weights * mask.unsqueeze(-1)
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)
        pooled = (x * attn_weights).sum(dim=1)  # [batch, features]
        return pooled

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object: [batch_size, features]
        input_dim = sample_object.shape[1]  # Dynamic input dimension

        # Object-aware processing (treat first 92 features specially)
        self.obj_encoder = nn.Sequential(
            nn.Linear(5, 32),  # Process each object's 5 features
            nn.GELU(),
            nn.Linear(32, 64),
            nn.GELU()
        )

        # Attention pooling for objects
        self.obj_attention = AttentionPooling(64)

        # Main feature processor
        self.main_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            ResidualBlock(256, dropout=0.2),
            ResidualBlock(256, dropout=0.2),
            nn.LayerNorm(256),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 128),  # Combine main features + object features
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, batch_x):
        # batch_x: [batch_size, features]
        batch_size = batch_x.shape[0]

        # Process object features separately
        obj_features_list = []
        obj_masks = []

        for i in range(18):
            obj_start = 2 + i * 5
            obj_data = batch_x[:, obj_start:obj_start+5]  # [batch, 5]
            obj_id = batch_x[:, obj_start]  # First feature is object ID

            # Create mask for non-zero objects
            mask = (obj_id != 0).float()  # [batch]
            obj_masks.append(mask.unsqueeze(-1))  # [batch, 1]

            # Encode object
            encoded = self.obj_encoder(obj_data)  # [batch, 64]
            obj_features_list.append(encoded.unsqueeze(1))  # [batch, 1, 64]

        # Stack object features: [batch, 18, 64]
        obj_features = torch.cat(obj_features_list, dim=1)
        obj_mask = torch.cat(obj_masks, dim=1)  # [batch, 18]

        # Apply attention pooling
        obj_pooled = self.obj_attention(obj_features, obj_mask)  # [batch, 64]

        # Process all features
        main_features = self.main_encoder(batch_x)  # [batch, 256]

        # Combine features
        combined = torch.cat([main_features, obj_pooled], dim=1)  # [batch, 320]

        # Final classification
        logits = self.classifier(combined)  # [batch, 1]
        logits = logits.squeeze(-1)  # [batch]

        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 100

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # Loss function with label smoothing
    criterion = nn.BCEWithLogitsLoss()

    # Mixed precision training
    scaler = GradScaler()

    # For tracking best model
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    max_patience = 15

    # Lists for metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device).float()

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # Calculate accuracy
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

                # Store for AUC calculation
                all_outputs.append(torch.sigmoid(outputs).cpu())
                all_targets.append(targets.cpu())

                # Calculate accuracy
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # Calculate AUC
        all_outputs = torch.cat(all_outputs).numpy()
        all_targets = torch.cat(all_targets).numpy()

        # Simple AUC calculation (area under ROC using trapezoidal rule)
        sorted_indices = np.argsort(all_outputs)[::-1]
        sorted_outputs = all_outputs[sorted_indices]
        sorted_targets = all_targets[sorted_indices]

        tpr = np.zeros(len(sorted_outputs) + 1)
        fpr = np.zeros(len(sorted_outputs) + 1)

        tp = fp = 0
        total_pos = np.sum(sorted_targets == 1)
        total_neg = np.sum(sorted_targets == 0)

        for i in range(len(sorted_outputs)):
            if sorted_targets[i] == 1:
                tp += 1
            else:
                fp += 1
            tpr[i+1] = tp / total_pos if total_pos > 0 else 0
            fpr[i+1] = fp / total_neg if total_neg > 0 else 0

        auc = np.trapz(tpr, fpr)

        # Early stopping based on AUC
        if auc > best_val_auc + 0.0001:  # Significant improvement
            best_val_auc = auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Update scheduler based on AUC
        scheduler.step(auc)

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

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

