
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops

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
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ---------- IMPORTS ----------
# Additional imports for enhanced preprocessing and model
import math
from typing import Tuple, Optional
from collections import OrderedDict

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_scaler = StandardScaler()
        self.obj_scaler = StandardScaler()
        self.pca = None
        self.feature_mask = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 1024,  # Increased batch size for efficiency
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True if torch.cuda.is_available() else False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {
                "shuffle": False,
                "batch_size": 2048
            }
        }

    def fit(self, X, y=None):
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Extract global features (first 2)
        global_features = X_np[:, :2]

        # Extract object features (reshape to [n_events * 18, 5])
        obj_features = X_np[:, 2:].reshape(-1, 18, 5)  # [n_events, 18, 5]
        obj_features_flat = obj_features.reshape(-1, 5)  # [n_events * 18, 5]

        # Only fit on real objects (non-zero objects)
        non_zero_mask = obj_features_flat[:, 0] != 0  # obj_id != 0
        real_obj_features = obj_features_flat[non_zero_mask]

        # Fit scalers
        self.global_scaler.fit(global_features)

        if len(real_obj_features) > 0:
            # Only scale continuous features (E, pT, eta, phi), not obj_id
            continuous_features = real_obj_features[:, 1:]
            self.obj_scaler.fit(continuous_features)

        # Feature engineering: create pairwise features
        # We'll compute these during transform
        return self

    def _compute_pairwise_features(self, event_objects):
        """Compute pairwise invariant mass and deltaR for objects in an event"""
        n_objects = event_objects.shape[0]
        if n_objects < 2:
            return np.zeros((0, 2))

        # Extract physics quantities
        obj_ids = event_objects[:, 0]  # [n_objects]
        E = event_objects[:, 1]  # [n_objects], MeV
        pT = event_objects[:, 2]  # [n_objects], MeV
        eta = event_objects[:, 3]  # [n_objects]
        phi = event_objects[:, 4]  # [n_objects]

        # Convert to GeV for numerical stability
        E = E / 1000.0
        pT = pT / 1000.0

        # Compute px, py, pz from pT, eta, phi
        px = pT * np.cos(phi)  # [n_objects]
        py = pT * np.sin(phi)  # [n_objects]
        pz = pT * np.sinh(eta)  # [n_objects]

        pairwise_features = []

        # Compute for all unique pairs (i < j)
        for i in range(n_objects):
            for j in range(i + 1, n_objects):
                # Skip if either object is padded (obj_id == 0)
                if obj_ids[i] == 0 or obj_ids[j] == 0:
                    continue

                # Invariant mass m_ij = sqrt((E_i + E_j)^2 - (p_i + p_j)^2)
                E_sum = E[i] + E[j]
                px_sum = px[i] + px[j]
                py_sum = py[i] + py[j]
                pz_sum = pz[i] + pz[j]

                # Handle numerical stability
                m2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
                if m2 < 0:
                    m2 = max(m2, 1e-6)  # Small positive value for sqrt
                m_ij = np.sqrt(abs(m2))

                # Delta R = sqrt((Δη)^2 + (Δφ)^2)
                # Handle φ periodicity
                dphi = abs(phi[i] - phi[j])
                if dphi > np.pi:
                    dphi = 2 * np.pi - dphi

                deta = eta[i] - eta[j]
                deltaR = np.sqrt(deta**2 + dphi**2)

                pairwise_features.append([m_ij, deltaR])

        return np.array(pairwise_features) if pairwise_features else np.zeros((0, 2))

    def _extract_high_level_features(self, global_feats, event_objects):
        """Extract high-level physics features"""
        features = []

        # Global features
        features.extend(global_feats)

        # Object statistics (only for real objects)
        real_mask = event_objects[:, 0] != 0
        if np.any(real_mask):
            real_objects = event_objects[real_mask]

            # Sum of transverse momenta (HT)
            HT = np.sum(real_objects[:, 2])  # Sum pT
            features.append(HT / 1000.0)  # Convert to GeV

            # Number of objects
            features.append(len(real_objects))

            # Mean and std of pT
            pT_values = real_objects[:, 2] / 1000.0  # GeV
            features.append(np.mean(pT_values))
            features.append(np.std(pT_values))

            # Mean and std of eta
            features.append(np.mean(real_objects[:, 3]))
            features.append(np.std(real_objects[:, 3]))
        else:
            # No real objects
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(features)

    def transform(self, X):
        X_np = X.numpy() if torch.is_tensor(X) else X
        n_events = X_np.shape[0]

        # Transform global features
        global_feats = self.global_scaler.transform(X_np[:, :2])  # [n_events, 2]

        # Transform object features
        obj_features = X_np[:, 2:].reshape(-1, 18, 5)  # [n_events, 18, 5]
        transformed_objects = np.zeros_like(obj_features)

        for i in range(n_events):
            event_objects = obj_features[i]  # [18, 5]
            real_mask = event_objects[:, 0] != 0

            if np.any(real_mask):
                # Scale continuous features for real objects
                continuous = event_objects[real_mask, 1:]  # [n_real, 4]
                if continuous.shape[0] > 0:
                    scaled_continuous = self.obj_scaler.transform(continuous)
                    transformed_objects[i, real_mask, 0] = event_objects[real_mask, 0]  # Keep obj_id
                    transformed_objects[i, real_mask, 1:] = scaled_continuous

        # Flatten objects back
        obj_flat = transformed_objects.reshape(n_events, -1)  # [n_events, 90]

        # Extract high-level features
        high_level_feats = []
        for i in range(n_events):
            feats = self._extract_high_level_features(
                global_feats[i], 
                obj_features[i]
            )
            high_level_feats.append(feats)

        high_level_feats = np.array(high_level_feats)  # [n_events, n_high_level]

        # Combine all features
        combined = np.concatenate([
            global_feats,          # [n_events, 2]
            high_level_feats,      # [n_events, 6]
            obj_flat              # [n_events, 90]
        ], axis=1)  # Total: [n_events, 98]

        return torch.FloatTensor(combined)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention
        src2, _ = self.self_attn(src, src, src, 
                                attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Input shape: [batch_size, 98]
        # We'll reshape to treat objects as sequence

        # Projection to higher dimension
        self.input_proj = nn.Sequential(
            nn.Linear(98, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Positional encoding for objects (treat first 2+6=8 features as global, rest 90/5=18 objects)
        self.pos_encoder = nn.Parameter(torch.randn(1, 20, 256))  # 18 objects + 2 global tokens

        # Transformer encoder
        self.transformer = nn.ModuleList([
            TransformerBlock(256, 8, 512, dropout=0.1) for _ in range(6)
        ])

        # Pooling layer
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch_x):
        # batch_x shape: [B, 98]
        B = batch_x.shape[0]

        # Project to higher dimension
        x = self.input_proj(batch_x)  # [B, 256]

        # Reshape for transformer: treat as sequence of 20 tokens
        # First token: global features (first 8)
        # Next 18 tokens: object features (each 5 features -> 256 dim via projection)
        x_global = batch_x[:, :8]  # [B, 8]
        x_objects = batch_x[:, 8:].reshape(B, 18, 5)  # [B, 18, 5]

        # Project objects
        obj_proj = nn.Linear(5, 256).to(x.device)
        x_objects = obj_proj(x_objects)  # [B, 18, 256]

        # Create mask for padded objects (obj_id == 0)
        obj_ids = batch_x[:, 8::5]  # [B, 18] - obj_id for each object
        padding_mask = (obj_ids == 0)  # [B, 18]

        # Combine global and object tokens
        global_token = torch.zeros(B, 1, 256).to(x.device)  # Placeholder
        global_token = self.input_proj(x_global.unsqueeze(1))  # [B, 1, 256]

        # Concatenate global token with object tokens
        x_seq = torch.cat([global_token, x_objects], dim=1)  # [B, 19, 256]

        # Add positional encoding
        x_seq = x_seq + self.pos_encoder[:, :x_seq.size(1), :]

        # Create attention mask (global token can attend to all, objects cannot attend to padded objects)
        # For transformer, we need [B, seq_len] mask where True means ignore
        extended_padding_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool).to(x.device),  # Global token
            padding_mask
        ], dim=1)  # [B, 19]

        # Apply transformer blocks
        for transformer_block in self.transformer:
            x_seq = transformer_block(x_seq, src_key_padding_mask=extended_padding_mask)

        # Global average pooling
        x_pooled = x_seq.mean(dim=1)  # [B, 256]

        # Classify
        logits = self.classifier(x_pooled).squeeze(-1)  # [B]
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 100

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # For early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience = 15
    patience_counter = 0

    # Metrics storage
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).float()

            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            # Calculate accuracy
            preds = (torch.sigmoid(output) > 0.5).float()
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device).float()
                output = model(data).squeeze()
                loss = criterion(output, target)

                total_val_loss += loss.item()

                preds = (torch.sigmoid(output) > 0.5).float()
                val_correct += (preds == target).sum().item()
                val_total += target.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        # Early stopping check
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

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

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane_fourtops(spec, first_batch)
    view = make_view_by_lane_fourtops(mode, first_batch, device)

    # Build model
    model = make_model(view.batch_x).to(device)

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
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

