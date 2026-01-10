
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
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------- IMPORTS ----------
# Additional imports
import math
from typing import List, Tuple
from collections import OrderedDict

# -------- CUSTOM DATASET FOR GRAPH CONSTRUCTION --------
class ParticleGraphDataset(Dataset):
    """Dataset that converts flat features to graph representation."""

    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        # Transform through preprocessor
        if pre is not None:
            X_transformed = pre.transform(X)
        else:
            X_transformed = X

        # Handle different return types from transform
        if isinstance(X_transformed, tuple):
            self.global_features, self.particle_features, self.adj_matrix = X_transformed
        else:
            self.global_features = X_transformed
            self.particle_features = None
            self.adj_matrix = None

        self.y = y.long() if torch.is_tensor(y) else torch.tensor(y, dtype=torch.long)
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.particle_features is not None:
            # Return graph representation
            return (self.global_features[idx], 
                   self.particle_features[idx], 
                   self.adj_matrix[idx] if self.adj_matrix is not None else None), self.y[idx]
        else:
            # Return flat features
            return self.global_features[idx], self.y[idx]

# ----------- PRE-PROCESSING -----------
class MyPreprocessor:
    """Preprocessor with feature engineering for particle physics."""

    def __init__(self):
        self.global_scaler = StandardScaler()
        self.particle_scaler = StandardScaler()
        self.pairwise_scaler = StandardScaler()
        self.compute_pairwise = True
        self.max_particles = 18
        self.global_feat_dim = 2
        self.particle_feat_dim = 5

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:ParticleGraphDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 256,  # Reduced for memory efficiency
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True if torch.cuda.is_available() else False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {
                "shuffle": False,
                "batch_size": 512
            }
        }

    def _extract_particles(self, X_flat):
        """Extract particle features from flat representation."""
        batch_size = X_flat.shape[0]

        # Global features: E_T_miss and phi_E_T_miss
        global_features = X_flat[:, :2]  # [B, 2]

        # Particle features: reshape to [B, 18, 5]
        particle_features = X_flat[:, 2:].reshape(batch_size, self.max_particles, self.particle_feat_dim)

        # Create mask for real particles (object id > 0)
        particle_mask = particle_features[:, :, 0] > 0  # [B, 18]

        return global_features, particle_features, particle_mask

    def _compute_pairwise_features(self, particle_features, particle_mask):
        """Compute pairwise features (invariant mass and deltaR)."""
        batch_size = particle_features.shape[0]
        device = particle_features.device if torch.is_tensor(particle_features) else 'cpu'

        # Extract kinematics for real particles
        # obj_id, E, pT, eta, phi
        E = particle_features[:, :, 1]  # [B, 18]
        pT = particle_features[:, :, 2]  # [B, 18]
        eta = particle_features[:, :, 3]  # [B, 18]
        phi = particle_features[:, :, 4]  # [B, 18]

        # Convert to cartesian coordinates
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)

        # Initialize pairwise features
        n_particles = self.max_particles
        n_pairs = n_particles * (n_particles - 1) // 2
        pairwise_features = torch.zeros(batch_size, n_pairs, 2, device=device)

        idx = 0
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                # Only compute for pairs where both particles are real
                mask_i = particle_mask[:, i]
                mask_j = particle_mask[:, j]
                valid_pair = mask_i & mask_j

                if valid_pair.any():
                    # Compute invariant mass
                    E_sum = E[:, i] + E[:, j]
                    px_sum = px[:, i] + px[:, j]
                    py_sum = py[:, i] + py[:, j]
                    pz_sum = pz[:, i] + pz[:, j]

                    m2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
                    m = torch.sqrt(torch.clamp(m2, min=1e-6))

                    # Compute deltaR
                    deta = eta[:, i] - eta[:, j]
                    dphi = phi[:, i] - phi[:, j]
                    # Normalize dphi to [-pi, pi]
                    dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
                    deltaR = torch.sqrt(deta**2 + dphi**2)

                    # Fill features
                    pairwise_features[valid_pair, idx, 0] = m[valid_pair]
                    pairwise_features[valid_pair, idx, 1] = deltaR[valid_pair]

                idx += 1

        return pairwise_features

    def _build_adjacency_matrix(self, particle_features, particle_mask):
        """Build adjacency matrix based on particle relationships."""
        batch_size = particle_features.shape[0]
        device = particle_features.device if torch.is_tensor(particle_features) else 'cpu'

        # Simple adjacency: connect particles within deltaR < 2.0
        eta = particle_features[:, :, 3]  # [B, 18]
        phi = particle_features[:, :, 4]  # [B, 18]

        adj_matrix = torch.zeros(batch_size, self.max_particles, self.max_particles, device=device)

        for b in range(batch_size):
            # Get real particles in this event
            real_indices = torch.where(particle_mask[b])[0]
            n_real = len(real_indices)

            if n_real > 1:
                # Compute deltaR between all real particles
                for i_idx, i in enumerate(real_indices):
                    for j_idx, j in enumerate(real_indices[i_idx+1:], i_idx+1):
                        deta = eta[b, i] - eta[b, j]
                        dphi = phi[b, i] - phi[b, j]
                        dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
                        deltaR = torch.sqrt(deta**2 + dphi**2)

                        # Connect if deltaR < 2.0
                        if deltaR < 2.0:
                            adj_matrix[b, i, j] = 1.0
                            adj_matrix[b, j, i] = 1.0

        return adj_matrix

    def fit(self, X, y=None):
        # Convert to numpy for sklearn scaling
        if torch.is_tensor(X):
            X_np = X.numpy()
        else:
            X_np = X

        # Extract features
        global_features, particle_features, particle_mask = self._extract_particles(torch.tensor(X_np))

        # Flatten particle features for scaling (only real particles)
        batch_size = particle_features.shape[0]
        real_particles = []
        for b in range(batch_size):
            real_indices = torch.where(particle_mask[b])[0]
            if len(real_indices) > 0:
                # Use all 5 features including object ID
                real_particles.append(particle_features[b, real_indices].numpy())

        if len(real_particles) > 0:
            real_particles_np = np.vstack(real_particles)
            # Fit particle scaler
            self.particle_scaler.fit(real_particles_np)

        # Fit global scaler
        self.global_scaler.fit(global_features.numpy())

        if self.compute_pairwise:
            # Fit pairwise scaler
            pairwise_features = self._compute_pairwise_features(particle_features, particle_mask)
            # Flatten and remove zeros
            pairwise_flat = pairwise_features.reshape(-1, 2)
            nonzero_mask = pairwise_flat[:, 0] > 0
            if nonzero_mask.any():
                self.pairwise_scaler.fit(pairwise_flat[nonzero_mask].numpy())

        return self

    def transform(self, X):
        # Convert to tensor if needed
        if not torch.is_tensor(X):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X.float()

        batch_size = X_tensor.shape[0]

        # Extract features
        global_features, particle_features, particle_mask = self._extract_particles(X_tensor)

        # Scale global features
        global_features_np = global_features.numpy()
        global_features_scaled = self.global_scaler.transform(global_features_np)
        global_features_tensor = torch.tensor(global_features_scaled, dtype=torch.float32)

        # Scale particle features
        particle_features_np = particle_features.numpy()
        orig_shape = particle_features_np.shape
        particle_features_flat = particle_features_np.reshape(-1, orig_shape[-1])
        particle_features_scaled = self.particle_scaler.transform(particle_features_flat)
        particle_features_tensor = torch.tensor(
            particle_features_scaled.reshape(orig_shape), dtype=torch.float32
        )

        # Compute pairwise features
        pairwise_features = self._compute_pairwise_features(particle_features_tensor, particle_mask)

        # Scale pairwise features
        if self.compute_pairwise and pairwise_features.sum() > 0:
            pairwise_np = pairwise_features.numpy()
            pairwise_flat = pairwise_np.reshape(-1, 2)
            nonzero_mask = pairwise_flat[:, 0] != 0
            if nonzero_mask.any():
                pairwise_scaled = pairwise_flat.copy()
                pairwise_scaled[nonzero_mask] = self.pairwise_scaler.transform(pairwise_flat[nonzero_mask])
                pairwise_features_tensor = torch.tensor(
                    pairwise_scaled.reshape(pairwise_np.shape), dtype=torch.float32
                )
            else:
                pairwise_features_tensor = pairwise_features
        else:
            pairwise_features_tensor = pairwise_features

        # Build adjacency matrix
        adj_matrix = self._build_adjacency_matrix(particle_features_tensor, particle_mask)

        return global_features_tensor, particle_features_tensor, pairwise_features_tensor

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class ParticleAttentionLayer(nn.Module):
    """Self-attention layer for particle features."""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: [B, N, D]
        attn_output, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm(x + self.dropout(attn_output))
        return x

class BinaryClassifier(nn.Module):
    """Hybrid model with attention and MLP for particle classification."""

    def __init__(self, sample_object):
        super().__init__()

        # Input dimensions
        self.global_dim = 2
        self.particle_dim = 5  # Original particle features
        self.max_particles = 18
        self.pairwise_dim = 306  # 153 pairs * 2 features

        # Particle feature encoder
        self.particle_encoder = nn.Sequential(
            nn.Linear(self.particle_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Attention layers for particles
        self.particle_attention = nn.ModuleList([
            ParticleAttentionLayer(128, n_heads=4, dropout=0.1),
            ParticleAttentionLayer(128, n_heads=4, dropout=0.1)
        ])

        # Pairwise feature encoder
        self.pairwise_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Final classifier
        total_features = 128 + 64 + 64  # Particle + Pairwise + Global
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, batch_x):
        # Unpack input
        if isinstance(batch_x, tuple) and len(batch_x) == 3:
            global_features, particle_features, pairwise_features = batch_x
        else:
            # Fallback for flat features
            global_features = batch_x[:, :2]
            particle_features = batch_x[:, 2:].reshape(-1, 18, 5)
            pairwise_features = None

        batch_size = global_features.shape[0]

        # 1. Process global features
        global_encoded = self.global_encoder(global_features)  # [B, 64]

        # 2. Process particle features with attention
        # Create mask for zero-padded particles
        particle_mask = particle_features[:, :, 0] == 0  # [B, 18]

        # Encode particle features
        particle_encoded = self.particle_encoder(particle_features)  # [B, 18, 128]

        # Apply attention
        for attn_layer in self.particle_attention:
            particle_encoded = attn_layer(particle_encoded, key_padding_mask=particle_mask)

        # Pool particle features (mean pooling, ignoring padded particles)
        particle_pooled = torch.zeros(batch_size, 128, device=global_features.device)
        for b in range(batch_size):
            real_indices = ~particle_mask[b]
            if real_indices.any():
                particle_pooled[b] = particle_encoded[b, real_indices].mean(dim=0)

        # 3. Process pairwise features if available
        if pairwise_features is not None and pairwise_features.sum() > 0:
            # Reshape pairwise features [B, 153, 2]
            pairwise_flat = pairwise_features.reshape(batch_size * 153, 2)
            pairwise_encoded = self.pairwise_encoder(pairwise_flat)
            pairwise_encoded = pairwise_encoded.reshape(batch_size, 153, 64)

            # Pool pairwise features (mean pooling)
            pairwise_mask = pairwise_features[:, :, 0] != 0
            pairwise_pooled = torch.zeros(batch_size, 64, device=global_features.device)
            for b in range(batch_size):
                valid_pairs = pairwise_mask[b]
                if valid_pairs.any():
                    pairwise_pooled[b] = pairwise_encoded[b, valid_pairs].mean(dim=0)
        else:
            pairwise_pooled = torch.zeros(batch_size, 64, device=global_features.device)

        # 4. Concatenate all features
        combined = torch.cat([particle_pooled, pairwise_pooled, global_encoded], dim=-1)  # [B, 256]

        # 5. Final classification
        logits = self.classifier(combined)  # [B, 1]

        return logits.squeeze(-1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Early stopping
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None

    # Metrics tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move to device
            if isinstance(data, tuple):
                data = tuple(d.to(device) for d in data)
            else:
                data = data.to(device)
            target = target.to(device).float()

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            predictions = (torch.sigmoid(output) > 0.5).float()
            train_correct += (predictions == target).sum().item()
            train_total += target.size(0)

            # Update LR scheduler per batch
            scheduler.step(epoch + batch_idx / len(train_loader))

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                if isinstance(data, tuple):
                    data = tuple(d.to(device) for d in data)
                else:
                    data = data.to(device)
                target = target.to(device).float()

                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                predictions = (torch.sigmoid(output) > 0.5).float()
                val_correct += (predictions == target).sum().item()
                val_total += target.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_accuracy)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if epoch % 5 == 0:
            print(f'Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, '
                  f'Train Acc: {train_accuracy:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Val Acc: {val_accuracy:.4f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')

        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
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

