
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------- IMPORTS ----------
# Additional imports needed for the model
from typing import Optional, Tuple
import math

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler_global = StandardScaler()
        self.scaler_kinematic = StandardScaler()
        self.obj_id_mapping = {}
        self.obj_id_counter = 0

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": "ragged_xy",
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def _extract_features(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract global, object, and mask from flat tensor."""
        batch_size = X.shape[0]
        # Global features: E_T_miss, phi_Et_miss
        global_feats = X[:, :2]  # [batch_size, 2]

        # Object features: reshape to [batch_size, 18, 5]
        objects = X[:, 2:].reshape(batch_size, 18, 5)  # [batch_size, 18, 5]

        # Create mask: non-zero object IDs indicate real objects
        obj_ids = objects[:, :, 0]  # [batch_size, 18]
        mask = (obj_ids != 0).float()  # [batch_size, 18]

        # Kinematic features: E, pT, eta, phi for each object
        kinematic_feats = objects[:, :, 1:]  # [batch_size, 18, 4]

        return global_feats, objects, kinematic_feats, mask

    def _compute_pairwise_features(self, kinematic_feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute invariant mass and deltaR for all object pairs."""
        batch_size, n_objects, _ = kinematic_feats.shape
        device = kinematic_feats.device

        # Extract kinematic features
        E = kinematic_feats[:, :, 0]  # [batch_size, 18]
        pT = kinematic_feats[:, :, 1]  # [batch_size, 18]
        eta = kinematic_feats[:, :, 2]  # [batch_size, 18]
        phi = kinematic_feats[:, :, 3]  # [batch_size, 18]

        # Compute px, py, pz from pT, eta, phi
        px = pT * torch.cos(phi)  # [batch_size, 18]
        py = pT * torch.sin(phi)  # [batch_size, 18]
        pz = pT * torch.sinh(eta)  # [batch_size, 18]

        # Initialize pairwise features tensor
        pairwise_feats = torch.zeros(batch_size, n_objects, n_objects, 2, device=device)

        # Compute for all pairs
        for i in range(n_objects):
            for j in range(n_objects):
                if i <= j:  # Compute symmetric pairs once
                    # Delta R
                    deta = eta[:, i] - eta[:, j]  # [batch_size]
                    dphi = phi[:, i] - phi[:, j]  # [batch_size]
                    # Wrap phi difference to [-pi, pi]
                    dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
                    deltaR = torch.sqrt(deta**2 + dphi**2)  # [batch_size]

                    # Invariant mass
                    E_sum = E[:, i] + E[:, j]  # [batch_size]
                    px_sum = px[:, i] + px[:, j]  # [batch_size]
                    py_sum = py[:, i] + py[:, j]  # [batch_size]
                    pz_sum = pz[:, i] + pz[:, j]  # [batch_size]

                    # m^2 = E^2 - p^2
                    m2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
                    # Protect against negative due to numerical errors
                    m2 = torch.clamp(m2, min=1e-6)
                    inv_mass = torch.sqrt(m2)  # [batch_size]

                    pairwise_feats[:, i, j, 0] = inv_mass
                    pairwise_feats[:, i, j, 1] = deltaR
                    pairwise_feats[:, j, i, 0] = inv_mass
                    pairwise_feats[:, j, i, 1] = deltaR

        # Apply mask: zero out features for pairs involving padded objects
        mask_expanded_i = mask.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 18, 1, 1]
        mask_expanded_j = mask.unsqueeze(1).unsqueeze(-1)   # [batch_size, 1, 18, 1]
        pair_mask = mask_expanded_i * mask_expanded_j      # [batch_size, 18, 18, 1]
        pairwise_feats = pairwise_feats * pair_mask  # [batch_size, 18, 18, 2]

        return pairwise_feats

    def fit(self, X, y=None):
        # Extract features
        global_feats, objects, kinematic_feats, mask = self._extract_features(X)

        # Fit scaler for global features
        self.scaler_global.fit(global_feats.numpy())

        # Fit scaler for kinematic features (only for real objects)
        # Flatten kinematic features for all real objects across all events
        kinematic_flat = kinematic_feats[mask.bool()].numpy()
        if len(kinematic_flat) > 0:
            self.scaler_kinematic.fit(kinematic_flat)

        # Build object ID mapping
        obj_ids = objects[:, :, 0].flatten().int().numpy()
        unique_ids = np.unique(obj_ids[obj_ids != 0])
        self.obj_id_mapping = {id_val: idx+1 for idx, id_val in enumerate(unique_ids)}
        self.obj_id_counter = len(self.obj_id_mapping) + 1  # +1 for padding

        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        batch_size = X.shape[0]

        # Extract features
        global_feats, objects, kinematic_feats, mask = self._extract_features(torch.from_numpy(X))

        # Normalize global features
        global_feats_norm = torch.from_numpy(
            self.scaler_global.transform(global_feats.numpy())
        ).float()

        # Normalize kinematic features
        kinematic_norm = kinematic_feats.clone()
        # Apply normalization only to real objects
        mask_bool = mask.bool()
        if mask_bool.any():
            kinematic_flat = kinematic_feats[mask_bool].numpy()
            if len(kinematic_flat) > 0:
                kinematic_norm_flat = self.scaler_kinematic.transform(kinematic_flat)
                kinematic_norm[mask_bool] = torch.from_numpy(kinematic_norm_flat).float()

        # Encode object IDs
        obj_ids = objects[:, :, 0].clone()
        for orig_id, mapped_id in self.obj_id_mapping.items():
            obj_ids[obj_ids == orig_id] = mapped_id
        # Unknown IDs become 0 (treated as padding)
        obj_ids[~torch.isin(obj_ids.int(), torch.tensor(list(self.obj_id_mapping.values())).int())] = 0

        # Create object features with encoded IDs
        obj_feats = torch.cat([
            obj_ids.unsqueeze(-1).float(),  # [batch_size, 18, 1]
            kinematic_norm  # [batch_size, 18, 4]
        ], dim=-1)  # [batch_size, 18, 5]

        # Compute pairwise features
        pairwise_feats = self._compute_pairwise_features(kinematic_norm, mask)  # [batch_size, 18, 18, 2]

        # Flatten pairwise features for transformer input
        # Take upper triangular part (excluding diagonal) to avoid redundancy
        triu_idx = torch.triu_indices(18, 18, offset=1)
        pairwise_flat = pairwise_feats[:, triu_idx[0], triu_idx[1], :]  # [batch_size, 153, 2]

        # Create final feature tensor
        # [batch_size, 2 (global) + 18*5 (objects) + 153*2 (pairwise)]
        output = torch.cat([
            global_feats_norm,  # [batch_size, 2]
            obj_feats.reshape(batch_size, -1),  # [batch_size, 90]
            pairwise_flat.reshape(batch_size, -1)  # [batch_size, 306]
        ], dim=-1)  # [batch_size, 398]

        return output

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[-1]

        # Embedding for object IDs (0 is padding)
        self.obj_id_embedding = nn.Embedding(num_embeddings=20, embedding_dim=16)  # Max 20 object types

        # Feature projections
        self.global_proj = nn.Linear(2, 32)
        self.kinematic_proj = nn.Linear(4, 64)
        self.pairwise_proj = nn.Linear(2, 32)

        # Transformer encoder
        self.transformer_input_dim = 32 + 64 + 32  # Global + kinematic + pairwise
        self.pos_encoding = nn.Parameter(torch.zeros(1, 18, self.transformer_input_dim))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.transformer_input_dim, nhead=8, dim_feedforward=512, dropout=0.1)
            for _ in range(4)
        ])

        # Output layers
        self.transformer_out_dim = self.transformer_input_dim
        self.final_layers = nn.Sequential(
            nn.Linear(self.transformer_out_dim * 18, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # Initialize positional encoding
        nn.init.normal_(self.pos_encoding, std=0.02)

    def forward(self, batch_x):
        batch_size = batch_x.shape[0]

        # Split features back to original components
        global_feats = batch_x[:, :2]  # [batch_size, 2]
        obj_feats = batch_x[:, 2:92].reshape(batch_size, 18, 5)  # [batch_size, 18, 5]
        pairwise_feats = batch_x[:, 92:].reshape(batch_size, 153, 2)  # [batch_size, 153, 2]

        # Extract components
        obj_ids = obj_feats[:, :, 0].long()  # [batch_size, 18]
        kinematic_feats = obj_feats[:, :, 1:]  # [batch_size, 18, 4]

        # Create mask for padded objects
        mask = (obj_ids == 0)  # [batch_size, 18]

        # Embed object IDs
        obj_id_emb = self.obj_id_embedding(obj_ids.clamp(0, 19))  # [batch_size, 18, 16]

        # Project features
        global_proj = self.global_proj(global_feats).unsqueeze(1).expand(-1, 18, -1)  # [batch_size, 18, 32]
        kinematic_proj = self.kinematic_proj(kinematic_feats)  # [batch_size, 18, 64]

        # Process pairwise features: map to object space
        # We need to reconstruct the 18x18 pairwise matrix from the flattened upper triangular
        pairwise_matrix = torch.zeros(batch_size, 18, 18, 2, device=batch_x.device)
        triu_idx = torch.triu_indices(18, 18, offset=1)
        pairwise_matrix[:, triu_idx[0], triu_idx[1], :] = pairwise_feats
        pairwise_matrix[:, triu_idx[1], triu_idx[0], :] = pairwise_feats  # Make symmetric

        # For each object, take mean of pairwise features with other objects
        pairwise_obj = pairwise_matrix.mean(dim=2)  # [batch_size, 18, 2]
        pairwise_proj = self.pairwise_proj(pairwise_obj)  # [batch_size, 18, 32]

        # Combine all features
        combined = torch.cat([global_proj, kinematic_proj, pairwise_proj], dim=-1)  # [batch_size, 18, 32+64+32=128]

        # Add positional encoding
        combined = combined + self.pos_encoding

        # Apply transformer blocks
        x = combined
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, key_padding_mask=mask)

        # Flatten transformer outputs
        x_flat = x.reshape(batch_size, -1)  # [batch_size, 18*128=2304]

        # Final classification layers
        output = self.final_layers(x_flat)  # [batch_size, 1]

        return output.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 150

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    # For storing metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Compute accuracy
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == batch_y.bool()).sum().item()
            total += batch_y.size(0)
            running_loss += loss.item() * batch_x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                # Compute accuracy
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == batch_y.bool()).sum().item()
                val_total += batch_y.size(0)
                running_val_loss += loss.item() * batch_x.size(0)

        val_loss = running_val_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pth')
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # Load best model
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, train_losses, val_losses, train_accs, val_accs

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


