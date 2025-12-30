
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

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch_geometric
from torch_geometric.nn import GATConv, global_mean_pool

# ---------- IMPORTS ----------
# Additional imports
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

#  -------- CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
        self.train = train

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        x = self.X[idx]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x, self.y[idx]

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler_global = StandardScaler()
        self.scaler_kinematic = StandardScaler()
        self.obj_id_stats = defaultdict(list)
        self.valid_obj_ids = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "collate": "ragged_xy",
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512},
        }

    def fit(self, X, y=None):
        # Extract global features (ET_miss, phi_ET_miss)
        global_features = X[:, :2].numpy()  # (n_samples, 2)

        # Extract kinematic features for all objects
        kinematic_features = []
        for i in range(18):
            start_idx = 2 + i * 5 + 1  # Skip obj_id, start at E
            end_idx = start_idx + 4  # E, pT, eta, phi
            kinematic_features.append(X[:, start_idx:end_idx])

        kinematic_features = torch.cat(kinematic_features, dim=0).numpy()  # (n_samples*18, 4)

        # Filter out zero-padded objects (E > 0)
        mask = kinematic_features[:, 0] > 0  # E > 0
        kinematic_features_valid = kinematic_features[mask]

        # Fit scalers
        self.scaler_global.fit(global_features)
        self.scaler_kinematic.fit(kinematic_features_valid)

        # Collect valid object IDs
        obj_ids = []
        for i in range(18):
            obj_id_col = X[:, 2 + i * 5].numpy()  # obj_id column
            obj_ids.append(obj_id_col)

        obj_ids = np.concatenate(obj_ids)  # (n_samples*18,)
        mask = kinematic_features[:, 0] > 0
        valid_obj_ids = obj_ids[mask]
        self.valid_obj_ids = np.unique(valid_obj_ids)

        return self

    def transform(self, X):
        # Extract components
        batch_size = X.shape[0]

        # Global features
        global_features = X[:, :2].numpy()  # (batch_size, 2)
        global_features = self.scaler_global.transform(global_features)  # (batch_size, 2)

        # Process each object
        obj_features_list = []
        masks = []

        for i in range(18):
            start_idx = 2 + i * 5
            obj_id = X[:, start_idx].numpy()  # (batch_size,)
            kinematic = X[:, start_idx+1:start_idx+5].numpy()  # (batch_size, 4)

            # Create mask: valid if E > 0
            mask = kinematic[:, 0] > 0  # (batch_size,)
            masks.append(mask)

            # Normalize kinematic features
            kinematic_norm = self.scaler_kinematic.transform(kinematic)  # (batch_size, 4)

            # One-hot encode object ID
            obj_id_onehot = np.zeros((batch_size, len(self.valid_obj_ids) + 1))  # +1 for unknown
            for j, valid_id in enumerate(self.valid_obj_ids):
                obj_id_onehot[:, j] = (obj_id == valid_id).astype(float)
            obj_id_onehot[:, -1] = (obj_id == 0).astype(float)  # padding

            # Combine features
            obj_features = np.concatenate([
                kinematic_norm,
                obj_id_onehot
            ], axis=1)  # (batch_size, 4 + n_obj_ids+1)

            obj_features_list.append(obj_features)

        # Stack objects
        obj_features_stacked = np.stack(obj_features_list, axis=1)  # (batch_size, 18, 4+n_obj_ids+1)
        mask_stacked = np.stack(masks, axis=1)  # (batch_size, 18)

        # Compute pairwise features (invariant mass and deltaR)
        n_objs = 18
        n_pairs = n_objs * (n_objs - 1) // 2
        pairwise_features = np.zeros((batch_size, n_pairs, 2))

        for b in range(batch_size):
            pair_idx = 0
            for i in range(n_objs):
                if not mask_stacked[b, i]:
                    continue

                # Get 4-vector components
                E_i = X[b, 2 + i*5 + 1].item()
                pT_i = X[b, 2 + i*5 + 2].item()
                eta_i = X[b, 2 + i*5 + 3].item()
                phi_i = X[b, 2 + i*5 + 4].item()

                # Convert to Cartesian (approximate for invariant mass)
                # px = pT * cos(phi), py = pT * sin(phi), pz = pT * sinh(eta)
                px_i = pT_i * math.cos(phi_i)
                py_i = pT_i * math.sin(phi_i)
                pz_i = pT_i * math.sinh(eta_i)

                for j in range(i+1, n_objs):
                    if not mask_stacked[b, j]:
                        continue

                    E_j = X[b, 2 + j*5 + 1].item()
                    pT_j = X[b, 2 + j*5 + 2].item()
                    eta_j = X[b, 2 + j*5 + 3].item()
                    phi_j = X[b, 2 + j*5 + 4].item()

                    px_j = pT_j * math.cos(phi_j)
                    py_j = pT_j * math.sin(phi_j)
                    pz_j = pT_j * math.sinh(eta_j)

                    # Invariant mass
                    E_sum = E_i + E_j
                    px_sum = px_i + px_j
                    py_sum = py_i + py_j
                    pz_sum = pz_i + pz_j

                    m2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
                    m = math.sqrt(max(0, m2)) if m2 > 0 else 0

                    # DeltaR
                    delta_eta = eta_i - eta_j
                    delta_phi = phi_i - phi_j
                    while delta_phi > math.pi:
                        delta_phi -= 2*math.pi
                    while delta_phi < -math.pi:
                        delta_phi += 2*math.pi

                    deltaR = math.sqrt(delta_eta**2 + delta_phi**2)

                    pairwise_features[b, pair_idx, 0] = m
                    pairwise_features[b, pair_idx, 1] = deltaR
                    pair_idx += 1

        # Normalize pairwise features
        pairwise_features_flat = pairwise_features.reshape(-1, 2)
        mask_pairwise = pairwise_features_flat[:, 0] > 0  # Valid pairs have m > 0
        if mask_pairwise.any():
            valid_pairs = pairwise_features_flat[mask_pairwise]
            pairwise_mean = valid_pairs.mean(axis=0)
            pairwise_std = valid_pairs.std(axis=0) + 1e-8
            pairwise_features_flat = (pairwise_features_flat - pairwise_mean) / pairwise_std
            pairwise_features = pairwise_features_flat.reshape(batch_size, n_pairs, 2)

        # Return as dictionary for easy access
        return {
            'global': torch.FloatTensor(global_features),
            'objects': torch.FloatTensor(obj_features_stacked),
            'mask': torch.BoolTensor(mask_stacked),
            'pairwise': torch.FloatTensor(pairwise_features)
        }

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class ParticleTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Object encoder
        obj_feat_dim = 4 + 21  # kinematic + one-hot (max 20 obj types + padding)
        self.obj_encoder = nn.Linear(obj_feat_dim, d_model)

        # Global feature encoder
        self.global_encoder = nn.Linear(2, d_model)

        # Pairwise feature encoder
        self.pairwise_encoder = nn.Linear(2, d_model)

        # Transformer
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout)

    def forward(self, batch_x):
        # Unpack batch
        global_feat = batch_x['global']  # (batch_size, 2)
        objects = batch_x['objects']  # (batch_size, 18, obj_feat_dim)
        mask = batch_x['mask']  # (batch_size, 18)
        pairwise = batch_x['pairwise']  # (batch_size, n_pairs, 2)

        batch_size = global_feat.size(0)

        # Encode objects
        obj_encoded = self.obj_encoder(objects)  # (batch_size, 18, d_model)

        # Add positional encoding
        obj_encoded = self.pos_encoder(obj_encoded)  # (batch_size, 18, d_model)

        # Apply transformer with mask
        obj_padding_mask = ~mask  # (batch_size, 18) True for padding
        obj_transformed = self.transformer(
            obj_encoded, 
            src_key_padding_mask=obj_padding_mask
        )  # (batch_size, 18, d_model)

        # Attention pooling over objects
        attention_weights = self.attention_pool(obj_transformed)  # (batch_size, 18, 1)
        attention_weights = attention_weights.masked_fill(obj_padding_mask.unsqueeze(-1), float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=1)
        obj_pooled = torch.sum(attention_weights * obj_transformed, dim=1)  # (batch_size, d_model)

        # Process pairwise features
        n_pairs = pairwise.size(1)
        pairwise_encoded = self.pairwise_encoder(pairwise)  # (batch_size, n_pairs, d_model)

        # Pool pairwise features
        pairwise_pooled = torch.mean(pairwise_encoded, dim=1)  # (batch_size, d_model)

        # Combine features
        combined = torch.cat([obj_pooled, pairwise_pooled], dim=1)  # (batch_size, 2*d_model)

        # Classify
        logits = self.classifier(combined)  # (batch_size, 1)

        return logits.squeeze(-1)  # (batch_size,)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Determine feature dimensions from sample
        d_model = 128

        self.model = ParticleTransformer(
            d_model=d_model,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1
        )

    def forward(self, batch_x):
        return self.model(batch_x)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=False
    )

    # For early stopping
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0
    best_model_state = None

    # For metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            outputs = model(xb)  # (batch_size,)
            loss = criterion(outputs, yb.float())
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).long()
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                outputs = model(xb)
                loss = criterion(outputs, yb.float())

                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update scheduler
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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

