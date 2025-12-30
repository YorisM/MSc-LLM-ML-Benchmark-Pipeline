
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.metrics import roc_auc_score

# ---------- IMPORTS ----------
# Additional imports for preprocessing and model
from torch import nn, Tensor
from typing import Tuple, Optional
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

#  -------- CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        # Preprocess the entire batch
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
        self.train = train

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.obj_mean = None
        self.obj_std = None
        self.obj_present_threshold = 1.0  # MeV

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # X shape: [n_samples, 92]
        # Global features: E_T_miss, phi_Et_miss
        global_features = X[:, :2]
        self.global_mean = global_features.mean(axis=0, keepdims=True)
        self.global_std = global_features.std(axis=0, keepdims=True) + 1e-8

        # Object features: reshape to [n_samples, 18, 5]
        obj_features = X[:, 2:].reshape(-1, 18, 5)

        # Only consider objects with E > threshold
        mask = obj_features[:, :, 1] > self.obj_present_threshold  # E > threshold
        masked_obj = obj_features[mask]  # [n_real_objects, 5]

        # Normalize object features (skip obj_id which is index 0)
        obj_kinematics = masked_obj[:, 1:]  # [n_real_objects, 4]
        self.obj_mean = obj_kinematics.mean(axis=0, keepdims=True)
        self.obj_std = obj_kinematics.std(axis=0, keepdims=True) + 1e-8

        return self

    def transform(self, X):
        # X shape: [n_samples, 92]
        batch_size = X.shape[0]

        # Normalize global features
        global_feats = X[:, :2]  # [batch, 2]
        global_feats = (global_feats - self.global_mean) / self.global_std

        # Reshape object features
        obj_features = X[:, 2:].reshape(batch_size, 18, 5)  # [batch, 18, 5]

        # Separate object ids and kinematics
        obj_ids = obj_features[:, :, 0].long()  # [batch, 18]
        obj_kinematics = obj_features[:, :, 1:]  # [batch, 18, 4]

        # Normalize kinematics
        obj_kinematics = (obj_kinematics - self.obj_mean) / self.obj_std

        # Create mask for real objects (E > threshold)
        obj_mask = (obj_features[:, :, 1] > self.obj_present_threshold).float()  # [batch, 18]

        # Compute derived features for each object
        # 1. Mass from four-vector (assuming massless approximation for now)
        # px = pT * cos(phi), py = pT * sin(phi), pz = pT * sinh(eta)
        # For mass calculation: m^2 = E^2 - (px^2 + py^2 + pz^2)
        # But we don't have full 3-momentum, so we'll compute approximate mass
        pT = obj_kinematics[:, :, 1]  # [batch, 18]
        eta = obj_kinematics[:, :, 2]  # [batch, 18]
        phi = obj_kinematics[:, :, 3]  # [batch, 18]
        E = obj_kinematics[:, :, 0]  # [batch, 18]

        # Calculate pz = pT * sinh(eta)
        pz = pT * torch.sinh(eta)
        # Calculate p = sqrt(pT^2 + pz^2)
        p = torch.sqrt(pT**2 + pz**2 + 1e-8)
        # Approximate mass (safe calculation)
        mass_sq = torch.clamp(E**2 - p**2, min=0.0)
        mass = torch.sqrt(mass_sq + 1e-8)  # [batch, 18]

        # Add mass as an extra feature
        obj_features_enhanced = torch.cat([
            obj_kinematics,
            mass.unsqueeze(-1)  # [batch, 18, 1]
        ], dim=-1)  # [batch, 18, 5]

        # Compute pairwise features between objects
        # We'll compute invariant mass and deltaR for top K pairs per event
        # to keep dimensionality manageable
        max_pairs = 20

        batch_pair_features = []
        for i in range(batch_size):
            # Get real objects for this event
            real_mask = obj_mask[i] > 0.5
            n_real = int(real_mask.sum().item())

            if n_real < 2:
                # Not enough real objects for pairs
                pair_feats = torch.zeros(max_pairs, 4, device=X.device)
                batch_pair_features.append(pair_feats)
                continue

            # Get kinematics for real objects
            real_kinematics = obj_kinematics[i, real_mask]  # [n_real, 4]
            real_eta = eta[i, real_mask]  # [n_real]
            real_phi = phi[i, real_mask]  # [n_real]
            real_E = E[i, real_mask]  # [n_real]
            real_pT = pT[i, real_mask]  # [n_real]

            # Compute all pairs
            n_real = real_kinematics.shape[0]
            pairs = []
            pair_features = []

            for j in range(n_real):
                for k in range(j+1, n_real):
                    # Compute deltaR
                    delta_eta = real_eta[j] - real_eta[k]
                    delta_phi = real_phi[j] - real_phi[k]
                    # Adjust delta_phi to be in [-pi, pi]
                    delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))
                    deltaR = torch.sqrt(delta_eta**2 + delta_phi**2 + 1e-8)

                    # Compute invariant mass from pT, eta, phi, E
                    # First reconstruct approximate 4-vectors
                    px1 = real_pT[j] * torch.cos(real_phi[j])
                    py1 = real_pT[j] * torch.sin(real_phi[j])
                    pz1 = real_pT[j] * torch.sinh(real_eta[j])

                    px2 = real_pT[k] * torch.cos(real_phi[k])
                    py2 = real_pT[k] * torch.sin(real_phi[k])
                    pz2 = real_pT[k] * torch.sinh(real_eta[k])

                    # Sum 4-vectors
                    sum_E = real_E[j] + real_E[k]
                    sum_px = px1 + px2
                    sum_py = py1 + py2
                    sum_pz = pz1 + pz2

                    # Invariant mass
                    inv_mass_sq = torch.clamp(sum_E**2 - (sum_px**2 + sum_py**2 + sum_pz**2), min=0.0)
                    inv_mass = torch.sqrt(inv_mass_sq + 1e-8)

                    # Also include pT sum and other features
                    pT_sum = real_pT[j] + real_pT[k]

                    pairs.append((j, k))
                    pair_features.append([inv_mass.item(), deltaR.item(), pT_sum.item(), 
                                         abs(real_eta[j] - real_eta[k]).item()])

            if not pair_features:
                pair_feats = torch.zeros(max_pairs, 4, device=X.device)
            else:
                pair_features = torch.tensor(pair_features, device=X.device)  # [n_pairs, 4]

                # Select top max_pairs by pT_sum
                if len(pair_features) > max_pairs:
                    # Get indices of pairs with highest pT_sum
                    _, indices = torch.topk(pair_features[:, 2], max_pairs)
                    pair_features = pair_features[indices]

                # Pad if needed
                if len(pair_features) < max_pairs:
                    padding = torch.zeros(max_pairs - len(pair_features), 4, device=X.device)
                    pair_features = torch.cat([pair_features, padding], dim=0)

            batch_pair_features.append(pair_features)

        # Stack all pair features
        pair_features_tensor = torch.stack(batch_pair_features, dim=0)  # [batch, max_pairs, 4]

        # Combine all features
        # Global features expanded to match object dimension
        global_expanded = global_feats.unsqueeze(1).expand(-1, 18, -1)  # [batch, 18, 2]

        # Final object features: [obj_ids (embedded), kinematics, mass, global_feats]
        # We'll embed obj_ids in the model, so just pass them as integers

        # Return tuple of features
        return (global_feats, obj_ids, obj_features_enhanced, pair_features_tensor, obj_mask)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
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

    def forward(self, src, src_key_padding_mask=None):
        # Self attention
        src2, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
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

        # Unpack sample to get dimensions
        global_feats, obj_ids, obj_features, pair_features, obj_mask = sample_object

        # Embedding for object types (assuming max 100 object types)
        self.obj_embedding = nn.Embedding(100, 16)

        # Object branch
        obj_feat_dim = obj_features.shape[-1]  # 5 (kinematics + mass)
        self.obj_proj = nn.Linear(16 + obj_feat_dim + 2, 64)  # +2 for global features

        # Pair branch
        pair_feat_dim = pair_features.shape[-1]  # 4
        self.pair_proj = nn.Linear(pair_feat_dim, 32)

        # Transformer for objects
        self.obj_transformer = nn.ModuleList([
            TransformerBlock(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1)
            for _ in range(2)
        ])

        # Transformer for pairs
        self.pair_transformer = nn.ModuleList([
            TransformerBlock(d_model=32, nhead=2, dim_feedforward=64, dropout=0.1)
            for _ in range(2)
        ])

        # Global processing
        self.global_proj = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32 + 32, 128),  # obj features + pair features + global
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        global_feats, obj_ids, obj_features, pair_features, obj_mask = batch_x

        batch_size = global_feats.shape[0]

        # Process global features
        global_processed = self.global_proj(global_feats)  # [batch, 32]

        # Process objects
        obj_emb = self.obj_embedding(obj_ids)  # [batch, 18, 16]

        # Expand global features to match objects
        global_expanded = global_feats.unsqueeze(1).expand(-1, 18, -1)  # [batch, 18, 2]

        # Combine object features
        obj_combined = torch.cat([obj_emb, obj_features, global_expanded], dim=-1)  # [batch, 18, 16+5+2=23]
        obj_projected = self.obj_proj(obj_combined)  # [batch, 18, 64]

        # Apply transformer to objects
        obj_mask_bool = obj_mask < 0.5  # True for padding
        for transformer in self.obj_transformer:
            obj_projected = transformer(obj_projected, src_key_padding_mask=obj_mask_bool)

        # Pool object features
        # Use mask to exclude padding
        obj_mask_expanded = obj_mask.unsqueeze(-1)  # [batch, 18, 1]
        obj_pooled = (obj_projected * obj_mask_expanded).sum(dim=1) / (obj_mask_expanded.sum(dim=1) + 1e-8)  # [batch, 64]

        # Process pair features
        pair_projected = self.pair_proj(pair_features)  # [batch, max_pairs, 32]

        # Apply transformer to pairs
        # Create mask for pair padding (all-zero rows)
        pair_mask = (pair_features.abs().sum(dim=-1) > 0).float()  # [batch, max_pairs]
        pair_mask_bool = pair_mask < 0.5

        for transformer in self.pair_transformer:
            pair_projected = transformer(pair_projected, src_key_padding_mask=pair_mask_bool)

        # Pool pair features
        pair_mask_expanded = pair_mask.unsqueeze(-1)  # [batch, max_pairs, 1]
        pair_pooled = (pair_projected * pair_mask_expanded).sum(dim=1) / (pair_mask_expanded.sum(dim=1) + 1e-8)  # [batch, 32]

        # Combine all features
        combined = torch.cat([obj_pooled, pair_pooled, global_processed], dim=-1)  # [batch, 64+32+32=128]

        # Final classification
        logits = self.classifier(combined).squeeze(-1)  # [batch]

        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_aucs = []
    val_aucs = []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

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

            epoch_train_loss += loss.item()

            # Calculate accuracy
            preds = torch.sigmoid(logits) > 0.5
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

            # Store for AUC
            all_train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_train_labels.extend(yb.cpu().numpy())

        train_loss = epoch_train_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_auc = roc_auc_score(all_train_labels, all_train_preds)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_aucs.append(train_auc)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                logits = model(xb)
                loss = criterion(logits, yb.float())
                epoch_val_loss += loss.item()

                preds = torch.sigmoid(logits) > 0.5
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

                all_val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_val_labels.extend(yb.cpu().numpy())

        val_loss = epoch_val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(all_val_labels, all_val_preds)

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)

        # Update scheduler
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

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

