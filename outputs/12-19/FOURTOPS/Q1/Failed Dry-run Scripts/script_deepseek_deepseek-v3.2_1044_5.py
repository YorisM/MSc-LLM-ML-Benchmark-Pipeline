
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
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from typing import Tuple, List

# ---------- IMPORTS ----------
import math
from scipy import stats
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

#  -------- (OPTIONAL) CUSTOM DATASET  --------
def make_dataset(events, pre, train: bool, **kwargs):
    return FourTopsDataset(events, pre, train=train)

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.kinematic_means = None
        self.kinematic_stds = None
        self.obj_id_stats = None
        self.pca = None
        self.event_aggregations = None
        self.obj_type_encoders = None
        self.feature_names = []

    def _extract_objects(self, X):
        batch_size = X.shape[0]
        # Reshape: [batch, 92] -> [batch, 18, 5] + 2 global
        objects = X[:, 2:].reshape(batch_size, 18, 5)  # [batch, 18, 5]
        global_feats = X[:, :2]  # [batch, 2]
        return global_feats, objects

    def _compute_physics_features(self, global_feats, objects):
        batch_size = objects.shape[0]

        # Basic kinematic stats per object
        obj_energy = objects[:, :, 1]  # E
        obj_pt = objects[:, :, 2]  # pT
        obj_eta = objects[:, :, 3]  # eta
        obj_phi = objects[:, :, 4]  # phi
        obj_ids = objects[:, :, 0].long()  # object IDs

        # Physics-inspired features
        features = []

        # 1. Global event features
        features.append(global_feats)  # [batch, 2]

        # 2. Object multiplicity by type (6 most common types)
        for obj_type in [1, 2, 3, 4, 5, 6]:
            count = (obj_ids == obj_type).sum(dim=1, keepdim=True).float()
            features.append(count)  # [batch, 1]

        # 3. Kinematic aggregations
        # Total transverse momentum
        total_pt = obj_pt.sum(dim=1, keepdim=True)  # [batch, 1]
        features.append(total_pt)

        # Average pt of top 4 objects
        top4_pt, _ = torch.topk(obj_pt, k=min(4, obj_pt.shape[1]), dim=1)
        avg_top4_pt = top4_pt.mean(dim=1, keepdim=True)  # [batch, 1]
        features.append(avg_top4_pt)

        # pt variance
        pt_variance = torch.var(obj_pt, dim=1, keepdim=True)  # [batch, 1]
        features.append(pt_variance)

        # 4. Angular correlations
        # Delta R between leading pt objects
        if obj_pt.shape[1] >= 2:
            # Get indices of top 2 pt objects
            _, idx1 = torch.topk(obj_pt, k=1, dim=1)
            mask = torch.ones_like(obj_pt, dtype=torch.bool)
            mask.scatter_(1, idx1, False)
            second_pt = obj_pt[mask].reshape(batch_size, -1)[:, :1]
            second_idx = torch.topk(second_pt, k=1, dim=0)[1]

            # Compute deltaR = sqrt((eta1-eta2)^2 + (phi1-phi2)^2)
            eta1 = obj_eta.gather(1, idx1)
            phi1 = obj_phi.gather(1, idx1)
            eta2 = obj_eta.gather(1, second_idx.unsqueeze(1))
            phi2 = obj_phi.gather(1, second_idx.unsqueeze(1))

            delta_eta = eta1 - eta2
            delta_phi = torch.atan2(torch.sin(phi1 - phi2), torch.cos(phi1 - phi2))
            delta_r = torch.sqrt(delta_eta**2 + delta_phi**2)  # [batch, 1]
            features.append(delta_r)
        else:
            features.append(torch.zeros(batch_size, 1, device=obj_pt.device))

        # 5. Missing ET significance
        met = global_feats[:, 0:1]  # E_T^miss
        met_sig = met / torch.sqrt(total_pt + 1e-8)  # [batch, 1]
        features.append(met_sig)

        # 6. Object type pt-weighted sums
        for obj_type in [1, 2, 3]:
            mask = (obj_ids == obj_type).float()
            type_pt_sum = (obj_pt * mask).sum(dim=1, keepdim=True)  # [batch, 1]
            features.append(type_pt_sum)

        # 7. Sphericity-like feature (momentum tensor)
        px = obj_pt * torch.cos(obj_phi)
        py = obj_pt * torch.sin(obj_phi)
        pz = obj_pt * torch.sinh(obj_eta)

        p_norm = torch.sqrt(px**2 + py**2 + pz**2 + 1e-8)
        px_norm = px / p_norm
        py_norm = py / p_norm
        pz_norm = pz / p_norm

        # Momentum tensor eigenvalues
        M = torch.stack([
            px_norm**2, px_norm*py_norm, px_norm*pz_norm,
            py_norm*px_norm, py_norm**2, py_norm*pz_norm,
            pz_norm*px_norm, pz_norm*py_norm, pz_norm**2
        ], dim=2).reshape(batch_size, 18, 3, 3)
        M_sum = M.sum(dim=1)  # [batch, 3, 3]

        # Compute eigenvalues (simplified)
        eigenvalues = torch.linalg.eigvalsh(M_sum)
        sphericity = 1.5 * (eigenvalues[:, 1] + eigenvalues[:, 2]).unsqueeze(1)  # [batch, 1]
        features.append(sphericity)

        # 8. Hadronic activity
        central_mask = (torch.abs(obj_eta) < 2.5).float()
        hadronic_activity = (obj_pt * central_mask).sum(dim=1, keepdim=True)  # [batch, 1]
        features.append(hadronic_activity)

        return torch.cat(features, dim=1)  # [batch, total_features]

    def fit(self, X, y=None):
        # Convert to numpy for scikit-learn compatibility
        X_np = X.numpy()

        # Extract objects
        global_feats = X_np[:, :2]
        objects = X_np[:, 2:].reshape(-1, 18, 5)

        # Store global feature statistics
        self.global_mean = np.mean(global_feats, axis=0)
        self.global_std = np.std(global_feats, axis=0) + 1e-8

        # Compute kinematics statistics (ignoring padded zeros)
        valid_mask = objects[:, :, 0] != 0  # obj_id != 0
        kinematic_feats = objects[:, :, 1:][valid_mask]  # E, pT, eta, phi

        self.kinematic_means = np.mean(kinematic_feats, axis=0)
        self.kinematic_stds = np.std(kinematic_feats, axis=0) + 1e-8

        # Compute PCA on aggregated features for dimensionality reduction
        sample_tensor = torch.from_numpy(X_np[:1000])
        with torch.no_grad():
            aggregated = self._compute_physics_features(
                torch.from_numpy(global_feats[:1000]), 
                torch.from_numpy(objects[:1000])
            ).numpy()

        # Keep 95% variance
        self.pca = PCA(n_components=0.95, random_state=42)
        self.pca.fit(aggregated)

        # Store object ID statistics for embedding
        obj_ids = objects[:, :, 0].flatten()
        valid_ids = obj_ids[obj_ids != 0]
        unique_ids = np.unique(valid_ids)
        self.obj_id_stats = {
            'unique_ids': unique_ids,
            'num_types': len(unique_ids),
            'id_to_idx': {int(id_): i for i, id_ in enumerate(unique_ids)}
        }

        return self

    def transform(self, X):
        # Convert to numpy for processing
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
            return_tensor = True
        else:
            X_np = X
            return_tensor = False

        batch_size = X_np.shape[0]

        # Extract components
        global_feats = X_np[:, :2]
        objects = X_np[:, 2:].reshape(batch_size, 18, 5)

        # Normalize global features
        global_feats_norm = (global_feats - self.global_mean) / self.global_std

        # Normalize kinematic features (ignore padded objects)
        objects_norm = objects.copy()
        valid_mask = objects[:, :, 0] != 0

        # Normalize E, pT, eta, phi (indices 1-4)
        for i in range(1, 5):
            objects_norm[:, :, i][valid_mask] = (
                objects[:, :, i][valid_mask] - self.kinematic_means[i-1]
            ) / self.kinematic_stds[i-1]

        # Convert to torch for physics feature computation
        global_tensor = torch.from_numpy(global_feats_norm).float()
        objects_tensor = torch.from_numpy(objects_norm).float()

        # Compute physics-inspired features
        physics_features = self._compute_physics_features(global_tensor, objects_tensor)

        # Apply PCA dimensionality reduction
        physics_np = physics_features.numpy()
        reduced_features = self.pca.transform(physics_np)

        # Convert back to tensor if needed
        if return_tensor:
            result = torch.from_numpy(reduced_features).float()
        else:
            result = reduced_features

        # Log feature count for model building
        if not self.feature_names:
            self.feature_names = [f"pca_{i}" for i in range(reduced_features.shape[1])]

        return result

    def make_loader_cfg(self):
        return {
            "dataset_builder": "llm_script:make_dataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True if torch.cuda.is_available() else False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Input size determined by PCA output
        input_size = sample_object.shape[0]

        # Deep network with residual connections
        hidden_size = 256
        self.input_norm = nn.BatchNorm1d(input_size)
        self.input_fc = nn.Linear(input_size, hidden_size)

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        num_blocks = 6
        dropout_rate = 0.3

        for i in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
            )
            self.res_blocks.append(block)

        # Attention pooling across features (self-attention)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1, bias=False)
        )

        # Final layers
        self.final_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.final_fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.final_fc3 = nn.Linear(hidden_size // 4, 1)

        # Regularization
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input normalization and projection
        x = self.input_norm(x)
        x = F.relu(self.input_fc(x))

        # Residual blocks with skip connections
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = F.relu(x + residual)
            x = self.dropout(x)

        # Attention pooling
        attn_weights = torch.softmax(self.attention(x).squeeze(-1), dim=0)
        x = (x * attn_weights.unsqueeze(-1)).sum(dim=0, keepdim=True)

        # Final classification layers
        x = F.relu(self.final_fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)

        x = F.relu(self.final_fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)

        # Output logit
        x = self.final_fc3(x)

        return x.squeeze(-1)  # [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 100

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return float(epoch) / warmup_epochs
        else:
            return 0.1 ** (epoch // 30)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function with label smoothing
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))

    # Early stopping
    best_val_auc = 0.0
    patience = 20
    patience_counter = 0
    best_model_state = None

    # Metrics storage
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_aucs, val_aucs = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Metrics
            train_loss += loss.item() * batch_x.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds.float() == batch_y).sum().item()
            train_total += batch_y.size(0)

            all_train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_train_labels.extend(batch_y.detach().cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds.float() == batch_y).sum().item()
                val_total += batch_y.size(0)

                all_val_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                all_val_labels.extend(batch_y.detach().cpu().numpy())

        # Calculate metrics
        train_loss_avg = train_loss / train_total
        val_loss_avg = val_loss / val_total

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # Calculate AUC
        try:
            train_auc = roc_auc_score(all_train_labels, all_train_preds)
            val_auc = roc_auc_score(all_val_labels, all_val_preds)
        except:
            train_auc = 0.5
            val_auc = 0.5

        # Store metrics
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        # Update learning rate
        scheduler.step()

        # Early stopping based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, "
                  f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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


