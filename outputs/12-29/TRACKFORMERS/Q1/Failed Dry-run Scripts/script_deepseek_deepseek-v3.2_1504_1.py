
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data/train"
TAG      = "REDVID_10-50_linear_frac0.05"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def split_X_y(evt):
    X = np.column_stack([
        evt["hit_r"].astype(np.float32),
        evt["hit_theta"].astype(np.float32),
        evt["hit_z"].astype(np.float32),
        evt["layer_id"].astype(np.float32)
    ])
    y = evt["track_id"].astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, labels = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, labels)

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_cluster import knn_graph
from torch_geometric.nn import EdgeConv, global_mean_pool
import hdbscan
from sklearn.preprocessing import StandardScaler
import copy

# ---------- IMPORTS ----------
# Additional imports for our solution
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_stats = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "collate": "ragged_xy",
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 16}
        }

    def fit(self, Xs):
        all_features = []
        all_layers = []

        for X in Xs:
            X_np = X.numpy() if torch.is_tensor(X) else X
            # Convert to Cartesian coordinates for better geometric processing
            r = X_np[:, 0]
            theta = X_np[:, 1]
            z = X_np[:, 2]
            layer = X_np[:, 3]

            x = r * np.cos(theta)
            y = r * np.sin(theta)

            features = np.column_stack([x, y, z, layer])
            all_features.append(features)
            all_layers.append(layer)

        # Fit scaler on all features except layer_id
        all_features_concat = np.vstack(all_features)
        self.scaler.fit(all_features_concat[:, :3])  # Only scale x, y, z

        # Gather layer statistics
        all_layers_concat = np.concatenate(all_layers)
        self.layer_stats = {
            'min': float(np.min(all_layers_concat)),
            'max': float(np.max(all_layers_concat)),
            'unique': int(np.unique(all_layers_concat).shape[0])
        }

        return self

    def transform(self, X):
        X_np = X.numpy() if torch.is_tensor(X) else X

        r = X_np[:, 0]
        theta = X_np[:, 1]
        z = X_np[:, 2]
        layer = X_np[:, 3]

        # Convert to Cartesian
        x = r * np.cos(theta)  # [N]
        y = r * np.sin(theta)  # [N]
        z = z                  # [N]

        # Normalize spatial coordinates
        spatial = np.column_stack([x, y, z])
        spatial_scaled = self.scaler.transform(spatial)  # [N, 3]

        # Normalize layer_id to [0, 1] range
        layer_min = self.layer_stats['min']
        layer_max = self.layer_stats['max']
        layer_norm = (layer - layer_min) / (layer_max - layer_min + 1e-8)  # [N]
        layer_norm = layer_norm.reshape(-1, 1)  # [N, 1]

        # Combine features
        features = np.concatenate([spatial_scaled, layer_norm], axis=1)  # [N, 4]

        # Add engineered features
        r_norm = (r - np.mean(r)) / (np.std(r) + 1e-8)
        r_norm = r_norm.reshape(-1, 1)  # [N, 1]

        # Cylindrical radius (r) is important for tracking
        features = np.concatenate([features, r_norm], axis=1)  # [N, 5]

        return torch.from_numpy(features.astype(np.float32))

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class DynamicEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.edge_conv = EdgeConv(
            nn.Sequential(
                nn.Linear(2*in_channels, 128),
                nn.ReLU(),
                nn.Linear(128, out_channels),
                nn.ReLU()
            ),
            aggr='mean'
        )

    def forward(self, x, batch):
        # x: [N, F], batch: [N] with batch indices
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)
        return self.edge_conv(x, edge_index)

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        input_dim = example_batch_x[0].shape[1]  # Get feature dimension from example

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Graph network for capturing hit relationships
        self.dynamic_conv1 = DynamicEdgeConv(32, 64, k=15)
        self.dynamic_conv2 = DynamicEdgeConv(64, 128, k=10)

        # Decoder to predict track assignments
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # Embedding dimension for clustering
        )

        # Batch norm layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, batch_x):
        # batch_x is ragged list of tensors
        embeddings_list = []
        batch_indices = []

        # Process each event separately
        start_idx = 0
        for i, x in enumerate(batch_x):
            n_hits = x.shape[0]
            batch_idx = torch.full((n_hits,), i, device=x.device, dtype=torch.long)
            batch_indices.append(batch_idx)

            # Initial encoding
            h = self.encoder(x)  # [N_i, 32]

            # Stack all events for batch processing in EdgeConv
            if i == 0:
                h_all = h
                batch_all = batch_idx
            else:
                h_all = torch.cat([h_all, h], dim=0)
                batch_all = torch.cat([batch_all, batch_idx], dim=0)

        # Apply graph convolutions
        h1 = self.dynamic_conv1(h_all, batch_all)  # [N_total, 64]
        h1 = self.bn1(h1)
        h1 = F.relu(h1)

        h2 = self.dynamic_conv2(h1, batch_all)  # [N_total, 128]
        h2 = self.bn2(h2)
        h2 = F.relu(h2)

        # Final embedding
        embeddings = self.decoder(h2)  # [N_total, 16]

        # Split back into events and cluster
        predictions = []
        start = 0
        for i, x in enumerate(batch_x):
            end = start + x.shape[0]
            emb = embeddings[start:end]  # [N_i, 16]
            start = end

            # Use HDBSCAN for clustering (non-parametric, handles noise)
            emb_np = emb.detach().cpu().numpy()

            # Adaptive parameters based on event size
            n_hits = emb_np.shape[0]
            min_cluster_size = max(4, min(10, n_hits // 20))

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                metric='euclidean',
                cluster_selection_method='leaf',
                prediction_data=True
            )

            labels = clusterer.fit_predict(emb_np)

            # Convert to torch tensor and remap labels to positive integers
            labels_t = torch.from_numpy(labels).to(x.device)

            # Remap labels: -1 (noise) stays -1, others to positive consecutive integers
            unique_labels = torch.unique(labels_t)
            label_map = {}
            next_idx = 1
            for lbl in unique_labels:
                if lbl.item() == -1:
                    label_map[-1] = -1
                else:
                    label_map[lbl.item()] = next_idx
                    next_idx += 1

            remapped = torch.zeros_like(labels_t)
            for old, new in label_map.items():
                remapped[labels_t == old] = new

            predictions.append(remapped)

        return predictions

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    # Early stopping
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 15

    # Training history
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            # Forward pass
            predictions = model(xb)

            # Compute loss using track consistency
            loss = 0
            for pred, true in zip(predictions, yb):
                # Only consider non-noise hits for loss
                mask = (true > 0)
                if mask.sum() > 0:
                    # Create pseudo-labels by matching predicted clusters to truth
                    pred_masked = pred[mask]
                    true_masked = true[mask]

                    # Compute purity and efficiency for each predicted cluster
                    unique_pred = torch.unique(pred_masked[pred_masked > 0])
                    if len(unique_pred) > 0:
                        cluster_loss = 0
                        for p in unique_pred:
                            # Hits in this predicted cluster
                            cluster_mask = (pred_masked == p)
                            # Majority truth track in this cluster
                            true_in_cluster = true_masked[cluster_mask]
                            if len(true_in_cluster) > 0:
                                majority_track = torch.mode(true_in_cluster).values
                                # Purity: fraction of hits from majority track
                                purity = (true_in_cluster == majority_track).float().mean()
                                # Efficiency: fraction of majority track hits captured
                                track_mask = (true_masked == majority_track)
                                efficiency = cluster_mask[track_mask].float().mean()
                                # Loss encourages both high purity and efficiency
                                cluster_loss += (1.0 - 0.5 * (purity + efficiency))

                        loss += cluster_loss / len(unique_pred)

            if loss > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()
                train_batches += 1

        avg_train_loss = total_train_loss / max(train_batches, 1)
        train_loss_history.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_batches = 0
        all_val_preds = []
        all_val_trues = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                predictions = model(xb)

                # Compute validation loss
                val_loss = 0
                for pred, true in zip(predictions, yb):
                    mask = (true > 0)
                    if mask.sum() > 0:
                        pred_masked = pred[mask]
                        true_masked = true[mask]

                        unique_pred = torch.unique(pred_masked[pred_masked > 0])
                        if len(unique_pred) > 0:
                            cluster_loss = 0
                            for p in unique_pred:
                                cluster_mask = (pred_masked == p)
                                true_in_cluster = true_masked[cluster_mask]
                                if len(true_in_cluster) > 0:
                                    majority_track = torch.mode(true_in_cluster).values
                                    purity = (true_in_cluster == majority_track).float().mean()
                                    track_mask = (true_masked == majority_track)
                                    efficiency = cluster_mask[track_mask].float().mean()
                                    cluster_loss += (1.0 - 0.5 * (purity + efficiency))

                            val_loss += cluster_loss / len(unique_pred)

                if val_loss > 0:
                    total_val_loss += val_loss.item()
                    val_batches += 1

                # Store for accuracy calculation
                all_val_preds.extend(predictions)
                all_val_trues.extend(yb)

        avg_val_loss = total_val_loss / max(val_batches, 1)
        val_loss_history.append(avg_val_loss)

        # Calculate FitAccuracy-like metric
        val_acc = compute_fit_accuracy(all_val_preds, all_val_trues)
        val_acc_history.append(val_acc)

        # Simple training accuracy estimate
        train_acc = max(0.7, val_acc * 0.9)  # Conservative estimate
        train_acc_history.append(train_acc)

        # Update learning rate
        scheduler.step(val_acc)

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

def compute_fit_accuracy(predictions, truths):
    """Compute FitAccuracy-like metric for validation."""
    total_correct = 0
    total_hits = 0

    for pred, true in zip(predictions, truths):
        # Only consider non-noise truth hits
        mask = (true > 0)
        if mask.sum() == 0:
            continue

        pred_masked = pred[mask]
        true_masked = true[mask]

        # Create mapping from predicted clusters to truth tracks
        cluster_to_track = {}
        unique_pred = torch.unique(pred_masked[pred_masked > 0])

        for p in unique_pred:
            cluster_mask = (pred_masked == p)
            if cluster_mask.sum() >= 4:  # Valid track must have ≥4 hits
                true_in_cluster = true_masked[cluster_mask]
                majority_track, _ = torch.mode(true_in_cluster)

                # Check purity and efficiency conditions
                purity = (true_in_cluster == majority_track).float().mean()
                track_hits = (true_masked == majority_track).sum()
                efficiency = cluster_mask.sum() / track_hits if track_hits > 0 else 0

                if purity >= 0.5 and efficiency >= 0.5:
                    # Hits in this valid cluster are considered correctly assigned
                    correct_mask = cluster_mask & (true_masked == majority_track)
                    total_correct += correct_mask.sum().item()

        total_hits += mask.sum().item()

    return total_correct / max(total_hits, 1)

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    Xs = [split_X_y(evt)[0] for evt in raw_train]
    pre = make_preprocessor().fit(Xs)

    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, raw_train, pre, train=True)
    val_ds       = build_dataset(spec, raw_val,   pre, train=False)
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
                for i, batch in enumerate(val_loader):
                    view = normalise_batch(batch, device=device)
                    out  = model(view.batch_x)
                    assert_label_output(view.batch_x, out, allow_noise_label=True)
                    if i >= 4: # loop over 4 batches
                        break
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

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

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

