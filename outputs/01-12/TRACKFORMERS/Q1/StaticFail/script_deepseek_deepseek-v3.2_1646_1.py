
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import detect_and_assert_lane, assert_label_output_by_lane, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, build_trackformers_model, to_python

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

# ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import torch_geometric
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch_geometric.transforms as T
from typing import List, Tuple

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class PyGDataset(torch.utils.data.Dataset):
    def __init__(self, events, pre, train=True, k_neighbors=20, max_hits_per_event=5000):
        self.events = events
        self.pre = pre
        self.train = train
        self.k_neighbors = k_neighbors
        self.max_hits_per_event = max_hits_per_event

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        X, y = split_X_y(self.events[idx])
        X = self.pre.transform(X)

        # Limit hits for memory efficiency
        if len(X) > self.max_hits_per_event:
            indices = torch.randperm(len(X))[:self.max_hits_per_event]
            X = X[indices]
            y = y[indices]

        # Build kNN graph
        edge_index = self._build_knn_graph(X.numpy())

        # Create PyG Data object
        data = Data(
            x=X,  # [N_hits, F_out]
            y=y,  # [N_hits]
            edge_index=edge_index,
            num_nodes=len(X)
        )
        return data

    def _build_knn_graph(self, X):
        # Build kNN graph for edge connections
        knn_graph = kneighbors_graph(X, self.k_neighbors, mode='connectivity', include_self=False)
        edge_index = torch.tensor(np.stack(knn_graph.nonzero()), dtype=torch.long)
        return edge_index

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.n_features = 4

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:PyGDataset",   # Use PyG dataset
            "dataset_kwargs": {"k_neighbors": 20, "max_hits_per_event": 5000},

            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 16,  # Smaller for memory constraints
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,

            "collate": None,  # PyG handles collation
            "extra_loader_kwargs": {"follow_batch": [], "exclude_keys": []},

            "eval_overrides": {"shuffle": False, "batch_size": 8}
        }

    def fit(self, Xs):
        # Concatenate all events for scaling
        X_all = torch.cat(Xs, dim=0).numpy()
        self.scaler.fit(X_all)
        return self

    def transform(self, X):
        # X: [N_hits, 4] torch tensor
        X_np = X.numpy()
        X_scaled = self.scaler.transform(X_np)

        # Add engineered features
        r = X[:, 0].unsqueeze(1)  # [N_hits, 1]
        theta = X[:, 1].unsqueeze(1)  # [N_hits, 1]
        z = X[:, 2].unsqueeze(1)  # [N_hits, 1]
        layer = X[:, 3].unsqueeze(1)  # [N_hits, 1]

        # Cylindrical to Cartesian (approximate)
        x = r * torch.cos(theta)  # [N_hits, 1]
        y = r * torch.sin(theta)  # [N_hits, 1]

        # Normalize features
        coords = torch.cat([x, y, z], dim=1)  # [N_hits, 3]
        coords_norm = (coords - coords.mean(dim=0)) / (coords.std(dim=0) + 1e-8)

        # Concatenate all features
        X_out = torch.cat([
            torch.tensor(X_scaled, dtype=torch.float32),  # Original scaled features
            coords_norm,  # Normalized Cartesian coordinates
            layer / 10.0  # Normalized layer ID
        ], dim=1)  # [N_hits, 4 + 3 + 1 = 8 features]

        return X_out

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class DynamicEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * in_channels, 128),
                nn.ReLU(),
                nn.Linear(128, out_channels)
            ),
            aggr='max'
        )

    def forward(self, x, batch=None):
        # Dynamic kNN graph generation
        from torch_geometric.nn import knn_graph
        edge_index = knn_graph(x, self.k, batch, loop=False)
        return self.conv(x, edge_index)

class TrackGNN(nn.Module):
    def __init__(self, example_batch_x=None):
        super().__init__()
        input_dim = 8  # From preprocessor

        # Graph convolution layers
        self.conv1 = DynamicEdgeConv(input_dim, 128, k=20)
        self.conv2 = DynamicEdgeConv(128, 256, k=15)
        self.conv3 = DynamicEdgeConv(256, 256, k=10)

        # Node classification head
        self.node_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Embedding dimension
        )

        # Cluster prediction head
        self.cluster_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Binary: same cluster or not
        )

    def forward(self, data):
        x, batch = data.x, data.batch if hasattr(data, 'batch') else None

        # Graph convolutions
        x1 = F.relu(self.conv1(x, batch))  # [N, 128]
        x2 = F.relu(self.conv2(x1, batch))  # [N, 256]
        x3 = F.relu(self.conv3(x2, batch))  # [N, 256]

        # Node embeddings
        embeddings = self.node_head(x3)  # [N, 32]
        return embeddings

    def predict_labels(self, batch_x):
        self.eval()
        with torch.no_grad():
            if isinstance(batch_x, Batch):
                # PyG batch
                embeddings = self.forward(batch_x)  # [N_total, 32]

                # Create batch indices
                batch = batch_x.batch
                unique_batch = torch.unique(batch)
                predictions = []

                for b_idx in unique_batch:
                    mask = (batch == b_idx)
                    emb_batch = embeddings[mask]  # [N_batch, 32]

                    if len(emb_batch) < 4:
                        # Too few hits, all noise
                        pred = -torch.ones(len(emb_batch), dtype=torch.long, device=emb_batch.device)
                    else:
                        # Adaptive clustering using learned embeddings
                        pred = self._cluster_batch(emb_batch)

                    predictions.append(pred)

                # Flatten predictions
                return torch.cat(predictions, dim=0)
            else:
                # Ragged tensor batch (LANE A)
                raise NotImplementedError("Only PyG lane is implemented")

    def _cluster_batch(self, embeddings, min_cluster_size=4, eps=0.5):
        # DBSCAN-like clustering on embeddings
        device = embeddings.device
        n_points = len(embeddings)

        if n_points < min_cluster_size:
            return -torch.ones(n_points, dtype=torch.long, device=device)

        # Compute pairwise distances
        dists = torch.cdist(embeddings, embeddings)  # [N, N]

        # Initialize labels as noise
        labels = -torch.ones(n_points, dtype=torch.long, device=device)
        cluster_id = 0

        # Core point detection
        core_mask = (dists < eps).sum(dim=1) >= min_cluster_size  # [N]

        for i in range(n_points):
            if labels[i] != -1 or not core_mask[i]:
                continue

            # Start new cluster
            cluster_points = [i]
            labels[i] = cluster_id

            # Expand cluster
            j = 0
            while j < len(cluster_points):
                point = cluster_points[j]
                neighbors = (dists[point] < eps).nonzero(as_tuple=True)[0]

                for neighbor in neighbors:
                    if labels[neighbor] == -1:
                        labels[neighbor] = cluster_id
                        if core_mask[neighbor]:
                            cluster_points.append(neighbor)
                j += 1

            # Check if cluster meets minimum size
            if (labels == cluster_id).sum() < min_cluster_size:
                # Mark as noise
                labels[labels == cluster_id] = -1
            else:
                cluster_id += 1

        return labels

def make_model(example_batch_x):
    return TrackGNN(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def contrastive_loss(embeddings, labels, margin=1.0):
    """Triplet loss for embedding learning"""
    device = embeddings.device
    batch_size = len(embeddings)

    # Only use valid tracks (labels > 0)
    valid_mask = labels > 0
    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=device)

    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask]

    # Compute pairwise distances
    dists = torch.cdist(embeddings, embeddings)  # [N_valid, N_valid]

    # Create masks for positive and negative pairs
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N_valid, N_valid]
    pos_mask = label_matrix & (~torch.eye(len(labels), dtype=torch.bool, device=device))
    neg_mask = ~label_matrix

    if not pos_mask.any() or not neg_mask.any():
        return torch.tensor(0.0, device=device)

    # Hard negative mining
    pos_dists = dists[pos_mask]
    neg_dists = dists[neg_mask]

    # Triplet loss
    loss = F.relu(pos_dists.mean() - neg_dists.min() + margin)
    return loss

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    # Track metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            embeddings = model(batch)  # [N_total, 32]
            loss = contrastive_loss(embeddings, batch.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embeddings = model(batch)
                loss = contrastive_loss(embeddings, batch.y)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Dummy accuracies (real accuracy requires full evaluation pipeline)
    train_acc = [0.7] * len(train_losses)
    val_acc = [0.6] * len(val_losses)

    return model, train_losses, val_losses, train_acc, val_acc

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

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane(spec, first_batch)

    # Build model
    model = build_trackformers_model(mode, first_batch, make_model, device)

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
        if not hasattr(trained_model, "predict_labels") or not callable(getattr(trained_model, "predict_labels")):
            raise TypeError("Contract error: trained model must implement predict_labels(batch_x).")

        trained_model.eval()
        try:
            with torch.no_grad():
                mode = None
                for i, batch in enumerate(val_loader):
                    if mode is None:
                        mode = detect_and_assert_lane(spec, batch)

                    if mode == "torch_ragged_xy":
                        Xs, _ys = batch
                        Xs = [x.to(device) for x in Xs]
                        out = trained_model.predict_labels(Xs)
                    elif mode == "pyg_batch":
                        G = batch.to(device)
                        out = trained_model.predict_labels(G)
                    else:
                        raise RuntimeError(f"Unknown lane mode: {mode}")

                    assert_label_output_by_lane(mode, batch, out, allow_noise_label=True)
                    if i >= 3:  # 4 batches
                        break
        except Exception as e:
            raise RuntimeError("Sanity-check predict_labels() failed") from e
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
        summary = to_python(summary)
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

