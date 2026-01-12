
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.utils import to_dense_batch, dense_to_sparse
import math

# ---------- IMPORTS ----------
# Additional imports needed for model
from typing import List, Tuple, Dict
import copy

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # State for feature scaling
        self.scaler = StandardScaler()
        # Graph construction parameters
        self.k_neighbors = 16
        self.max_layer_diff = 2  # Max layer difference for edge connections

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 16,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 8}
        }

    def _build_graph(self, X: torch.Tensor) -> Data:
        """Convert point cloud to graph using k-NN with layer constraints"""
        n_points = X.shape[0]

        # Extract features
        pos = X[:, :3].numpy()  # [N, 3]: r, theta, z
        layers = X[:, 3].numpy()  # [N]: layer_id

        # Build adjacency with layer constraints
        edge_index = []
        for i in range(n_points):
            layer_i = layers[i]
            # Find candidate neighbors within layer difference constraint
            candidates = []
            for j in range(n_points):
                if i == j:
                    continue
                layer_j = layers[j]
                if abs(layer_i - layer_j) <= self.max_layer_diff:
                    candidates.append(j)

            if not candidates:
                continue

            # Calculate distances to candidates
            dists = []
            pos_i = pos[i]
            for j in candidates:
                pos_j = pos[j]
                # Cylindrical distance metric
                dr = pos_j[0] - pos_i[0]
                dz = pos_j[2] - pos_i[2]
                # Use angular difference in theta
                dtheta = min(abs(pos_j[1] - pos_i[1]), 
                           2*math.pi - abs(pos_j[1] - pos_i[1]))
                # Weighted distance metric
                dist = math.sqrt(dr**2 + (pos_i[0]*dtheta)**2 + dz**2)
                dists.append((dist, j))

            # Select k nearest neighbors
            dists.sort(key=lambda x: x[0])
            k = min(self.k_neighbors, len(dists))
            for _, j in dists[:k]:
                edge_index.append([i, j])
                edge_index.append([j, i])  # Undirected graph

        if not edge_index:
            # Fallback: connect to self
            edge_index = [[i, i] for i in range(n_points)]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create node features with enhanced geometry info
        r = X[:, 0:1]  # [N, 1]
        theta = X[:, 1:2]  # [N, 1]
        z = X[:, 2:3]  # [N, 1]
        layer = X[:, 3:4]  # [N, 1]

        # Additional features
        r_norm = (r - r.mean()) / (r.std() + 1e-8)
        z_norm = (z - z.mean()) / (z.std() + 1e-8)

        # Angular features
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        # Combine features
        node_features = torch.cat([
            r_norm,
            sin_theta,
            cos_theta,
            z_norm,
            layer / 10.0,  # Normalize layer_id
            torch.ones_like(r),  # Bias term
        ], dim=1)  # [N, 6]

        return Data(x=node_features, edge_index=edge_index)

    def fit(self, Xs):
        # Collect statistics from training data
        all_features = []
        for X in Xs:
            if len(X) > 0:
                all_features.append(X.numpy())

        if all_features:
            all_features = np.vstack(all_features)
            # Fit scaler on positions only
            self.scaler.fit(all_features[:, :3])

        return self

    def transform(self, X):
        # X: [N_hits, 4]
        # Scale positional features
        pos_scaled = self.scaler.transform(X[:, :3].numpy())
        pos_scaled = torch.from_numpy(pos_scaled).float()

        # Keep layer_id unchanged
        layer_id = X[:, 3:4]

        # Combine scaled features
        X_scaled = torch.cat([pos_scaled, layer_id], dim=1)  # [N, 4]

        # Build graph
        data = self._build_graph(X_scaled)
        return data

# ---------- MODEL ARCHITECTURE ----------
class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels, out_channels // heads,
            heads=heads, dropout=dropout, edge_dim=1, add_self_loops=True
        )
        self.norm = GraphNorm(out_channels)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 2, out_channels)
        )
        self.skip = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        identity = x
        x = F.relu(self.norm(self.conv(x, edge_index, edge_attr), batch))
        x = x + self.ffn(x)
        x = x + self.skip(identity)
        return x

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Extract feature dimension from example
        if isinstance(example_batch_x, Batch):
            input_dim = example_batch_x.x.shape[1]
        else:
            input_dim = 6  # From our preprocessing

        hidden_dim = 256
        embedding_dim = 128

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # GNN backbone
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, hidden_dim, heads=8, dropout=0.2),
            GNNLayer(hidden_dim, hidden_dim, heads=8, dropout=0.2),
            GNNLayer(hidden_dim, hidden_dim, heads=8, dropout=0.2),
            GNNLayer(hidden_dim, embedding_dim, heads=8, dropout=0.2),
        ])

        # Attention pooling for global context
        self.attention_pool = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Final projection for clustering
        self.cluster_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Output dimension for clustering space
        )

        # Learnable prototype vectors
        self.prototypes = nn.Parameter(torch.randn(64, 64) * 0.01)

        # For stability
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Add edge features (distance)
        if edge_index.shape[1] > 0:
            row, col = edge_index
            edge_attr = torch.norm(x[row, :3] - x[col, :3], dim=1, keepdim=True)
            edge_attr = self.edge_encoder(edge_attr)
        else:
            edge_attr = None

        # Apply GNN layers
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index, edge_attr, batch)
            if i < len(self.gnn_layers) - 1:
                x = F.dropout(x, p=0.2, training=self.training)

        return x  # [N, embedding_dim]

    def _compute_assignments(self, embeddings, batch):
        """Compute cluster assignments using learned prototypes"""
        # embeddings: [N, 64]
        # Normalize embeddings and prototypes
        embeddings_norm = F.normalize(embeddings, dim=1)  # [N, 64]
        prototypes_norm = F.normalize(self.prototypes, dim=1)  # [64, 64]

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings_norm, prototypes_norm.t())  # [N, 64]
        sim_matrix = sim_matrix / (self.temperature.clamp(min=0.01))

        # Get global context
        global_feat = global_mean_pool(embeddings, batch)  # [B, embedding_dim]
        global_feat = global_feat[batch]  # [N, embedding_dim]

        # Combine with embeddings
        combined = torch.cat([embeddings, global_feat], dim=1)  # [N, embedding_dim*2]
        cluster_space = self.cluster_head(combined)  # [N, 64]
        cluster_norm = F.normalize(cluster_space, dim=1)

        # Final similarity
        final_sim = torch.matmul(cluster_norm, prototypes_norm.t())  # [N, 64]
        final_sim = final_sim / (self.temperature.clamp(min=0.01))

        return final_sim

    def predict_labels(self, batch):
        # batch: PyG Batch object
        with torch.no_grad():
            self.eval()
            embeddings = self(batch)  # [N, embedding_dim]
            sim_scores = self._compute_assignments(embeddings, batch.batch)

            # Get assignments (hard clustering)
            assignments = torch.argmax(sim_scores, dim=1)  # [N]

            # Apply per-event constraints
            unique_batches = torch.unique(batch.batch)
            final_labels = torch.full_like(assignments, -1)

            for b in unique_batches:
                mask = (batch.batch == b)
                batch_assignments = assignments[mask]
                batch_embeddings = embeddings[mask]

                # DBSCAN-like clustering within batch
                from sklearn.cluster import DBSCAN
                try:
                    emb_np = batch_embeddings.cpu().numpy()
                    # Adaptive epsilon based on density
                    k = min(5, len(emb_np)-1)
                    if k > 0:
                        from sklearn.neighbors import NearestNeighbors
                        nn = NearestNeighbors(n_neighbors=k)
                        nn.fit(emb_np)
                        distances, _ = nn.kneighbors(emb_np)
                        eps = distances[:, 1:].mean() * 2.0
                    else:
                        eps = 0.5

                    # Cluster with DBSCAN
                    clustering = DBSCAN(
                        eps=max(eps, 0.1),
                        min_samples=4,
                        metric='euclidean'
                    ).fit(emb_np)

                    batch_labels = torch.from_numpy(clustering.labels_).to(assignments.device)

                    # Remap labels to positive integers, keep -1 for noise
                    unique_labels = torch.unique(batch_labels)
                    next_label = 0
                    for lbl in unique_labels:
                        if lbl == -1:
                            continue
                        final_labels[mask][batch_labels == lbl] = next_label
                        next_label += 1

                except Exception:
                    # Fallback: assign all to same track if clustering fails
                    final_labels[mask] = 0

            return final_labels

# ---------- MODEL TRAINING ----------
EPOCHS = 100

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, batch):
        device = features.device

        # Normalize features
        features = F.normalize(features, dim=1)

        loss = torch.tensor(0.0, device=device)
        count = 0

        unique_batches = torch.unique(batch)
        for b in unique_batches:
            batch_mask = (batch == b)
            batch_features = features[batch_mask]
            batch_labels = labels[batch_mask]

            if len(batch_features) < 2:
                continue

            # Compute similarity matrix
            sim_matrix = torch.matmul(batch_features, batch_features.t()) / self.temperature

            # Create mask for positive pairs (same track, non-noise)
            label_matrix = (batch_labels.unsqueeze(1) == batch_labels.unsqueeze(0))
            noise_mask = (batch_labels > 0).unsqueeze(1) & (batch_labels > 0).unsqueeze(0)
            pos_mask = label_matrix & noise_mask
            pos_mask.fill_diagonal_(False)  # Remove self

            # Create mask for negative pairs (different track or noise)
            neg_mask = ~label_matrix

            if not pos_mask.any() or not neg_mask.any():
                continue

            # Compute logits
            exp_sim = torch.exp(sim_matrix)

            # Positive logits
            pos_logits = (sim_matrix * pos_mask.float()).sum(dim=1)

            # Negative logits
            neg_logits = torch.log(exp_sim * neg_mask.float() + 1e-8).sum(dim=1)

            # Loss for this batch
            batch_loss = -pos_logits + neg_logits
            valid_samples = (pos_mask.sum(dim=1) > 0)
            if valid_samples.any():
                loss += batch_loss[valid_samples].mean()
                count += 1

        return loss / max(count, 1)

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )

    # Loss functions
    contrastive_loss_fn = SupConLoss(temperature=0.1)

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Early stopping
    best_val_acc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        train_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            # Forward pass
            embeddings = model(batch)

            # Compute contrastive loss
            loss = contrastive_loss_fn(embeddings, batch.y, batch.batch)

            # Regularization: diversity loss on prototypes
            prototypes_norm = F.normalize(model.prototypes, dim=1)
            proto_sim = torch.matmul(prototypes_norm, prototypes_norm.t())
            eye_mask = torch.eye(proto_sim.size(0), device=device)
            div_loss = (proto_sim * (1 - eye_mask)).abs().mean()
            loss = loss + 0.1 * div_loss

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            train_batches += 1

        train_losses.append(epoch_train_loss / max(train_batches, 1))

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embeddings = model(batch)
                loss = contrastive_loss_fn(embeddings, batch.y, batch.batch)
                epoch_val_loss += loss.item()
                val_batches += 1

        val_losses.append(epoch_val_loss / max(val_batches, 1))

        # Compute clustering accuracy
        def compute_fit_accuracy(loader):
            total_hits = 0
            correct_hits = 0

            for batch in loader:
                batch = batch.to(device)
                pred_labels = model.predict_labels(batch)
                true_labels = batch.y.cpu().numpy()
                pred_labels = pred_labels.cpu().numpy()

                # Simple accuracy calculation (for monitoring)
                # Match predicted clusters to true tracks using Hungarian algorithm
                from scipy.optimize import linear_sum_assignment

                # Group by batch
                batch_indices = batch.batch.cpu().numpy()
                unique_batches = np.unique(batch_indices)

                for b in unique_batches:
                    mask = (batch_indices == b)
                    batch_pred = pred_labels[mask]
                    batch_true = true_labels[mask]

                    # Filter noise (true label 0 and pred label -1)
                    valid_mask = (batch_true > 0)
                    if not valid_mask.any():
                        continue

                    batch_pred_valid = batch_pred[valid_mask]
                    batch_true_valid = batch_true[valid_mask]

                    # Create confusion matrix between predicted and true clusters
                    unique_pred = np.unique(batch_pred_valid[batch_pred_valid >= 0])
                    unique_true = np.unique(batch_true_valid)

                    if len(unique_pred) == 0 or len(unique_true) == 0:
                        continue

                    conf_matrix = np.zeros((len(unique_pred), len(unique_true)))
                    for i, p in enumerate(unique_pred):
                        pred_mask = (batch_pred_valid == p)
                        for j, t in enumerate(unique_true):
                            true_mask = (batch_true_valid == t)
                            conf_matrix[i, j] = np.sum(pred_mask & true_mask)

                    # Hungarian assignment
                    row_ind, col_ind = linear_sum_assignment(-conf_matrix)

                    # Count correctly assigned hits
                    for r, c in zip(row_ind, col_ind):
                        if conf_matrix[r, c] > 0:
                            correct_hits += conf_matrix[r, c]

                    total_hits += len(batch_true_valid)

            return correct_hits / max(total_hits, 1)

        # Compute accuracy on subsets for efficiency
        train_acc = compute_fit_accuracy(
            DataLoader(train_loader.dataset[:100], batch_size=8, shuffle=False)
        ) if len(train_loader.dataset) > 0 else 0

        val_acc = compute_fit_accuracy(
            DataLoader(val_loader.dataset[:50], batch_size=8, shuffle=False)
        ) if len(val_loader.dataset) > 0 else 0

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Update scheduler
        scheduler.step()

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}, "
                  f"Val Loss = {val_losses[-1]:.4f}, "
                  f"Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

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

