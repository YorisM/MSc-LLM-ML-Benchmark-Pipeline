
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
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
import torch_scatter
from torch_cluster import knn_graph, radius_graph

# -------- CUSTOM DATASET --------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        self.events = events
        self.pre = pre
        self.train = train

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        X, y = split_X_y(self.events[idx])
        if self.pre is not None:
            X = self.pre.transform(X)
        # Create PyG Data object
        data = Data(x=X, y=y)
        return data

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_means = None
        self.layer_stds = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {"follow_batch": []},
            "eval_overrides": {"shuffle": False, "batch_size": 32}
        }

    def fit(self, Xs):
        # Concatenate all events for global statistics
        all_X = torch.cat(Xs, dim=0).numpy()  # [total_hits, 4]

        # Fit scaler on all features except layer_id
        self.scaler.fit(all_X[:, :3])

        # Compute layer-specific statistics
        layer_ids = all_X[:, 3]
        unique_layers = np.unique(layer_ids)
        self.layer_means = {}
        self.layer_stds = {}

        for layer in unique_layers:
            mask = layer_ids == layer
            if mask.sum() > 1:
                self.layer_means[layer] = all_X[mask, :3].mean(axis=0)
                self.layer_stds[layer] = all_X[mask, :3].std(axis=0)
                # Avoid division by zero
                self.layer_stds[layer][self.layer_stds[layer] == 0] = 1.0
        return self

    def transform(self, X):
        # X: [N_hits, 4]
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X

        # Separate features
        spatial = X_np[:, :3]  # r, theta, z
        layers = X_np[:, 3]    # layer_id

        # Global normalization
        spatial_normalized = self.scaler.transform(spatial)

        # Layer-wise normalization
        spatial_layer_norm = np.zeros_like(spatial_normalized)
        for i, layer in enumerate(layers):
            if layer in self.layer_means:
                spatial_layer_norm[i] = (spatial_normalized[i] - self.layer_means[layer]) / self.layer_stds[layer]
            else:
                spatial_layer_norm[i] = spatial_normalized[i]

        # Add engineered features
        r = spatial[:, 0]
        theta = spatial[:, 1]
        z = spatial[:, 2]

        # Convert to cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Create final feature matrix [N_hits, 8]
        features = np.column_stack([
            x, y, z,                         # Cartesian coordinates
            spatial_layer_norm,              # Normalized features [3]
            layers.reshape(-1, 1),          # Layer ID
            np.sqrt(x**2 + y**2 + z**2)     # Distance from origin
        ])

        return torch.from_numpy(features.astype(np.float32))

def make_preprocessor():
    return MyPreprocessor()

# ---------- GNN LAYERS ----------
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)  # [E, 2*in_channels]
        return self.mlp(edge_features)  # [E, out_channels]

class TrackGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EdgeConv(hidden_dim, hidden_dim))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Final projection for clustering
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64)  # Embedding dimension for clustering
        )

        # Edge prediction head for auxiliary loss
        self.edge_pred = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch):
        # x: [N, input_dim], edge_index: [2, E], batch: [N]
        h = F.relu(self.input_proj(x))  # [N, hidden_dim]

        # Store intermediate representations for skip connections
        intermediates = []
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)  # [N, hidden_dim]
            h = norm(h + h_new)  # Residual connection
            h = F.relu(h)
            intermediates.append(h)

        # Final embeddings
        embeddings = self.final_proj(h)  # [N, 64]

        # Edge features for auxiliary loss
        src, dst = edge_index
        edge_features = torch.cat([h[src], h[dst]], dim=1)  # [E, 2*hidden_dim]
        edge_scores = self.edge_pred(edge_features)  # [E, 1]

        return embeddings, edge_scores.squeeze(-1)

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Extract input dimension from example batch
        input_dim = example_batch_x.x.size(1)

        self.gnn = TrackGNN(input_dim=input_dim, hidden_dim=128, num_layers=5)

        # Learnable clustering parameters
        self.cluster_centers = nn.Parameter(torch.randn(200, 64))  # Max clusters
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Edge construction parameters
        self.k = 20
        self.r = 0.1

    def build_graph(self, x, batch):
        # Build multi-scale graph connectivity
        edge_index_knn = knn_graph(x[:, :3], k=self.k, batch=batch)  # Spatial KNN

        # Radius graph for local connections
        edge_index_radius = radius_graph(x[:, :3], r=self.r, batch=batch)

        # Combine edges
        edge_index = torch.cat([edge_index_knn, edge_index_radius], dim=1)
        edge_index = torch.unique(edge_index, dim=1)  # Remove duplicates

        return edge_index

    def forward(self, data):
        # Extract from PyG batch
        x, batch = data.x, data.batch

        # Build graph
        edge_index = self.build_graph(x, batch)

        # Get embeddings and edge scores
        embeddings, edge_scores = self.gnn(x, edge_index, batch)

        return embeddings, edge_scores, edge_index

    def predict_labels(self, batch_x):
        # batch_x is PyG Batch object
        self.eval()
        with torch.no_grad():
            # Get embeddings
            embeddings, _, edge_index = self.forward(batch_x)

            # Soft clustering assignment
            distances = torch.cdist(embeddings, self.cluster_centers)  # [N, num_centers]
            cluster_probs = F.softmax(-distances / self.temperature, dim=1)  # [N, num_centers]

            # Assign to nearest cluster
            cluster_assignments = torch.argmax(cluster_probs, dim=1)  # [N]

            # Apply DBSCAN-like post-processing using embeddings
            labels = self._dbscan_postprocess(
                embeddings, 
                cluster_assignments,
                batch_x.batch,
                eps=0.5,
                min_samples=3
            )

            # Handle noise (assign -1 to isolated points)
            return labels

    def _dbscan_postprocess(self, embeddings, initial_labels, batch, eps=0.5, min_samples=3):
        # Simple DBSCAN implementation for post-processing
        batch_size = batch.max().item() + 1
        all_labels = []

        for b in range(batch_size):
            mask = batch == b
            emb_batch = embeddings[mask]  # [N_b, 64]

            if len(emb_batch) == 0:
                all_labels.append(torch.tensor([], device=embeddings.device, dtype=torch.long))
                continue

            # Compute pairwise distances
            dists = torch.cdist(emb_batch, emb_batch)

            # Find core points
            n_neighbors = (dists <= eps).sum(dim=1) - 1  # Exclude self
            core_mask = n_neighbors >= min_samples

            # Initialize labels: -1 for noise, 0 for unassigned
            labels_batch = torch.full((len(emb_batch),), -1, 
                                    device=embeddings.device, dtype=torch.long)
            cluster_id = 0

            # Expand clusters from core points
            visited = torch.zeros(len(emb_batch), dtype=torch.bool, device=embeddings.device)

            for i in torch.where(core_mask)[0]:
                if visited[i]:
                    continue

                # Start new cluster
                stack = [i]
                labels_batch[i] = cluster_id
                visited[i] = True

                while stack:
                    current = stack.pop()

                    # Find neighbors
                    neighbors = torch.where(dists[current] <= eps)[0]

                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            labels_batch[neighbor] = cluster_id
                            visited[neighbor] = True

                            # If neighbor is core, expand from it
                            if core_mask[neighbor]:
                                stack.append(neighbor)

                cluster_id += 1

            # Assign border points to nearest cluster
            for i in torch.where(~visited)[0]:
                if not core_mask[i]:
                    neighbors = torch.where(dists[i] <= eps)[0]
                    neighbor_labels = labels_batch[neighbors]
                    valid_labels = neighbor_labels[neighbor_labels != -1]

                    if len(valid_labels) > 0:
                        # Assign to most common neighbor label
                        unique_labels, counts = torch.unique(valid_labels, return_counts=True)
                        labels_batch[i] = unique_labels[torch.argmax(counts)]

            all_labels.append(labels_batch)

        return torch.cat(all_labels, dim=0)

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss weights
    lambda_cluster = 1.0
    lambda_edge = 0.5
    lambda_separation = 0.1

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct_hits = 0
        total_hits = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            embeddings, edge_scores, edge_index = model(batch)

            # Build ground truth edges (same track = 1, different = 0)
            track_ids = batch.y
            src, dst = edge_index

            # Edge labels: 1 if same track (excluding noise), 0 otherwise
            edge_labels = ((track_ids[src] == track_ids[dst]) & 
                          (track_ids[src] != 0) & 
                          (track_ids[dst] != 0)).float()

            # Edge prediction loss
            edge_loss = F.binary_cross_entropy_with_logits(
                edge_scores, edge_labels, reduction='mean'
            )

            # Cluster separation loss
            unique_tracks = torch.unique(track_ids[track_ids != 0])
            if len(unique_tracks) > 1:
                cluster_means = []
                for track in unique_tracks:
                    mask = track_ids == track
                    if mask.sum() > 0:
                        cluster_means.append(embeddings[mask].mean(dim=0, keepdim=True))

                if len(cluster_means) > 1:
                    cluster_means = torch.cat(cluster_means, dim=0)
                    intra_dists = torch.cdist(cluster_means, cluster_means)
                    mask = torch.eye(len(cluster_means), device=device).bool()
                    intra_dists = intra_dists[~mask].view(len(cluster_means), -1)

                    separation_loss = -torch.mean(torch.min(intra_dists, dim=1)[0])
                else:
                    separation_loss = torch.tensor(0.0, device=device)
            else:
                separation_loss = torch.tensor(0.0, device=device)

            # Combine losses
            loss = lambda_edge * edge_loss + lambda_separation * separation_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy (approximate)
            with torch.no_grad():
                pred_labels = model.predict_labels(batch)
                true_labels = batch.y

                # Simple accuracy calculation (ignoring permutation invariance)
                for b in range(batch.batch.max().item() + 1):
                    mask = batch.batch == b
                    pred_b = pred_labels[mask]
                    true_b = true_labels[mask]

                    # Match clusters to tracks using Hungarian algorithm
                    from scipy.optimize import linear_sum_assignment
                    import numpy as np

                    pred_unique = torch.unique(pred_b[pred_b != -1])
                    true_unique = torch.unique(true_b[true_b != 0])

                    if len(pred_unique) > 0 and len(true_unique) > 0:
                        # Build cost matrix
                        cost_matrix = np.zeros((len(pred_unique), len(true_unique)))
                        for i, p_cluster in enumerate(pred_unique):
                            for j, t_track in enumerate(true_unique):
                                mask_p = pred_b == p_cluster
                                mask_t = true_b == t_track
                                intersection = (mask_p & mask_t).sum().item()
                                union = (mask_p | mask_t).sum().item()
                                cost_matrix[i, j] = -intersection / max(union, 1)

                        # Hungarian matching
                        row_ind, col_ind = linear_sum_assignment(cost_matrix)

                        # Count correctly assigned hits
                        for i, j in zip(row_ind, col_ind):
                            p_cluster = pred_unique[i]
                            t_track = true_unique[j]
                            mask_p = pred_b == p_cluster
                            mask_t = true_b == t_track
                            correct_hits += (mask_p & mask_t).sum().item()

                    total_hits += mask.sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_hits / max(total_hits, 1)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                embeddings, edge_scores, edge_index = model(batch)
                track_ids = batch.y
                src, dst = edge_index

                edge_labels = ((track_ids[src] == track_ids[dst]) & 
                              (track_ids[src] != 0) & 
                              (track_ids[dst] != 0)).float()

                edge_loss = F.binary_cross_entropy_with_logits(
                    edge_scores, edge_labels, reduction='mean'
                )

                val_loss += edge_loss.item()

                # Calculate validation accuracy
                pred_labels = model.predict_labels(batch)
                true_labels = batch.y

                for b in range(batch.batch.max().item() + 1):
                    mask = batch.batch == b
                    pred_b = pred_labels[mask]
                    true_b = true_labels[mask]

                    pred_unique = torch.unique(pred_b[pred_b != -1])
                    true_unique = torch.unique(true_b[true_b != 0])

                    if len(pred_unique) > 0 and len(true_unique) > 0:
                        cost_matrix = np.zeros((len(pred_unique), len(true_unique)))
                        for i, p_cluster in enumerate(pred_unique):
                            for j, t_track in enumerate(true_unique):
                                mask_p = pred_b == p_cluster
                                mask_t = true_b == t_track
                                intersection = (mask_p & mask_t).sum().item()
                                union = (mask_p | mask_t).sum().item()
                                cost_matrix[i, j] = -intersection / max(union, 1)

                        row_ind, col_ind = linear_sum_assignment(cost_matrix)

                        for i, j in zip(row_ind, col_ind):
                            p_cluster = pred_unique[i]
                            t_track = true_unique[j]
                            mask_p = pred_b == p_cluster
                            mask_t = true_b == t_track
                            val_correct += (mask_p & mask_t).sum().item()

                    val_total += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / max(val_total, 1)

        # Update learning rate
        scheduler.step()

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

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

