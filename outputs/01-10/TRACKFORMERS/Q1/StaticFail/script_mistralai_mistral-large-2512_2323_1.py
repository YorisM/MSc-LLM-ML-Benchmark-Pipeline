
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
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from collections import defaultdict

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_stats = None
        self.global_mean = None
        self.global_std = None

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

            "eval_overrides": {"shuffle": False, "num_workers": 0}
        }

    def fit(self, Xs):
        # Compute global statistics
        all_X = np.concatenate(Xs, axis=0)
        self.global_mean = np.mean(all_X, axis=0)
        self.global_std = np.std(all_X, axis=0)

        # Compute per-layer statistics
        layer_ids = all_X[:, 3]
        unique_layers = np.unique(layer_ids)
        self.layer_stats = {}

        for layer in unique_layers:
            mask = layer_ids == layer
            layer_data = all_X[mask]
            self.layer_stats[layer] = {
                'mean': np.mean(layer_data, axis=0),
                'std': np.std(layer_data, axis=0)
            }

        # Fit scaler on all features except layer_id
        self.scaler.fit(all_X[:, :3])
        return self

    def transform(self, X):
        # X: [N_hits, 4] - r, theta, z, layer_id
        X = X.clone().numpy() if torch.is_tensor(X) else X.copy()

        # Normalize r, theta, z using global scaler
        X[:, :3] = self.scaler.transform(X[:, :3])

        # Add layer-aware features
        layer_ids = X[:, 3]
        new_features = []

        for i, layer in enumerate(layer_ids):
            layer = int(layer)
            if layer in self.layer_stats:
                stats = self.layer_stats[layer]
                # Add layer-relative features
                layer_rel = (X[i, :3] - stats['mean'][:3]) / (stats['std'][:3] + 1e-8)
                new_features.append(np.concatenate([X[i], layer_rel]))
            else:
                new_features.append(np.concatenate([X[i], np.zeros(3)]))

        new_features = np.array(new_features, dtype=np.float32)

        # Add cylindrical coordinate features
        r = new_features[:, 0]
        theta = new_features[:, 1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        new_features = np.column_stack([new_features, x, y])

        return torch.from_numpy(new_features).float()

# ---------- MODEL ARCHITECTURE ----------
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Determine input dimension from example batch
        if isinstance(example_batch_x, list) and len(example_batch_x) > 0:
            input_dim = example_batch_x[0].shape[1]
        else:
            input_dim = 8  # fallback

        # Feature extraction
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Graph layers
        self.conv1 = EdgeConv(64, 64)
        self.conv2 = EdgeConv(64, 64)
        self.conv3 = EdgeConv(64, 64)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Output layers
        self.cluster_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.track_mlp = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Edge construction parameters
        self.k = 16  # number of nearest neighbors
        self.threshold = 0.5  # distance threshold

    def build_edges(self, x):
        # x: [N, F]
        N = x.size(0)
        if N <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=x.device)

        # Use KDTree for efficient nearest neighbor search
        pos = x[:, :3].cpu().numpy()  # use r, theta, z for distance
        tree = KDTree(pos)

        # Find k-nearest neighbors
        distances, indices = tree.query(pos, k=min(self.k + 1, N))

        # Create edge list
        edge_list = []
        for i in range(N):
            for j in range(1, len(indices[i])):  # skip self
                if distances[i][j] < self.threshold:
                    edge_list.append([i, indices[i][j]])

        if len(edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=x.device)

        edge_index = torch.tensor(edge_list, dtype=torch.long, device=x.device).t()
        return edge_index

    def forward(self, batch_x):
        # batch_x is list of tensors [N_i, F]
        all_x = []
        all_edge_indices = []
        batch_offsets = [0]
        current_offset = 0

        # Process each event in the batch
        for x in batch_x:
            if x.size(0) == 0:
                continue

            # Build edges for this event
            edge_index = self.build_edges(x)
            if edge_index.size(1) > 0:
                edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

            # Store data
            all_x.append(x)
            all_edge_indices.append(edge_index + current_offset)
            current_offset += x.size(0)
            batch_offsets.append(current_offset)

        if len(all_x) == 0:
            return torch.empty(0, 32, device=device)

        # Concatenate all events
        x = torch.cat(all_x, dim=0)
        batch = torch.cat([torch.full((x.size(0),), i, device=x.device)
                          for i in range(len(batch_x))], dim=0)

        # Create edge index for the whole batch
        if len(all_edge_indices) > 0 and all_edge_indices[0].size(1) > 0:
            edge_index = torch.cat(all_edge_indices, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)

        # Node features
        h = self.node_encoder(x)

        # Graph convolutions
        h1 = self.conv1(h, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Attention
        attn_weights = self.attention(h3)
        h_attn = h3 * attn_weights

        # Cluster features
        cluster_feats = self.cluster_head(h_attn)

        return cluster_feats

    def predict_labels(self, batch_x):
        if len(batch_x) == 0:
            return []

        # Get cluster features
        cluster_feats = self.forward(batch_x)

        # Split by event
        batch_offsets = [0]
        current_offset = 0
        for x in batch_x:
            current_offset += x.size(0)
            batch_offsets.append(current_offset)

        # Process each event separately
        all_labels = []
        for i in range(len(batch_x)):
            start = batch_offsets[i]
            end = batch_offsets[i+1]

            if start >= end:
                all_labels.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            feats = cluster_feats[start:end]

            # Simple clustering - in practice you might use more sophisticated methods
            # Here we use a simple approach for demonstration
            if feats.size(0) > 0:
                # Use cosine similarity for clustering
                sim_matrix = F.cosine_similarity(feats.unsqueeze(1), feats.unsqueeze(0), dim=2)
                sim_matrix = sim_matrix.cpu().numpy()

                # Threshold-based clustering
                clusters = []
                visited = set()
                for j in range(sim_matrix.shape[0]):
                    if j not in visited:
                        cluster = np.where(sim_matrix[j] > 0.7)[0]
                        clusters.append(cluster)
                        visited.update(cluster)

                # Assign labels
                labels = torch.full((feats.size(0),), -1, dtype=torch.long, device=device)
                for cluster_id, cluster in enumerate(clusters):
                    labels[cluster] = cluster_id
            else:
                labels = torch.empty(0, dtype=torch.long, device=device)

            all_labels.append(labels)

        return all_labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
def train_model(model, train_loader, val_loader, epochs=10):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    best_val_acc = 0
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_idx, (Xs, ys) in enumerate(train_loader):
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            optimizer.zero_grad()

            # Forward pass
            cluster_feats = model(Xs)

            # Compute loss - we'll use a simple contrastive loss
            loss = 0
            batch_offsets = [0]
            current_offset = 0

            for x, y in zip(Xs, ys):
                current_offset += x.size(0)
                batch_offsets.append(current_offset)

            for i in range(len(Xs)):
                start = batch_offsets[i]
                end = batch_offsets[i+1]

                if start >= end:
                    continue

                feats = cluster_feats[start:end]
                labels = ys[i]

                # Skip noise hits
                mask = labels != 0
                if mask.sum() == 0:
                    continue

                feats = feats[mask]
                labels = labels[mask]

                # Create positive and negative pairs
                unique_labels = torch.unique(labels)
                if len(unique_labels) < 2:
                    continue

                # Compute pairwise distances
                dist_matrix = torch.cdist(feats, feats)

                # Create positive and negative masks
                pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
                neg_mask = ~pos_mask

                # Contrastive loss
                pos_dist = dist_matrix[pos_mask].pow(2)
                neg_dist = (1 - dist_matrix[neg_mask]).clamp(min=0).pow(2)

                if pos_dist.size(0) > 0 and neg_dist.size(0) > 0:
                    loss += pos_dist.mean() + F.relu(1 - neg_dist.sqrt()).pow(2).mean()

            if loss > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() if loss > 0 else 0

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                cluster_feats = model(Xs)
                batch_offsets = [0]
                current_offset = 0

                for x, y in zip(Xs, ys):
                    current_offset += x.size(0)
                    batch_offsets.append(current_offset)

                for i in range(len(Xs)):
                    start = batch_offsets[i]
                    end = batch_offsets[i+1]

                    if start >= end:
                        continue

                    feats = cluster_feats[start:end]
                    labels = ys[i]

                    mask = labels != 0
                    if mask.sum() == 0:
                        continue

                    feats = feats[mask]
                    labels = labels[mask]

                    if feats.size(0) > 0:
                        # Simple accuracy - in practice you'd compute FitAccuracy
                        pred_labels = model.predict_labels([Xs[i]])[0]
                        if pred_labels.size(0) > 0:
                            pred_labels = pred_labels[mask]

                            # Count correct predictions (simplified)
                            correct = (pred_labels != -1).float().sum()
                            val_correct += correct.item()
                            val_samples += mask.sum().item()

        # Compute metrics
        train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

        train_acc = 0  # Simplified - in practice compute FitAccuracy
        val_acc = val_correct / val_samples if val_samples > 0 else 0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        scheduler.step(val_acc)

    # Load best model
    if 'best_model' in locals():
        model.load_state_dict(best_model)

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

