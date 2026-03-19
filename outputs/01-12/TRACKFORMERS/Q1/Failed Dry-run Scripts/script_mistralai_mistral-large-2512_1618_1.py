
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
        self.layer_mean = None
        self.layer_std = None
        self.r_mean = None
        self.r_std = None
        self.z_mean = None
        self.z_std = None
        self.theta_mean = None
        self.theta_std = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": "ragged_xy",
            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Compute global statistics for normalization
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Only scale r, theta, z

        # Compute per-layer statistics
        layer_ids = np.unique(all_X[:, 3])
        layer_stats = {}
        for lid in layer_ids:
            mask = all_X[:, 3] == lid
            layer_data = all_X[mask, :3]
            layer_stats[lid] = {
                'mean': np.mean(layer_data, axis=0),
                'std': np.std(layer_data, axis=0)
            }

        self.layer_mean = {lid: stats['mean'] for lid, stats in layer_stats.items()}
        self.layer_std = {lid: stats['std'] for lid, stats in layer_stats.items()}

        # Store global statistics for layer_id (won't be scaled)
        self.r_mean, self.theta_mean, self.z_mean = np.mean(all_X[:, :3], axis=0)
        self.r_std, self.theta_std, self.z_std = np.std(all_X[:, :3], axis=0)

        return self

    def transform(self, X):
        # X shape: [N_hits, 4]
        X = X.clone().detach().numpy() if torch.is_tensor(X) else X

        # Normalize r, theta, z using global scaler
        X[:, :3] = self.scaler.transform(X[:, :3])

        # Add layer-aware features
        layer_ids = X[:, 3]
        for i, lid in enumerate(layer_ids):
            lid = int(lid)
            if lid in self.layer_mean:
                # Add layer-relative position features
                X[i, 0] = (X[i, 0] - self.layer_mean[lid][0]) / (self.layer_std[lid][0] + 1e-8)
                X[i, 1] = (X[i, 1] - self.layer_mean[lid][1]) / (self.layer_std[lid][1] + 1e-8)
                X[i, 2] = (X[i, 2] - self.layer_mean[lid][2]) / (self.layer_std[lid][2] + 1e-8)

        # Add cylindrical coordinate features
        r = X[:, 0]
        theta = X[:, 1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = X[:, 2]

        # Stack all features
        features = np.column_stack([
            X[:, :3],  # normalized r, theta, z
            x, y, z,   # cartesian coordinates
            X[:, 3],   # layer_id (not normalized)
            r * np.cos(theta) * z,  # interaction terms
            r * np.sin(theta) * z,
            r * r,
            z * z
        ])

        return torch.FloatTensor(features)

def make_preprocessor():
    return MyPreprocessor()

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

        # Determine input feature dimension from example batch
        if isinstance(example_batch_x, list):
            input_dim = example_batch_x[0].shape[1]
        else:
            input_dim = example_batch_x.shape[1]

        # Graph neural network layers
        self.conv1 = EdgeConv(input_dim, 64)
        self.conv2 = EdgeConv(64, 64)
        self.conv3 = EdgeConv(64, 64)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Output layers
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 1)  # Will predict similarity score

        # Track embedding
        self.track_embedding = nn.Embedding(1, 64)  # Dummy embedding for track initialization

        # Dropout
        self.dropout = nn.Dropout(0.2)

    def build_graph(self, x):
        # Build k-NN graph (k=10)
        k = 10
        batch_size = len(x)
        edge_indices = []

        for i in range(batch_size):
            # Use KDTree for efficient nearest neighbor search
            coords = x[i][:, :3].cpu().numpy()  # Use first 3 features for spatial proximity
            tree = KDTree(coords)
            distances, indices = tree.query(coords, k=k+1)  # +1 to include self

            # Create edge index for this event
            src = np.repeat(np.arange(len(coords)), k)
            dst = indices[:, 1:].flatten()  # exclude self
            edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=x[i].device)
            edge_indices.append(edge_index)

        return edge_indices

    def forward(self, Xs):
        # Xs: list of tensors [N_i, F]
        batch_size = len(Xs)
        all_embeddings = []

        for i in range(batch_size):
            x = Xs[i]
            edge_index = self.build_graph([x])[0]

            # Graph convolutions
            x = F.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv3(x, edge_index))

            # Attention mechanism
            attn_weights = self.attention(x)
            attn_weights = F.softmax(attn_weights, dim=0)
            x = x * attn_weights

            # Further processing
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))

            all_embeddings.append(x)

        return all_embeddings

    def predict_labels(self, Xs):
        # Get embeddings
        embeddings = self.forward(Xs)

        # Perform clustering on each event
        all_labels = []
        for i in range(len(Xs)):
            x = embeddings[i]
            coords = Xs[i][:, :3]  # Use spatial coordinates for clustering

            # Simple clustering: find high-density regions
            # In practice, we'd use a more sophisticated clustering algorithm
            # Here we use a simple approach for demonstration

            # Compute pairwise distances
            dist_matrix = torch.cdist(x, x)

            # Find hits that are close in embedding space and physical space
            sim_matrix = torch.exp(-dist_matrix / 1.0)  # Similarity matrix

            # Threshold similarity
            adj_matrix = (sim_matrix > 0.5).float()

            # Connected components as clusters
            n_hits = adj_matrix.shape[0]
            visited = torch.zeros(n_hits, dtype=torch.bool, device=x.device)
            labels = -torch.ones(n_hits, dtype=torch.long, device=x.device)
            current_label = 0

            for j in range(n_hits):
                if not visited[j]:
                    # BFS to find connected component
                    queue = [j]
                    visited[j] = True
                    labels[j] = current_label

                    while queue:
                        node = queue.pop(0)
                        neighbors = (adj_matrix[node] > 0).nonzero().squeeze()
                        if neighbors.dim() == 0:
                            continue
                        for neighbor in neighbors:
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                labels[neighbor] = current_label
                                queue.append(neighbor)
                    current_label += 1

            # Filter small clusters (less than 4 hits)
            unique_labels, counts = torch.unique(labels, return_counts=True)
            for label in unique_labels:
                if counts[label] < 4:
                    labels[labels == label] = -1

            all_labels.append(labels)

        return all_labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for Xs, ys in train_loader:
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            optimizer.zero_grad()

            # Forward pass
            embeddings = model(Xs)

            # Compute loss (contrastive loss)
            loss = 0.0
            for i in range(len(Xs)):
                x = embeddings[i]
                y = ys[i]

                # Get positive and negative pairs
                pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & (y.unsqueeze(0) != 0)
                neg_mask = (y.unsqueeze(0) != y.unsqueeze(1)) | (y.unsqueeze(0) == 0)

                # Compute pairwise distances
                dist = torch.cdist(x, x)

                # Contrastive loss
                pos_dist = dist[pos_mask]
                neg_dist = dist[neg_mask]

                pos_loss = torch.mean(F.relu(pos_dist - 0.1))
                neg_loss = torch.mean(F.relu(1.0 - neg_dist))
                loss += pos_loss + neg_loss

                # Accuracy estimation (not exact)
                pred_labels = model.predict_labels([Xs[i]])[0]
                correct += ((pred_labels == ys[i]) & (ys[i] != 0)).sum().item()
                total += (ys[i] != 0).sum().item()

            loss = loss / len(Xs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0.0
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                embeddings = model(Xs)
                loss = 0.0

                for i in range(len(Xs)):
                    x = embeddings[i]
                    y = ys[i]

                    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & (y.unsqueeze(0) != 0)
                    neg_mask = (y.unsqueeze(0) != y.unsqueeze(1)) | (y.unsqueeze(0) == 0)

                    dist = torch.cdist(x, x)
                    pos_dist = dist[pos_mask]
                    neg_dist = dist[neg_mask]

                    pos_loss = torch.mean(F.relu(pos_dist - 0.1))
                    neg_loss = torch.mean(F.relu(1.0 - neg_dist))
                    loss += pos_loss + neg_loss

                    pred_labels = model.predict_labels([Xs[i]])[0]
                    val_correct += ((pred_labels == ys[i]) & (ys[i] != 0)).sum().item()
                    val_total += (ys[i] != 0).sum().item()

                val_loss += (loss / len(Xs)).item()

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
        else:
            if epoch > 10 and val_acc < best_val_acc * 0.95:
                print("Early stopping triggered")
                break

    # Load best model
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

