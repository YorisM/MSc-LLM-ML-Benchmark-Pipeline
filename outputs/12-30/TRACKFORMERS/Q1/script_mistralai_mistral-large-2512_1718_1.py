
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import detect_and_assert_lane, assert_label_output_by_lane, build_dataset, build_dataloader
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
        self.layer_mean = None
        self.layer_std = None
        self.r_mean = None
        self.r_std = None
        self.z_mean = None
        self.z_std = None

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

            "eval_overrides": {"shuffle": False, "num_workers": 2}
        }

    def fit(self, Xs):
        # Stack all events to compute global statistics
        all_X = np.concatenate(Xs, axis=0)

        # Fit standard scaler on all features
        self.scaler.fit(all_X)

        # Compute per-layer statistics for layer_id
        layer_ids = all_X[:, 3]
        self.layer_mean = np.mean(layer_ids)
        self.layer_std = np.std(layer_ids)

        # Compute r and z statistics
        self.r_mean = np.mean(all_X[:, 0])
        self.r_std = np.std(all_X[:, 0])
        self.z_mean = np.mean(all_X[:, 2])
        self.z_std = np.std(all_X[:, 2])

        return self

    def transform(self, X):
        # X shape: [N_hits, 4]
        X = self.scaler.transform(X)

        # Normalize layer_id separately
        X[:, 3] = (X[:, 3] - self.layer_mean) / (self.layer_std + 1e-8)

        # Normalize r and z separately
        X[:, 0] = (X[:, 0] - self.r_mean) / (self.r_std + 1e-8)
        X[:, 2] = (X[:, 2] - self.z_mean) / (self.z_std + 1e-8)

        return torch.FloatTensor(X)

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

        # Determine input dimension from example batch
        input_dim = example_batch_x[0].shape[1]

        # Edge convolution layers
        self.conv1 = EdgeConv(input_dim, 64)
        self.conv2 = EdgeConv(64, 128)
        self.conv3 = EdgeConv(128, 256)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Output layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

        # Track embedding layer
        self.track_embedding = nn.Linear(32, 32)

        # Final classification head
        self.classifier = nn.Linear(32, 1)  # Will predict similarity scores

        # Dropout
        self.dropout = nn.Dropout(0.2)

    def build_kdtree_graph(self, x):
        # Build KDTree for spatial neighbors
        coords = x[:, :3].cpu().numpy()
        tree = KDTree(coords)

        # Find 10 nearest neighbors (including self)
        distances, indices = tree.query(coords, k=10)

        # Create edge_index
        edge_index = []
        for i in range(len(indices)):
            for j in range(len(indices[i])):
                edge_index.append([i, indices[i][j]])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return edge_index.to(x.device)

    def forward(self, batch_x):
        outputs = []

        for x in batch_x:
            # x shape: [N_hits, 4]
            N = x.size(0)

            # Build spatial graph
            edge_index = self.build_kdtree_graph(x)

            # Edge convolution layers
            x1 = F.relu(self.conv1(x, edge_index))
            x1 = self.dropout(x1)

            x2 = F.relu(self.conv2(x1, edge_index))
            x2 = self.dropout(x2)

            x3 = F.relu(self.conv3(x2, edge_index))
            x3 = self.dropout(x3)

            # Attention mechanism
            attn_weights = F.softmax(self.attention(x3), dim=0)
            x_attn = x3 * attn_weights

            # Fully connected layers
            x_fc = F.relu(self.fc1(x_attn))
            x_fc = self.dropout(x_fc)

            x_fc = F.relu(self.fc2(x_fc))
            x_fc = self.dropout(x_fc)

            x_fc = F.relu(self.fc3(x_fc))

            # Store embeddings for this event
            outputs.append(x_fc)

        return outputs

class TrackClusterer:
    def __init__(self, embeddings, threshold=0.5):
        self.embeddings = embeddings
        self.threshold = threshold

    def cluster(self):
        # Convert embeddings to numpy
        emb_np = self.embeddings.detach().cpu().numpy()

        # Compute pairwise cosine similarity
        norm = np.linalg.norm(emb_np, axis=1, keepdims=True)
        norm_emb = emb_np / (norm + 1e-8)
        similarity = np.dot(norm_emb, norm_emb.T)

        # Create adjacency matrix
        adj = similarity > self.threshold

        # Connected components for clustering
        n = adj.shape[0]
        visited = np.zeros(n, dtype=bool)
        clusters = []
        current_cluster = []

        for i in range(n):
            if not visited[i]:
                # BFS to find connected component
                queue = [i]
                visited[i] = True
                current_cluster = [i]

                while queue:
                    node = queue.pop(0)
                    neighbors = np.where(adj[node])[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                            current_cluster.append(neighbor)

                clusters.append(current_cluster)

        # Create cluster labels
        labels = -np.ones(n, dtype=np.int64)
        for cluster_id, cluster in enumerate(clusters):
            for hit_idx in cluster:
                labels[hit_idx] = cluster_id

        return torch.tensor(labels, dtype=torch.long)

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
def compute_loss(predictions, targets, embeddings):
    # predictions: list of embeddings for each event
    # targets: list of true labels for each event

    loss = 0.0
    total_pairs = 0

    for pred_emb, true_labels in zip(predictions, targets):
        # Get unique track IDs (excluding noise)
        unique_tracks = torch.unique(true_labels[true_labels > 0])

        if len(unique_tracks) == 0:
            continue

        # Create positive and negative pairs
        pos_pairs = []
        neg_pairs = []

        # Create track to hits mapping
        track_to_hits = defaultdict(list)
        for hit_idx, track_id in enumerate(true_labels):
            if track_id > 0:
                track_to_hits[track_id.item()].append(hit_idx)

        # Create positive pairs (hits from same track)
        for track_id, hits in track_to_hits.items():
            if len(hits) >= 2:
                for i in range(len(hits)):
                    for j in range(i+1, len(hits)):
                        pos_pairs.append((hits[i], hits[j]))

        # Create negative pairs (hits from different tracks)
        track_ids = list(track_to_hits.keys())
        for i in range(len(track_ids)):
            for j in range(i+1, len(track_ids)):
                hits_i = track_to_hits[track_ids[i]]
                hits_j = track_to_hits[track_ids[j]]
                for hit_i in hits_i:
                    for hit_j in hits_j:
                        neg_pairs.append((hit_i, hit_j))

        if not pos_pairs or not neg_pairs:
            continue

        # Convert to tensors
        pos_pairs = torch.tensor(pos_pairs, dtype=torch.long)
        neg_pairs = torch.tensor(neg_pairs, dtype=torch.long)

        # Compute similarity scores
        emb = pred_emb
        pos_sim = F.cosine_similarity(emb[pos_pairs[:, 0]], emb[pos_pairs[:, 1]])
        neg_sim = F.cosine_similarity(emb[neg_pairs[:, 0]], emb[neg_pairs[:, 1]])

        # Contrastive loss
        pos_loss = (1 - pos_sim).pow(2)
        neg_loss = F.relu(neg_sim - 0.2).pow(2)
        pair_loss = torch.cat([pos_loss, neg_loss]).mean()

        # Add regularization
        reg_loss = emb.pow(2).mean()

        # Total loss
        loss += pair_loss + 0.01 * reg_loss
        total_pairs += 1

    if total_pairs > 0:
        return loss / total_pairs
    else:
        return torch.tensor(0.0, device=predictions[0].device)

def train_model(model, train_loader, val_loader, epochs=10):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for xs, ys in train_loader:
            xs = [x.to(device) for x in xs]
            ys = [y.to(device) for y in ys]

            optimizer.zero_grad()
            embeddings = model(xs)

            loss = compute_loss(embeddings, ys, embeddings)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for xs, ys in val_loader:
                xs = [x.to(device) for x in xs]
                ys = [y.to(device) for y in ys]

                embeddings = model(xs)
                loss = compute_loss(embeddings, ys, embeddings)
                val_loss += loss.item()
                val_batches += 1

                # Cluster embeddings to get predictions
                for emb, y in zip(embeddings, ys):
                    clusterer = TrackClusterer(emb, threshold=0.5)
                    pred = clusterer.cluster()
                    all_preds.append(pred.cpu())
                    all_targets.append(y.cpu())

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        # Compute accuracy (simplified - real metric would need proper matching)
        # This is just for monitoring, real evaluation uses FitAccuracy
        correct = 0
        total = 0
        for pred, target in zip(all_preds, all_targets):
            # Simple accuracy - not the real metric but useful for monitoring
            non_noise = target > 0
            if non_noise.sum() > 0:
                correct += (pred[non_noise] == target[non_noise]).sum().item()
                total += non_noise.sum().item()

        val_acc = correct / total if total > 0 else 0
        val_accs.append(val_acc)

        # Training accuracy (simplified)
        model.train()
        train_correct = 0
        train_total = 0
        for xs, ys in train_loader:
            xs = [x.to(device) for x in xs]
            ys = [y.to(device) for y in ys]

            embeddings = model(xs)
            for emb, y in zip(embeddings, ys):
                clusterer = TrackClusterer(emb, threshold=0.5)
                pred = clusterer.cluster()
                non_noise = y > 0
                if non_noise.sum() > 0:
                    train_correct += (pred[non_noise].cpu() == y[non_noise].cpu()).sum().item()
                    train_total += non_noise.sum().item()

        train_acc = train_correct / train_total if train_total > 0 else 0
        train_accs.append(train_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
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
    if mode == "torch_ragged_xy":
        Xs, ys = first_batch
        Xs = [x.to(device) for x in Xs]
        ys = [y.to(device) for y in ys]
        model = make_model(Xs).to(device)
        out = model(Xs)
    elif mode == "pyg_batch":
        G = first_batch.to(device)
        model = make_model(G).to(device)
        out = model(G)
    else:
        raise RuntimeError(f"Unknown lane mode: {mode}")

    assert_label_output_by_lane(mode, first_batch, out, allow_noise_label=True)

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
                        out = trained_model(Xs)
                    elif mode == "pyg_batch":
                        G = batch.to(device)
                        out = trained_model(G)
                    else:
                        raise RuntimeError(f"Unknown lane mode: {mode}")
                    assert_label_output_by_lane(mode, batch, out, allow_noise_label=True)
                    if i >= 3:  # 4 batches
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

