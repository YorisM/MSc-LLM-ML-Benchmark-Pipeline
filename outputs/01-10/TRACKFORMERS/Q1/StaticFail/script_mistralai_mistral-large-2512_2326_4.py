
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
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from collections import defaultdict

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_mean = None
        self.layer_std = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,

            "collate": None,
            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False, "batch_size": 64}
        }

    def fit(self, Xs):
        # Compute global statistics for normalization
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Only scale r, theta, z

        # Compute layer statistics
        layer_ids = all_X[:, 3]
        self.layer_mean = np.mean(layer_ids)
        self.layer_std = np.std(layer_ids)

        return self

    def transform(self, X):
        # X shape: [N_hits, 4]
        X = X.clone().detach()

        # Normalize r, theta, z
        X[:, :3] = torch.from_numpy(self.scaler.transform(X[:, :3].numpy())).float()

        # Normalize layer_id
        if self.layer_std > 0:
            X[:, 3] = (X[:, 3] - self.layer_mean) / self.layer_std

        # Create edge_index for PyG
        coords = X[:, :3].numpy()
        tree = KDTree(coords)
        edge_index = tree.query_pairs(r=0.5, output_type='ndarray').T
        edge_index = torch.from_numpy(edge_index).long()

        # Create PyG Data object
        data = Data(x=X, edge_index=edge_index)
        return data

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
        num_features = example_batch_x.x.shape[1]

        # Graph layers
        self.conv1 = EdgeConv(num_features, 64)
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

        # Track embedding
        self.track_embedding = nn.Linear(32, 32)

        # Final classification
        self.classifier = nn.Linear(32, 1)  # Will use sigmoid for binary classification

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Attention
        attn_weights = F.softmax(self.attention(x), dim=0)
        x = x * attn_weights

        # MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

    def predict_labels(self, data):
        with torch.no_grad():
            embeddings = self.forward(data)  # [N_hits, 32]

            # Simple clustering using embeddings
            # First, try to find clusters using distance threshold
            coords = data.x[:, :3].cpu().numpy()
            embeddings_np = embeddings.cpu().numpy()

            # Use HDBSCAN for clustering
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=4,
                    min_samples=2,
                    cluster_selection_epsilon=0.5,
                    metric='euclidean'
                )
                labels = clusterer.fit_predict(embeddings_np)
                labels = torch.from_numpy(labels).to(data.x.device)

                # Convert noise (-1) to new cluster IDs
                max_label = labels.max()
                noise_mask = (labels == -1)
                if noise_mask.any():
                    labels[noise_mask] = torch.arange(
                        max_label + 1,
                        max_label + 1 + noise_mask.sum(),
                        device=labels.device
                    )
            except:
                # Fallback to simple distance-based clustering
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=0.5, min_samples=4).fit(embeddings_np)
                labels = torch.from_numpy(clustering.labels_).to(data.x.device)

                # Convert noise (-1) to new cluster IDs
                max_label = labels.max()
                noise_mask = (labels == -1)
                if noise_mask.any():
                    labels[noise_mask] = torch.arange(
                        max_label + 1,
                        max_label + 1 + noise_mask.sum(),
                        device=labels.device
                    )

            return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            embeddings = model(batch)

            # Create target embeddings for contrastive learning
            # We'll use the true track IDs to create positive pairs
            track_ids = batch.y
            unique_tracks = torch.unique(track_ids)

            # Create positive pairs within each track
            pos_pairs = []
            for track in unique_tracks:
                if track == 0:  # Skip noise
                    continue
                mask = (track_ids == track)
                if mask.sum() >= 2:
                    indices = torch.where(mask)[0]
                    for i in range(len(indices)):
                        for j in range(i+1, len(indices)):
                            pos_pairs.append((indices[i], indices[j]))

            if len(pos_pairs) == 0:
                continue

            # Sample negative pairs
            neg_pairs = []
            for _ in range(len(pos_pairs)):
                i, j = torch.randint(0, len(track_ids), (2,))
                while track_ids[i] == track_ids[j] and track_ids[i] != 0:
                    i, j = torch.randint(0, len(track_ids), (2,))
                neg_pairs.append((i, j))

            # Compute contrastive loss
            pos_pairs = torch.tensor(pos_pairs, device=device)
            neg_pairs = torch.tensor(neg_pairs, device=device)

            pos_emb_i = embeddings[pos_pairs[:, 0]]
            pos_emb_j = embeddings[pos_pairs[:, 1]]
            neg_emb_i = embeddings[neg_pairs[:, 0]]
            neg_emb_j = embeddings[neg_pairs[:, 1]]

            pos_sim = F.cosine_similarity(pos_emb_i, pos_emb_j)
            neg_sim = F.cosine_similarity(neg_emb_i, neg_emb_j)

            loss = F.relu(neg_sim - pos_sim + 0.5).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embeddings = model(batch)

                # Simple validation metric: fraction of hits with correct nearest neighbor
                track_ids = batch.y
                unique_tracks = torch.unique(track_ids)

                correct = 0
                total = 0

                for track in unique_tracks:
                    if track == 0:
                        continue
                    mask = (track_ids == track)
                    if mask.sum() < 2:
                        continue

                    track_emb = embeddings[mask]
                    dist_matrix = torch.cdist(track_emb, track_emb)
                    np.fill_diagonal(dist_matrix.cpu().numpy(), np.inf)
                    min_indices = dist_matrix.argmin(dim=1)

                    # Check if nearest neighbor is from same track
                    correct += (track_ids[mask][min_indices] == track).sum().item()
                    total += len(min_indices)

                if total > 0:
                    val_correct += correct
                    val_samples += total

        train_loss = total_loss / len(train_loader)
        val_acc = val_correct / val_samples if val_samples > 0 else 0

        train_losses.append(train_loss)
        val_accs.append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model)

    return model, train_losses, [0]*len(train_losses), [0]*len(train_losses), val_accs

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

