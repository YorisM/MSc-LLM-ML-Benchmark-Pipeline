
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
        self.r_scale = 1000.0  # Scale r to be similar to z
        self.theta_scale = 1.0  # Theta is already in radians

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
        X = X.clone().detach().numpy() if torch.is_tensor(X) else X

        # Scale r, theta, z
        X[:, 0] = X[:, 0] * self.r_scale  # Scale r
        X[:, :3] = self.scaler.transform(X[:, :3])

        # Normalize layer_id
        X[:, 3] = (X[:, 3] - self.layer_mean) / (self.layer_std + 1e-8)

        return torch.from_numpy(X).float()

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
        if isinstance(example_batch_x, list):
            in_dim = example_batch_x[0].shape[1]
        else:
            in_dim = example_batch_x.shape[1]

        # EdgeConv layers
        self.conv1 = EdgeConv(in_dim, 64)
        self.conv2 = EdgeConv(64, 64)
        self.conv3 = EdgeConv(64, 64)

        # MLP for final classification
        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Track embedding head
        self.track_head = nn.Linear(64, 1)  # For track classification

        # Noise classification head
        self.noise_head = nn.Linear(64, 1)

        # Clustering parameters
        self.max_tracks = 50  # Maximum number of tracks per event
        self.threshold = 0.5  # Threshold for track assignment

    def build_kd_tree_edges(self, x, k=10):
        # x shape: [N_hits, 4]
        coords = x[:, :3].cpu().numpy()
        tree = KDTree(coords)
        distances, indices = tree.query(coords, k=k+1)  # +1 to include self

        # Create edge_index
        edge_index = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # Skip self
                edge_index.append([i, indices[i][j]])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return edge_index.to(x.device)

    def forward(self, batch_x):
        if isinstance(batch_x, list):
            # Handle ragged batch
            all_embeddings = []
            for x in batch_x:
                x = x.to(next(self.parameters()).device)
                edge_index = self.build_kd_tree_edges(x)

                # EdgeConv layers
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))

                # MLP
                x = self.mlp(x)
                all_embeddings.append(x)
            return all_embeddings
        else:
            # Single event
            edge_index = self.build_kd_tree_edges(batch_x)
            x = F.relu(self.conv1(batch_x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))
            x = self.mlp(x)
            return x

    def predict_labels(self, batch_x):
        if isinstance(batch_x, list):
            # Handle ragged batch
            all_labels = []
            for x in batch_x:
                x = x.to(next(self.parameters()).device)
                embeddings = self.forward(x)

                # Predict track scores
                track_scores = torch.sigmoid(self.track_head(embeddings)).squeeze()

                # Predict noise scores
                noise_scores = torch.sigmoid(self.noise_head(embeddings)).squeeze()

                # Combine scores
                combined_scores = track_scores - noise_scores

                # Simple clustering: assign to top tracks
                n_hits = x.shape[0]
                labels = torch.full((n_hits,), -1, dtype=torch.long, device=x.device)

                # Get top hits for each potential track
                for track_id in range(self.max_tracks):
                    if len(combined_scores) == 0:
                        break

                    # Find hit with highest score
                    max_idx = torch.argmax(combined_scores)
                    if combined_scores[max_idx] < self.threshold:
                        break

                    # Assign to track
                    labels[max_idx] = track_id

                    # Remove from consideration
                    combined_scores[max_idx] = -float('inf')

                    # Find nearby hits in embedding space
                    dists = torch.norm(embeddings - embeddings[max_idx], dim=1)
                    nearby = dists < 0.5  # Threshold in embedding space

                    # Assign nearby hits to same track if not already assigned
                    for idx in torch.where(nearby)[0]:
                        if labels[idx] == -1 and combined_scores[idx] > -1:
                            labels[idx] = track_id
                            combined_scores[idx] = -float('inf')

                all_labels.append(labels)
            return all_labels
        else:
            # Single event
            embeddings = self.forward(batch_x)
            track_scores = torch.sigmoid(self.track_head(embeddings)).squeeze()
            noise_scores = torch.sigmoid(self.noise_head(embeddings)).squeeze()
            combined_scores = track_scores - noise_scores

            n_hits = batch_x.shape[0]
            labels = torch.full((n_hits,), -1, dtype=torch.long, device=batch_x.device)

            for track_id in range(self.max_tracks):
                if len(combined_scores) == 0:
                    break

                max_idx = torch.argmax(combined_scores)
                if combined_scores[max_idx] < self.threshold:
                    break

                labels[max_idx] = track_id
                combined_scores[max_idx] = -float('inf')

                dists = torch.norm(embeddings - embeddings[max_idx], dim=1)
                nearby = dists < 0.5

                for idx in torch.where(nearby)[0]:
                    if labels[idx] == -1 and combined_scores[idx] > -1:
                        labels[idx] = track_id
                        combined_scores[idx] = -float('inf')

            return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def compute_loss(embeddings, true_labels, noise_mask):
    # embeddings: [N_hits, embedding_dim]
    # true_labels: [N_hits] (0 = noise)
    # noise_mask: [N_hits] (True for noise hits)

    # Create positive and negative pairs
    device = embeddings.device
    n_hits = embeddings.shape[0]

    # Get unique track IDs (excluding noise)
    track_ids = torch.unique(true_labels[~noise_mask])
    if len(track_ids) == 0:
        return torch.tensor(0.0, device=device)

    # Create track centers
    track_centers = []
    for track_id in track_ids:
        mask = (true_labels == track_id)
        center = embeddings[mask].mean(dim=0)
        track_centers.append(center)
    track_centers = torch.stack(track_centers)  # [N_tracks, embedding_dim]

    # Compute distances to track centers
    dists = torch.cdist(embeddings, track_centers)  # [N_hits, N_tracks]

    # Create target distances
    target_dists = torch.full((n_hits,), float('inf'), device=device)
    for i, track_id in enumerate(track_ids):
        mask = (true_labels == track_id)
        target_dists[mask] = dists[mask, i]

    # Noise hits should be far from all tracks
    target_dists[noise_mask] = 2.0  # Target distance for noise

    # Compute loss
    loss = F.mse_loss(dists, target_dists.unsqueeze(1).expand(-1, len(track_ids)))

    # Add noise classification loss
    noise_logits = torch.sigmoid(torch.randn(n_hits, device=device))  # Placeholder
    noise_loss = F.binary_cross_entropy_with_logits(
        noise_logits, noise_mask.float())

    return loss + 0.1 * noise_loss

def train_model(model, train_loader, val_loader, epochs):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (Xs, ys) in enumerate(train_loader):
            optimizer.zero_grad()

            # Move data to device
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            # Forward pass
            embeddings = model(Xs)

            # Compute loss for each event in batch
            batch_loss = 0.0
            for i in range(len(Xs)):
                noise_mask = (ys[i] == 0)
                loss = compute_loss(embeddings[i], ys[i], noise_mask)
                batch_loss += loss

            batch_loss /= len(Xs)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()

            # Simple accuracy monitoring (not used for optimization)
            with torch.no_grad():
                pred_labels = model.predict_labels(Xs)
                for i in range(len(Xs)):
                    pred = pred_labels[i]
                    true = ys[i]

                    # Simple accuracy: count hits assigned to correct track
                    correct = 0
                    for track_id in torch.unique(true[true > 0]):
                        true_mask = (true == track_id)
                        pred_mask = (pred == track_id)
                        if true_mask.sum() > 0 and pred_mask.sum() > 0:
                            intersection = (true_mask & pred_mask).sum()
                            union = (true_mask | pred_mask).sum()
                            if union > 0:
                                correct += intersection.item()

                    total_correct += correct
                    total_samples += true[true > 0].shape[0]

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                embeddings = model(Xs)
                batch_loss = 0.0

                for i in range(len(Xs)):
                    noise_mask = (ys[i] == 0)
                    loss = compute_loss(embeddings[i], ys[i], noise_mask)
                    batch_loss += loss

                    pred_labels = model.predict_labels([Xs[i]])[0]
                    true = ys[i]

                    correct = 0
                    for track_id in torch.unique(true[true > 0]):
                        true_mask = (true == track_id)
                        pred_mask = (pred_labels == track_id)
                        if true_mask.sum() > 0 and pred_mask.sum() > 0:
                            intersection = (true_mask & pred_mask).sum()
                            union = (true_mask | pred_mask).sum()
                            if union > 0:
                                correct += intersection.item()

                    val_correct += correct
                    val_samples += true[true > 0].shape[0]

                val_loss += batch_loss.item() / len(Xs)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_samples if val_samples > 0 else 0.0
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping and learning rate scheduling
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

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

