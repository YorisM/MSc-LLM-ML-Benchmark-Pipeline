
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
        self.layer_stats = None
        self.r_stats = None
        self.z_stats = None
        self.theta_stats = None

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
        # Collect all hits for statistics
        all_X = np.concatenate(Xs, axis=0)

        # Fit scaler on all features except layer_id
        self.scaler.fit(all_X[:, :3])

        # Compute layer-wise statistics
        layer_ids = all_X[:, 3]
        unique_layers = np.unique(layer_ids)
        self.layer_stats = {}
        for layer in unique_layers:
            mask = layer_ids == layer
            self.layer_stats[layer] = {
                'r_mean': np.mean(all_X[mask, 0]),
                'r_std': np.std(all_X[mask, 0]),
                'z_mean': np.mean(all_X[mask, 2]),
                'z_std': np.std(all_X[mask, 2])
            }

        # Compute global statistics
        self.r_stats = {'mean': np.mean(all_X[:, 0]), 'std': np.std(all_X[:, 0])}
        self.z_stats = {'mean': np.mean(all_X[:, 2]), 'std': np.std(all_X[:, 2])}
        self.theta_stats = {'mean': np.mean(all_X[:, 1]), 'std': np.std(all_X[:, 1])}

        return self

    def transform(self, X):
        # X shape: [N_hits, 4]
        X = X.clone().detach()

        # Normalize r, theta, z features
        X[:, :3] = torch.from_numpy(self.scaler.transform(X[:, :3].numpy())).float()

        # Add layer-wise normalized features
        layer_ids = X[:, 3].long()
        for i, layer in enumerate(layer_ids):
            if layer.item() in self.layer_stats:
                stats = self.layer_stats[layer.item()]
                X[i, 0] = (X[i, 0] - stats['r_mean']) / (stats['r_std'] + 1e-8)
                X[i, 2] = (X[i, 2] - stats['z_mean']) / (stats['z_std'] + 1e-8)

        # Add relative features
        r = X[:, 0]
        theta = X[:, 1]
        z = X[:, 2]

        # Add cylindrical coordinates differences
        X = torch.cat([
            X,
            torch.sin(theta).unsqueeze(1),
            torch.cos(theta).unsqueeze(1),
            (r - self.r_stats['mean']) / (self.r_stats['std'] + 1e-8).unsqueeze(1),
            (z - self.z_stats['mean']) / (self.z_stats['std'] + 1e-8).unsqueeze(1)
        ], dim=1)  # [N_hits, 9]

        return X

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

        # Determine input features from example batch
        if isinstance(example_batch_x, list):
            in_features = example_batch_x[0].shape[1]
        else:
            in_features = example_batch_x.shape[1]

        # Feature embedding
        self.embedding = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
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
        self.output = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Track classification head
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Noise classification head
        self.noise_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def build_edges(self, x):
        # Build edges based on spatial proximity
        pos = x[:, :3]  # Use r, theta, z for edge building
        kdtree = KDTree(pos.cpu().numpy())
        edge_list = []

        # Find 10 nearest neighbors for each hit
        for i in range(pos.shape[0]):
            _, indices = kdtree.query(pos[i].cpu().numpy(), k=11)  # 10 neighbors + self
            for j in indices[1:]:  # Skip self
                edge_list.append([i, j])

        edge_index = torch.tensor(edge_list, dtype=torch.long, device=x.device).t()
        return edge_index

    def forward(self, batch_x):
        if isinstance(batch_x, list):
            # Handle ragged batch
            all_embeddings = []
            for x in batch_x:
                x = x.to(next(self.parameters()).device)
                edge_index = self.build_edges(x)

                # Embed features
                h = self.embedding(x)  # [N_hits, 64]

                # Graph convolutions
                h1 = self.conv1(h, edge_index)
                h2 = self.conv2(h1, edge_index)
                h3 = self.conv3(h2, edge_index)

                # Attention
                attn = self.attention(h3)
                h = h3 * attn

                all_embeddings.append(h)
            return all_embeddings
        else:
            # Handle single event
            edge_index = self.build_edges(batch_x)
            h = self.embedding(batch_x)
            h1 = self.conv1(h, edge_index)
            h2 = self.conv2(h1, edge_index)
            h3 = self.conv3(h2, edge_index)
            attn = self.attention(h3)
            h = h3 * attn
            return h

    def predict_labels(self, batch_x):
        if isinstance(batch_x, list):
            # Handle ragged batch
            all_labels = []
            for x in batch_x:
                x = x.to(next(self.parameters()).device)
                embeddings = self.forward(x)

                # Predict track scores
                track_scores = self.classifier(embeddings).squeeze(-1)  # [N_hits]

                # Predict noise probabilities
                noise_probs = self.noise_classifier(embeddings).squeeze(-1)  # [N_hits]

                # Combine scores
                scores = track_scores * (1 - noise_probs)

                # Simple clustering: assign labels based on score thresholds
                # This is a simplified approach - in practice you'd use a proper clustering algorithm
                labels = torch.zeros(x.shape[0], dtype=torch.long, device=x.device) - 1  # -1 for noise

                # Threshold for noise
                noise_mask = noise_probs > 0.5
                labels[noise_mask] = -1

                # For non-noise hits, assign track IDs based on score similarity
                non_noise_mask = ~noise_mask
                if non_noise_mask.sum() > 0:
                    non_noise_scores = scores[non_noise_mask]
                    non_noise_embeddings = embeddings[non_noise_mask]

                    # Simple clustering: assign same label to hits with similar scores
                    # This is a placeholder - replace with proper clustering
                    threshold = 0.1
                    current_label = 1
                    assigned = torch.zeros(non_noise_scores.shape[0], dtype=torch.bool, device=x.device)

                    while not assigned.all():
                        # Find highest score unassigned hit
                        max_idx = torch.argmax(non_noise_scores * (~assigned).float())
                        if assigned[max_idx]:
                            break

                        # Assign same label to hits with similar scores
                        similar_mask = torch.abs(non_noise_scores - non_noise_scores[max_idx]) < threshold
                        labels[non_noise_mask.nonzero().squeeze(1)[similar_mask]] = current_label
                        assigned[similar_mask] = True
                        current_label += 1

                all_labels.append(labels)
            return all_labels
        else:
            # Handle single event
            embeddings = self.forward(batch_x)
            track_scores = self.classifier(embeddings).squeeze(-1)
            noise_probs = self.noise_classifier(embeddings).squeeze(-1)
            scores = track_scores * (1 - noise_probs)

            labels = torch.zeros(batch_x.shape[0], dtype=torch.long, device=batch_x.device) - 1
            noise_mask = noise_probs > 0.5
            labels[noise_mask] = -1

            non_noise_mask = ~noise_mask
            if non_noise_mask.sum() > 0:
                non_noise_scores = scores[non_noise_mask]
                threshold = 0.1
                current_label = 1
                assigned = torch.zeros(non_noise_scores.shape[0], dtype=torch.bool, device=batch_x.device)

                while not assigned.all():
                    max_idx = torch.argmax(non_noise_scores * (~assigned).float())
                    if assigned[max_idx]:
                        break

                    similar_mask = torch.abs(non_noise_scores - non_noise_scores[max_idx]) < threshold
                    labels[non_noise_mask.nonzero().squeeze(1)[similar_mask]] = current_label
                    assigned[similar_mask] = True
                    current_label += 1

            return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_hits = 0

        for batch in train_loader:
            Xs, ys = batch
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            optimizer.zero_grad()

            # Forward pass
            embeddings = model(Xs)

            # Compute loss for each event in batch
            batch_loss = 0
            batch_hits = 0

            for i, (x, y) in enumerate(zip(Xs, ys)):
                # Create target for noise classification
                noise_target = (y == 0).float().unsqueeze(1)

                # Get embeddings for this event
                h = embeddings[i]

                # Noise classification loss
                noise_logits = model.noise_classifier(h)
                loss = criterion(noise_logits, noise_target)

                # Track classification loss (simplified)
                # We'll use a contrastive loss approach
                track_mask = (y > 0)
                if track_mask.sum() > 1:
                    # Get embeddings for track hits
                    track_embeddings = h[track_mask]
                    track_ids = y[track_mask]

                    # Create pairs of hits from same track
                    unique_tracks = torch.unique(track_ids)
                    for track in unique_tracks:
                        mask = (track_ids == track)
                        if mask.sum() > 1:
                            # Positive pairs
                            pos_pairs = track_embeddings[mask]
                            pos_dist = F.pairwise_distance(pos_pairs.unsqueeze(1), pos_pairs.unsqueeze(0))
                            pos_loss = pos_dist.mean()

                            # Negative pairs (simplified)
                            neg_mask = (track_ids != track)
                            if neg_mask.sum() > 0:
                                neg_embeddings = track_embeddings[neg_mask]
                                neg_dist = F.pairwise_distance(pos_pairs.unsqueeze(1), neg_embeddings.unsqueeze(0))
                                neg_loss = F.relu(1.0 - neg_dist).mean()
                                loss += 0.1 * (pos_loss + neg_loss)

                batch_loss += loss * x.shape[0]
                batch_hits += x.shape[0]

            batch_loss = batch_loss / batch_hits
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item() * batch_hits
            total_hits += batch_hits

        train_loss = total_loss / total_hits
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_hits = 0

        with torch.no_grad():
            for batch in val_loader:
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                embeddings = model(Xs)

                for i, (x, y) in enumerate(zip(Xs, ys)):
                    noise_target = (y == 0).float().unsqueeze(1)
                    h = embeddings[i]
                    noise_logits = model.noise_classifier(h)
                    loss = criterion(noise_logits, noise_target)

                    track_mask = (y > 0)
                    if track_mask.sum() > 1:
                        track_embeddings = h[track_mask]
                        track_ids = y[track_mask]
                        unique_tracks = torch.unique(track_ids)

                        for track in unique_tracks:
                            mask = (track_ids == track)
                            if mask.sum() > 1:
                                pos_pairs = track_embeddings[mask]
                                pos_dist = F.pairwise_distance(pos_pairs.unsqueeze(1), pos_pairs.unsqueeze(0))
                                pos_loss = pos_dist.mean()

                                neg_mask = (track_ids != track)
                                if neg_mask.sum() > 0:
                                    neg_embeddings = track_embeddings[neg_mask]
                                    neg_dist = F.pairwise_distance(pos_pairs.unsqueeze(1), neg_embeddings.unsqueeze(0))
                                    neg_loss = F.relu(1.0 - neg_dist).mean()
                                    loss += 0.1 * (pos_loss + neg_loss)

                    val_loss += loss * x.shape[0]
                    val_hits += x.shape[0]

        val_loss = val_loss / val_hits
        val_losses.append(val_loss)

        # Simple accuracy monitoring (not the actual FitAccuracy)
        train_acc = compute_simple_accuracy(model, train_loader, device)
        val_acc = compute_simple_accuracy(model, val_loader, device)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    return model, train_losses, val_losses, train_accs, val_accs

def compute_simple_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            Xs, ys = batch
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            preds = model.predict_labels(Xs)

            for pred, y in zip(preds, ys):
                # Simple accuracy: count hits where noise prediction matches
                noise_mask = (y == 0)
                pred_noise = (pred == -1)
                correct += (noise_mask == pred_noise).sum().item()
                total += y.shape[0]

    return correct / total if total > 0 else 0

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

