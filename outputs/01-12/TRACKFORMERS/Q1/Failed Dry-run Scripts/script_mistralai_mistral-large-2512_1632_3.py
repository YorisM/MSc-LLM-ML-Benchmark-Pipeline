
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
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from collections import defaultdict

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_means = None
        self.layer_stds = None

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
        # Compute global statistics for normalization
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Only scale r, theta, z

        # Compute per-layer statistics
        layer_ids = np.unique(all_X[:, 3])
        self.layer_means = {}
        self.layer_stds = {}

        for layer in layer_ids:
            layer_mask = all_X[:, 3] == layer
            layer_data = all_X[layer_mask, :3]
            if len(layer_data) > 0:
                self.layer_means[layer] = np.mean(layer_data, axis=0)
                self.layer_stds[layer] = np.std(layer_data, axis=0) + 1e-6

        return self

    def transform(self, X):
        # X shape: [N_hits, 4]
        X = X.clone().detach()

        # Normalize r, theta, z
        X[:, :3] = torch.from_numpy(self.scaler.transform(X[:, :3].numpy())).float()

        # Add layer-relative features
        layer_ids = X[:, 3].unique()
        for layer in layer_ids:
            layer_mask = X[:, 3] == layer
            if layer in self.layer_means:
                mean = torch.tensor(self.layer_means[layer], dtype=torch.float32)
                std = torch.tensor(self.layer_stds[layer], dtype=torch.float32)
                X[layer_mask, :3] = (X[layer_mask, :3] - mean) / std

        # Add cylindrical coordinate features
        r = X[:, 0]
        theta = X[:, 1]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        X = torch.cat([X, x.unsqueeze(1), y.unsqueeze(1)], dim=1)  # [N_hits, 6]

        return X.float()

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

        # Feature extraction
        self.node_encoder = nn.Sequential(
            nn.Linear(in_features, 64),
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

        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Will use sigmoid for clustering
        )

        # Track embedding
        self.track_embed = nn.Linear(64, 32)

        # Noise detection
        self.noise_detector = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def build_edges(self, x, batch_idx=None, k=16):
        # Build k-NN graph
        if batch_idx is None:
            batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        edge_indices = []
        for b in batch_idx.unique():
            mask = batch_idx == b
            x_batch = x[mask]

            # Use KDTree for efficient nearest neighbor search
            tree = KDTree(x_batch.cpu().numpy())
            _, indices = tree.query(x_batch.cpu().numpy(), k=k+1)  # +1 to include self

            # Convert to edge_index format
            src = torch.arange(mask.sum(), device=x.device).repeat(k)
            dst = torch.from_numpy(indices[:, 1:].flatten()).to(x.device)
            edge_indices.append(torch.stack([src, dst], dim=0))

        edge_index = torch.cat(edge_indices, dim=1)
        return edge_index

    def forward(self, Xs):
        # Handle ragged batch
        device = next(self.parameters()).device
        batch_sizes = [x.size(0) for x in Xs]
        batch_idx = torch.cat([torch.full((n,), i, device=device) for i, n in enumerate(batch_sizes)])

        # Concatenate all hits
        x = torch.cat(Xs, dim=0).to(device)
        batch_idx = batch_idx.to(device)

        # Build edges
        edge_index = self.build_edges(x, batch_idx, k=16)

        # Node features
        x = self.node_encoder(x)

        # Graph convolutions
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)

        # Combine features
        x = x1 + x2 + x3
        return x

    def predict_labels(self, Xs):
        device = next(self.parameters()).device
        embeddings = self.forward(Xs)

        # Predict track embeddings
        track_embeds = self.track_embed(embeddings)

        # Predict noise probability
        noise_probs = self.noise_detector(embeddings).squeeze()

        # Cluster hits
        labels = []
        for i, x in enumerate(Xs):
            mask = torch.ones(x.size(0), dtype=torch.bool, device=device)
            batch_embeds = track_embeds[mask]

            # Simple clustering - in practice would use more sophisticated method
            # Here we use a simple threshold-based approach for demonstration
            if batch_embeds.size(0) > 0:
                # Normalize embeddings
                batch_embeds = F.normalize(batch_embeds, p=2, dim=1)

                # Compute similarity matrix
                sim_matrix = torch.mm(batch_embeds, batch_embeds.t())

                # Threshold similarity for clustering
                cluster_matrix = sim_matrix > 0.7

                # Convert to labels
                n_hits = cluster_matrix.size(0)
                labels_i = torch.zeros(n_hits, dtype=torch.long, device=device)
                current_label = 1

                for j in range(n_hits):
                    if labels_i[j] == 0:
                        # Find all connected hits
                        connected = cluster_matrix[j].nonzero().squeeze()
                        if connected.dim() == 0:
                            connected = connected.unsqueeze(0)

                        # Assign label
                        labels_i[connected] = current_label
                        current_label += 1

                # Mark noise hits
                noise_mask = noise_probs[mask] > 0.5
                labels_i[noise_mask] = -1
            else:
                labels_i = torch.zeros(0, dtype=torch.long, device=device)

            labels.append(labels_i.cpu())

        return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_hits = 0

        for Xs, ys in train_loader:
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            optimizer.zero_grad()

            # Forward pass
            embeddings = model(Xs)

            # Compute loss - we'll use a simple contrastive loss
            loss = 0
            total_pairs = 0

            for i, (x, y) in enumerate(zip(Xs, ys)):
                mask = y > 0  # Ignore noise hits
                if mask.sum() < 2:
                    continue

                # Get embeddings for this event
                event_embeds = embeddings[total_hits:total_hits + x.size(0)]
                event_embeds = event_embeds[mask]
                event_labels = y[mask]

                # Compute pairwise distances
                dist_matrix = torch.cdist(event_embeds, event_embeds)

                # Create positive and negative masks
                same_track = event_labels.unsqueeze(0) == event_labels.unsqueeze(1)
                pos_mask = same_track.fill_diagonal_(False)
                neg_mask = ~same_track

                # Compute contrastive loss
                pos_dist = dist_matrix[pos_mask]
                neg_dist = dist_matrix[neg_mask]

                if pos_dist.numel() > 0 and neg_dist.numel() > 0:
                    pos_loss = torch.mean(F.relu(pos_dist - 0.1))
                    neg_loss = torch.mean(F.relu(0.5 - neg_dist))
                    loss += pos_loss + neg_loss
                    total_pairs += 1

                total_hits += x.size(0)

            if total_pairs > 0:
                loss = loss / total_pairs
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * total_pairs

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_pairs = 0

        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                embeddings = model(Xs)

                for i, (x, y) in enumerate(zip(Xs, ys)):
                    mask = y > 0
                    if mask.sum() < 2:
                        continue

                    event_embeds = embeddings[val_pairs:val_pairs + mask.sum()]
                    event_labels = y[mask]

                    dist_matrix = torch.cdist(event_embeds, event_embeds)
                    same_track = event_labels.unsqueeze(0) == event_labels.unsqueeze(1)
                    pos_mask = same_track.fill_diagonal_(False)
                    neg_mask = ~same_track

                    pos_dist = dist_matrix[pos_mask]
                    neg_dist = dist_matrix[neg_mask]

                    if pos_dist.numel() > 0 and neg_dist.numel() > 0:
                        pos_loss = torch.mean(F.relu(pos_dist - 0.1))
                        neg_loss = torch.mean(F.relu(0.5 - neg_dist))
                        val_loss += (pos_loss + neg_loss).item()
                        val_pairs += 1

                    val_pairs += 1

        # Calculate average losses
        avg_train_loss = total_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, val_pairs)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Simple accuracy estimation (not true FitAccuracy but for monitoring)
        train_acc = estimate_accuracy(model, train_loader)
        val_acc = estimate_accuracy(model, val_loader)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Load best model
    model.load_state_dict(best_model)

    return model, train_losses, val_losses, train_accs, val_accs

def estimate_accuracy(model, loader):
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0

    with torch.no_grad():
        for Xs, ys in loader:
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            pred_labels = model.predict_labels(Xs)

            for pred, true in zip(pred_labels, ys):
                # Simple accuracy - not true FitAccuracy but for monitoring
                pred = pred.to(device)
                true = true.to(device)

                # Only consider non-noise hits
                mask = true > 0
                if mask.sum() == 0:
                    continue

                pred = pred[mask]
                true = true[mask]

                # Count correct predictions (very rough estimate)
                correct += (pred == true).sum().item()
                total += true.size(0)

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

