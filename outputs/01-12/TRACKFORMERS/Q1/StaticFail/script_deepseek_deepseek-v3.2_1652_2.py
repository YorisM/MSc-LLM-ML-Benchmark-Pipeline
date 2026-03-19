
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
from collections import defaultdict
import scipy.spatial.distance as distance

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

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
            "eval_overrides": {"shuffle": False, "batch_size": 32}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, F_raw]
        # Collect all hit features for standardization
        all_X = torch.cat([x for x in Xs], dim=0).numpy()  # [total_hits, 4]
        self.scaler.fit(all_X)
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, F_raw]
        X_np = X.numpy()
        X_scaled = self.scaler.transform(X_np)
        return torch.from_numpy(X_scaled).float()  # [N_hits, 4]

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
# Using LANE A: Torch ragged tensors
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Analyze input feature dimension from example batch
        first_event = example_batch_x[0]  # [N_hits, 4]
        in_features = first_event.shape[1]

        # Embedding network
        hidden_dim = 128
        self.embedding_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Output projection for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32)  # Final embedding dimension
        )

        # Attention pooling for global context
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, batch_x):
        # batch_x: list of tensors, each [N_i, F]
        embeddings = []
        for x in batch_x:
            emb = self.embedding_net(x)  # [N_i, hidden_dim]
            # Attention-weighted features
            weights = self.attention(emb)  # [N_i, 1]
            weighted_emb = emb * weights  # [N_i, hidden_dim]
            embeddings.append(weighted_emb)
        return embeddings

    def predict_labels(self, batch_x):
        # Get embeddings for each event
        embeddings_list = self.forward(batch_x)
        all_labels = []

        for emb in embeddings_list:
            # emb: [N_i, hidden_dim]
            if emb.shape[0] == 0:
                all_labels.append(torch.tensor([], dtype=torch.long))
                continue

            # Project to final embedding space
            final_emb = self.projection(emb)  # [N_i, 32]

            # Simple clustering based on cosine similarity
            # Normalize embeddings
            norms = torch.norm(final_emb, dim=1, keepdim=True) + 1e-8
            normalized_emb = final_emb / norms  # [N_i, 32]

            # Compute cosine similarity matrix
            sim_matrix = torch.mm(normalized_emb, normalized_emb.t())  # [N_i, N_i]

            # DBSCAN-like clustering using cosine similarity
            eps = 0.5
            min_samples = 4
            labels = self._dbscan_cosine(sim_matrix.numpy(), eps, min_samples)

            # Convert to torch tensor
            all_labels.append(torch.from_numpy(labels).long())

        return all_labels

    def _dbscan_cosine(self, sim_matrix, eps, min_samples):
        # Simple DBSCAN implementation using cosine similarity
        n_points = sim_matrix.shape[0]
        visited = np.zeros(n_points, dtype=bool)
        labels = -np.ones(n_points, dtype=int)
        cluster_id = 0

        for i in range(n_points):
            if visited[i]:
                continue
            visited[i] = True

            # Find neighbors
            neighbors = np.where(sim_matrix[i] > eps)[0]
            if len(neighbors) < min_samples:
                continue  # Noise point

            # Start new cluster
            labels[i] = cluster_id
            seed_set = set(neighbors)
            seed_set.discard(i)

            while seed_set:
                j = seed_set.pop()
                if not visited[j]:
                    visited[j] = True
                    j_neighbors = np.where(sim_matrix[j] > eps)[0]
                    if len(j_neighbors) >= min_samples:
                        seed_set.update(j_neighbors)
                if labels[j] == -1:
                    labels[j] = cluster_id

            cluster_id += 1

        return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss function - contrastive loss
    def contrastive_loss(embeddings, labels, margin=1.0):
        # embeddings: list of [N_i, D]
        # labels: list of [N_i]
        total_loss = 0.0
        total_pairs = 0

        for emb, lbl in zip(embeddings, labels):
            if emb.shape[0] < 2:
                continue

            # Only use non-noise points (label > 0)
            mask = lbl > 0
            if mask.sum() < 2:
                continue

            emb_pos = emb[mask]  # [M, D]
            lbl_pos = lbl[mask]  # [M]

            # Normalize embeddings
            norms = torch.norm(emb_pos, dim=1, keepdim=True) + 1e-8
            emb_norm = emb_pos / norms

            # Compute similarity matrix
            sim = torch.mm(emb_norm, emb_norm.t())  # [M, M]

            # Create positive mask (same track)
            pos_mask = (lbl_pos.unsqueeze(1) == lbl_pos.unsqueeze(0)).float()
            pos_mask.fill_diagonal_(0)  # Remove self

            # Create negative mask (different tracks)
            neg_mask = (lbl_pos.unsqueeze(1) != lbl_pos.unsqueeze(0)).float()

            # Positive pairs loss
            pos_loss = (1 - sim) * pos_mask
            pos_loss = pos_loss.sum() / (pos_mask.sum() + 1e-8)

            # Negative pairs loss
            neg_loss = F.relu(sim - margin) * neg_mask
            neg_loss = neg_loss.sum() / (neg_mask.sum() + 1e-8)

            total_loss += pos_loss + neg_loss
            total_pairs += 1

        return total_loss / max(total_pairs, 1)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        batch_count = 0

        for batch_idx, (Xs, ys) in enumerate(train_loader):
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            # Forward pass
            embeddings = model(Xs)

            # Compute loss
            loss = contrastive_loss(embeddings, ys)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Compute accuracy for this batch
            with torch.no_grad():
                pred_labels = model.predict_labels(Xs)
                batch_acc = 0.0
                for pred, true in zip(pred_labels, ys):
                    pred_cpu = pred.cpu().numpy()
                    true_cpu = true.cpu().numpy()
                    batch_acc += self._compute_fit_accuracy_single(pred_cpu, true_cpu)
                batch_acc /= len(pred_labels)

            epoch_train_loss += loss.item()
            epoch_train_acc += batch_acc
            batch_count += 1

            if batch_count % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}")

        avg_train_loss = epoch_train_loss / max(batch_count, 1)
        avg_train_acc = epoch_train_acc / max(batch_count, 1)
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                embeddings = model(Xs)
                loss = contrastive_loss(embeddings, ys)

                pred_labels = model.predict_labels(Xs)
                batch_acc = 0.0
                for pred, true in zip(pred_labels, ys):
                    pred_cpu = pred.cpu().numpy()
                    true_cpu = true.cpu().numpy()
                    batch_acc += self._compute_fit_accuracy_single(pred_cpu, true_cpu)
                batch_acc /= len(pred_labels)

                epoch_val_loss += loss.item()
                epoch_val_acc += batch_acc
                val_batch_count += 1

        avg_val_loss = epoch_val_loss / max(val_batch_count, 1)
        avg_val_acc = epoch_val_acc / max(val_batch_count, 1)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        # Update learning rate
        scheduler.step()

        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model_state = model.state_dict().copy()

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# Helper function to compute FitAccuracy for single event
def _compute_fit_accuracy_single(self, pred_labels, true_labels):
    """
    Compute FitAccuracy for a single event.
    pred_labels: [N] integer labels, -1 for noise
    true_labels: [N] integer track ids, 0 for noise
    """
    # Only consider non-noise true hits (track_id > 0)
    non_noise_mask = true_labels > 0
    if not non_noise_mask.any():
        return 0.0

    pred_labels = pred_labels[non_noise_mask]
    true_labels = true_labels[non_noise_mask]

    # Group predicted clusters
    pred_clusters = {}
    for i, (pred, true) in enumerate(zip(pred_labels, true_labels)):
        if pred == -1:
            continue
        if pred not in pred_clusters:
            pred_clusters[pred] = []
        pred_clusters[pred].append((i, true))

    # Filter clusters with < 4 hits
    valid_clusters = {}
    for cluster_id, hits in pred_clusters.items():
        if len(hits) >= 4:
            valid_clusters[cluster_id] = hits

    if not valid_clusters:
        return 0.0

    # Compute matching and correctness
    total_correct = 0
    matched_true_tracks = set()

    for cluster_id, hits in valid_clusters.items():
        # Count hits per true track
        true_counts = {}
        for _, true_id in hits:
            true_counts[true_id] = true_counts.get(true_id, 0) + 1

        # Find dominant true track
        dominant_true = max(true_counts.items(), key=lambda x: x[1])
        dominant_id, dominant_count = dominant_true

        # Check purity and coverage
        purity = dominant_count / len(hits)
        total_hits_in_true = np.sum(true_labels == dominant_id)
        coverage = dominant_count / total_hits_in_true

        if purity >= 0.5 and coverage >= 0.5:
            # Count correct hits
            total_correct += dominant_count
            matched_true_tracks.add(dominant_id)

    # Compute accuracy
    total_non_noise = np.sum(non_noise_mask)
    accuracy = total_correct / total_non_noise

    return accuracy

# Attach helper method to train_model function for convenience
train_model._compute_fit_accuracy_single = _compute_fit_accuracy_single

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

