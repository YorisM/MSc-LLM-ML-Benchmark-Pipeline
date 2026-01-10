
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import detect_and_assert_lane, assert_label_output_by_lane, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, build_trackformers_model

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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
import hdbscan

# ---------- IMPORTS ----------

#  -------- CUSTOM DATASET  --------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, events, pre, train=True, **kwargs):
        self.events = events
        self.pre = pre
        self.train = train

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        evt = self.events[idx]
        X = np.column_stack([
            evt["hit_r"].astype(np.float32),
            evt["hit_theta"].astype(np.float32),
            evt["hit_z"].astype(np.float32),
            evt["layer_id"].astype(np.float32)
        ])
        y = evt["track_id"].astype(np.int64)

        X = torch.FloatTensor(X)
        if self.pre is not None:
            X = self.pre.transform(X)

        # Create PyG Data object
        pos = X[:, :3].clone()  # xyz coordinates for geometric features
        features = X.clone()

        # Create kNN graph
        k = min(20, len(X) - 1)  # adaptive k
        if len(X) > 1:
            coords = pos.numpy()
            dists = cdist(coords, coords)
            knn_idx = np.argpartition(dists, k, axis=1)[:, :k+1]

            edge_index = []
            for i in range(len(X)):
                for j in knn_idx[i]:
                    if i != j:
                        edge_index.append([i, j])

            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data = Data(x=features, y=torch.LongTensor(y), pos=pos, 
                   edge_index=edge_index, num_nodes=len(X))
        return data

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.coord_scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 32}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, F_raw]
        all_coords = []
        all_features = []

        for X in Xs:
            if isinstance(X, torch.Tensor):
                X = X.numpy()

            # Convert cylindrical to Cartesian
            r = X[:, 0]
            theta = X[:, 1]
            z = X[:, 2]
            layer = X[:, 3]

            x = r * np.cos(theta)  # [N]
            y = r * np.sin(theta)  # [N]

            coords = np.column_stack([x, y, z])  # [N, 3]
            features = np.column_stack([x, y, z, layer, r, theta])  # [N, 6]

            all_coords.append(coords)
            all_features.append(features)

        # Fit scalers
        self.coord_scaler.fit(np.vstack(all_coords))
        self.scaler.fit(np.vstack(all_features))

        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, F_raw]
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X

        # Convert cylindrical to Cartesian
        r = X_np[:, 0]  # [N]
        theta = X_np[:, 1]  # [N]
        z = X_np[:, 2]  # [N]
        layer = X_np[:, 3]  # [N]

        x = r * np.cos(theta)  # [N]
        y = r * np.sin(theta)  # [N]

        coords = np.column_stack([x, y, z])  # [N, 3]
        features = np.column_stack([x, y, z, layer, r, theta])  # [N, 6]

        # Normalize
        coords_norm = self.coord_scaler.transform(coords)  # [N, 3]
        features_norm = self.scaler.transform(features)  # [N, 6]

        # Combine features
        combined = np.column_stack([coords_norm, features_norm])  # [N, 9]

        return torch.FloatTensor(combined)  # [N, 9]

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        input_dim = example_batch_x.x.size(1)  # 9
        hidden_dim = 256
        embedding_dim = 128

        # GNN layers
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, embedding_dim)

        # Batch norms
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(embedding_dim)

        # MLP for edge weights (optional)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Final embedding for clustering
        )

        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GNN forward pass
        x = F.relu(self.bn1(self.conv1(x, edge_index)))  # [N, hidden_dim]
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x, edge_index)))  # [N, hidden_dim]
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x, edge_index)))  # [N, hidden_dim]
        x = self.dropout(x)

        x = F.relu(self.bn4(self.conv4(x, edge_index)))  # [N, embedding_dim]

        # Project to clustering space
        embeddings = self.projection(x)  # [N, 32]

        return embeddings

    def predict_labels(self, batch_x):
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(batch_x)  # [N_total, 32]

            # Get batch indices
            if hasattr(batch_x, 'batch'):
                batch_indices = batch_x.batch.cpu().numpy()
            else:
                # Fallback: assume single batch
                batch_indices = np.zeros(len(embeddings))

            # Cluster per event
            all_labels = []
            for batch_idx in range(int(batch_indices.max()) + 1):
                mask = batch_indices == batch_idx
                event_emb = embeddings[mask].cpu().numpy()

                if len(event_emb) < 4:
                    # Too few points, mark all as noise
                    labels = np.full(len(event_emb), -1)
                else:
                    # HDBSCAN clustering
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=4,
                        min_samples=1,
                        metric='euclidean',
                        cluster_selection_method='leaf',
                        cluster_selection_epsilon=0.5
                    )
                    labels = clusterer.fit_predict(event_emb)

                    # Convert HDBSCAN noise (-1) to our noise label (-1)
                    # Already matches

                all_labels.append(labels)

            # Combine labels
            combined_labels = np.concatenate(all_labels)
            return torch.from_numpy(combined_labels.astype(np.int64))

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Contrastive loss (triplet loss)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        train_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Get embeddings
            embeddings = model(batch)  # [N_total, 32]

            # Create triplets for contrastive learning
            if len(embeddings) < 3:
                continue

            # Simple random triplet sampling
            batch_indices = batch.batch.cpu().numpy()
            y = batch.y.cpu().numpy()

            # Sample triplets within each event
            anchors = []
            positives = []
            negatives = []

            for batch_idx in range(int(batch_indices.max()) + 1):
                mask = batch_indices == batch_idx
                event_emb = embeddings[mask]
                event_y = y[mask]

                # Only use hits with valid track IDs (>0)
                valid_mask = event_y > 0
                if valid_mask.sum() < 2:
                    continue

                valid_emb = event_emb[valid_mask]
                valid_y = event_y[valid_mask]

                # Create triplets
                n_valid = len(valid_emb)
                if n_valid < 2:
                    continue

                # Sample anchors and positives from same track
                for _ in range(min(10, n_valid)):
                    # Random anchor
                    anchor_idx = torch.randint(0, n_valid, (1,)).item()
                    anchor_y = valid_y[anchor_idx]

                    # Find positives (same track)
                    pos_mask = valid_y == anchor_y
                    pos_indices = np.where(pos_mask)[0]
                    if len(pos_indices) > 1:
                        # Remove anchor from positives
                        pos_indices = pos_indices[pos_indices != anchor_idx]
                        if len(pos_indices) > 0:
                            pos_idx = np.random.choice(pos_indices)

                            # Find negatives (different track)
                            neg_mask = valid_y != anchor_y
                            neg_indices = np.where(neg_mask)[0]
                            if len(neg_indices) > 0:
                                neg_idx = np.random.choice(neg_indices)

                                anchors.append(valid_emb[anchor_idx])
                                positives.append(valid_emb[pos_idx])
                                negatives.append(valid_emb[neg_idx])

            if len(anchors) == 0:
                continue

            anchors = torch.stack(anchors)
            positives = torch.stack(positives)
            negatives = torch.stack(negatives)

            # Compute triplet loss
            loss = criterion(anchors, positives, negatives)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / max(train_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                embeddings = model(batch)

                if len(embeddings) < 3:
                    continue

                # Similar triplet sampling for validation
                batch_indices = batch.batch.cpu().numpy()
                y = batch.y.cpu().numpy()

                anchors = []
                positives = []
                negatives = []

                for batch_idx in range(int(batch_indices.max()) + 1):
                    mask = batch_indices == batch_idx
                    event_emb = embeddings[mask]
                    event_y = y[mask]

                    valid_mask = event_y > 0
                    if valid_mask.sum() < 2:
                        continue

                    valid_emb = event_emb[valid_mask]
                    valid_y = event_y[valid_mask]

                    n_valid = len(valid_emb)
                    if n_valid < 2:
                        continue

                    # Sample a few triplets
                    for _ in range(min(5, n_valid)):
                        anchor_idx = torch.randint(0, n_valid, (1,)).item()
                        anchor_y = valid_y[anchor_idx]

                        pos_mask = valid_y == anchor_y
                        pos_indices = np.where(pos_mask)[0]
                        if len(pos_indices) > 1:
                            pos_indices = pos_indices[pos_indices != anchor_idx]
                            if len(pos_indices) > 0:
                                pos_idx = np.random.choice(pos_indices)

                                neg_mask = valid_y != anchor_y
                                neg_indices = np.where(neg_mask)[0]
                                if len(neg_indices) > 0:
                                    neg_idx = np.random.choice(neg_indices)

                                    anchors.append(valid_emb[anchor_idx])
                                    positives.append(valid_emb[pos_idx])
                                    negatives.append(valid_emb[neg_idx])

                if len(anchors) == 0:
                    continue

                anchors = torch.stack(anchors)
                positives = torch.stack(positives)
                negatives = torch.stack(negatives)

                loss = criterion(anchors, positives, negatives)
                total_val_loss += loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)

        # Simple accuracy monitoring (track purity)
        train_acc = 0.0
        val_acc = 0.0

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step()

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Load best model
    if 'best_model_state' in locals():
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
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

