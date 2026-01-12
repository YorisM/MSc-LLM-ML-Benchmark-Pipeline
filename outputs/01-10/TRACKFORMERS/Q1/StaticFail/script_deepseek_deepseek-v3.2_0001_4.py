
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

# ---------- IMPORTS ----------
# All imports are already in the prefix or added above

# -------- CUSTOM DATASET (PyG Graph) --------
class TrackGraphDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events = events
        self.pre = pre
        self.train = train

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        X, y = split_X_y(self.events[idx])

        # Apply preprocessing
        if self.pre is not None:
            X = self.pre.transform(X)

        # Convert to Cartesian coordinates for better geometry awareness
        x_vals = X[:, 0] * torch.cos(X[:, 1])  # r * cos(theta)
        y_vals = X[:, 0] * torch.sin(X[:, 1])  # r * sin(theta)
        z_vals = X[:, 2]
        layer_vals = X[:, 3]

        # Create combined features: Cartesian + cylindrical + layer
        features = torch.stack([x_vals, y_vals, z_vals, 
                               X[:, 0], X[:, 1], X[:, 2], 
                               layer_vals], dim=1)  # [N_hits, 7]

        # Build k-NN graph (k=20)
        k = min(20, len(features) - 1)
        if k > 0:
            pos_np = features[:, :3].cpu().numpy()
            tree = cKDTree(pos_np)
            distances, indices = tree.query(pos_np, k=k+1)

            # Create edge_index from k-NN
            rows = []
            cols = []
            for i in range(len(features)):
                for j in indices[i][1:]:  # Skip self
                    rows.append(i)
                    cols.append(j)

            edge_index = torch.tensor([rows, cols], dtype=torch.long)

            # Edge attributes: relative position and distance
            src_pos = features[edge_index[0], :3]
            dst_pos = features[edge_index[1], :3]
            edge_attr = torch.cat([
                dst_pos - src_pos,  # [E, 3]
                torch.norm(dst_pos - src_pos, dim=1, keepdim=True)  # [E, 1]
            ], dim=1)  # [E, 4]
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float32)

        # Create PyG Data object
        data = Data(
            x=features.float(),
            edge_index=edge_index,
            edge_attr=edge_attr.float(),
            y=y.long(),
            pos=features[:, :3].float()
        )

        return data

# ----------- PRE-PROCESSING -----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_scaler = StandardScaler()
        self.fitted = False

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:TrackGraphDataset",
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
        # Collect all hits for statistics
        all_hits = []
        for X in Xs:
            if isinstance(X, torch.Tensor):
                X = X.numpy()
            all_hits.append(X)

        all_hits = np.vstack(all_hits)

        # Fit scalers
        self.scaler.fit(all_hits[:, :3])  # r, theta, z
        self.layer_scaler.fit(all_hits[:, 3:4])  # layer_id

        self.fitted = True
        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X.copy()

        # Apply scaling
        if self.fitted:
            X_np[:, :3] = self.scaler.transform(X_np[:, :3])
            X_np[:, 3:4] = self.layer_scaler.transform(X_np[:, 3:4])

        return torch.from_numpy(X_np.astype(np.float32))

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, edge_dim=4)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=4)
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, edge_dim=4)
        self.bn1 = BatchNorm(hidden_channels * heads)
        self.bn2 = BatchNorm(hidden_channels * heads)

    def forward(self, x, edge_index, edge_attr, batch=None):
        # x: [N, in_channels], edge_index: [2, E], edge_attr: [E, 4]
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_attr)  # [N, out_channels]
        return x

class TrackClusteringModel(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Extract feature dimension from example
        if isinstance(example_batch_x, Batch):
            in_channels = example_batch_x.x.shape[1]
        else:
            in_channels = 7  # Our constructed features

        self.encoder = GATEncoder(in_channels, 128, 64)

        # Edge prediction head
        self.edge_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Node clustering head
        self.cluster_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

    def forward(self, data):
        # Encode nodes
        node_emb = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)  # [N, 64]

        # Predict edge scores
        src_emb = node_emb[data.edge_index[0]]  # [E, 64]
        dst_emb = node_emb[data.edge_index[1]]  # [E, 64]
        edge_features = torch.cat([src_emb, dst_emb], dim=1)  # [E, 128]
        edge_scores = self.edge_predictor(edge_features)  # [E, 1]

        # Get cluster embeddings
        cluster_emb = self.cluster_head(node_emb)  # [N, 16]

        return edge_scores.squeeze(), cluster_emb, node_emb

    def predict_labels(self, batch_x):
        self.eval()
        with torch.no_grad():
            if isinstance(batch_x, list):
                # Convert list of tensors to PyG Batch
                data_list = []
                for x in batch_x:
                    # Simple conversion - in practice should match dataset construction
                    pos = x[:, :3]
                    features = torch.cat([pos, x[:, 3:]], dim=1)
                    data = Data(x=features, pos=pos)
                    data_list.append(data)
                batch_x = Batch.from_data_list(data_list)

            # Get embeddings
            edge_scores, cluster_emb, _ = self(batch_x)

            # Use HDBSCAN-like clustering on embeddings per batch
            labels = []
            start_idx = 0

            for b in range(batch_x.batch.max().item() + 1):
                # Get nodes for this batch
                mask = batch_x.batch == b
                batch_emb = cluster_emb[mask].cpu().numpy()

                if len(batch_emb) > 0:
                    # Simple but effective clustering: DBSCAN in embedding space
                    from sklearn.cluster import DBSCAN
                    clustering = DBSCAN(eps=0.5, min_samples=4, metric='euclidean').fit(batch_emb)
                    batch_labels = clustering.labels_

                    # Convert -1 (noise) to -1, keep cluster ids
                    batch_labels = batch_labels.astype(int)

                    # Ensure clusters have at least 4 hits
                    unique_labels, counts = np.unique(batch_labels[batch_labels != -1], return_counts=True)
                    for label, count in zip(unique_labels, counts):
                        if count < 4:
                            batch_labels[batch_labels == label] = -1
                else:
                    batch_labels = np.array([], dtype=int)

                labels.append(torch.from_numpy(batch_labels).to(batch_x.x.device))
                start_idx += mask.sum().item()

            # Combine labels
            if len(labels) > 1:
                all_labels = torch.cat(labels, dim=0)
            else:
                all_labels = labels[0] if labels else torch.tensor([], device=batch_x.x.device)

            return all_labels

def make_model(example_batch_x):
    return TrackClusteringModel(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss weights
    edge_weight = 1.0
    contrastive_weight = 0.5

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        correct_hits = 0
        total_hits = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            edge_scores, cluster_emb, node_emb = model(batch)

            # Edge loss: supervised edge classification
            edge_labels = (batch.y[batch.edge_index[0]] == batch.y[batch.edge_index[1]]).float()
            edge_labels = edge_labels * (batch.y[batch.edge_index[0]] > 0).float()  # Ignore noise-to-noise edges
            edge_loss = F.binary_cross_entropy_with_logits(edge_scores, edge_labels)

            # Contrastive loss for node embeddings
            pos_mask = (batch.y.unsqueeze(0) == batch.y.unsqueeze(1)) & (batch.y > 0).unsqueeze(1)
            neg_mask = (batch.y.unsqueeze(0) != batch.y.unsqueeze(1)) & (batch.y > 0).unsqueeze(1)

            # Simple contrastive loss
            if pos_mask.any() and neg_mask.any():
                pos_pairs = node_emb[pos_mask]
                neg_pairs = node_emb[neg_mask]
                if len(pos_pairs) > 0 and len(neg_pairs) > 0:
                    pos_dist = torch.pdist(pos_pairs).mean() if len(pos_pairs) > 1 else torch.tensor(0.0, device=device)
                    neg_dist = torch.pdist(neg_pairs[:min(1000, len(neg_pairs))]).mean() if len(neg_pairs) > 1 else torch.tensor(1.0, device=device)
                    contrastive_loss = torch.relu(pos_dist - neg_dist + 0.5)
                else:
                    contrastive_loss = torch.tensor(0.0, device=device)
            else:
                contrastive_loss = torch.tensor(0.0, device=device)

            # Total loss
            loss = edge_weight * edge_loss + contrastive_weight * contrastive_loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

            # Calculate training accuracy (approximate)
            with torch.no_grad():
                pred_labels = model.predict_labels(batch)
                true_labels = batch.y.cpu().numpy()
                pred_labels_np = pred_labels.cpu().numpy()

                # Simple accuracy: same track ID assignment (permutation invariant)
                from sklearn.metrics import adjusted_rand_score
                if len(np.unique(true_labels)) > 1 and len(np.unique(pred_labels_np)) > 1:
                    acc = adjusted_rand_score(true_labels, pred_labels_np)
                    correct_hits += int(acc * len(true_labels))
                    total_hits += len(true_labels)

        # Validation
        model.eval()
        epoch_val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                edge_scores, cluster_emb, node_emb = model(batch)

                # Edge loss
                edge_labels = (batch.y[batch.edge_index[0]] == batch.y[batch.edge_index[1]]).float()
                edge_labels = edge_labels * (batch.y[batch.edge_index[0]] > 0).float()
                edge_loss = F.binary_cross_entropy_with_logits(edge_scores, edge_labels)
                loss = edge_loss

                epoch_val_loss += loss.item()

                # Validation accuracy
                pred_labels = model.predict_labels(batch)
                true_labels = batch.y.cpu().numpy()
                pred_labels_np = pred_labels.cpu().numpy()

                if len(np.unique(true_labels)) > 1 and len(np.unique(pred_labels_np)) > 1:
                    acc = adjusted_rand_score(true_labels, pred_labels_np)
                    val_correct += int(acc * len(true_labels))
                    val_total += len(true_labels)

        # Calculate metrics
        train_loss = epoch_train_loss / len(train_loader)
        val_loss = epoch_val_loss / len(val_loader)
        train_acc = correct_hits / total_hits if total_hits > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step()

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            # Load best model
            model.load_state_dict(best_model_state)
            break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

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

