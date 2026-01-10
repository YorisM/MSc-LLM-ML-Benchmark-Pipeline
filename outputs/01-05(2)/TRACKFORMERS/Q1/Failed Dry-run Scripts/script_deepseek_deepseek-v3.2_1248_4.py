
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

# ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, EdgeConv
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import warnings
warnings.filterwarnings('ignore')

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.knn_k = 16

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomPyGDataset",
            "dataset_kwargs": {"preprocessor": self},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 32}
        }

    def fit(self, Xs):
        all_features = []
        for X in Xs[:1000]:  # Use subset for fitting
            xyz = self._cylindrical_to_cartesian(X)  # [N, 3]
            features = self._extract_features(X, xyz)  # [N, 9]
            all_features.append(features)
        self.scaler.fit(np.vstack(all_features))
        return self

    def transform(self, X):
        xyz = self._cylindrical_to_cartesian(X)  # [N, 3]
        features = self._extract_features(X, xyz)  # [N, 9]
        scaled = self.scaler.transform(features)  # [N, 9]
        return torch.FloatTensor(scaled)

    def _cylindrical_to_cartesian(self, X):
        r, theta, z = X[:, 0], X[:, 1], X[:, 2]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y, z], dim=1)  # [N, 3]

    def _extract_features(self, X, xyz):
        r, theta, z, layer = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        features = torch.cat([
            xyz,  # x, y, z [N, 3]
            X,    # r, theta, z, layer [N, 4]
            r.unsqueeze(1) * torch.cos(theta).unsqueeze(1),  # x alternative
            r.unsqueeze(1) * torch.sin(theta).unsqueeze(1),  # y alternative
            torch.sqrt(r**2 + z**2).unsqueeze(1)  # radius_3d [N, 1]
        ], dim=1)  # [N, 9]
        return features.numpy()

    def build_graph(self, X_scaled, xyz):
        X_np = X_scaled.numpy()
        xyz_np = xyz.numpy()

        # KNN graph in 3D space
        adj = kneighbors_graph(xyz_np, self.knn_k, mode='connectivity', include_self=True)
        edge_index = torch.LongTensor(np.array(adj.nonzero()))  # [2, E]

        # Add reverse edges for undirected graph
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return edge_index

def make_preprocessor():
    return MyPreprocessor()

# Custom PyG Dataset
class CustomPyGDataset(Dataset):
    def __init__(self, events, preprocessor, train=True):
        self.events = events
        self.pre = preprocessor
        self.train = train

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        X_raw, y = self.events[idx]  # X_raw: [N, 4], y: [N]
        xyz = self.pre._cylindrical_to_cartesian(X_raw)  # [N, 3]
        X_scaled = self.pre.transform(X_raw)  # [N, 9]
        edge_index = self.pre.build_graph(X_scaled, xyz)  # [2, E]

        # Create PyG Data object
        return Data(
            x=X_scaled,  # [N, 9]
            edge_index=edge_index,  # [2, E]
            y=y,  # [N]
            pos=xyz  # [N, 3]
        )

# ---------- MODEL ARCHITECTURE ----------
class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = EdgeConv(
            nn.Sequential(
                nn.Linear(2*in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ), aggr='max')

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Extract feature dimension from example batch
        if isinstance(example_batch_x, Batch):
            in_channels = example_batch_x.x.shape[1]
        else:
            in_channels = 9  # From preprocessor

        # Graph convolution layers
        self.conv1 = EdgeConvBlock(in_channels, 128)
        self.conv2 = EdgeConvBlock(128, 256)
        self.conv3 = EdgeConvBlock(256, 512)

        # Global pooling
        self.pool = global_mean_pool

        # Per-node classifiers
        self.node_mlp = nn.Sequential(
            nn.Linear(512 + 128, 256),  # Combine local and global
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Prototype-based classification
        self.prototype_layer = nn.Linear(64, 32, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Local features
        x1 = self.conv1(x, edge_index)  # [N, 128]
        x2 = self.conv2(x1, edge_index)  # [N, 256]
        x3 = self.conv3(x2, edge_index)  # [N, 512]

        # Global context
        global_feat = self.pool(x3, batch)  # [B, 512]
        global_expanded = global_feat[batch]  # [N, 512]

        # Combine
        combined = torch.cat([x3, global_expanded], dim=1)  # [N, 512+512]
        node_features = self.node_mlp(combined)  # [N, 64]

        # Prototype similarity
        prototypes = self.prototype_layer.weight  # [32, 64]
        similarities = F.cosine_similarity(
            node_features.unsqueeze(1),  # [N, 1, 64]
            prototypes.unsqueeze(0),     # [1, 32, 64]
            dim=2
        )  # [N, 32]

        logits = similarities / (self.temperature + 1e-8)
        return logits, node_features

    def predict_labels(self, batch_x):
        self.eval()
        with torch.no_grad():
            if isinstance(batch_x, list):
                # Handle ragged batch case (should not happen in PyG lane)
                batch_x = Batch.from_data_list(batch_x)

            logits, embeddings = self.forward(batch_x)

            # DBSCAN clustering on embeddings
            from sklearn.cluster import DBSCAN
            eps = 0.5
            min_samples = 4

            emb_np = embeddings.cpu().numpy()
            batch_indices = batch_x.batch.cpu().numpy()

            all_labels = []
            for batch_idx in range(batch_x.num_graphs):
                mask = batch_indices == batch_idx
                emb_batch = emb_np[mask]

                if len(emb_batch) < min_samples:
                    labels = -np.ones(len(emb_batch), dtype=int)
                else:
                    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                    labels = clustering.fit_predict(emb_batch)
                    # Ensure noise is -1
                    labels[labels == -1] = -1

                all_labels.append(labels)

            # Flatten and return as tensor
            flat_labels = np.concatenate(all_labels)
            return torch.from_numpy(flat_labels).to(batch_x.x.device)

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    # Loss functions
    def contrastive_loss(embeddings, batch_idx, margin=1.0):
        """Pull same-track hits together, push different-track apart"""
        batch_size = embeddings.shape[0]
        similarities = torch.cdist(embeddings, embeddings, p=2)  # [N, N]

        # Create mask for same track (excluding noise)
        same_track = (batch_idx.unsqueeze(1) == batch_idx.unsqueeze(0)) & \
                    (batch_idx.unsqueeze(1) > 0) & (batch_idx.unsqueeze(0) > 0)

        # Create mask for different tracks
        diff_track = (batch_idx.unsqueeze(1) != batch_idx.unsqueeze(0)) & \
                    (batch_idx.unsqueeze(1) > 0) & (batch_idx.unsqueeze(0) > 0)

        pos_loss = (similarities * same_track.float()).sum() / (same_track.sum() + 1e-8)
        neg_loss = F.relu(margin - similarities * diff_track.float()).sum() / (diff_track.sum() + 1e-8)

        return pos_loss + 0.5 * neg_loss

    def classification_loss(logits, y, valid_mask):
        """Cross-entropy for non-noise hits"""
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Dynamic class assignment: assign each track unique label within batch
        unique_tracks = torch.unique(y[valid_mask])
        track_to_class = {track.item(): i for i, track in enumerate(unique_tracks)}

        class_labels = torch.zeros_like(y)
        for track, cls in track_to_class.items():
            class_labels[y == track] = cls

        valid_logits = logits[valid_mask]
        valid_labels = class_labels[valid_mask]

        return F.cross_entropy(valid_logits, valid_labels.long())

    best_val_acc = 0
    patience_counter = 0
    patience = 10

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits, embeddings = model(batch)

            # Valid hits (non-noise)
            valid_mask = batch.y > 0

            # Combined loss
            loss_cls = classification_loss(logits, batch.y, valid_mask)
            loss_cont = contrastive_loss(embeddings, batch.y)
            loss = loss_cls + 0.1 * loss_cont

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

            # Training accuracy (track matching)
            with torch.no_grad():
                preds = model.predict_labels(batch)
                for i in range(batch.num_graphs):
                    mask = batch.batch == i
                    y_batch = batch.y[mask]
                    p_batch = preds[mask]

                    # Simple accuracy: require same number of unique tracks
                    if torch.unique(y_batch[y_batch > 0]).shape[0] == torch.unique(p_batch[p_batch >= 0]).shape[0]:
                        correct_train += 1
                    total_train += 1

        # Validation
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, embeddings = model(batch)

                valid_mask = batch.y > 0
                loss_cls = classification_loss(logits, batch.y, valid_mask)
                loss_cont = contrastive_loss(embeddings, batch.y)
                loss = loss_cls + 0.1 * loss_cont

                epoch_val_loss += loss.item()

                # Validation accuracy
                preds = model.predict_labels(batch)
                for i in range(batch.num_graphs):
                    mask = batch.batch == i
                    y_batch = batch.y[mask]
                    p_batch = preds[mask]

                    if torch.unique(y_batch[y_batch > 0]).shape[0] == torch.unique(p_batch[p_batch >= 0]).shape[0]:
                        correct_val += 1
                    total_val += 1

        # Metrics
        train_loss = epoch_train_loss / len(train_loader)
        val_loss = epoch_val_loss / len(val_loader)
        train_acc = correct_train / total_train if total_train > 0 else 0
        val_acc = correct_val / total_val if total_val > 0 else 0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

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

