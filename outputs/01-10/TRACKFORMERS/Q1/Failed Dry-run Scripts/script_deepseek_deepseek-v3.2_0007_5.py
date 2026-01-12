
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
from scipy.spatial.distance import cdist
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, EdgeConv, DynamicEdgeConv, global_mean_pool, graclus, max_pool
from torch_cluster import knn_graph, radius_graph
import warnings
warnings.filterwarnings('ignore')

class MyPreprocessor:
    def __init__(self):
        self.r_scaler = StandardScaler()
        self.theta_scaler = StandardScaler()
        self.z_scaler = StandardScaler()
        self.layer_scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:PyGDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {"follow_batch": []},
            "eval_overrides": {"shuffle": False, "batch_size": 16}
        }

    def fit(self, Xs):
        all_features = torch.cat([x for x in Xs], dim=0).numpy()
        self.r_scaler.fit(all_features[:, 0:1])
        self.theta_scaler.fit(all_features[:, 1:2])
        self.z_scaler.fit(all_features[:, 2:3])
        self.layer_scaler.fit(all_features[:, 3:4])
        return self

    def transform(self, X):
        X_np = X.numpy()
        r_norm = self.r_scaler.transform(X_np[:, 0:1])
        theta_norm = self.theta_scaler.transform(X_np[:, 1:2])
        z_norm = self.z_scaler.transform(X_np[:, 2:3])
        layer_norm = self.layer_scaler.transform(X_np[:, 3:4])

        features = np.hstack([r_norm, theta_norm, z_norm, layer_norm])
        return torch.from_numpy(features.astype(np.float32))

def make_preprocessor():
    return MyPreprocessor()

class PyGDataset(torch.utils.data.Dataset):
    def __init__(self, events, pre, train=True):
        self.events = events
        self.pre = pre
        self.train = train

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        X, y = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X

        pos = torch.stack([
            X[:, 0] * torch.cos(X[:, 1]),
            X[:, 0] * torch.sin(X[:, 1]),
            X[:, 2]
        ], dim=1)  # [N, 3] Cartesian coordinates

        # Edge construction: k-NN graph with multiple scales
        edge_index1 = knn_graph(pos, k=8, loop=False)
        edge_index2 = knn_graph(pos, k=16, loop=False)
        edge_index3 = radius_graph(pos, r=2.0, loop=False)
        edge_index = torch.cat([edge_index1, edge_index2, edge_index3], dim=1)
        edge_index = torch.unique(edge_index, dim=1)

        # Edge features: relative differences
        row, col = edge_index
        edge_attr = torch.cat([
            pos[row] - pos[col],
            torch.norm(pos[row] - pos[col], dim=1, keepdim=True),
            X[row, 3:4] - X[col, 3:4]
        ], dim=1)  # [E, 7]

        # Additional node features
        node_features = torch.cat([
            X,
            torch.sqrt(X[:, 0:1]),  # sqrt(r)
            torch.sin(X[:, 1:2]),   # sin(theta)
            torch.cos(X[:, 1:2]),   # cos(theta)
            X[:, 0:1] * X[:, 2:3],  # r*z
        ], dim=1)  # [N, 9]

        data = Data(
            x=node_features,        # [N, 9]
            edge_index=edge_index,  # [2, E]
            edge_attr=edge_attr,    # [E, 7]
            y=y,                     # [N]
            pos=pos                  # [N, 3]
        )
        return data

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__()
        self.attn = GATConv(in_channels, out_channels//heads, heads=heads, edge_dim=7)

    def forward(self, x, edge_index, edge_attr):
        return F.elu(self.attn(x, edge_index, edge_attr))

class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = EdgeConv(
            nn.Sequential(
                nn.Linear(2*in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ELU(),
                nn.Linear(out_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ELU()
            ), aggr='max'
        )

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Initial projection
        self.node_proj = nn.Sequential(
            nn.Linear(9, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ELU()
        )

        # EdgeConv layers
        self.edge_conv1 = EdgeConvBlock(128, 256)
        self.edge_conv2 = EdgeConvBlock(256, 256)

        # GAT layers
        self.gat1 = GraphAttentionLayer(256, 256)
        self.gat2 = GraphAttentionLayer(256, 256)

        # Decoder layers
        self.decoder1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.2)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.1)
        )

        # Final projection for clustering
        self.final_proj = nn.Linear(64, 32)

        # Edge prediction head for auxiliary loss
        self.edge_pred = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Node encoding
        x = self.node_proj(x)  # [N, 128]

        # EdgeConv processing
        x1 = self.edge_conv1(x, edge_index)  # [N, 256]
        x2 = self.edge_conv2(x1, edge_index)  # [N, 256]

        # GAT processing
        x3 = self.gat1(x2, edge_index, edge_attr)  # [N, 256]
        x4 = self.gat2(x3, edge_index, edge_attr)  # [N, 256]

        # Global context
        attn_weights = torch.softmax(self.attention_pool(x4), dim=0)
        global_context = (x4 * attn_weights).sum(dim=0, keepdim=True)
        global_context = global_context.expand(x4.size(0), -1)

        # Combine with skip connections
        x_combined = x4 + x2 + x  # [N, 256]

        # Decoder
        x_dec = self.decoder1(x_combined)  # [N, 128]
        x_dec = self.decoder2(x_dec)        # [N, 64]
        embeddings = self.final_proj(x_dec)  # [N, 32]

        # Edge predictions for auxiliary loss
        row, col = edge_index
        edge_feats = torch.cat([x_dec[row], x_dec[col]], dim=1)  # [E, 128]
        edge_pred = self.edge_pred(edge_feats)  # [E, 1]

        return embeddings, edge_pred.squeeze(-1)

    def predict_labels(self, data):
        self.eval()
        with torch.no_grad():
            embeddings, _ = self.forward(data)
            embeddings = embeddings.cpu().numpy()

            # DBSCAN clustering
            from sklearn.cluster import DBSCAN
            labels = -1 * np.ones(embeddings.shape[0], dtype=np.int32)

            # Cluster with adaptive parameters
            if embeddings.shape[0] > 10:
                clustering = DBSCAN(eps=0.5, min_samples=3, metric='euclidean')
                cluster_labels = clustering.fit_predict(embeddings)

                # Relabel to ensure -1 for noise
                unique = np.unique(cluster_labels)
                label_map = {old: new for new, old in enumerate(unique) if old != -1}
                for i in range(len(cluster_labels)):
                    if cluster_labels[i] in label_map:
                        labels[i] = label_map[cluster_labels[i]]
                    else:
                        labels[i] = -1

            # Post-processing: remove small clusters
            from collections import Counter
            counts = Counter(labels)
            for label, count in counts.items():
                if label != -1 and count < 4:
                    labels[labels == label] = -1

            # Relabel consecutively
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]
            if len(unique_labels) > 0:
                label_map = {old: new for new, old in enumerate(unique_labels)}
                for i in range(len(labels)):
                    if labels[i] != -1:
                        labels[i] = label_map[labels[i]]

            return torch.from_numpy(labels.astype(np.int64))

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_hits = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            embeddings, edge_pred = model(batch)

            # Create edge labels (1 if same track, 0 otherwise)
            row, col = batch.edge_index
            edge_labels = (batch.y[row] == batch.y[col]).float()
            edge_labels[batch.y[row] == 0] = 0  # Noise hits
            edge_labels[batch.y[col] == 0] = 0

            # Edge classification loss
            edge_loss = F.binary_cross_entropy_with_logits(edge_pred, edge_labels)

            # Contrastive loss for node embeddings
            pos_mask = edge_labels.bool()
            if pos_mask.sum() > 0:
                pos_pairs = embeddings[row[pos_mask]], embeddings[col[pos_mask]]
                pos_dist = F.pairwise_distance(pos_pairs[0], pos_pairs[1], p=2)
                pos_loss = pos_dist.mean()

                # Negative sampling
                neg_mask = ~pos_mask
                if neg_mask.sum() > 0 and pos_mask.sum() > 0:
                    neg_samples = min(neg_mask.sum(), pos_mask.sum())
                    neg_idx = torch.where(neg_mask)[0][:neg_samples]
                    pos_idx = torch.where(pos_mask)[0][:neg_samples]

                    neg_pairs = embeddings[row[neg_idx]], embeddings[col[neg_idx]]
                    neg_dist = F.pairwise_distance(neg_pairs[0], neg_pairs[1], p=2)

                    # Contrastive loss
                    margin = 1.0
                    contrastive_loss = torch.clamp(margin + pos_dist[:neg_samples] - neg_dist, min=0).mean()
                else:
                    contrastive_loss = torch.tensor(0.0, device=device)
            else:
                pos_loss = torch.tensor(0.0, device=device)
                contrastive_loss = torch.tensor(0.0, device=device)

            # Total loss
            loss = edge_loss + 0.5 * pos_loss + 0.1 * contrastive_loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track accuracy
            with torch.no_grad():
                pred_labels = model.predict_labels(batch.cpu())
                true_labels = batch.y.cpu()
                mask = true_labels != 0  # Ignore noise in training accuracy
                if mask.sum() > 0:
                    correct = (pred_labels[mask] == true_labels[mask]).sum().item()
                    total_train_correct += correct
                    total_train_hits += mask.sum().item()

            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_hits = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                embeddings, edge_pred = model(batch)

                # Edge loss for validation
                row, col = batch.edge_index
                edge_labels = (batch.y[row] == batch.y[col]).float()
                edge_labels[batch.y[row] == 0] = 0
                edge_labels[batch.y[col] == 0] = 0

                edge_loss = F.binary_cross_entropy_with_logits(edge_pred, edge_labels)
                total_val_loss += edge_loss.item()

                # Accuracy
                pred_labels = model.predict_labels(batch.cpu())
                true_labels = batch.y.cpu()
                mask = true_labels != 0
                if mask.sum() > 0:
                    correct = (pred_labels[mask] == true_labels[mask]).sum().item()
                    total_val_correct += correct
                    total_val_hits += mask.sum().item()

        # Metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_acc = total_train_correct / total_train_hits if total_train_hits > 0 else 0
        val_acc = total_val_correct / total_val_hits if total_val_hits > 0 else 0

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        scheduler.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Load best model
    model.load_state_dict(torch.load('best_model.pt', map_location=device))

    return model, train_losses, val_losses, train_accs, val_accs

EPOCHS = 50

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

