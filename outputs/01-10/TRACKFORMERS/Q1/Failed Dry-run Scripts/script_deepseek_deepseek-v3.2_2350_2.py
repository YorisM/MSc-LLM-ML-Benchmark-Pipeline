
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
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import EdgeConv, global_mean_pool, GATConv, TransformerConv
from torch_geometric.utils import to_dense_batch
from torch_cluster import knn_graph, radius_graph
from torch_scatter import scatter_mean
import hdbscan

class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.knn_k = 10
        self.radius = 0.1

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 16,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 8}
        }

    def fit(self, Xs):
        all_features = []
        for X in Xs:
            if isinstance(X, torch.Tensor):
                X = X.numpy()
            # Feature engineering
            r = X[:, 0:1]
            theta = X[:, 1:2]
            z = X[:, 2:3]
            layer = X[:, 3:4]

            # Create new features
            x = r * np.cos(theta)  # [N,1]
            y = r * np.sin(theta)  # [N,1]
            r2 = r ** 2  # [N,1]
            abs_z = np.abs(z)  # [N,1]

            features = np.concatenate([x, y, z, layer, r2, abs_z], axis=1)  # [N,6]
            all_features.append(features)

        self.scaler.fit(np.vstack(all_features))
        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X

        r = X_np[:, 0:1]
        theta = X_np[:, 1:2]
        z = X_np[:, 2:3]
        layer = X_np[:, 3:4]

        # Same feature engineering as in fit
        x = r * np.cos(theta)  # [N,1]
        y = r * np.sin(theta)  # [N,1]
        r2 = r ** 2  # [N,1]
        abs_z = np.abs(z)  # [N,1]

        features = np.concatenate([x, y, z, layer, r2, abs_z], axis=1)  # [N,6]
        scaled = self.scaler.transform(features)  # [N,6]

        return torch.from_numpy(scaled).float()  # [N,6]

def make_preprocessor():
    return MyPreprocessor()

class GraphConstructionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index):
        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=1)  # [E, 2*input_dim]
        return self.mlp(edge_features)  # [E, output_dim]

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x=None):
        super().__init__()
        input_dim = 6
        hidden_dim = 128
        latent_dim = 64
        num_heads = 8

        # Initial feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Graph attention layers
        self.conv1 = TransformerConv(hidden_dim, hidden_dim, heads=num_heads, dropout=0.1)
        self.conv2 = TransformerConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=0.1)
        self.conv3 = TransformerConv(hidden_dim, latent_dim, heads=1, dropout=0.1)

        # Edge prediction head
        self.edge_pred = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1)
        )

        # Node embedding projection
        self.node_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim)
        )

        # Clustering projection
        self.cluster_proj = nn.Linear(latent_dim, 32)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input_proj(x)  # [N, hidden_dim]

        x = F.relu(self.conv1(x, edge_index))  # [N, hidden_dim*heads]
        x = F.relu(self.conv2(x, edge_index))  # [N, hidden_dim]
        node_embeddings = F.relu(self.conv3(x, edge_index))  # [N, latent_dim]

        # Edge scores
        src, dst = edge_index
        edge_features = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=1)  # [E, 2*latent_dim]
        edge_scores = self.edge_pred(edge_features).squeeze()  # [E]

        # Node features for clustering
        cluster_features = self.cluster_proj(node_embeddings)  # [N, 32]

        return {
            'node_emb': node_embeddings,
            'cluster_features': cluster_features,
            'edge_scores': edge_scores,
            'batch': batch
        }

    def predict_labels(self, batch_data):
        self.eval()
        with torch.no_grad():
            if isinstance(batch_data, Batch):
                out = self.forward(batch_data)
                cluster_features = out['cluster_features']
                batch = out['batch']

                # Process each event separately
                labels = []
                for i in range(batch.max().item() + 1):
                    mask = (batch == i)
                    event_features = cluster_features[mask].cpu().numpy()

                    if len(event_features) < 4:
                        labels.append(torch.full((len(event_features),), -1, dtype=torch.long))
                        continue

                    # HDBSCAN clustering
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=4,
                        min_samples=1,
                        metric='euclidean',
                        cluster_selection_method='eom',
                        prediction_data=True
                    )
                    cluster_labels = clusterer.fit_predict(event_features)

                    # Convert to torch tensor, -1 for noise
                    cluster_labels_t = torch.from_numpy(cluster_labels).long()
                    labels.append(cluster_labels_t)

                # Concatenate all labels
                return torch.cat(labels, dim=0)

            elif isinstance(batch_data, list):
                all_labels = []
                for data in batch_data:
                    out = self.forward(data)
                    features = out['cluster_features'].cpu().numpy()

                    if len(features) < 4:
                        all_labels.append(torch.full((len(features),), -1, dtype=torch.long))
                        continue

                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=4,
                        min_samples=1,
                        metric='euclidean',
                        cluster_selection_method='eom',
                        prediction_data=True
                    )
                    cluster_labels = clusterer.fit_predict(features)
                    all_labels.append(torch.from_numpy(cluster_labels).long())

                return all_labels
            else:
                raise ValueError("Unsupported batch type")

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

def compute_edge_labels(data, threshold=0.05):
    """Compute ground truth edge labels based on track membership and spatial proximity."""
    pos = data.x[:, :3]  # [N,3]
    track_ids = data.y  # [N]
    edge_index = data.edge_index  # [2,E]

    src, dst = edge_index
    edge_labels = torch.zeros(len(src), device=data.x.device)  # [E]

    # Edges between hits from same track
    same_track = (track_ids[src] == track_ids[dst]) & (track_ids[src] > 0)  # [E]
    edge_labels[same_track] = 1.0

    # Edges between hits from different tracks but spatially close
    pos_diff = torch.norm(pos[src] - pos[dst], dim=1)  # [E]
    close_edges = (pos_diff < threshold) & (~same_track)  # [E]
    edge_labels[close_edges] = 0.5

    return edge_labels

def build_graph_for_event(x, y=None, k=15, r=0.2):
    """Build graph for a single event."""
    device = x.device if isinstance(x, torch.Tensor) else torch.device('cpu')

    if isinstance(x, torch.Tensor):
        pos = x[:, :3]  # Use first 3 features as position
    else:
        pos = torch.tensor(x[:, :3], device=device)
        x = torch.tensor(x, device=device)

    # Build k-NN graph
    edge_index_knn = knn_graph(pos, k=k, loop=False)  # [2,E_knn]

    # Build radius graph
    edge_index_radius = radius_graph(pos, r=r, loop=False)  # [2,E_radius]

    # Combine graphs
    edge_index = torch.cat([edge_index_knn, edge_index_radius], dim=1)  # [2,E]
    edge_index = torch.unique(edge_index, dim=1)  # Remove duplicates

    # Create Data object
    if y is not None:
        if isinstance(y, torch.Tensor):
            y_tensor = y.long()
        else:
            y_tensor = torch.tensor(y, device=device).long()
    else:
        y_tensor = None

    return Data(x=x, edge_index=edge_index, y=y_tensor)

class PyGDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events = events
        self.pre = pre
        self.train = train
        self.graphs = []

        # Pre-build graphs for faster training
        print("Building graphs...")
        for evt in events:
            X, y = split_X_y(evt)
            X_processed = self.pre.transform(X)
            graph = build_graph_for_event(X_processed, y, k=15, r=0.2)
            self.graphs.append(graph)
        print(f"Built {len(self.graphs)} graphs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

def make_custom_dataset(events, pre, train=True, **kwargs):
    return PyGDataset(events, pre, train)

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)
            edge_scores = out['edge_scores']

            # Edge classification loss
            edge_labels = compute_edge_labels(batch, threshold=0.05)
            edge_loss = F.binary_cross_entropy_with_logits(edge_scores, edge_labels)

            # Node contrastive loss
            node_emb = out['node_emb']
            batch_idx = out['batch']

            # Normalize embeddings
            node_emb = F.normalize(node_emb, dim=1)

            # Simple contrastive loss (pull same track together, push different apart)
            pos = batch.x[:, :3]
            track_ids = batch.y

            # Create positive pairs (same track, nearby in space)
            with torch.no_grad():
                dist_matrix = torch.cdist(pos, pos)  # [N,N]
                same_track = (track_ids.unsqueeze(1) == track_ids.unsqueeze(0)) & (track_ids > 0).unsqueeze(1)
                close = dist_matrix < 0.1
                positive_mask = same_track & close

                # Negative pairs (different tracks)
                diff_track = (track_ids.unsqueeze(1) != track_ids.unsqueeze(0)) & (track_ids > 0).unsqueeze(1) & (track_ids > 0).unsqueeze(0)
                negative_mask = diff_track

            # Compute contrastive loss
            if positive_mask.any() and negative_mask.any():
                sim_matrix = torch.mm(node_emb, node_emb.t())  # [N,N]

                positive_sim = sim_matrix[positive_mask]
                negative_sim = sim_matrix[negative_mask]

                if len(positive_sim) > 0 and len(negative_sim) > 0:
                    pos_loss = -positive_sim.mean()
                    neg_loss = F.relu(negative_sim - 0.5).mean()
                    contrastive_loss = pos_loss + neg_loss
                else:
                    contrastive_loss = torch.tensor(0.0, device=device)
            else:
                contrastive_loss = torch.tensor(0.0, device=device)

            # Total loss
            loss = edge_loss + 0.5 * contrastive_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_total += 1

            # Compute edge accuracy
            pred_edges = (torch.sigmoid(edge_scores) > 0.5).float()
            edge_acc = (pred_edges == edge_labels).float().mean()
            train_correct += edge_acc.item()

        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                edge_scores = out['edge_scores']

                edge_labels = compute_edge_labels(batch, threshold=0.05)
                edge_loss = F.binary_cross_entropy_with_logits(edge_scores, edge_labels)

                val_loss += edge_loss.item()

                pred_edges = (torch.sigmoid(edge_scores) > 0.5).float()
                edge_acc = (pred_edges == edge_labels).float().mean()
                val_correct += edge_acc.item()
                val_total += 1

        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')

        scheduler.step()

        # Early stopping
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                model.load_state_dict(best_model_state)
                break

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

