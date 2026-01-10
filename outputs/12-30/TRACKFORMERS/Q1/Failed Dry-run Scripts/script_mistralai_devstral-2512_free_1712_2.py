
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import detect_and_assert_lane, assert_label_output_by_lane, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

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
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import hdbscan
from scipy.spatial import cKDTree
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_means = None
        self.layer_stds = None
        self.global_mean = None
        self.global_std = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Collect all hits for global statistics
        all_hits = np.concatenate(Xs, axis=0)

        # Compute global normalization
        self.global_mean = all_hits.mean(axis=0)
        self.global_std = all_hits.std(axis=0)
        self.global_std[self.global_std == 0] = 1.0  # Avoid division by zero

        # Layer-wise normalization
        layer_ids = all_hits[:, 3]
        unique_layers = np.unique(layer_ids)
        self.layer_means = {}
        self.layer_stds = {}

        for layer in unique_layers:
            mask = (layer_ids == layer)
            layer_hits = all_hits[mask]
            self.layer_means[layer] = layer_hits.mean(axis=0)
            self.layer_stds[layer] = layer_hits.std(axis=0)
            self.layer_stds[layer][self.layer_stds[layer] == 0] = 1.0

        return self

    def transform(self, X):
        # Convert to numpy if tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        # Apply global normalization first
        X_normalized = (X - self.global_mean) / self.global_std

        # Apply layer-wise normalization to spatial features
        layer_ids = X[:, 3]
        X_transformed = X_normalized.copy()

        for i, layer in enumerate(layer_ids):
            if layer in self.layer_means:
                mean = self.layer_means[layer][:3]  # Only spatial features
                std = self.layer_stds[layer][:3]
                X_transformed[i, :3] = (X_transformed[i, :3] - mean) / std

        return torch.from_numpy(X_transformed).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()

        # Determine input features
        if isinstance(example_batch_x, list):
            # Torch ragged lane
            self.lane = "torch_ragged_xy"
            example_x = example_batch_x[0]
        else:
            # PyG lane
            self.lane = "pyg_batch"
            example_x = example_batch_x.x

        input_dim = example_x.shape[1]

        # Graph-based architecture
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.conv4 = GCNConv(256, 128)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Output head
        self.output = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Clustering parameters
        self.min_cluster_size = 4
        self.min_samples = 1
        self.cluster_selection_epsilon = 0.1

    def build_graph(self, X, batch_idx=None):
        if self.lane == "torch_ragged_xy":
            # For ragged lane, we need to process each event separately
            graphs = []
            for x in X:
                # Convert to PyG Data format
                num_nodes = x.shape[0]

                # Create edges based on spatial proximity
                pos = x[:, :3]  # r, theta, z
                tree = cKDTree(pos.cpu().numpy())
                edges = tree.query_pairs(r=0.5)  # Adjust radius as needed

                if len(edges) > 0:
                    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
                    edge_index = edge_index.to(x.device)
                else:
                    # Create self-loops if no edges found
                    edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0).to(x.device)

                data = Data(x=x, edge_index=edge_index)
                graphs.append(data)
            return graphs
        else:
            # For PyG lane, batch is already a graph
            return X

    def forward(self, batch_x):
        if self.lane == "torch_ragged_xy":
            graphs = self.build_graph(batch_x)
            outputs = []
            for data in graphs:
                x, edge_index = data.x, data.edge_index

                # Graph convolutions
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = F.relu(self.conv4(x, edge_index))

                # Attention mechanism
                attention_weights = self.attention(x)
                x = x * attention_weights

                # Global pooling for graph-level features
                graph_feat = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))

                # Expand graph features to node level
                graph_feat = graph_feat.expand(x.size(0), -1)

                # Combine node and graph features
                x = torch.cat([x, graph_feat], dim=1)

                # Output
                logits = self.output(x).squeeze(-1)

                # Clustering
                with torch.no_grad():
                    # Convert to numpy for clustering
                    features = x.cpu().numpy()

                    # Use HDBSCAN for clustering
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=self.min_cluster_size,
                        min_samples=self.min_samples,
                        cluster_selection_epsilon=self.cluster_selection_epsilon,
                        metric='euclidean',
                        gen_min_span_tree=True
                    )
                    cluster_labels = clusterer.fit_predict(features)

                    # Convert to tensor and handle noise
                    cluster_labels = torch.from_numpy(cluster_labels).to(x.device)
                    cluster_labels[cluster_labels == -1] = -1  # Noise label

                    # Make labels contiguous starting from 0
                    unique_labels = torch.unique(cluster_labels[cluster_labels != -1])
                    if len(unique_labels) > 0:
                        label_mapping = torch.zeros(unique_labels.max() + 2, dtype=torch.long, device=x.device)
                        label_mapping[unique_labels] = torch.arange(len(unique_labels))
                        cluster_labels = label_mapping[cluster_labels + 1] - 1

                outputs.append(cluster_labels)

            return outputs
        else:
            # PyG lane
            x, edge_index, batch = batch_x.x, batch_x.edge_index, batch_x.batch

            # Graph convolutions
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))
            x = F.relu(self.conv4(x, edge_index))

            # Attention mechanism
            attention_weights = self.attention(x)
            x = x * attention_weights

            # Global pooling for graph-level features
            graph_feat = global_mean_pool(x, batch)

            # Expand graph features to node level
            graph_feat = graph_feat[batch]

            # Combine node and graph features
            x = torch.cat([x, graph_feat], dim=1)

            # Output
            logits = self.output(x).squeeze(-1)

            # Clustering
            with torch.no_grad():
                # Convert to numpy for clustering
                features = x.cpu().numpy()

                # Use HDBSCAN for clustering
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    cluster_selection_epsilon=self.cluster_selection_epsilon,
                    metric='euclidean',
                    gen_min_span_tree=True
                )
                cluster_labels = clusterer.fit_predict(features)

                # Convert to tensor and handle noise
                cluster_labels = torch.from_numpy(cluster_labels).to(x.device)
                cluster_labels[cluster_labels == -1] = -1  # Noise label

                # Make labels contiguous starting from 0
                unique_labels = torch.unique(cluster_labels[cluster_labels != -1])
                if len(unique_labels) > 0:
                    label_mapping = torch.zeros(unique_labels.max() + 2, dtype=torch.long, device=x.device)
                    label_mapping[unique_labels] = torch.arange(len(unique_labels))
                    cluster_labels = label_mapping[cluster_labels + 1] - 1

            return cluster_labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    best_model = None

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        train_samples = 0

        for batch in train_loader:
            if model.lane == "torch_ragged_xy":
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                optimizer.zero_grad()
                outputs = model(Xs)

                loss = 0.0
                correct = 0
                total_hits = 0

                for out, y in zip(outputs, ys):
                    # Create target for clustering loss
                    unique_labels = torch.unique(y[y != 0])
                    if len(unique_labels) == 0:
                        continue

                    # Map truth labels to contiguous range
                    label_mapping = torch.zeros(y.max() + 1, dtype=torch.long, device=y.device)
                    label_mapping[unique_labels] = torch.arange(len(unique_labels))
                    y_mapped = label_mapping[y]

                    # Only consider non-noise hits
                    mask = y_mapped != -1
                    if mask.sum() == 0:
                        continue

                    # Compute loss
                    loss += criterion(out[mask], y_mapped[mask])

                    # Compute accuracy (for monitoring)
                    pred = out[mask]
                    correct += (pred == y_mapped).float().sum().item()
                    total_hits += mask.sum().item()

                if total_hits > 0:
                    loss /= len(Xs)
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
                    total_train_acc += correct / total_hits
                    train_samples += 1

            else:
                # PyG lane
                G = batch.to(device)
                optimizer.zero_grad()

                out = model(G)
                y = G.y

                # Create target for clustering loss
                unique_labels = torch.unique(y[y != 0])
                if len(unique_labels) > 0:
                    label_mapping = torch.zeros(y.max() + 1, dtype=torch.long, device=y.device)
                    label_mapping[unique_labels] = torch.arange(len(unique_labels))
                    y_mapped = label_mapping[y]

                    # Only consider non-noise hits
                    mask = y_mapped != -1
                    if mask.sum() > 0:
                        loss = criterion(out[mask], y_mapped[mask])
                        loss.backward()
                        optimizer.step()
                        total_train_loss += loss.item()

                        # Compute accuracy
                        correct = (out[mask] == y_mapped).float().sum().item()
                        total_train_acc += correct / mask.sum().item()
                        train_samples += 1

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                if model.lane == "torch_ragged_xy":
                    Xs, ys = batch
                    Xs = [x.to(device) for x in Xs]
                    ys = [y.to(device) for y in ys]

                    outputs = model(Xs)

                    loss = 0.0
                    correct = 0
                    total_hits = 0

                    for out, y in zip(outputs, ys):
                        unique_labels = torch.unique(y[y != 0])
                        if len(unique_labels) == 0:
                            continue

                        label_mapping = torch.zeros(y.max() + 1, dtype=torch.long, device=y.device)
                        label_mapping[unique_labels] = torch.arange(len(unique_labels))
                        y_mapped = label_mapping[y]

                        mask = y_mapped != -1
                        if mask.sum() == 0:
                            continue

                        loss += criterion(out[mask], y_mapped[mask])
                        pred = out[mask]
                        correct += (pred == y_mapped).float().sum().item()
                        total_hits += mask.sum().item()

                    if total_hits > 0:
                        loss /= len(Xs)
                        total_val_loss += loss.item()
                        total_val_acc += correct / total_hits
                        val_samples += 1

                else:
                    G = batch.to(device)
                    out = model(G)
                    y = G.y

                    unique_labels = torch.unique(y[y != 0])
                    if len(unique_labels) > 0:
                        label_mapping = torch.zeros(y.max() + 1, dtype=torch.long, device=y.device)
                        label_mapping[unique_labels] = torch.arange(len(unique_labels))
                        y_mapped = label_mapping[y]

                        mask = y_mapped != -1
                        if mask.sum() > 0:
                            loss = criterion(out[mask], y_mapped[mask])
                            total_val_loss += loss.item()

                            correct = (out[mask] == y_mapped).float().sum().item()
                            total_val_acc += correct / mask.sum().item()
                            val_samples += 1

        # Calculate averages
        if train_samples > 0:
            avg_train_loss = total_train_loss / train_samples
            avg_train_acc = total_train_acc / train_samples
        else:
            avg_train_loss = 0.0
            avg_train_acc = 0.0

        if val_samples > 0:
            avg_val_loss = total_val_loss / val_samples
            avg_val_acc = total_val_acc / val_samples
        else:
            avg_val_loss = 0.0
            avg_val_acc = 0.0

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)

        # Update learning rate
        scheduler.step(avg_val_acc)

        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model = model.state_dict()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Acc: {avg_val_acc:.4f}")

    # Load best model
    if best_model is not None:
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
    if mode == "torch_ragged_xy":
        Xs, ys = first_batch
        Xs = [x.to(device) for x in Xs]
        ys = [y.to(device) for y in ys]
        model = make_model(Xs).to(device)
        out = model(Xs)
    elif mode == "pyg_batch":
        G = first_batch.to(device)
        model = make_model(G).to(device)
        out = model(G)
    else:
        raise RuntimeError(f"Unknown lane mode: {mode}")

    assert_label_output_by_lane(mode, first_batch, out, allow_noise_label=True)

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
                        out = trained_model(Xs)
                    elif mode == "pyg_batch":
                        G = batch.to(device)
                        out = trained_model(G)
                    else:
                        raise RuntimeError(f"Unknown lane mode: {mode}")
                    assert_label_output_by_lane(mode, batch, out, allow_noise_label=True)
                    if i >= 3:  # 4 batches
                        break
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
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

