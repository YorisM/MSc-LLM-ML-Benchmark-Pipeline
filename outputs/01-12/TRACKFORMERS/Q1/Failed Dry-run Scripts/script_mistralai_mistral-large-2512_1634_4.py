
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
from torch_scatter import scatter_add
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# Using default EventDataset as it meets requirements

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_mean = None
        self.layer_std = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,

            "collate": None,
            "extra_loader_kwargs": {"follow_batch": ["x"]},

            "eval_overrides": {"shuffle": False, "num_workers": 0}
        }

    def fit(self, Xs):
        # Compute global statistics
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Only scale r, theta, z

        # Compute layer statistics
        layers = all_X[:, 3]
        self.layer_mean = np.mean(layers)
        self.layer_std = np.std(layers)

        return self

    def transform(self, X):
        # Scale r, theta, z
        X_scaled = X.clone()
        X_scaled[:, :3] = torch.from_numpy(self.scaler.transform(X[:, :3].numpy()))

        # Normalize layer_id
        if self.layer_std > 0:
            X_scaled[:, 3] = (X[:, 3] - self.layer_mean) / self.layer_std

        # Add hit distance features
        r = X_scaled[:, 0]
        theta = X_scaled[:, 1]
        z = X_scaled[:, 2]

        # Convert to cartesian for distance calculations
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # Create edge_index for kNN graph (k=8)
        coords = torch.stack([x, y, z], dim=1).numpy()
        tree = cKDTree(coords)
        distances, indices = tree.query(coords, k=9)  # k=9 to include self

        # Create edge_index (undirected)
        edge_index = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # skip self
                edge_index.append([i, indices[i][j]])
                edge_index.append([indices[i][j], i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

        # Create PyG Data object
        from torch_geometric.data import Data
        data = Data(x=X_scaled, edge_index=edge_index)

        return data

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

        # Determine input features from example
        num_features = example_batch_x.x.shape[1]

        # Graph neural network layers
        self.conv1 = EdgeConv(num_features, 64)
        self.conv2 = EdgeConv(64, 64)
        self.conv3 = EdgeConv(64, 64)

        # Track embedding layers
        self.track_embed = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Output layer for track classification
        self.output = nn.Linear(32, 1)  # Will use sigmoid for clustering

        # Noise detection head
        self.noise_head = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Track embedding
        embeddings = self.track_embed(x)

        # Output for clustering
        cluster_logits = self.output(embeddings).squeeze(-1)

        # Noise prediction
        noise_logits = self.noise_head(embeddings).squeeze(-1)

        return cluster_logits, noise_logits

    def predict_labels(self, data):
        with torch.no_grad():
            cluster_logits, noise_logits = self.forward(data)

            # Get noise predictions
            noise_probs = torch.sigmoid(noise_logits)
            is_noise = noise_probs > 0.5

            # Get cluster assignments
            cluster_probs = torch.sigmoid(cluster_logits)
            cluster_ids = torch.argmax(cluster_probs.unsqueeze(0) - cluster_probs.unsqueeze(1), dim=1)

            # Convert to unique cluster labels
            unique_clusters = torch.unique(cluster_ids, return_inverse=True)[1]

            # Assign -1 to noise hits
            labels = unique_clusters.clone()
            labels[is_noise] = -1

            return labels.cpu()

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    # Loss functions
    cluster_loss_fn = nn.BCEWithLogitsLoss()
    noise_loss_fn = nn.BCEWithLogitsLoss()

    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            cluster_logits, noise_logits = model(data)

            # Create target matrices for clustering
            y = data.y
            is_noise = (y == 0)

            # Create cluster targets (same track = 1, different = 0)
            cluster_target = (y.unsqueeze(0) == y.unsqueeze(1)).float()
            cluster_target[is_noise] = 0
            cluster_target[:, is_noise] = 0

            # Create noise targets
            noise_target = is_noise.float()

            # Compute losses
            cluster_loss = cluster_loss_fn(cluster_logits, cluster_target)
            noise_loss = noise_loss_fn(noise_logits, noise_target)
            loss = cluster_loss + noise_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute accuracy (simplified)
            with torch.no_grad():
                pred_labels = model.predict_labels(data)
                # Simple accuracy calculation (not exact FitAccuracy)
                correct += (pred_labels == data.y.cpu()).sum().item()
                total += len(data.y)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                cluster_logits, noise_logits = model(data)

                y = data.y
                is_noise = (y == 0)

                cluster_target = (y.unsqueeze(0) == y.unsqueeze(1)).float()
                cluster_target[is_noise] = 0
                cluster_target[:, is_noise] = 0

                noise_target = is_noise.float()

                cluster_loss = cluster_loss_fn(cluster_logits, cluster_target)
                noise_loss = noise_loss_fn(noise_logits, noise_target)
                loss = cluster_loss + noise_loss

                val_loss += loss.item()

                pred_labels = model.predict_labels(data)
                val_correct += (pred_labels == data.y.cpu()).sum().item()
                val_total += len(data.y)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
        else:
            if epoch > 10 and val_acc < best_val_acc * 0.95:
                print("Early stopping triggered")
                break

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

