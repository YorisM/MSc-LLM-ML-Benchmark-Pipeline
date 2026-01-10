
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

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
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
        X = self.pre.transform(X) if self.pre is not None else X
        return torch.from_numpy(X), torch.from_numpy(y)

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_encoder = None
        self.n_layers = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
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
        # Concatenate all events for global statistics
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Scale r, theta, z

        # Encode layer_id as one-hot
        unique_layers = np.unique(all_X[:, 3])
        self.n_layers = len(unique_layers)
        self.layer_encoder = {l: i for i, l in enumerate(unique_layers)}
        return self

    def transform(self, X):
        X = X.numpy() if isinstance(X, torch.Tensor) else X
        # Scale spatial coordinates
        X_scaled = self.scaler.transform(X[:, :3])
        # One-hot encode layer_id
        layer_onehot = np.zeros((X.shape[0], self.n_layers), dtype=np.float32)
        layer_ids = X[:, 3].astype(int)
        for i, lid in enumerate(layer_ids):
            if lid in self.layer_encoder:
                layer_onehot[i, self.layer_encoder[lid]] = 1.0
        # Combine features
        X_out = np.hstack([X_scaled, layer_onehot])
        return torch.from_numpy(X_out).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Determine input features from example batch
        if isinstance(example_batch_x, list):
            # Torch ragged lane
            self.input_dim = example_batch_x[0].shape[1]
            self.lane = "torch_ragged_xy"
        else:
            # PyG lane
            self.input_dim = example_batch_x.x.shape[1]
            self.lane = "pyg_batch"

        # GNN architecture
        self.conv1 = GCNConv(self.input_dim, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc_out = nn.Linear(16, 1)  # Predict cluster embedding

        # Clustering parameters
        self.cluster_min_size = 4
        self.cluster_min_samples = 2

    def forward(self, batch_x):
        if self.lane == "torch_ragged_xy":
            # Convert ragged batch to PyG format
            xs, ys = batch_x
            batch_list = []
            for i, (x, y) in enumerate(zip(xs, ys)):
                data = Data(x=x, y=y)
                data.batch = torch.full((x.shape[0],), i, dtype=torch.long)
                batch_list.append(data)
            batch = Batch.from_data_list(batch_list).to(device)
        else:
            batch = batch_x

        # GNN forward pass
        x = batch.x
        edge_index = torch_geometric.utils.dense_to_sparse(batch.batch)[0].to(device)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch.batch)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        embeddings = self.fc_out(x)

        # Get cluster assignments
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.cluster_min_size,
            min_samples=self.cluster_min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        ).fit(embeddings.cpu().numpy())

        # Assign clusters to nodes
        cluster_labels = torch.tensor(clusterer.labels_, dtype=torch.long, device=device)

        # Map cluster labels to track IDs (noise gets -1)
        unique_clusters = torch.unique(cluster_labels[cluster_labels >= 0])
        if len(unique_clusters) > 0:
            # Simple mapping: cluster_id -> track_id (1-based)
            cluster_to_track = {c.item(): i+1 for i, c in enumerate(unique_clusters)}
            track_labels = torch.zeros_like(cluster_labels)
            for c, t in cluster_to_track.items():
                track_labels[cluster_labels == c] = t
            track_labels[cluster_labels == -1] = -1
        else:
            track_labels = torch.full_like(cluster_labels, -1)

        if self.lane == "torch_ragged_xy":
            # Split back into ragged format
            split_sizes = [data.x.shape[0] for data in batch.to_data_list()]
            return torch.split(track_labels, split_sizes)
        else:
            return track_labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    best_val_loss = float('inf')
    best_model = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        count = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if model.lane == "torch_ragged_xy":
                xs, ys = batch
                xs = [x.to(device) for x in xs]
                ys = [y.to(device) for y in ys]
                out = model(xs)
                # Convert to single tensor for loss calculation
                pred_embeddings = torch.cat([model.fc_out(model.fc2(F.relu(model.fc1(
                    global_mean_pool(model.conv3(F.relu(model.conv2(F.relu(model.conv1(
                        x, torch_geometric.utils.dense_to_sparse(torch.full((x.shape[0],), i, dtype=torch.long))[0].to(device)
                    )))), torch_geometric.utils.dense_to_sparse(torch.full((x.shape[0],), i, dtype=torch.long))[0].to(device)
                ))), torch.full((x.shape[0],), i, dtype=torch.long))).to(device)) for i, x in enumerate(xs)], dim=0)

                target_embeddings = torch.cat([torch.mean(x[y == tid].float(), dim=0, keepdim=True)
                                              for x, y in zip(xs, ys) for tid in torch.unique(y[y > 0])], dim=0)

                loss = criterion(pred_embeddings, target_embeddings)
            else:
                batch = batch.to(device)
                out = model(batch)
                # Similar embedding loss for PyG
                pred_embeddings = model.fc_out(model.fc2(F.relu(model.fc1(
                    global_mean_pool(model.conv3(F.relu(model.conv2(F.relu(model.conv1(
                        batch.x, batch.edge_index
                    ))))), batch.batch)
                ))))

                unique_tracks = torch.unique(batch.y[batch.y > 0])
                target_embeddings = torch.stack([
                    torch.mean(batch.x[batch.y == tid], dim=0)
                    for tid in unique_tracks
                ], dim=0).to(device)

                loss = criterion(pred_embeddings, target_embeddings)

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            count += 1

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                if model.lane == "torch_ragged_xy":
                    xs, ys = batch
                    xs = [x.to(device) for x in xs]
                    ys = [y.to(device) for y in ys]
                    out = model(xs)
                    # Calculate accuracy
                    correct = 0
                    total = 0
                    for pred, true in zip(out, ys):
                        mask = true > 0
                        if mask.sum() > 0:
                            # Simple accuracy: count matching labels
                            correct += (pred[mask] == true[mask]).float().sum().item()
                            total += mask.sum().item()
                    epoch_val_acc += correct / total if total > 0 else 0
                    val_count += 1
                else:
                    batch = batch.to(device)
                    out = model(batch)
                    # Calculate accuracy
                    mask = batch.y > 0
                    if mask.sum() > 0:
                        epoch_val_acc += (out[mask] == batch.y[mask]).float().mean().item()
                    val_count += 1

        # Update metrics
        epoch_train_loss /= count
        epoch_val_acc /= val_count
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        train_acc.append(0)  # Placeholder
        val_acc.append(epoch_val_acc)

        # Early stopping and model saving
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step(epoch_val_loss)

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_loss, val_loss, train_acc, val_acc

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

