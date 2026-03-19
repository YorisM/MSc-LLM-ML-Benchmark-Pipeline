
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
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import hdbscan

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
        # Stack all events for global statistics
        all_hits = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_hits[:, :3])  # Scale r, theta, z

        # Compute layer-wise statistics
        layer_stats = []
        for layer_id in np.unique(all_hits[:, 3]):
            mask = all_hits[:, 3] == layer_id
            layer_hits = all_hits[mask, :3]
            layer_stats.append((layer_id, layer_hits.mean(axis=0), layer_hits.std(axis=0)))

        self.layer_means = {lid: mean for lid, mean, _ in layer_stats}
        self.layer_stds = {lid: std for lid, _, std in layer_stats}
        return self

    def transform(self, X):
        X = X.numpy() if isinstance(X, torch.Tensor) else X
        # Scale coordinates
        scaled = self.scaler.transform(X[:, :3])
        # Layer normalization
        layer_id = X[:, 3]
        normalized = np.zeros_like(scaled)
        for lid in np.unique(layer_id):
            mask = layer_id == lid
            normalized[mask] = (scaled[mask] - self.layer_means[lid]) / (self.layer_stds[lid] + 1e-6)
        # Combine features
        features = np.hstack([normalized, X[:, 3:]])
        return torch.FloatTensor(features)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Determine input features
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
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.fc = nn.Linear(16, 8)  # Embedding dimension

        # Clustering head
        self.cluster_head = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Predict cluster assignment probability
        )

    def forward(self, batch_x):
        if self.lane == "torch_ragged_xy":
            # Convert to PyG batch for processing
            graphs = []
            for i, x in enumerate(batch_x):
                # Create edge indices based on spatial proximity
                dist = torch.cdist(x[:, :3], x[:, :3])
                edges = (dist < 0.5).nonzero(as_tuple=False).t()
                graphs.append(Data(x=x, edge_index=edges))
            batch = Batch.from_data_list(graphs)
        else:
            batch = batch_x

        # GNN forward
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = F.relu(self.conv2(x, batch.edge_index))
        x = F.relu(self.conv3(x, batch.edge_index))
        x = global_mean_pool(x, batch.batch)  # [B, 16]
        x = self.fc(x)  # [B, 8]

        # Get node embeddings
        node_emb = self.fc(x[batch.batch])  # [N, 8]

        # Clustering scores
        scores = self.cluster_head(node_emb).squeeze(-1)
        return scores, node_emb

    def predict_labels(self, batch_x):
        self.eval()
        with torch.no_grad():
            if self.lane == "torch_ragged_xy":
                # Process each event separately
                all_labels = []
                for x in batch_x:
                    x = x.to(device)
                    # Create graph
                    dist = torch.cdist(x[:, :3], x[:, :3])
                    edges = (dist < 0.5).nonzero(as_tuple=False).t()
                    data = Data(x=x, edge_index=edges).to(device)

                    # Get embeddings
                    _, emb = self.forward(data)
                    emb = emb.cpu().numpy()

                    # Cluster with HDBSCAN
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=4,
                        min_samples=2,
                        cluster_selection_epsilon=0.1,
                        prediction_data=True
                    )
                    clusterer.fit(emb)
                    labels = clusterer.labels_

                    # Convert to track IDs (-1 for noise)
                    labels = torch.LongTensor([l if l != -1 else -1 for l in labels])
                    all_labels.append(labels)
                return all_labels
            else:
                # PyG batch processing
                batch_x = batch_x.to(device)
                _, emb = self.forward(batch_x)
                emb = emb.cpu().numpy()

                # Cluster with HDBSCAN
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=4,
                    min_samples=2,
                    cluster_selection_epsilon=0.1,
                    prediction_data=True
                )
                clusterer.fit(emb)
                labels = clusterer.labels_

                # Convert to track IDs (-1 for noise)
                labels = torch.LongTensor([l if l != -1 else -1 for l in labels])
                return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.BCEWithLogitsLoss()

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        epoch_val_loss = 0

        # Training
        for batch in train_loader:
            if model.lane == "torch_ragged_xy":
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                # Create graphs and compute targets
                graphs = []
                targets = []
                for x, y in zip(Xs, ys):
                    dist = torch.cdist(x[:, :3], x[:, :3])
                    edges = (dist < 0.5).nonzero(as_tuple=False).t()
                    graphs.append(Data(x=x, edge_index=edges))

                    # Create binary targets (1 if part of track, 0 if noise)
                    target = (y > 0).float()
                    targets.append(target)

                batch = Batch.from_data_list(graphs)
                targets = torch.cat(targets).to(device)
            else:
                batch = batch.to(device)
                targets = (batch.y > 0).float()

            optimizer.zero_grad()
            scores, _ = model(batch)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if model.lane == "torch_ragged_xy":
                    Xs, ys = batch
                    Xs = [x.to(device) for x in Xs]
                    ys = [y.to(device) for y in ys]

                    graphs = []
                    targets = []
                    for x, y in zip(Xs, ys):
                        dist = torch.cdist(x[:, :3], x[:, :3])
                        edges = (dist < 0.5).nonzero(as_tuple=False).t()
                        graphs.append(Data(x=x, edge_index=edges))
                        target = (y > 0).float()
                        targets.append(target)

                    batch = Batch.from_data_list(graphs)
                    targets = torch.cat(targets).to(device)
                else:
                    batch = batch.to(device)
                    targets = (batch.y > 0).float()

                scores, _ = model(batch)
                loss = criterion(scores, targets)
                epoch_val_loss += loss.item()

        # Compute metrics
        train_loss.append(epoch_train_loss / len(train_loader))
        val_loss.append(epoch_val_loss / len(val_loader))

        # Accuracy is computed during evaluation by the harness
        train_acc.append(0)  # Placeholder
        val_acc.append(0)    # Placeholder

        scheduler.step(epoch_val_loss)

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

