
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
from scipy.spatial.distance import pdist, squareform

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_encodings = None
        self.n_layers = None

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
        # Collect all hits for scaling
        all_hits = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_hits[:, :3])  # Scale r, theta, z

        # Get unique layer IDs and create embeddings
        unique_layers = np.unique(np.concatenate([x[:, 3].numpy() for x in Xs]))
        self.n_layers = len(unique_layers)
        self.layer_encodings = {layer: idx for idx, layer in enumerate(unique_layers)}
        return self

    def transform(self, X):
        # X: [N_hits, 4] (r, theta, z, layer_id)
        X = X.numpy() if isinstance(X, torch.Tensor) else X

        # Scale spatial coordinates
        scaled = self.scaler.transform(X[:, :3])

        # One-hot encode layer_id
        layer_onehot = np.zeros((X.shape[0], self.n_layers))
        layer_indices = np.array([self.layer_encodings[lid] for lid in X[:, 3]])
        layer_onehot[np.arange(X.shape[0]), layer_indices] = 1

        # Combine features: [r, theta, z, layer_onehot]
        transformed = np.hstack([scaled, layer_onehot])
        return torch.from_numpy(transformed).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Determine input dimensions from example batch
        if isinstance(example_batch_x, list):
            # Torch ragged lane
            x = example_batch_x[0]
            input_dim = x.shape[1]
        else:
            # PyG lane
            input_dim = example_batch_x.x.shape[1]

        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)

        # Node classification head
        self.classifier = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Predict cluster ID offset
        )

        # Edge construction parameters
        self.radius = 0.5
        self.min_samples = 4

    def build_graph(self, X, batch=None):
        # X: [N, F] features
        # Build k-NN graph based on spatial distance
        if batch is None:
            # Single event case
            pos = X[:, :3]  # r, theta, z
            dist = squareform(pdist(pos))
            adj = (dist < self.radius).astype(float)
            edge_index = torch.nonzero(adj).t().contiguous()
            return Data(x=X, edge_index=edge_index)
        else:
            # Batched case - build separate graphs per event
            graphs = []
            for i in range(batch.max().item() + 1):
                mask = (batch == i)
                x_i = X[mask]
                pos = x_i[:, :3]
                dist = squareform(pdist(pos))
                adj = (dist < self.radius).astype(float)
                edge_index = torch.nonzero(torch.from_numpy(adj)).t().contiguous()
                graphs.append(Data(x=x_i, edge_index=edge_index))
            return Batch.from_data_list(graphs)

    def forward(self, batch_x):
        if isinstance(batch_x, list):
            # Torch ragged lane
            Xs = batch_x
            outs = []
            for X in Xs:
                # Build graph for this event
                data = self.build_graph(X)
                data = data.to(device)

                # Graph convolutions
                x = F.relu(self.conv1(data.x, data.edge_index))
                x = F.relu(self.conv2(x, data.edge_index))
                x = F.relu(self.conv3(x, data.edge_index))

                # Node features
                node_feats = x

                # Predict cluster ID offset
                logits = self.classifier(node_feats).squeeze(-1)

                # Apply DBSCAN clustering on learned features
                feats = node_feats.cpu().numpy()
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_samples,
                    min_samples=1,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                cluster_labels = clusterer.fit_predict(feats)

                # Convert to tensor and handle noise (-1)
                cluster_labels = torch.from_numpy(cluster_labels).to(device)
                outs.append(cluster_labels)
            return outs
        else:
            # PyG lane
            data = batch_x
            data = data.to(device)

            # Graph convolutions
            x = F.relu(self.conv1(data.x, data.edge_index))
            x = F.relu(self.conv2(x, data.edge_index))
            x = F.relu(self.conv3(x, data.edge_index))

            # Node features
            node_feats = x

            # Predict cluster ID offset
            logits = self.classifier(node_feats).squeeze(-1)

            # Apply DBSCAN clustering on learned features per batch element
            cluster_labels = []
            for i in range(data.batch.max().item() + 1):
                mask = (data.batch == i)
                feats = node_feats[mask].cpu().numpy()
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_samples,
                    min_samples=1,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                labels = clusterer.fit_predict(feats)
                cluster_labels.append(torch.from_numpy(labels).to(device))

            # Combine results
            cluster_labels = torch.cat(cluster_labels)
            return cluster_labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.MSELoss()

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # Training loop
        for batch in train_loader:
            optimizer.zero_grad()

            if isinstance(batch, list):
                # Torch ragged lane
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                # Forward pass
                outs = model(Xs)

                # Compute loss (we'll use a proxy loss since clustering is non-differentiable)
                loss = 0.0
                for out, y in zip(outs, ys):
                    # Create pseudo-targets for training
                    unique_labels = torch.unique(y[y > 0])
                    pseudo_target = torch.zeros_like(out).float()
                    for lbl in unique_labels:
                        mask = (y == lbl)
                        pseudo_target[mask] = lbl.float()
                    loss += criterion(out.float(), pseudo_target)

                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            else:
                # PyG lane
                data = batch.to(device)
                out = model(data)

                # Create pseudo-targets
                y = data.y
                unique_labels = torch.unique(y[y > 0])
                pseudo_target = torch.zeros_like(out).float()
                for lbl in unique_labels:
                    mask = (y == lbl)
                    pseudo_target[mask] = lbl.float()

                loss = criterion(out.float(), pseudo_target)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, list):
                    Xs, ys = batch
                    Xs = [x.to(device) for x in Xs]
                    outs = model(Xs)

                    loss = 0.0
                    for out, y in zip(outs, ys):
                        unique_labels = torch.unique(y[y > 0])
                        pseudo_target = torch.zeros_like(out).float()
                        for lbl in unique_labels:
                            mask = (y == lbl)
                            pseudo_target[mask] = lbl.float()
                        loss += criterion(out.float(), pseudo_target)
                    epoch_val_loss += loss.item()
                else:
                    data = batch.to(device)
                    out = model(data)
                    y = data.y
                    unique_labels = torch.unique(y[y > 0])
                    pseudo_target = torch.zeros_like(out).float()
                    for lbl in unique_labels:
                        mask = (y == lbl)
                        pseudo_target[mask] = lbl.float()
                    epoch_val_loss += criterion(out.float(), pseudo_target).item()

        # Calculate metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)

        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)

        # Dummy accuracy for monitoring (will be replaced by actual FitAccuracy)
        train_acc.append(0.0)
        val_acc.append(0.0)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

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

