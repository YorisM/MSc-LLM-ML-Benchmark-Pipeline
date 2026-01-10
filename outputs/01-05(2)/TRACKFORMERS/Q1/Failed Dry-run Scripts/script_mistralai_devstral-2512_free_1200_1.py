
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
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import scipy.spatial.distance as dist

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
        all_X = torch.cat(Xs, dim=0).numpy()
        self.scaler.fit(all_X)

        # Compute layer-wise statistics for normalization
        layer_ids = all_X[:, 3]
        unique_layers = np.unique(layer_ids)
        self.layer_means = {}
        self.layer_stds = {}

        for layer in unique_layers:
            mask = (layer_ids == layer)
            layer_data = all_X[mask, :3]
            self.layer_means[layer] = np.mean(layer_data, axis=0)
            self.layer_stds[layer] = np.std(layer_data, axis=0)

        return self

    def transform(self, X):
        # Apply global scaling
        X_scaled = self.scaler.transform(X.numpy())
        X_scaled = torch.from_numpy(X_scaled).float()

        # Apply layer-wise normalization
        layer_ids = X[:, 3].numpy()
        normalized_features = X_scaled.clone()

        for i, layer in enumerate(layer_ids):
            if layer in self.layer_means:
                mean = torch.from_numpy(self.layer_means[layer]).float()
                std = torch.from_numpy(self.layer_stds[layer]).float()
                normalized_features[i, :3] = (X_scaled[i, :3] - mean) / (std + 1e-8)

        return normalized_features

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
        else:
            # PyG lane
            self.input_dim = example_batch_x.x.shape[1]

        # Graph convolution layers
        self.conv1 = GCNConv(self.input_dim, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Output layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

        # Cluster embedding
        self.cluster_embedding = nn.Linear(64, 32)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, GCNConv):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, batch_x):
        if isinstance(batch_x, list):
            # Torch ragged lane - convert to PyG batch
            graphs = []
            for i, x in enumerate(batch_x):
                # Create edge indices based on spatial proximity
                pos = x[:, :3]
                distance_matrix = dist.cdist(pos, pos)
                edge_index = torch.nonzero(distance_matrix < 0.5).t()

                # Create graph
                graph = Data(x=x, edge_index=edge_index)
                graphs.append(graph)

            batch = Batch.from_data_list(graphs).to(device)
        else:
            # PyG lane
            batch = batch_x.to(device)

        # Graph convolutions
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = F.relu(self.conv2(x, batch.edge_index))
        x = self.conv3(x, batch.edge_index)

        # Attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Global pooling for cluster embedding
        cluster_emb = global_mean_pool(x, batch.batch)
        cluster_emb = self.cluster_embedding(cluster_emb)

        # Node-level features
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze()

        return x, cluster_emb

    def predict_labels(self, batch_x):
        with torch.no_grad():
            if isinstance(batch_x, list):
                # Torch ragged lane
                all_labels = []
                for x in batch_x:
                    # Create graph for this event
                    pos = x[:, :3]
                    distance_matrix = dist.cdist(pos, pos)
                    edge_index = torch.nonzero(distance_matrix < 0.5).t()
                    graph = Data(x=x, edge_index=edge_index).to(device)

                    # Get embeddings
                    node_scores, cluster_emb = self.forward(graph)

                    # DBSCAN clustering on node embeddings
                    node_emb = node_scores.cpu().numpy().reshape(-1, 1)
                    clustering = DBSCAN(eps=0.3, min_samples=4).fit(node_emb)
                    labels = torch.from_numpy(clustering.labels_).to(device)

                    # Assign -1 to noise
                    labels[labels == -1] = -1
                    all_labels.append(labels)
                return all_labels
            else:
                # PyG lane
                node_scores, cluster_emb = self.forward(batch_x)
                node_emb = node_scores.cpu().numpy().reshape(-1, 1)
                clustering = DBSCAN(eps=0.3, min_samples=4).fit(node_emb)
                labels = torch.from_numpy(clustering.labels_).to(device)
                labels[labels == -1] = -1
                return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float('inf')
    best_model = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # Training
        for batch in train_loader:
            optimizer.zero_grad()

            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                # Torch ragged lane
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                # Convert to PyG batch for training
                graphs = []
                for i, x in enumerate(Xs):
                    pos = x[:, :3]
                    distance_matrix = dist.cdist(pos, pos)
                    edge_index = torch.nonzero(distance_matrix < 0.5).t()
                    graph = Data(x=x, edge_index=edge_index, y=ys[i])
                    graphs.append(graph)

                batch = Batch.from_data_list(graphs).to(device)
            else:
                # PyG lane
                batch = batch.to(device)

            # Forward pass
            node_scores, cluster_emb = model(batch)

            # Create target (simplified for training)
            # We'll use the track IDs as targets, but need to handle noise
            y = batch.y
            y = y.float()
            y[y == 0] = -1  # Set noise to -1

            # Loss calculation
            loss = criterion(node_scores, y.unsqueeze(1))

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # Torch ragged lane
                    Xs, ys = batch
                    Xs = [x.to(device) for x in Xs]
                    ys = [y.to(device) for y in ys]

                    graphs = []
                    for i, x in enumerate(Xs):
                        pos = x[:, :3]
                        distance_matrix = dist.cdist(pos, pos)
                        edge_index = torch.nonzero(distance_matrix < 0.5).t()
                        graph = Data(x=x, edge_index=edge_index, y=ys[i])
                        graphs.append(graph)

                    batch = Batch.from_data_list(graphs).to(device)
                else:
                    # PyG lane
                    batch = batch.to(device)

                node_scores, cluster_emb = model(batch)
                y = batch.y
                y = y.float()
                y[y == 0] = -1

                loss = criterion(node_scores, y.unsqueeze(1))
                epoch_val_loss += loss.item()

        # Calculate metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Update scheduler
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Calculate accuracy (simplified)
        train_acc = 0.0
        val_acc = 0.0

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

