
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
        X, y = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        # Convert to PyG Data format
        data = Data(x=X, y=y)
        return data

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_encoder = None
        self.layer_ids = None

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
        # Collect all data for fitting
        all_data = torch.cat(Xs, dim=0).numpy()
        # Fit scaler on coordinates (excluding layer_id)
        self.scaler.fit(all_data[:, :3])
        # Encode layer_id as one-hot
        unique_layers = torch.unique(torch.cat([x[:, 3] for x in Xs]))
        self.layer_ids = unique_layers
        return self

    def transform(self, X):
        # Scale coordinates
        coords = self.scaler.transform(X[:, :3].numpy())
        coords = torch.from_numpy(coords).float()
        # One-hot encode layer_id
        layer_onehot = F.one_hot(X[:, 3].long(), num_classes=len(self.layer_ids)).float()
        # Combine features
        X_transformed = torch.cat([coords, layer_onehot], dim=1)
        return X_transformed

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
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 1)  # For clustering embedding

        # Clustering parameters
        self.cluster_min_samples = 4
        self.cluster_min_cluster_size = 4

    def forward(self, batch_x):
        if self.lane == "torch_ragged_xy":
            # Handle ragged batch
            embeddings = []
            for x in batch_x:
                # Create dummy edge_index for fully connected graph
                n_nodes = x.shape[0]
                edge_index = torch.combinations(torch.arange(n_nodes), r=2).t()
                edge_index = edge_index.to(x.device)

                # GNN forward
                h = F.relu(self.conv1(x, edge_index))
                h = F.relu(self.conv2(h, edge_index))
                h = F.relu(self.conv3(h, edge_index))
                h = global_mean_pool(h, torch.zeros(n_nodes, dtype=torch.long, device=x.device))
                h = F.relu(self.fc1(h))
                emb = self.fc2(h)
                embeddings.append(emb)
            return torch.cat(embeddings, dim=0)
        else:
            # PyG batch
            h = F.relu(self.conv1(batch_x.x, batch_x.edge_index))
            h = F.relu(self.conv2(h, batch_x.edge_index))
            h = F.relu(self.conv3(h, batch_x.edge_index))
            h = global_mean_pool(h, batch_x.batch)
            h = F.relu(self.fc1(h))
            return self.fc2(h)

    def predict_labels(self, batch_x):
        if self.lane == "torch_ragged_xy":
            # Get embeddings for each event
            all_labels = []
            for x in batch_x:
                # Create fully connected graph
                n_nodes = x.shape[0]
                edge_index = torch.combinations(torch.arange(n_nodes), r=2).t()
                edge_index = edge_index.to(x.device)

                # Get node embeddings
                with torch.no_grad():
                    h = F.relu(self.conv1(x, edge_index))
                    h = F.relu(self.conv2(h, edge_index))
                    h = F.relu(self.conv3(h, edge_index))
                    node_emb = h

                # Cluster using HDBSCAN
                node_emb_np = node_emb.cpu().numpy()
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.cluster_min_cluster_size,
                    min_samples=self.cluster_min_samples,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                labels = clusterer.fit_predict(node_emb_np)

                # Convert to torch and handle noise (-1)
                labels = torch.from_numpy(labels).to(x.device)
                labels[labels == -1] = -1  # Noise
                all_labels.append(labels)
            return all_labels
        else:
            # PyG batch processing
            with torch.no_grad():
                h = F.relu(self.conv1(batch_x.x, batch_x.edge_index))
                h = F.relu(self.conv2(h, batch_x.edge_index))
                h = F.relu(self.conv3(h, batch_x.edge_index))

            # Cluster per event
            labels = []
            unique_batches = torch.unique(batch_x.batch)
            for b in unique_batches:
                mask = (batch_x.batch == b)
                node_emb = h[mask].cpu().numpy()

                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.cluster_min_cluster_size,
                    min_samples=self.cluster_min_samples,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                pred_labels = clusterer.fit_predict(node_emb)
                pred_labels = torch.from_numpy(pred_labels).to(batch_x.x.device)
                pred_labels[pred_labels == -1] = -1  # Noise
                labels.append(pred_labels)

            return torch.cat(labels, dim=0)

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.MSELoss()

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    max_patience = 5

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            if model.lane == "torch_ragged_xy":
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                # Forward pass
                embeddings = model(Xs)
                loss = torch.tensor(0.0, device=device)

                # Simple reconstruction loss
                for x, y, emb in zip(Xs, ys, embeddings):
                    # Create target embeddings based on track IDs
                    unique_tracks = torch.unique(y[y != 0])
                    track_embeddings = torch.randn(len(unique_tracks), 1, device=device)
                    target_emb = torch.zeros_like(y, dtype=torch.float32).unsqueeze(1)
                    for i, track_id in enumerate(unique_tracks):
                        target_emb[y == track_id] = track_embeddings[i]

                    loss += criterion(emb.expand(x.shape[0], -1), target_emb)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Accuracy calculation (simplified)
                with torch.no_grad():
                    pred_labels = model.predict_labels(Xs)
                    for pred, true in zip(pred_labels, ys):
                        # Simple accuracy (not FitAccuracy)
                        correct = (pred == true).float().sum()
                        train_correct += correct
                        train_total += true.shape[0]

            else:
                # PyG training
                batch = batch.to(device)
                embeddings = model(batch)
                loss = torch.tensor(0.0, device=device)

                # Simple reconstruction loss
                unique_tracks = torch.unique(batch.y[batch.y != 0])
                track_embeddings = torch.randn(len(unique_tracks), 1, device=device)
                target_emb = torch.zeros_like(batch.y, dtype=torch.float32).unsqueeze(1)
                for i, track_id in enumerate(unique_tracks):
                    target_emb[batch.y == track_id] = track_embeddings[i]

                loss = criterion(embeddings, target_emb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Accuracy calculation
                with torch.no_grad():
                    pred_labels = model.predict_labels(batch)
                    correct = (pred_labels == batch.y).float().sum()
                    train_correct += correct
                    train_total += batch.y.shape[0]

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if model.lane == "torch_ragged_xy":
                    Xs, ys = batch
                    Xs = [x.to(device) for x in Xs]
                    ys = [y.to(device) for y in ys]

                    embeddings = model(Xs)
                    loss = torch.tensor(0.0, device=device)

                    for x, y, emb in zip(Xs, ys, embeddings):
                        unique_tracks = torch.unique(y[y != 0])
                        track_embeddings = torch.randn(len(unique_tracks), 1, device=device)
                        target_emb = torch.zeros_like(y, dtype=torch.float32).unsqueeze(1)
                        for i, track_id in enumerate(unique_tracks):
                            target_emb[y == track_id] = track_embeddings[i]

                        loss += criterion(emb.expand(x.shape[0], -1), target_emb)

                    val_loss += loss.item()

                    pred_labels = model.predict_labels(Xs)
                    for pred, true in zip(pred_labels, ys):
                        correct = (pred == true).float().sum()
                        val_correct += correct
                        val_total += true.shape[0]

                else:
                    batch = batch.to(device)
                    embeddings = model(batch)
                    loss = torch.tensor(0.0, device=device)

                    unique_tracks = torch.unique(batch.y[batch.y != 0])
                    track_embeddings = torch.randn(len(unique_tracks), 1, device=device)
                    target_emb = torch.zeros_like(batch.y, dtype=torch.float32).unsqueeze(1)
                    for i, track_id in enumerate(unique_tracks):
                        target_emb[batch.y == track_id] = track_embeddings[i]

                    loss = criterion(embeddings, target_emb)
                    val_loss += loss.item()

                    pred_labels = model.predict_labels(batch)
                    correct = (pred_labels == batch.y).float().sum()
                    val_correct += correct
                    val_total += batch.y.shape[0]

        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc.item())
        val_acc_history.append(val_acc.item())

        # Early stopping and model saving
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

