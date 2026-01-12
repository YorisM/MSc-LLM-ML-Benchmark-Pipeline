
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
        self.layer_encoder = None
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

        # Encode layer_id
        unique_layers = np.unique(all_hits[:, 3])
        self.n_layers = len(unique_layers)
        self.layer_encoder = {l: i for i, l in enumerate(unique_layers)}
        return self

    def transform(self, X):
        # X: [N_hits, 4] numpy array
        X = X.numpy() if isinstance(X, torch.Tensor) else X

        # Scale coordinates
        scaled = self.scaler.transform(X[:, :3])

        # Encode layer_id
        layer_encoded = np.array([self.layer_encoder[l] for l in X[:, 3]], dtype=np.float32)

        # Combine features
        X_out = np.column_stack([scaled, layer_encoded])
        return torch.from_numpy(X_out).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Determine input dimensions
        if isinstance(example_batch_x, list):
            # Torch ragged lane
            self.lane = "torch_ragged_xy"
            self.input_dim = example_batch_x[0].shape[1]
        else:
            # PyG lane
            self.lane = "pyg_batch"
            self.input_dim = example_batch_x.x.shape[1]

        # GNN architecture
        self.conv1 = GCNConv(self.input_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)

        # Cluster head
        self.cluster_head = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Edge prediction
        self.edge_mlp = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, batch_x):
        if self.lane == "torch_ragged_xy":
            # Convert to PyG format for processing
            graphs = []
            for x in batch_x:
                # Create complete graph (simplest approach)
                n_nodes = x.shape[0]
                edge_index = torch.combinations(torch.arange(n_nodes), r=2).t()
                edge_index = edge_index.to(x.device)

                # Add self-loops
                edge_index = torch.cat([
                    edge_index,
                    torch.arange(n_nodes).unsqueeze(0).repeat(2, 1).to(x.device)
                ], dim=1)

                graphs.append(Data(x=x, edge_index=edge_index))
            batch = Batch.from_data_list(graphs)
        else:
            batch = batch_x

        # GNN forward
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = F.relu(self.conv2(x, batch.edge_index))
        x = self.conv3(x, batch.edge_index)

        # Get node embeddings
        node_emb = self.cluster_head(x)

        # Edge prediction for clustering
        row, col = batch.edge_index
        edge_feat = torch.cat([node_emb[row], node_emb[col]], dim=1)
        edge_scores = self.edge_mlp(edge_feat).squeeze()

        return node_emb, edge_scores

    def predict_labels(self, batch_x):
        with torch.no_grad():
            if self.lane == "torch_ragged_xy":
                # Process each event separately
                all_labels = []
                for x in batch_x:
                    # Create graph
                    n_nodes = x.shape[0]
                    edge_index = torch.combinations(torch.arange(n_nodes), r=2).t()
                    edge_index = edge_index.to(x.device)
                    edge_index = torch.cat([
                        edge_index,
                        torch.arange(n_nodes).unsqueeze(0).repeat(2, 1).to(x.device)
                    ], dim=1)

                    data = Data(x=x, edge_index=edge_index)
                    batch = Batch.from_data_list([data])

                    # Get embeddings
                    node_emb, edge_scores = self.forward(batch)

                    # Convert to numpy for clustering
                    emb = node_emb.cpu().numpy()
                    scores = edge_scores.cpu().numpy()

                    # Use HDBSCAN for clustering
                    clusterer = hdbscan.HDBSCAN(
                        metric='precomputed',
                        min_cluster_size=4,
                        min_samples=1,
                        cluster_selection_epsilon=0.5,
                        gen_min_span_tree=True
                    )

                    # Create distance matrix from edge scores
                    dist_mat = np.ones((n_nodes, n_nodes)) * 2.0  # Large default distance
                    np.fill_diagonal(dist_mat, 0)
                    for i in range(len(edge_index[0])):
                        u, v = edge_index[0][i].item(), edge_index[1][i].item()
                        if u != v:
                            dist = 1.0 - scores[i]  # Convert score to distance
                            dist_mat[u, v] = dist
                            dist_mat[v, u] = dist

                    # Fit clusterer
                    clusterer.fit(dist_mat)
                    labels = clusterer.labels_

                    # Convert to torch and handle noise (-1)
                    labels = torch.from_numpy(labels).to(x.device)
                    all_labels.append(labels)

                return all_labels
            else:
                # PyG batch processing
                node_emb, edge_scores = self.forward(batch_x)

                # Get batch info
                batch_vec = batch_x.batch
                n_nodes = batch_x.x.shape[0]

                # Create distance matrix
                dist_mat = torch.ones((n_nodes, n_nodes), device=batch_x.x.device) * 2.0
                torch.fill_diagonal(dist_mat, 0)

                row, col = batch_x.edge_index
                for i in range(len(row)):
                    u, v = row[i].item(), col[i].item()
                    if u != v:
                        dist = 1.0 - edge_scores[i]
                        dist_mat[u, v] = dist
                        dist_mat[v, u] = dist

                # Process each event separately
                all_labels = []
                unique_batches = torch.unique(batch_vec)
                for b in unique_batches:
                    mask = (batch_vec == b)
                    event_dist = dist_mat[mask][:, mask]
                    event_emb = node_emb[mask].cpu().numpy()

                    # Use HDBSCAN
                    clusterer = hdbscan.HDBSCAN(
                        metric='precomputed',
                        min_cluster_size=4,
                        min_samples=1,
                        cluster_selection_epsilon=0.5
                    )
                    clusterer.fit(event_dist.cpu().numpy())
                    labels = clusterer.labels_

                    # Convert to torch
                    labels = torch.from_numpy(labels).to(batch_x.x.device)
                    all_labels.append(labels)

                # Concatenate all labels
                return torch.cat(all_labels)

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # Training
        for batch in train_loader:
            optimizer.zero_grad()

            if model.lane == "torch_ragged_xy":
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                # Create graphs
                graphs = []
                for x, y in zip(Xs, ys):
                    n_nodes = x.shape[0]
                    edge_index = torch.combinations(torch.arange(n_nodes), r=2).t()
                    edge_index = edge_index.to(x.device)
                    edge_index = torch.cat([
                        edge_index,
                        torch.arange(n_nodes).unsqueeze(0).repeat(2, 1).to(x.device)
                    ], dim=1)

                    # Create edge labels (1 if same track, 0 otherwise)
                    edge_labels = []
                    for i in range(edge_index.shape[1]):
                        u, v = edge_index[0][i], edge_index[1][i]
                        if y[u] == y[v] and y[u] != 0:  # Same track and not noise
                            edge_labels.append(1.0)
                        else:
                            edge_labels.append(0.0)
                    edge_labels = torch.tensor(edge_labels, dtype=torch.float32).to(x.device)

                    graphs.append(Data(x=x, edge_index=edge_index, y=edge_labels))

                batch_data = Batch.from_data_list(graphs)
                node_emb, edge_scores = model(batch_data)

                # Compute loss
                loss = criterion(edge_scores, batch_data.y)
            else:
                # PyG batch
                batch_data = batch.to(device)
                node_emb, edge_scores = model(batch_data)

                # Create edge labels
                y = batch_data.y
                row, col = batch_data.edge_index
                edge_labels = (y[row] == y[col]) & (y[row] != 0)
                edge_labels = edge_labels.float()

                loss = criterion(edge_scores, edge_labels)

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
                    for x, y in zip(Xs, ys):
                        n_nodes = x.shape[0]
                        edge_index = torch.combinations(torch.arange(n_nodes), r=2).t()
                        edge_index = edge_index.to(x.device)
                        edge_index = torch.cat([
                            edge_index,
                            torch.arange(n_nodes).unsqueeze(0).repeat(2, 1).to(x.device)
                        ], dim=1)

                        edge_labels = []
                        for i in range(edge_index.shape[1]):
                            u, v = edge_index[0][i], edge_index[1][i]
                            if y[u] == y[v] and y[u] != 0:
                                edge_labels.append(1.0)
                            else:
                                edge_labels.append(0.0)
                        edge_labels = torch.tensor(edge_labels, dtype=torch.float32).to(x.device)

                        graphs.append(Data(x=x, edge_index=edge_index, y=edge_labels))

                    batch_data = Batch.from_data_list(graphs)
                    _, edge_scores = model(batch_data)
                    loss = criterion(edge_scores, batch_data.y)
                else:
                    batch_data = batch.to(device)
                    _, edge_scores = model(batch_data)

                    y = batch_data.y
                    row, col = batch_data.edge_index
                    edge_labels = (y[row] == y[col]) & (y[row] != 0)
                    edge_labels = edge_labels.float()

                    loss = criterion(edge_scores, edge_labels)

                epoch_val_loss += loss.item()

        # Calculate metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Simple accuracy for monitoring (not the real FitAccuracy)
        train_acc = 0.0
        val_acc = 0.0

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

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

