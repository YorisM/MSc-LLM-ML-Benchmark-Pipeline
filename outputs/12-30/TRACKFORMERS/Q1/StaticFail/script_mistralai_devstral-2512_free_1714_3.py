
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
        X, y = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        # Convert to PyG Data format
        data = Data(x=X, y=y)
        return data

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_means = None
        self.layer_stds = None

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
        # Concatenate all events for global scaling
        all_X = torch.cat(Xs, dim=0).numpy()
        self.scaler.fit(all_X[:, :3])  # Scale r, theta, z

        # Compute layer-wise statistics
        layer_stats = {}
        for X in Xs:
            layers = X[:, 3].numpy()
            for i, layer in enumerate(layers):
                if layer not in layer_stats:
                    layer_stats[layer] = []
                layer_stats[layer].append(X[i, :3].numpy())

        self.layer_means = {}
        self.layer_stds = {}
        for layer, hits in layer_stats.items():
            hits = np.array(hits)
            self.layer_means[layer] = np.mean(hits, axis=0)
            self.layer_stds[layer] = np.std(hits, axis=0)

        return self

    def transform(self, X):
        # Scale features
        X = X.clone()
        X[:, :3] = torch.from_numpy(self.scaler.transform(X[:, :3].numpy())).float()

        # Layer-wise normalization
        for i in range(X.shape[0]):
            layer = X[i, 3].item()
            if layer in self.layer_means:
                X[i, :3] = (X[i, :3] - torch.from_numpy(self.layer_means[layer])) / (torch.from_numpy(self.layer_stds[layer]) + 1e-6)

        return X

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

        # GNN layers
        self.conv1 = GCNConv(self.input_dim, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 32)

        # Edge prediction head
        self.edge_mlp = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Node embedding head
        self.node_mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Clustering head
        self.cluster_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

    def forward(self, batch_x):
        if isinstance(batch_x, list):
            # Torch ragged lane - convert to PyG batch
            xs, ys = batch_x, None
            graphs = []
            for i, x in enumerate(xs):
                # Create complete graph for each event
                num_nodes = x.shape[0]
                edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
                graphs.append(Data(x=x, edge_index=edge_index))
            batch = Batch.from_data_list(graphs)
        else:
            # PyG lane
            batch = batch_x

        # GNN forward pass
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = F.relu(self.conv2(x, batch.edge_index))
        x = F.relu(self.conv3(x, batch.edge_index))

        # Edge prediction
        row, col = batch.edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        edge_scores = self.edge_mlp(edge_features).squeeze()

        # Filter edges based on learned scores
        edge_mask = edge_scores > 0.5
        filtered_edge_index = batch.edge_index[:, edge_mask]

        # Update graph with filtered edges
        batch.edge_index = filtered_edge_index

        # Node embedding
        node_embeddings = self.node_mlp(x)

        # Get cluster assignments
        cluster_features = self.cluster_head(node_embeddings).cpu().numpy()

        # Use HDBSCAN for clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=4,
            min_samples=1,
            cluster_selection_epsilon=0.1,
            prediction_data=True
        )
        clusterer.fit(cluster_features)

        # Get cluster labels
        labels = torch.tensor(clusterer.labels_, dtype=torch.long, device=batch.x.device)

        # Convert -1 (noise) to -1 in output
        labels[labels == -1] = -1

        # For ragged lane, split back into list
        if isinstance(batch_x, list):
            labels_list = []
            start = 0
            for x in batch_x:
                num_nodes = x.shape[0]
                labels_list.append(labels[start:start+num_nodes])
                start += num_nodes
            return labels_list
        else:
            return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

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
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            if isinstance(batch, list):
                xs, ys = batch, None
                out = model(xs)
                # For ragged lane, we need to flatten for loss calculation
                flat_out = torch.cat(out)
                flat_ys = torch.cat([y for y in ys])
            else:
                out = model(batch)
                flat_out = out
                flat_ys = batch.y

            # Filter out noise labels (-1) for loss calculation
            mask = flat_ys != -1
            if mask.sum() > 0:
                loss = criterion(flat_out[mask], flat_ys[mask])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Calculate accuracy
                pred = flat_out[mask].argmax(dim=1)
                correct += (pred == flat_ys[mask]).sum().item()
                total += mask.sum().item()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                if isinstance(batch, list):
                    xs, ys = batch, None
                    out = model(xs)
                    flat_out = torch.cat(out)
                    flat_ys = torch.cat([y for y in ys])
                else:
                    out = model(batch)
                    flat_out = out
                    flat_ys = batch.y

                mask = flat_ys != -1
                if mask.sum() > 0:
                    loss = criterion(flat_out[mask], flat_ys[mask])
                    val_loss += loss.item()

                    pred = flat_out[mask].argmax(dim=1)
                    val_correct += (pred == flat_ys[mask]).sum().item()
                    val_total += mask.sum().item()

        # Calculate metrics
        train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = correct / total if total > 0 else 0
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step(val_loss)

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

