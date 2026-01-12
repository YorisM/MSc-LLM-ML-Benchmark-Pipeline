
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
from scipy.spatial import KDTree

#  -------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_mean = None
        self.layer_std = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,

            "collate": "ragged_xy",
            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False, "num_workers": 0}
        }

    def fit(self, Xs):
        # Compute global statistics for normalization
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Only scale r, theta, z

        # Compute layer statistics
        layers = all_X[:, 3]
        self.layer_mean = np.mean(layers)
        self.layer_std = np.std(layers)

        return self

    def transform(self, X):
        # X shape: [N_hits, 4]
        X = X.numpy() if isinstance(X, torch.Tensor) else X

        # Normalize r, theta, z
        X[:, :3] = self.scaler.transform(X[:, :3])

        # Normalize layer_id
        if self.layer_std > 0:
            X[:, 3] = (X[:, 3] - self.layer_mean) / self.layer_std

        return torch.from_numpy(X).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
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
        # Determine input dimension from example batch
        input_dim = example_batch_x[0].shape[1] if isinstance(example_batch_x, list) else example_batch_x.shape[1]

        # Graph construction parameters
        self.k = 16  # Number of nearest neighbors
        self.edge_threshold = 0.5  # Distance threshold for edge creation

        # Network architecture
        self.edge_conv1 = EdgeConv(input_dim, 64)
        self.edge_conv2 = EdgeConv(64, 64)
        self.edge_conv3 = EdgeConv(64, 64)

        self.lin1 = nn.Linear(64, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 32)

        # Output layer for track classification
        self.output = nn.Linear(32, 1)  # Will use sigmoid for clustering

        # Track embedding dimension
        self.embedding_dim = 32

    def build_graph(self, x):
        # x shape: [N_hits, F]
        N = x.size(0)

        # Build KDTree for efficient nearest neighbor search
        coords = x[:, :3].cpu().numpy()
        tree = KDTree(coords)

        # Find k-nearest neighbors
        distances, indices = tree.query(coords, k=self.k+1)  # +1 to include self
        indices = indices[:, 1:]  # Remove self
        distances = distances[:, 1:]

        # Create edge_index
        row = np.repeat(np.arange(N), self.k)
        col = indices.flatten()
        edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long, device=x.device)

        # Filter edges by distance threshold
        mask = distances.flatten() < self.edge_threshold
        edge_index = edge_index[:, mask]

        return edge_index

    def forward(self, batch_x):
        # batch_x is list of tensors [N_i, F]
        all_x = []
        all_batch = []
        ptr = 0

        # Process each event in the batch
        for x in batch_x:
            # Build graph for this event
            edge_index = self.build_graph(x)

            # Apply EdgeConv layers
            x = F.relu(self.edge_conv1(x, edge_index))
            x = F.relu(self.edge_conv2(x, edge_index))
            x = F.relu(self.edge_conv3(x, edge_index))

            # Global pooling (mean)
            x = scatter_add(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device), dim=0) / x.size(0)

            # Store results
            all_x.append(x)
            all_batch.append(torch.full((x.size(0),), ptr, device=x.device))
            ptr += 1

        # Combine all events
        x = torch.cat(all_x, dim=0)
        batch = torch.cat(all_batch, dim=0)

        # Apply final MLP
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))

        return x

    def predict_labels(self, batch_x):
        # Get embeddings
        embeddings = self.forward(batch_x)

        # Convert to numpy for clustering
        embeddings_np = embeddings.detach().cpu().numpy()

        # Cluster using HDBSCAN (will be done per event)
        from hdbscan import HDBSCAN
        labels_list = []

        ptr = 0
        for x in batch_x:
            N = x.size(0)
            if N == 0:
                labels_list.append(torch.empty(0, dtype=torch.long, device=x.device))
                continue

            # Cluster this event
            clusterer = HDBSCAN(
                min_cluster_size=4,
                min_samples=1,
                cluster_selection_epsilon=0.5,
                metric='euclidean'
            )
            labels = clusterer.fit_predict(embeddings_np[ptr:ptr+N])

            # Convert to torch tensor and adjust labels
            labels = torch.tensor(labels, dtype=torch.long, device=x.device)
            labels[labels == -1] = -1  # Keep noise as -1

            # Remap labels to be contiguous
            unique_labels = torch.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]
            for new_label, old_label in enumerate(unique_labels):
                labels[labels == old_label] = new_label

            labels_list.append(labels)
            ptr += N

        return labels_list

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_idx, (Xs, ys) in enumerate(train_loader):
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            optimizer.zero_grad()

            # Forward pass
            embeddings = model(Xs)

            # Compute loss (contrastive learning)
            loss = 0
            ptr = 0
            for x, y in zip(Xs, ys):
                N = x.size(0)
                if N == 0:
                    continue

                # Get embeddings for this event
                emb = embeddings[ptr:ptr+N]
                y_event = y

                # Create positive and negative pairs
                pos_mask = y_event.unsqueeze(1) == y_event.unsqueeze(0)
                pos_mask.fill_diagonal_(False)

                if pos_mask.sum() > 0:
                    pos_pairs = emb[pos_mask]
                    pos_dist = F.pairwise_distance(pos_pairs[:, 0], pos_pairs[:, 1])

                    # Negative pairs (different tracks)
                    neg_mask = y_event.unsqueeze(1) != y_event.unsqueeze(0)
                    neg_mask.fill_diagonal_(False)

                    if neg_mask.sum() > 0:
                        neg_pairs = emb[neg_mask]
                        neg_dist = F.pairwise_distance(neg_pairs[:, 0], neg_pairs[:, 1])

                        # Triplet loss
                        margin = 1.0
                        loss += F.relu(pos_dist.mean() - neg_dist.mean() + margin)

                ptr += N

            if loss > 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                # Get predictions
                pred_labels = model.predict_labels(Xs)

                # Compute accuracy (simplified - real metric is FitAccuracy)
                for pred, true in zip(pred_labels, ys):
                    if pred.size(0) == 0:
                        continue

                    # Simple accuracy (not the real metric)
                    correct = (pred == true).sum().item()
                    val_correct += correct
                    val_samples += pred.size(0)

        # Compute metrics
        train_loss = total_loss / len(train_loader)
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        train_acc = 0  # Not computed for training
        val_acc = val_correct / val_samples if val_samples > 0 else 0

        # Update learning rate
        scheduler.step(val_acc)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
        else:
            if epoch > 10 and val_acc < best_val_acc * 0.95:
                print("Early stopping")
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

