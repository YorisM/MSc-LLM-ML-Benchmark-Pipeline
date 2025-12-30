
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader
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
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_max, scatter_add
import hdbscan
from sklearn.preprocessing import StandardScaler
import numpy as np

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
        all_features = np.concatenate([X.numpy() for X in Xs], axis=0)
        self.scaler.fit(all_features[:, :3])  # Only scale r, theta, z

        # Compute layer statistics
        layer_ids = all_features[:, 3]
        self.layer_mean = np.mean(layer_ids)
        self.layer_std = np.std(layer_ids)
        return self

    def transform(self, X):
        # X: [N_hits, 4] tensor
        X = X.clone()
        # Normalize r, theta, z
        X[:, :3] = torch.from_numpy(self.scaler.transform(X[:, :3].numpy())).float()
        # Normalize layer_id
        X[:, 3] = (X[:, 3] - self.layer_mean) / (self.layer_std + 1e-8)
        return X

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
        # Determine input feature dimension
        if isinstance(example_batch_x, list):
            in_features = example_batch_x[0].shape[1]
        else:
            in_features = example_batch_x.shape[1]

        # Graph neural network layers
        self.conv1 = EdgeConv(in_features, 64)
        self.conv2 = EdgeConv(64, 128)
        self.conv3 = EdgeConv(128, 256)

        # Track embedding
        self.track_embed = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Output layer
        self.output = nn.Linear(64, 1)  # Will use for clustering

        # Clustering parameters
        self.clusterer = None
        self.min_cluster_size = 5
        self.min_samples = 3

    def build_graph(self, x):
        # Create complete graph (fully connected)
        num_nodes = x.size(0)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return edge_index.to(x.device)

    def forward(self, batch_x):
        # Handle ragged batch
        if isinstance(batch_x, list):
            outputs = []
            for x in batch_x:
                x = x.to(device)
                out = self.forward_single(x)
                outputs.append(out)
            return outputs
        else:
            return self.forward_single(batch_x)

    def forward_single(self, x):
        # x: [N_hits, F] tensor
        edge_index = self.build_graph(x)

        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Track embeddings
        x = self.track_embed(x)

        # Get embeddings for clustering
        embeddings = x.detach().cpu().numpy()

        # HDBSCAN clustering
        if self.clusterer is None:
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )

        # Fit clusterer and predict
        self.clusterer.fit(embeddings)
        labels = self.clusterer.labels_

        # Convert to torch tensor and adjust labels
        labels = torch.from_numpy(labels).long().to(x.device)
        labels[labels == -1] = 0  # Noise to 0
        labels += 1  # Shift to make 0 available for noise

        # Ensure we have at least 4 hits per track (post-processing)
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            if label == 0:
                continue
            mask = (labels == label)
            if torch.sum(mask) < 4:
                labels[mask] = 0

        return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    best_val_acc = 0
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_hits = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()

            # Forward pass
            if isinstance(xb, list):
                preds = [model(x) for x in xb]
                loss = 0
                for pred, y in zip(preds, yb):
                    # Create pseudo-labels for training (simplified)
                    # In practice, we'd need a better way to handle this
                    mask = (y > 0)
                    if torch.sum(mask) > 0:
                        loss += F.cross_entropy(pred[mask].unsqueeze(0), y[mask].unsqueeze(0))
                loss = loss / len(xb)
            else:
                preds = model(xb)
                mask = (yb > 0)
                if torch.sum(mask) > 0:
                    loss = F.cross_entropy(preds[mask].unsqueeze(0), yb[mask].unsqueeze(0))
                else:
                    loss = torch.tensor(0.0, device=device)

            if loss > 0:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # Simple accuracy calculation (not perfect for this task)
            if isinstance(preds, list):
                for pred, y in zip(preds, yb):
                    mask = (y > 0)
                    if torch.sum(mask) > 0:
                        pred_labels = torch.argmax(pred[mask], dim=1)
                        total_correct += torch.sum(pred_labels == y[mask]).item()
                        total_hits += torch.sum(mask).item()
            else:
                mask = (yb > 0)
                if torch.sum(mask) > 0:
                    pred_labels = torch.argmax(preds[mask], dim=1)
                    total_correct += torch.sum(pred_labels == yb[mask]).item()
                    total_hits += torch.sum(mask).item()

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_hits if total_hits > 0 else 0

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_hits = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                if isinstance(xb, list):
                    preds = [model(x) for x in xb]
                    batch_loss = 0
                    for pred, y in zip(preds, yb):
                        mask = (y > 0)
                        if torch.sum(mask) > 0:
                            batch_loss += F.cross_entropy(pred[mask].unsqueeze(0), y[mask].unsqueeze(0))
                    batch_loss = batch_loss / len(xb)
                else:
                    preds = model(xb)
                    mask = (yb > 0)
                    if torch.sum(mask) > 0:
                        batch_loss = F.cross_entropy(preds[mask].unsqueeze(0), yb[mask].unsqueeze(0))
                    else:
                        batch_loss = torch.tensor(0.0, device=device)

                val_loss += batch_loss.item()

                if isinstance(preds, list):
                    for pred, y in zip(preds, yb):
                        mask = (y > 0)
                        if torch.sum(mask) > 0:
                            pred_labels = torch.argmax(pred[mask], dim=1)
                            val_correct += torch.sum(pred_labels == y[mask]).item()
                            val_hits += torch.sum(mask).item()
                else:
                    mask = (yb > 0)
                    if torch.sum(mask) > 0:
                        pred_labels = torch.argmax(preds[mask], dim=1)
                        val_correct += torch.sum(pred_labels == yb[mask]).item()
                        val_hits += torch.sum(mask).item()

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_hits if val_hits > 0 else 0

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        scheduler.step(val_acc)

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

    # Build model
    first_batch = next(iter(train_loader))
    view        = normalise_batch(first_batch, device=device)
    model       = make_model(view.batch_x).to(device)

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
        try:
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    view = normalise_batch(batch, device=device)
                    out  = model(view.batch_x)
                    assert_label_output(view.batch_x, out, allow_noise_label=True)
                    if i >= 4: # loop over 4 batches
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

