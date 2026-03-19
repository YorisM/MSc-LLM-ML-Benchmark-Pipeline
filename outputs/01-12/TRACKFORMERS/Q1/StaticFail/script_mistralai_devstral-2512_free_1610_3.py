
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

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# Not needed for this solution

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
        # Collect all hits for scaling
        all_hits = torch.cat(Xs, dim=0).numpy()
        self.scaler.fit(all_hits[:, :3])  # Scale r, theta, z

        # Calculate layer statistics
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
        scaled = self.scaler.transform(X[:, :3].numpy())
        X = torch.from_numpy(scaled).float()

        # Add layer features
        layer_features = torch.zeros((X.shape[0], 3))
        for i, layer in enumerate(X_layer[:, 3]):
            if layer in self.layer_means:
                layer_features[i] = torch.from_numpy((X[i] - self.layer_means[layer]) / (self.layer_stds[layer] + 1e-6)).float()

        # Combine features
        X = torch.cat([X, layer_features], dim=1)
        return X

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Graph convolution layers
        self.conv1 = GCNConv(6, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)

        # MLP for classification
        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Clustering parameters
        self.cluster_head = nn.Linear(32, 16)

    def forward(self, batch_x):
        # Graph convolution forward pass
        x, edge_index = batch_x.x, batch_x.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        # Global pooling for graph-level features
        graph_features = global_mean_pool(x, batch_x.batch)

        # MLP processing
        x = self.mlp(x)
        return x, graph_features

    def predict_labels(self, batch_x):
        with torch.no_grad():
            # Get node embeddings
            embeddings, _ = self.forward(batch_x)
            embeddings = embeddings.cpu().numpy()

            # Cluster using HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=4,
                min_samples=1,
                cluster_selection_epsilon=0.5,
                metric='euclidean'
            )
            labels = clusterer.fit_predict(embeddings)

            # Convert to torch tensor and map noise to -1
            labels = torch.from_numpy(labels).to(device)
            labels[labels == -1] = -1  # Noise label
            return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.CrossEntropyLoss()

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            embeddings, graph_features = model(batch)

            # Create pseudo-labels using DBSCAN for training
            embeddings_np = embeddings.detach().cpu().numpy()
            clustering = DBSCAN(eps=0.5, min_samples=4).fit(embeddings_np)
            pseudo_labels = torch.from_numpy(clustering.labels_).to(device)

            # Filter out noise
            mask = pseudo_labels != -1
            if mask.sum() > 0:
                # Create classification targets
                unique_labels = torch.unique(pseudo_labels[mask])
                label_mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
                targets = torch.tensor([label_mapping[label.item()] for label in pseudo_labels[mask]]).to(device)

                # Train on embeddings
                output = model.mlp(embeddings[mask])
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                pred = output.argmax(dim=1)
                correct_train += (pred == targets).sum().item()
                total_train += targets.size(0)

        # Validation
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embeddings, _ = model(batch)

                # Get predictions
                pred_labels = model.predict_labels(batch)

                # Calculate accuracy (simple matching for validation)
                mask = batch.y != 0  # Ignore noise
                if mask.sum() > 0:
                    correct_val += (pred_labels[mask] == batch.y[mask]).sum().item()
                    total_val += mask.sum().item()

        # Store metrics
        train_loss.append(epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0)
        val_loss.append(epoch_val_loss / len(val_loader) if len(val_loader) > 0 else 0)
        train_acc.append(correct_train / total_train if total_train > 0 else 0)
        val_acc.append(correct_val / total_val if total_val > 0 else 0)

        # Update scheduler
        scheduler.step(epoch_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}")

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

