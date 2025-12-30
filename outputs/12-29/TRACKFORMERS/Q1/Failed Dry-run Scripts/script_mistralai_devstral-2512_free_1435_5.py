
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
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np

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
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": "ragged_xy",
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Collect all hits for scaling
        all_hits = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_hits[:, :3])  # Scale r, theta, z

        # Encode layer_id as one-hot
        unique_layers = np.unique(all_hits[:, 3])
        self.n_layers = len(unique_layers)
        self.layer_encoder = {layer: i for i, layer in enumerate(unique_layers)}
        return self

    def transform(self, X):
        # X: [N_hits, 4] (r, theta, z, layer_id)
        X_np = X.numpy()
        scaled = self.scaler.transform(X_np[:, :3])
        layer_encoded = np.zeros((X_np.shape[0], self.n_layers))
        layer_encoded[np.arange(X_np.shape[0]), [self.layer_encoder[l] for l in X_np[:, 3]]] = 1
        return torch.FloatTensor(np.concatenate([scaled, layer_encoded], axis=1))

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # example_batch_x is a list of tensors, each [N_hits, F]
        # Get feature dimension from first event
        self.input_dim = example_batch_x[0].shape[1]

        # Graph convolution layers
        self.conv1 = GCNConv(self.input_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)

        # MLP for final classification
        self.mlp = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predict cluster ID
        )

        # DBSCAN for post-processing
        self.dbscan = DBSCAN(eps=0.5, min_samples=4)

    def forward(self, batch_x):
        # batch_x is list of [N_hits, F] tensors
        outputs = []
        for event in batch_x:
            # Create graph for this event
            x = event  # [N_hits, F]
            edge_index = self._build_knn_graph(x)  # [2, E]

            # Graph convolutions
            h = F.relu(self.conv1(x, edge_index))
            h = F.relu(self.conv2(h, edge_index))
            h = self.conv3(h, edge_index)  # [N_hits, 16]

            # Global pooling for event context
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            global_feat = global_mean_pool(h, batch)  # [1, 16]

            # Combine local and global features
            h = torch.cat([h, global_feat.expand(h.size(0), -1)], dim=1)

            # Predict cluster ID
            logits = self.mlp(h).squeeze(-1)  # [N_hits]

            # DBSCAN clustering
            with torch.no_grad():
                # Convert to numpy for DBSCAN
                features = h.cpu().numpy()
                clusters = self.dbscan.fit_predict(features)

                # Map cluster IDs to positive integers (DBSCAN uses -1 for noise)
                unique_clusters = np.unique(clusters[clusters >= 0])
                cluster_map = {c: i+1 for i, c in enumerate(unique_clusters)}
                pred_labels = np.array([cluster_map.get(c, -1) for c in clusters])

            outputs.append(torch.from_numpy(pred_labels).to(event.device))

        return outputs

    def _build_knn_graph(self, x, k=10):
        # x: [N_hits, F]
        # Compute pairwise distances
        dist = torch.cdist(x, x)  # [N_hits, N_hits]

        # Get k nearest neighbors for each node
        _, indices = torch.topk(dist, k=k+1, largest=False)  # +1 to include self
        indices = indices[:, 1:]  # Remove self-loop

        # Create edge_index
        src = torch.arange(x.size(0), device=x.device).repeat_interleave(k)
        dst = indices.flatten()
        edge_index = torch.stack([src, dst], dim=0)  # [2, N_hits*k]

        return edge_index

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
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        total_hits = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()

            # Forward pass
            out = model(xb)

            # Calculate loss (treat as classification problem)
            loss = 0
            for i in range(len(out)):
                # Get unique labels in this event
                unique_labels = torch.unique(yb[i])
                n_classes = len(unique_labels)

                # Create target tensor with class indices
                target = torch.zeros_like(yb[i])
                for j, label in enumerate(unique_labels):
                    target[yb[i] == label] = j

                # Predicted labels to class indices
                pred = out[i]
                pred_classes = torch.zeros_like(pred)
                for j, label in enumerate(unique_labels):
                    pred_classes[pred == label] = j

                # Compute loss
                loss += criterion(pred_classes.float(), target.long())

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            # Calculate accuracy
            for i in range(len(out)):
                pred = out[i]
                target = yb[i]

                # Count correct predictions (ignoring noise)
                mask = target > 0
                if mask.sum() > 0:
                    correct = (pred[mask] == target[mask]).float().sum()
                    epoch_train_acc += correct.item()
                    total_hits += mask.sum().item()

        train_loss.append(epoch_train_loss / len(train_loader))
        train_acc.append(epoch_train_acc / total_hits if total_hits > 0 else 0)

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        total_hits = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                out = model(xb)

                loss = 0
                for i in range(len(out)):
                    unique_labels = torch.unique(yb[i])
                    n_classes = len(unique_labels)

                    target = torch.zeros_like(yb[i])
                    for j, label in enumerate(unique_labels):
                        target[yb[i] == label] = j

                    pred = out[i]
                    pred_classes = torch.zeros_like(pred)
                    for j, label in enumerate(unique_labels):
                        pred_classes[pred == label] = j

                    loss += criterion(pred_classes.float(), target.long())

                epoch_val_loss += loss.item()

                for i in range(len(out)):
                    pred = out[i]
                    target = yb[i]

                    mask = target > 0
                    if mask.sum() > 0:
                        correct = (pred[mask] == target[mask]).float().sum()
                        epoch_val_acc += correct.item()
                        total_hits += mask.sum().item()

        val_loss.append(epoch_val_loss / len(val_loader))
        val_acc.append(epoch_val_acc / total_hits if total_hits > 0 else 0)

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

