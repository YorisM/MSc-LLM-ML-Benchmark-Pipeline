
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
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np

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
        # Stack all events for global statistics
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Scale r, theta, z

        # Compute per-layer statistics
        layer_stats = {}
        for X in Xs:
            layers = X[:, 3].astype(int)
            for i, layer in enumerate(layers):
                if layer not in layer_stats:
                    layer_stats[layer] = []
                layer_stats[layer].append(X[i, :3])

        self.layer_means = {}
        self.layer_stds = {}
        for layer, hits in layer_stats.items():
            hits = np.array(hits)
            self.layer_means[layer] = np.mean(hits, axis=0)
            self.layer_stds[layer] = np.std(hits, axis=0)

        return self

    def transform(self, X):
        # Apply global scaling
        X = X.clone()
        X[:, :3] = torch.from_numpy(self.scaler.transform(X[:, :3].numpy())).float()

        # Apply per-layer normalization
        for i in range(X.shape[0]):
            layer = int(X[i, 3].item())
            if layer in self.layer_means:
                X[i, :3] = (X[i, :3] - torch.from_numpy(self.layer_means[layer])) / (torch.from_numpy(self.layer_stds[layer]) + 1e-6)

        return X

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Graph-based approach
        self.conv1 = GCNConv(4, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Predict track ID

        # Cluster parameters
        self.eps = 0.5
        self.min_samples = 4

    def forward(self, batch_x):
        # Convert ragged batch to graph
        graphs = []
        for x in batch_x:
            # Create edges based on spatial proximity
            pos = x[:, :3]
            dist = torch.cdist(pos, pos)
            edges = (dist < self.eps).nonzero(as_tuple=False).t()
            edge_attr = dist[edges[0], edges[1]].unsqueeze(1)

            # Create graph
            graph = Data(x=x, edge_index=edges, edge_attr=edge_attr)
            graphs.append(graph)

        # Batch graphs
        batch = Batch.from_data_list(graphs)

        # Graph convolution
        x = F.relu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))
        x = F.relu(self.conv2(x, batch.edge_index, batch.edge_attr))
        x = F.relu(self.conv3(x, batch.edge_index, batch.edge_attr))

        # Global pooling
        x = global_mean_pool(x, batch.batch)

        # MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Get predictions per hit
        predictions = []
        start_idx = 0
        for i, graph in enumerate(graphs):
            num_hits = graph.x.shape[0]
            pred = x[start_idx:start_idx+num_hits]
            predictions.append(pred)
            start_idx += num_hits

        # Convert to track IDs using clustering
        final_predictions = []
        for i, x in enumerate(batch_x):
            pred = predictions[i].squeeze().cpu().numpy()
            # DBSCAN clustering
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(x[:, :3].cpu().numpy())
            labels = clustering.labels_
            # Map cluster labels to track IDs (0 is noise)
            track_ids = labels + 1  # Shift to make noise = 0
            track_ids[track_ids == 1] = 0  # Noise becomes 0
            final_predictions.append(torch.from_numpy(track_ids).long())

        return final_predictions

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        count = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            out = model(xb)

            # Calculate loss
            loss = 0
            for i in range(len(out)):
                # Convert track IDs to continuous values for loss calculation
                pred = out[i].float()
                true = yb[i].float()
                loss += criterion(pred, true)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            count += 1

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                out = model(xb)

                # Calculate loss
                loss = 0
                for i in range(len(out)):
                    pred = out[i].float()
                    true = yb[i].float()
                    loss += criterion(pred, true)

                epoch_val_loss += loss.item()
                val_count += 1

        # Calculate metrics
        avg_train_loss = epoch_train_loss / count
        avg_val_loss = epoch_val_loss / val_count

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Update scheduler
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

