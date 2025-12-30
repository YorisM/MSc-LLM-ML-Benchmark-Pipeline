
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
from scipy.spatial.distance import pdist, squareform
import networkx as nx

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_encoder = None
        self.layer_scaler = StandardScaler()

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

        # Layer encoding
        unique_layers = np.unique(all_hits[:, 3])
        self.layer_encoder = {l: i for i, l in enumerate(unique_layers)}
        layer_values = np.array([self.layer_encoder[l] for l in all_hits[:, 3]])
        self.layer_scaler.fit(layer_values.reshape(-1, 1))
        return self

    def transform(self, X):
        # X: [N_hits, 4] numpy array
        X = X.numpy() if isinstance(X, torch.Tensor) else X
        scaled_features = self.scaler.transform(X[:, :3])
        layer_encoded = self.layer_scaler.transform(
            np.array([self.layer_encoder[l] for l in X[:, 3]]).reshape(-1, 1)
        )
        return torch.FloatTensor(np.hstack([scaled_features, layer_encoded]))

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # example_batch_x is a list of tensors [N_hits_i, 4]
        # We'll use the first event to determine input dimensions
        input_dim = example_batch_x[0].shape[1]

        # Graph construction parameters
        self.k = 8  # Number of nearest neighbors for graph construction

        # Model architecture
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Predict track ID as continuous value

        # Cluster assignment head
        self.cluster_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def build_graph(self, x, batch=None):
        # x: [N_hits, 4] tensor
        # Build k-NN graph
        dist = squareform(pdist(x.numpy()))
        knn_indices = np.argpartition(dist, self.k, axis=1)[:, :self.k+1]
        knn_indices = knn_indices[:, 1:]  # Exclude self

        # Create edge index
        edge_index = []
        for i in range(x.shape[0]):
            for j in knn_indices[i]:
                edge_index.append([i, j])
        edge_index = torch.LongTensor(edge_index).t().contiguous()

        # Create PyG Data object
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, batch=batch)

    def forward(self, batch_x):
        # batch_x is a list of tensors [N_hits_i, 4]
        # We need to process each event separately
        all_predictions = []

        for event_x in batch_x:
            # Build graph for this event
            data = self.build_graph(event_x)

            # GCN layers
            x = F.relu(self.conv1(data.x, data.edge_index))
            x = F.relu(self.conv2(x, data.edge_index))
            x = F.relu(self.conv3(x, data.edge_index))

            # Global pooling to get event-level features
            event_features = global_mean_pool(x, data.batch)

            # Expand event features to each hit
            event_features = event_features.repeat(x.shape[0], 1)

            # Combine with local features
            combined = torch.cat([x, event_features], dim=1)

            # Predict track features
            track_features = F.relu(self.fc1(combined))
            track_features = F.relu(self.fc2(track_features))

            # Predict cluster assignment
            cluster_logits = self.cluster_head(track_features).squeeze()

            # Convert to track IDs (simple thresholding for now)
            # In practice, we'd want to use a clustering algorithm here
            # For now, we'll just use the continuous prediction
            predictions = cluster_logits

            # Convert to integer labels (this is a simplified approach)
            # In a real implementation, we'd use a proper clustering algorithm
            unique_preds = torch.unique(predictions)
            label_mapping = {v.item(): i+1 for i, v in enumerate(unique_preds)}
            int_labels = torch.tensor([label_mapping.get(p.item(), -1) for p in predictions],
                                    dtype=torch.long)

            all_predictions.append(int_labels)

        # Return concatenated predictions for all events
        return torch.cat(all_predictions)

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.MSELoss()  # Using MSE for continuous prediction

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        train_samples = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            # Convert labels to continuous values for training
            # This is a simplified approach - in practice we'd want proper clustering
            unique_labels = torch.unique(yb)
            label_mapping = {l.item(): float(i) for i, l in enumerate(unique_labels)}
            y_continuous = torch.tensor([label_mapping.get(l.item(), -1.0) for l in yb],
                                      dtype=torch.float32, device=device)

            optimizer.zero_grad()
            out = model(xb)

            # Calculate loss
            loss = criterion(out.float(), y_continuous)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * len(yb)
            train_samples += len(yb)

            # Calculate accuracy (simplified)
            # Convert predictions to labels
            unique_preds = torch.unique(out)
            pred_mapping = {v.item(): i+1 for i, v in enumerate(unique_preds)}
            pred_labels = torch.tensor([pred_mapping.get(p.item(), -1) for p in out],
                                     dtype=torch.long, device=device)

            # Compare with ground truth
            correct = (pred_labels == yb).float().sum()
            epoch_train_acc += correct.item()

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                # Convert labels to continuous values
                unique_labels = torch.unique(yb)
                label_mapping = {l.item(): float(i) for i, l in enumerate(unique_labels)}
                y_continuous = torch.tensor([label_mapping.get(l.item(), -1.0) for l in yb],
                                          dtype=torch.float32, device=device)

                out = model(xb)
                loss = criterion(out.float(), y_continuous)

                epoch_val_loss += loss.item() * len(yb)
                val_samples += len(yb)

                # Calculate accuracy
                unique_preds = torch.unique(out)
                pred_mapping = {v.item(): i+1 for i, v in enumerate(unique_preds)}
                pred_labels = torch.tensor([pred_mapping.get(p.item(), -1) for p in out],
                                         dtype=torch.long, device=device)

                correct = (pred_labels == yb).float().sum()
                epoch_val_acc += correct.item()

        # Update scheduler
        scheduler.step(epoch_val_loss)

        # Store metrics
        train_losses.append(epoch_train_loss / train_samples)
        val_losses.append(epoch_val_loss / val_samples)
        train_accs.append(epoch_train_acc / train_samples)
        val_accs.append(epoch_val_acc / val_samples)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
              f"Train Acc: {train_accs[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

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

