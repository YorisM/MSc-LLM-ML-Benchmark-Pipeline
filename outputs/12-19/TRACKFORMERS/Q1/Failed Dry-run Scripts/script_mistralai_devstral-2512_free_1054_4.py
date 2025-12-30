
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, torch, torch_geometric
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

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pickle

# -------- (OPTIONAL) CUSTOM DATASET  --------
def make_dataset(events, pre, train: bool, **kwargs):
    return EventDataset(events, pre, train=train)

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_means = None
        self.layer_stds = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:make_dataset",
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": "identity",

            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, 4]
        all_data = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_data[:, :3])  # Scale r, theta, z

        # Compute layer statistics
        layer_data = {}
        for X in Xs:
            layers = X[:, 3].numpy()
            for i, layer in enumerate(layers):
                if layer not in layer_data:
                    layer_data[layer] = []
                layer_data[layer].append(X[i, :3].numpy())

        self.layer_means = {}
        self.layer_stds = {}
        for layer, hits in layer_data.items():
            hits = np.array(hits)
            self.layer_means[layer] = np.mean(hits, axis=0)
            self.layer_stds[layer] = np.std(hits, axis=0)

        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, 4]
        X = X.numpy() if isinstance(X, torch.Tensor) else X

        # Scale features
        scaled = self.scaler.transform(X[:, :3])
        X_transformed = np.column_stack([scaled, X[:, 3]])

        # Add layer-relative features
        layer_features = []
        for i in range(len(X)):
            layer = X[i, 3]
            if layer in self.layer_means:
                rel_features = (X[i, :3] - self.layer_means[layer]) / (self.layer_stds[layer] + 1e-6)
                layer_features.append(rel_features)
            else:
                layer_features.append(np.zeros(3))

        X_transformed = np.column_stack([X_transformed, layer_features])

        return torch.from_numpy(X_transformed).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Determine input features from example batch
        self.in_features = example_batch_x[0].shape[1] if isinstance(example_batch_x, list) else example_batch_x.shape[1]

        # Graph convolution layers
        self.conv1 = GCNConv(self.in_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Output layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Predicts cluster ID

        # Layer normalization
        self.layer_norm = nn.LayerNorm(256)

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, batch_x):
        # Handle both ragged list and single tensor
        if isinstance(batch_x, list):
            # Process each event separately
            outputs = []
            for x in batch_x:
                outputs.append(self._forward_single(x))
            return outputs
        else:
            return self._forward_single(batch_x)

    def _forward_single(self, x):
        # x: [N_hits, F]
        N = x.shape[0]

        # Create graph structure (fully connected within event)
        edge_index = torch.combinations(torch.arange(N), r=2).t().contiguous()

        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.layer_norm(x)

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(x), dim=0)
        x = x * attention_weights

        # Global pooling
        x = global_mean_pool(x, torch.zeros(N, dtype=torch.long, device=x.device))

        # Final prediction
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Convert to cluster IDs (positive integers)
        cluster_ids = torch.round(x).long()
        cluster_ids = torch.clamp(cluster_ids, min=1)  # Ensure positive

        return cluster_ids

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        train_samples = 0

        for batch in train_loader:
            # Prepare data
            if isinstance(batch, list):
                # Ragged batch
                batch_x = [x.to(device) for x in batch[0]]
                batch_y = [y.to(device) for y in batch[1]]
            else:
                # Single batch
                batch_x = batch[0].to(device)
                batch_y = batch[1].to(device)

            optimizer.zero_grad()

            # Forward pass
            if isinstance(batch_x, list):
                outputs = []
                for x in batch_x:
                    outputs.append(model._forward_single(x))
                loss = 0
                for i, out in enumerate(outputs):
                    loss += criterion(out, batch_y[i])
                loss /= len(outputs)
            else:
                outputs = model._forward_single(batch_x)
                loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            if isinstance(outputs, list):
                acc = 0
                for i, out in enumerate(outputs):
                    pred = torch.argmax(out, dim=1)
                    acc += (pred == batch_y[i]).float().mean().item()
                acc /= len(outputs)
            else:
                pred = torch.argmax(outputs, dim=1)
                acc = (pred == batch_y).float().mean().item()

            epoch_train_loss += loss.item() * len(batch_y) if isinstance(batch_y, list) else loss.item() * batch_y.shape[0]
            epoch_train_acc += acc * (len(batch_y) if isinstance(batch_y, list) else batch_y.shape[0])
            train_samples += len(batch_y) if isinstance(batch_y, list) else batch_y.shape[0]

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                # Prepare data
                if isinstance(batch, list):
                    batch_x = [x.to(device) for x in batch[0]]
                    batch_y = [y.to(device) for y in batch[1]]
                else:
                    batch_x = batch[0].to(device)
                    batch_y = batch[1].to(device)

                # Forward pass
                if isinstance(batch_x, list):
                    outputs = []
                    for x in batch_x:
                        outputs.append(model._forward_single(x))
                    loss = 0
                    for i, out in enumerate(outputs):
                        loss += criterion(out, batch_y[i])
                    loss /= len(outputs)
                else:
                    outputs = model._forward_single(batch_x)
                    loss = criterion(outputs, batch_y)

                # Calculate accuracy
                if isinstance(outputs, list):
                    acc = 0
                    for i, out in enumerate(outputs):
                        pred = torch.argmax(out, dim=1)
                        acc += (pred == batch_y[i]).float().mean().item()
                    acc /= len(outputs)
                else:
                    pred = torch.argmax(outputs, dim=1)
                    acc = (pred == batch_y).float().mean().item()

                epoch_val_loss += loss.item() * len(batch_y) if isinstance(batch_y, list) else loss.item() * batch_y.shape[0]
                epoch_val_acc += acc * (len(batch_y) if isinstance(batch_y, list) else batch_y.shape[0])
                val_samples += len(batch_y) if isinstance(batch_y, list) else batch_y.shape[0]

        # Calculate epoch metrics
        train_loss = epoch_train_loss / train_samples
        train_acc = epoch_train_acc / train_samples
        val_loss = epoch_val_loss / val_samples
        val_acc = epoch_val_acc / val_samples

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model)

    return model, train_losses, val_losses, train_accs, val_accs

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

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

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

