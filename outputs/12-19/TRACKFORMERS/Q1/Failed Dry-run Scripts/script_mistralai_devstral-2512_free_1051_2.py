
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

        # Compute layer-wise statistics
        layer_stats = {}
        for X in Xs:
            for layer_id in np.unique(X[:, 3]):
                mask = X[:, 3] == layer_id
                if layer_id not in layer_stats:
                    layer_stats[layer_id] = []
                layer_stats[layer_id].append(X[mask, :3])

        self.layer_means = {}
        self.layer_stds = {}
        for layer_id, data_list in layer_stats.items():
            layer_data = np.concatenate(data_list, axis=0)
            self.layer_means[layer_id] = np.mean(layer_data, axis=0)
            self.layer_stds[layer_id] = np.std(layer_data, axis=0)

        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, 4]
        X = X.numpy() if isinstance(X, torch.Tensor) else X
        scaled_features = self.scaler.transform(X[:, :3])

        # Layer-wise normalization
        normalized_features = np.zeros_like(scaled_features)
        for layer_id in np.unique(X[:, 3]):
            mask = X[:, 3] == layer_id
            if layer_id in self.layer_means:
                mean = self.layer_means[layer_id]
                std = self.layer_stds[layer_id]
                normalized_features[mask] = (scaled_features[mask] - mean) / (std + 1e-8)
            else:
                normalized_features[mask] = scaled_features[mask]

        # Add layer_id as one-hot encoding
        layer_ids = X[:, 3].astype(int)
        num_layers = len(self.layer_means) if self.layer_means else 10
        one_hot = np.zeros((len(layer_ids), num_layers))
        one_hot[np.arange(len(layer_ids)), layer_ids % num_layers] = 1

        # Combine features
        processed_X = np.concatenate([
            normalized_features,
            one_hot
        ], axis=1)

        return torch.from_numpy(processed_X).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Infer input features from example batch
        self.in_features = example_batch_x[0].shape[1] if isinstance(example_batch_x, list) else example_batch_x.shape[1]

        # Graph convolution layers
        self.conv1 = GCNConv(self.in_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # Output layer
        self.out = nn.Linear(32, 1)

        # Cluster embedding
        self.cluster_embed = nn.Embedding(100, 32)  # Max 100 clusters per event

    def forward(self, batch_x):
        # Handle both ragged list and single tensor
        if isinstance(batch_x, list):
            # Process each event separately
            all_preds = []
            for x in batch_x:
                preds = self._forward_single(x)
                all_preds.append(preds)
            return all_preds
        else:
            return self._forward_single(batch_x)

    def _forward_single(self, x):
        # x: [N_hits, F]
        n_hits = x.shape[0]

        # Create graph structure (complete graph for simplicity)
        edge_index = torch.combinations(torch.arange(n_hits), r=2).t().contiguous()

        # Graph convolutions
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))

        # Attention scores
        attn_scores = self.attention(h).squeeze()
        attn_weights = F.softmax(attn_scores, dim=0)

        # Weighted features
        weighted_h = h * attn_weights.unsqueeze(1)

        # Predict cluster assignments
        logits = self.out(weighted_h).squeeze()

        # DBSCAN clustering on learned features
        features = h.detach().cpu().numpy()
        clustering = DBSCAN(eps=0.5, min_samples=4).fit(features)
        labels = torch.from_numpy(clustering.labels_).to(x.device)

        # Map cluster labels to positive integers (noise = -1)
        unique_labels = torch.unique(labels)
        label_map = torch.zeros_like(unique_labels) - 1
        label_map[unique_labels != -1] = torch.arange(1, len(unique_labels[unique_labels != -1]) + 1)
        preds = label_map[labels]

        return preds

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        train_samples = 0

        for batch in train_loader:
            # Normalize batch
            view = normalise_batch(batch, device=device)
            x = view.batch_x
            y = view.batch_y

            optimizer.zero_grad()

            # Forward pass
            if isinstance(x, list):
                preds = [model(xi.unsqueeze(0)) for xi in x]
                loss = 0
                for pred, yi in zip(preds, y):
                    loss += criterion(pred, yi)
                loss /= len(x)
            else:
                preds = model(x)
                loss = criterion(preds, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * len(y)
            train_samples += len(y)

            # Accuracy calculation
            if isinstance(preds, list):
                correct = sum((pred.argmax(dim=1) == yi).sum().item() for pred, yi in zip(preds, y))
            else:
                correct = (preds.argmax(dim=1) == y).sum().item()
            epoch_train_acc += correct

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                x = view.batch_x
                y = view.batch_y

                if isinstance(x, list):
                    preds = [model(xi.unsqueeze(0)) for xi in x]
                    loss = 0
                    for pred, yi in zip(preds, y):
                        loss += criterion(pred, yi)
                    loss /= len(x)
                else:
                    preds = model(x)
                    loss = criterion(preds, y)

                epoch_val_loss += loss.item() * len(y)
                val_samples += len(y)

                if isinstance(preds, list):
                    correct = sum((pred.argmax(dim=1) == yi).sum().item() for pred, yi in zip(preds, y))
                else:
                    correct = (preds.argmax(dim=1) == y).sum().item()
                epoch_val_acc += correct

        # Update metrics
        train_loss = epoch_train_loss / train_samples
        val_loss = epoch_val_loss / val_samples
        train_acc = epoch_train_acc / train_samples
        val_acc = epoch_val_acc / val_samples

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Early stopping and learning rate scheduling
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model)
                break

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

