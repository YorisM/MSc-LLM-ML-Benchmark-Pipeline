
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
from scipy.spatial.distance import cdist

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
            "pin_memory": True,

            "collate": None,

            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, 4]
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Scale r, theta, z

        # Compute layer-wise statistics
        layer_stats = {}
        for X in Xs:
            for layer in np.unique(X[:, 3]):
                mask = X[:, 3] == layer
                if layer not in layer_stats:
                    layer_stats[layer] = []
                layer_stats[layer].append(X[mask, :3])

        self.layer_means = {}
        self.layer_stds = {}
        for layer, hits in layer_stats.items():
            hits = np.concatenate(hits, axis=0)
            self.layer_means[layer] = np.mean(hits, axis=0)
            self.layer_stds[layer] = np.std(hits, axis=0)

        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, 4]
        X = X.numpy() if isinstance(X, torch.Tensor) else X

        # Standardize spatial coordinates
        X[:, :3] = self.scaler.transform(X[:, :3])

        # Add layer-relative features
        layer_feat = np.zeros((X.shape[0], 3))
        for layer in np.unique(X[:, 3]):
            mask = X[:, 3] == layer
            if layer in self.layer_means:
                layer_feat[mask] = (X[mask, :3] - self.layer_means[layer]) / (self.layer_stds[layer] + 1e-6)

        # Combine features
        X = np.concatenate([X, layer_feat], axis=1)

        return torch.from_numpy(X).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Infer input features from example batch
        in_features = example_batch_x[0].shape[1] if isinstance(example_batch_x, list) else example_batch_x.shape[1]

        # Graph convolution layers
        self.conv1 = GCNConv(in_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Cluster embedding
        self.cluster_embedding = nn.Embedding(100, 64)  # Max 100 clusters per event

    def build_graph(self, X, batch_idx=None):
        # X: [N_hits, F]
        # Build k-NN graph
        with torch.no_grad():
            dist = cdist(X[:, :3].cpu().numpy(), X[:, :3].cpu().numpy())
            k = min(10, X.shape[0] - 1)
            knn_indices = np.argpartition(dist, k, axis=1)[:, :k+1]
            edge_index = torch.tensor(np.stack([
                np.repeat(np.arange(X.shape[0]), k+1),
                knn_indices.ravel()
            ]), dtype=torch.long)

            # Remove self-loops
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]

        return edge_index

    def forward(self, batch_x):
        # batch_x: list of [N_hits_i, F] tensors
        if isinstance(batch_x, list):
            # Process each event separately
            all_preds = []
            for x in batch_x:
                preds = self.forward_single_event(x)
                all_preds.append(preds)
            return all_preds
        else:
            return self.forward_single_event(batch_x)

    def forward_single_event(self, x):
        # x: [N_hits, F]
        device = x.device

        # Build graph
        edge_index = self.build_graph(x).to(device)

        # Graph convolutions
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = self.conv3(h, edge_index)  # [N_hits, 256]

        # Attention weights
        attn_weights = torch.softmax(self.attention(h), dim=0)  # [N_hits, 1]

        # Global context
        global_ctx = global_mean_pool(h * attn_weights, torch.zeros(h.shape[0], dtype=torch.long, device=device))

        # Combine features
        h = torch.cat([h, global_ctx.expand(h.shape[0], -1)], dim=1)

        # Predict cluster assignments
        logits = self.classifier(h).squeeze(-1)  # [N_hits]

        # Convert to cluster IDs
        unique_logits = torch.unique(logits)
        cluster_map = {v.item(): i+1 for i, v in enumerate(unique_logits)}
        preds = torch.tensor([cluster_map.get(v.item(), -1) for v in logits], device=device, dtype=torch.long)

        return preds

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        train_samples = 0

        for batch in train_loader:
            # Prepare batch
            if isinstance(batch, list):
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]
            else:
                Xs = [batch.x.to(device)]
                ys = [batch.y.to(device)]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(Xs)

            # Compute loss
            loss = 0
            correct = 0
            total = 0

            for out, y in zip(outputs, ys):
                # Create target for loss (cluster IDs)
                unique_y = torch.unique(y[y > 0])
                target = torch.zeros_like(y, dtype=torch.long)
                for i, uid in enumerate(unique_y):
                    target[y == uid] = i + 1

                # Compute loss
                loss += criterion(out[target > 0], target[target > 0] - 1)

                # Compute accuracy
                pred_cluster = out.argmax(dim=0) if out.dim() > 1 else out
                correct += (pred_cluster == target).sum().item()
                total += target.numel()

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_acc += correct / total
            train_samples += 1

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, list):
                    Xs, ys = batch
                    Xs = [x.to(device) for x in Xs]
                    ys = [y.to(device) for y in ys]
                else:
                    Xs = [batch.x.to(device)]
                    ys = [batch.y.to(device)]

                outputs = model(Xs)

                loss = 0
                correct = 0
                total = 0

                for out, y in zip(outputs, ys):
                    unique_y = torch.unique(y[y > 0])
                    target = torch.zeros_like(y, dtype=torch.long)
                    for i, uid in enumerate(unique_y):
                        target[y == uid] = i + 1

                    loss += criterion(out[target > 0], target[target > 0] - 1)

                    pred_cluster = out.argmax(dim=0) if out.dim() > 1 else out
                    correct += (pred_cluster == target).sum().item()
                    total += target.numel()

                epoch_val_loss += loss.item()
                epoch_val_acc += correct / total
                val_samples += 1

        # Update metrics
        train_loss = epoch_train_loss / train_samples
        val_loss = epoch_val_loss / val_samples
        train_acc = epoch_train_acc / train_samples
        val_acc = epoch_val_acc / val_samples

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping and learning rate scheduling
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
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

