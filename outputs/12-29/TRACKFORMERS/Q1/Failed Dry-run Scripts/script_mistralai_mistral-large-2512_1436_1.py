
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_max, scatter_add
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree

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
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Only scale r, theta, z

        # Compute layer statistics
        layers = all_X[:, 3].reshape(-1, 1)
        self.layer_mean = np.mean(layers)
        self.layer_std = np.std(layers)

        return self

    def transform(self, X):
        # X: [N_hits, 4] - r, theta, z, layer_id
        X = X.clone().numpy() if torch.is_tensor(X) else X

        # Normalize r, theta, z
        X[:, :3] = self.scaler.transform(X[:, :3])

        # Normalize layer_id
        if self.layer_std > 0:
            X[:, 3] = (X[:, 3] - self.layer_mean) / self.layer_std
        else:
            X[:, 3] = X[:, 3] - self.layer_mean

        return torch.from_numpy(X).float()

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

        # Determine input dimension from example batch
        if isinstance(example_batch_x, list):
            input_dim = example_batch_x[0].shape[1]
        else:
            input_dim = example_batch_x.shape[1]

        # Graph construction parameters
        self.k = 16  # Number of nearest neighbors
        self.edge_threshold = 0.5  # Distance threshold for edge creation

        # Embedding layers
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Graph layers
        self.conv1 = EdgeConv(64, 64)
        self.conv2 = EdgeConv(64, 64)
        self.conv3 = EdgeConv(64, 64)

        # Output layers
        self.cluster_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Will use for clustering
        )

        self.track_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128)  # Will use for track classification
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def build_graph(self, x):
        # x: [N_hits, F]
        N = x.size(0)

        # Build KDTree for efficient nearest neighbor search
        pos = x[:, :3].cpu().numpy()
        tree = KDTree(pos)

        # Find k-nearest neighbors
        distances, indices = tree.query(pos, k=self.k+1)  # +1 to include self
        indices = indices[:, 1:]  # Remove self
        distances = distances[:, 1:]

        # Create edge_index
        row = np.repeat(np.arange(N), self.k)
        col = indices.flatten()
        mask = distances.flatten() < self.edge_threshold
        row = row[mask]
        col = col[mask]

        edge_index = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long, device=x.device)

        return edge_index

    def forward(self, batch_x):
        # Handle ragged batch
        if isinstance(batch_x, list):
            outputs = []
            for x in batch_x:
                out = self.forward_single(x)
                outputs.append(out)
            return outputs
        else:
            return self.forward_single(batch_x)

    def forward_single(self, x):
        # x: [N_hits, F]
        N = x.size(0)

        # Build graph
        edge_index = self.build_graph(x)

        # Node features
        h = self.node_encoder(x)  # [N_hits, 64]

        # Graph convolutions
        h1 = self.conv1(h, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Attention mechanism
        attn_weights = self.attention(h3)
        attn_weights = F.softmax(attn_weights, dim=0)
        h_attn = h3 * attn_weights

        # Cluster predictions
        cluster_logits = self.cluster_head(h_attn).squeeze(-1)  # [N_hits]

        # Convert to track IDs using connected components
        # This is a simplified approach - in practice you might use more sophisticated clustering
        with torch.no_grad():
            # Create affinity matrix
            affinity = torch.sigmoid(cluster_logits.unsqueeze(0) - cluster_logits.unsqueeze(1))
            affinity = affinity * (affinity > 0.5).float()

            # Simple connected components (this is a placeholder - consider using HDBSCAN or similar)
            track_ids = torch.arange(1, N+1, device=x.device)
            for i in range(N):
                for j in range(i+1, N):
                    if affinity[i, j] > 0.5:
                        track_ids[j] = track_ids[i]

            # Reassign IDs to be contiguous
            unique_ids = torch.unique(track_ids)
            for new_id, old_id in enumerate(unique_ids, 1):
                track_ids[track_ids == old_id] = new_id

        return track_ids.long()

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_hits = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()

            # Forward pass
            if isinstance(xb, list):
                preds = [model(x) for x in xb]
                targets = yb
            else:
                preds = [model(xb)]
                targets = [yb]

            # Compute loss (simplified - in practice you'd need a proper clustering loss)
            loss = 0.0
            for pred, target in zip(preds, targets):
                # Skip noise hits (track_id == 0)
                mask = target > 0
                if mask.sum() == 0:
                    continue

                # Convert predictions to one-hot for comparison
                pred_ids = pred[mask]
                target_ids = target[mask]

                # Simple cross-entropy-like loss (this is a placeholder)
                # In practice, you'd want a proper clustering loss
                unique_preds = torch.unique(pred_ids)
                for p in unique_preds:
                    mask_p = pred_ids == p
                    if mask_p.sum() == 0:
                        continue
                    target_p = target_ids[mask_p]
                    unique_targets = torch.unique(target_p)
                    for t in unique_targets:
                        mask_t = target_p == t
                        if mask_t.sum() == 0:
                            continue
                        # Simple IoU-like loss
                        intersection = mask_t.sum()
                        union = (pred_ids == p).sum() + (target_ids == t).sum() - intersection
                        iou = intersection.float() / union.float()
                        loss -= iou

            if loss > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            # Simple accuracy computation (placeholder)
            with torch.no_grad():
                for pred, target in zip(preds, targets):
                    mask = target > 0
                    if mask.sum() == 0:
                        continue
                    pred_ids = pred[mask]
                    target_ids = target[mask]

                    # Count correct assignments (simplified)
                    correct = 0
                    for p in torch.unique(pred_ids):
                        mask_p = pred_ids == p
                        target_p = target_ids[mask_p]
                        if target_p.numel() == 0:
                            continue
                        # Majority vote
                        values, counts = torch.unique(target_p, return_counts=True)
                        majority = values[counts.argmax()]
                        correct += (target_p == majority).sum().item()

                    total_correct += correct
                    total_hits += mask.sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_hits if total_hits > 0 else 0.0
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_hits = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                if isinstance(xb, list):
                    preds = [model(x) for x in xb]
                    targets = yb
                else:
                    preds = [model(xb)]
                    targets = [yb]

                # Compute validation loss
                val_loss = 0.0
                for pred, target in zip(preds, targets):
                    mask = target > 0
                    if mask.sum() == 0:
                        continue
                    pred_ids = pred[mask]
                    target_ids = target[mask]

                    unique_preds = torch.unique(pred_ids)
                    for p in unique_preds:
                        mask_p = pred_ids == p
                        if mask_p.sum() == 0:
                            continue
                        target_p = target_ids[mask_p]
                        unique_targets = torch.unique(target_p)
                        for t in unique_targets:
                            mask_t = target_p == t
                            if mask_t.sum() == 0:
                                continue
                            intersection = mask_t.sum()
                            union = (pred_ids == p).sum() + (target_ids == t).sum() - intersection
                            iou = intersection.float() / union.float()
                            val_loss -= iou

                total_val_loss += val_loss.item()

                # Compute validation accuracy
                for pred, target in zip(preds, targets):
                    mask = target > 0
                    if mask.sum() == 0:
                        continue
                    pred_ids = pred[mask]
                    target_ids = target[mask]

                    correct = 0
                    for p in torch.unique(pred_ids):
                        mask_p = pred_ids == p
                        target_p = target_ids[mask_p]
                        if target_p.numel() == 0:
                            continue
                        values, counts = torch.unique(target_p, return_counts=True)
                        majority = values[counts.argmax()]
                        correct += (target_p == majority).sum().item()

                    total_val_correct += correct
                    total_val_hits += mask.sum().item()

        val_loss = total_val_loss / len(val_loader)
        val_acc = total_val_correct / total_val_hits if total_val_hits > 0 else 0.0
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
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

    # added debugging
    print("NB meta:", view.meta, flush=True)
    print("type(batch_x):", type(view.batch_x), flush=True)
    if isinstance(view.batch_x, list) and len(view.batch_x) > 0:
        print("type(batch_x[0]):", type(view.batch_x[0]), flush=True)
        if isinstance(view.batch_x[0], (tuple, list)):
            print("len(batch_x[0]):", len(view.batch_x[0]), flush=True)
            print("inner types:", [type(z) for z in view.batch_x[0]], flush=True)

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

