
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
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.loader import DataLoader
import hdbscan
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.coord_scaler = StandardScaler()
        self.layer_scaler = StandardScaler()
        self.knn_k = 20
        self.max_radius = 5.0

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:GraphDataset",
            "dataset_kwargs": {"preprocessor": self},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 64}
        }

    def fit(self, Xs):
        all_coords = []
        all_layers = []
        for X in Xs:
            # Convert to Cartesian coordinates
            x = X[:, 0] * torch.cos(X[:, 1])  # r*cos(theta)
            y = X[:, 0] * torch.sin(X[:, 1])  # r*sin(theta)
            z = X[:, 2]                       # z
            coords = torch.stack([x, y, z], dim=1)  # [N, 3]
            all_coords.append(coords)
            all_layers.append(X[:, 3:4])  # layer_id
        all_coords = torch.cat(all_coords, dim=0).numpy()
        all_layers = torch.cat(all_layers, dim=0).numpy()
        self.coord_scaler.fit(all_coords)
        self.layer_scaler.fit(all_layers)
        return self

    def transform(self, X):
        # Convert cylindrical to Cartesian
        r = X[:, 0]
        theta = X[:, 1]
        z = X[:, 2]
        layer = X[:, 3:4]

        x = r * torch.cos(theta)  # [N]
        y = r * torch.sin(theta)  # [N]
        z = z                     # [N]

        coords = torch.stack([x, y, z], dim=1)  # [N, 3]
        coords_np = coords.numpy()
        layer_np = layer.numpy()

        # Normalize
        coords_norm = self.coord_scaler.transform(coords_np)  # [N, 3]
        layer_norm = self.layer_scaler.transform(layer_np)    # [N, 1]

        # Combine features
        features = np.concatenate([coords_norm, layer_norm], axis=1)  # [N, 4]
        return torch.FloatTensor(features)  # [N, 4]


# -------- CUSTOM DATASET FOR GRAPH CONSTRUCTION --------
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, events, preprocessor, train=True):
        self.events = events
        self.pre = preprocessor
        self.train = train

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        X, y = split_X_y(self.events[idx])
        X = self.pre.transform(X)  # [N, 4]

        # Build graph edges using k-NN in normalized Cartesian space
        coords = X[:, :3].numpy()  # [N, 3]
        tree = cKDTree(coords)

        # k-NN edges
        k = min(self.pre.knn_k, len(coords)-1)
        if k > 0:
            distances, indices = tree.query(coords, k=k+1)  # +1 to include self
            indices = indices[:, 1:]  # remove self
            distances = distances[:, 1:]

            # Create edge_index
            rows = np.repeat(np.arange(len(coords)), k)
            cols = indices.flatten()
            edge_index = torch.stack([
                torch.LongTensor(rows),
                torch.LongTensor(cols)
            ], dim=0)  # [2, E]

            # Add reciprocal edges for undirected graph
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Create PyG Data object
        data = Data(
            x=X,
            y=torch.LongTensor(y),
            edge_index=edge_index,
            num_nodes=len(X)
        )
        return data


def make_preprocessor():
    return MyPreprocessor()


# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        input_dim = example_batch_x.x.size(1)  # 4 features
        hidden_dim = 128
        embedding_dim = 64

        # GNN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        self.bn3 = BatchNorm(embedding_dim)

        # Attention pooling for global context
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Tanh()
        )

        # Final projection for pairwise affinity
        self.affinity = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        # data: PyG Batch object
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GNN forward pass
        x = self.conv1(x, edge_index)  # [N, hidden_dim]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)  # [N, hidden_dim]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)  # [N, embedding_dim]
        x = self.bn3(x)
        embeddings = F.normalize(x, p=2, dim=1)  # L2 normalize

        return embeddings

    def predict_labels(self, batch_x):
        # batch_x: PyG Batch object
        with torch.no_grad():
            embeddings = self.forward(batch_x)  # [N_total, embedding_dim]
            embeddings_np = embeddings.cpu().numpy()

            # Use HDBSCAN for clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=4,
                min_samples=3,
                cluster_selection_epsilon=0.5,
                metric='euclidean'
            )
            labels = clusterer.fit_predict(embeddings_np)

            # Convert to torch tensor
            labels = torch.from_numpy(labels).to(batch_x.x.device)

            # Ensure noise is -1
            labels[labels == -1] = -1

            # Remap labels to be contiguous per event
            batch = batch_x.batch
            remapped_labels = torch.full_like(labels, -1)

            for b in range(batch.max().item() + 1):
                mask = (batch == b)
                event_labels = labels[mask]
                unique_labels = torch.unique(event_labels)
                unique_labels = unique_labels[unique_labels != -1]

                # Create mapping
                mapping = {int(old): new for new, old in enumerate(unique_labels)}
                for old, new in mapping.items():
                    remapped_labels[mask & (labels == old)] = new

            return remapped_labels  # [N_total]


def make_model(example_batch_x):
    return HitClassifier(example_batch_x)


# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    patience = 10
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Custom contrastive loss
    def contrastive_loss(embeddings, batch, margin=1.0):
        # Simple contrastive loss using batch structure
        # Pull same-batch embeddings together, push different batches apart
        batch = batch.unsqueeze(0)  # [1, N]
        same_batch = (batch == batch.T).float()  # [N, N]

        dist = torch.cdist(embeddings, embeddings, p=2)  # [N, N]

        pos_loss = (dist ** 2) * same_batch
        neg_loss = F.relu(margin - dist) ** 2 * (1 - same_batch)

        return (pos_loss.mean() + neg_loss.mean()) / 2

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            embeddings = model(batch)
            loss = contrastive_loss(embeddings, batch.batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

            # Compute accuracy using predicted labels
            with torch.no_grad():
                pred_labels = model.predict_labels(batch)
                true_labels = batch.y

                # Only evaluate on non-noise true hits
                mask = (true_labels > 0)
                if mask.sum() > 0:
                    pred_masked = pred_labels[mask]
                    true_masked = true_labels[mask]

                    # Compute Hungarian matching for accuracy
                    from scipy.optimize import linear_sum_assignment

                    unique_pred = torch.unique(pred_masked[pred_masked != -1])
                    unique_true = torch.unique(true_masked)

                    if len(unique_pred) > 0 and len(unique_true) > 0:
                        cost_matrix = torch.zeros((len(unique_true), len(unique_pred)))

                        for i, t in enumerate(unique_true):
                            for j, p in enumerate(unique_pred):
                                mask_t = (true_masked == t)
                                mask_p = (pred_masked == p)
                                intersection = (mask_t & mask_p).sum().float()
                                union = (mask_t | mask_p).sum().float()
                                if union > 0:
                                    cost_matrix[i, j] = 1 - intersection / union

                        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
                        matched_cost = cost_matrix[row_ind, col_ind].sum().item()
                        acc = 1 - matched_cost / len(true_masked)
                        correct += (acc * len(true_masked))
                        total += len(true_masked)

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0

        # Validation
        model.eval()
        epoch_val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embeddings = model(batch)
                loss = contrastive_loss(embeddings, batch.batch)
                epoch_val_loss += loss.item()

                pred_labels = model.predict_labels(batch)
                true_labels = batch.y

                mask = (true_labels > 0)
                if mask.sum() > 0:
                    pred_masked = pred_labels[mask]
                    true_masked = true_labels[mask]

                    unique_pred = torch.unique(pred_masked[pred_masked != -1])
                    unique_true = torch.unique(true_masked)

                    if len(unique_pred) > 0 and len(unique_true) > 0:
                        cost_matrix = torch.zeros((len(unique_true), len(unique_pred)))

                        for i, t in enumerate(unique_true):
                            for j, p in enumerate(unique_pred):
                                mask_t = (true_masked == t)
                                mask_p = (pred_masked == p)
                                intersection = (mask_t & mask_p).sum().float()
                                union = (mask_t | mask_p).sum().float()
                                if union > 0:
                                    cost_matrix[i, j] = 1 - intersection / union

                        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
                        matched_cost = cost_matrix[row_ind, col_ind].sum().item()
                        acc = 1 - matched_cost / len(true_masked)
                        val_correct += (acc * len(true_masked))
                        val_total += len(true_masked)

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step()

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}")

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

