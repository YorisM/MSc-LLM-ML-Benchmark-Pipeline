
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import detect_and_assert_lane, assert_label_output_by_lane, build_dataset, build_dataloader
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
import networkx as nx

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_encoder = None
        self.layer_ids = None

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
        all_hits = np.concatenate(Xs)
        self.scaler.fit(all_hits[:, :3])  # Scale r, theta, z

        # Encode layer_id as one-hot
        unique_layers = np.unique(np.concatenate([x[:, 3] for x in Xs]))
        self.layer_ids = unique_layers
        return self

    def transform(self, X):
        # Scale coordinates
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        X_np[:, :3] = self.scaler.transform(X_np[:, :3])

        # One-hot encode layer_id
        layer_onehot = np.zeros((X_np.shape[0], len(self.layer_ids)))
        for i, lid in enumerate(self.layer_ids):
            layer_onehot[:, i] = (X_np[:, 3] == lid).astype(np.float32)

        # Combine features
        X_transformed = np.concatenate([X_np[:, :3], layer_onehot], axis=1)
        return torch.from_numpy(X_transformed).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Determine input features from example batch
        if isinstance(example_batch_x, list):
            # Torch ragged lane
            self.input_dim = example_batch_x[0].shape[1]
        else:
            # PyG lane
            self.input_dim = example_batch_x.x.shape[1]

        # GNN layers
        self.conv1 = GCNConv(self.input_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)

        # MLP for final classification
        self.mlp = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predict cluster ID offset
        )

        # Track embedding
        self.track_embedding = nn.Embedding(100, 16)  # Max 100 tracks per event

    def forward(self, batch_x):
        if isinstance(batch_x, list):
            # Torch ragged lane
            return self._forward_ragged(batch_x)
        else:
            # PyG lane
            return self._forward_pyg(batch_x)

    def _forward_ragged(self, Xs):
        # Process each event separately
        all_preds = []
        for X in Xs:
            # Create graph for this event
            num_hits = X.shape[0]
            edge_index = self._build_edges(X)

            # Convert to PyG data
            data = Data(x=X, edge_index=edge_index)

            # Process through GNN
            x = F.relu(self.conv1(data.x, data.edge_index))
            x = F.relu(self.conv2(x, data.edge_index))
            x = self.conv3(x, data.edge_index)

            # Get node features
            node_features = x

            # Predict cluster assignments
            logits = self.mlp(node_features).squeeze(-1)

            # Convert to cluster IDs
            preds = self._logits_to_clusters(logits, num_hits)
            all_preds.append(preds)

        return all_preds

    def _forward_pyg(self, data):
        # Process through GNN
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = self.conv3(x, data.edge_index)

        # Get node features
        node_features = x

        # Predict cluster assignments
        logits = self.mlp(node_features).squeeze(-1)

        # Convert to cluster IDs
        preds = self._logits_to_clusters(logits, data.x.shape[0])
        return preds

    def _build_edges(self, X, k=5):
        # Build k-NN graph
        dist = torch.cdist(X[:, :3], X[:, :3])  # Only use spatial coords
        _, indices = torch.topk(dist, k=k+1, largest=False)  # +1 to exclude self
        indices = indices[:, 1:]  # Remove self-loops

        # Create edge index
        src = torch.arange(X.shape[0]).unsqueeze(1).repeat(1, k).flatten()
        dst = indices.flatten()
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index

    def _logits_to_clusters(self, logits, num_hits):
        # Convert logits to cluster IDs using DBSCAN
        with torch.no_grad():
            # Normalize logits
            logits_np = logits.cpu().numpy()
            logits_np = (logits_np - logits_np.min()) / (logits_np.max() - logits_np.min() + 1e-8)

            # Reshape for DBSCAN
            features = logits_np.reshape(-1, 1)

            # Cluster
            clustering = DBSCAN(eps=0.1, min_samples=4).fit(features)
            labels = clustering.labels_

            # Convert to tensor and handle noise
            labels = torch.from_numpy(labels).to(logits.device)
            labels[labels == -1] = -1  # Noise
            return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    best_val_loss = float('inf')
    best_model = None
    patience = 5
    no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        count = 0

        for batch in train_loader:
            if isinstance(batch, list):
                # Torch ragged lane
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                optimizer.zero_grad()
                outs = model(Xs)

                loss = 0
                correct = 0
                total = 0

                for out, y in zip(outs, ys):
                    # Convert to classification problem
                    unique_labels = torch.unique(y[y != -1])
                    if len(unique_labels) == 0:
                        continue

                    # Create mapping from true labels to 0..K-1
                    label_map = {label.item(): i for i, label in enumerate(unique_labels)}
                    y_mapped = torch.tensor([label_map.get(label.item(), -1) for label in y], device=y.device)

                    # Filter out noise
                    mask = y_mapped != -1
                    if mask.sum() == 0:
                        continue

                    # Convert predictions to same space
                    out_mapped = torch.tensor([label_map.get(pred.item(), -1) for pred in out], device=out.device)
                    out_mapped = out_mapped[mask]
                    y_mapped = y_mapped[mask]

                    # Compute loss
                    loss += criterion(out_mapped.float(), y_mapped.long())
                    correct += (out_mapped == y_mapped).sum().item()
                    total += y_mapped.shape[0]

                if total > 0:
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                    epoch_train_acc += correct / total
                    count += 1

            else:
                # PyG lane
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)

                # Convert to classification problem
                unique_labels = torch.unique(batch.y[batch.y != -1])
                if len(unique_labels) == 0:
                    continue

                # Create mapping from true labels to 0..K-1
                label_map = {label.item(): i for i, label in enumerate(unique_labels)}
                y_mapped = torch.tensor([label_map.get(label.item(), -1) for label in batch.y], device=batch.y.device)

                # Filter out noise
                mask = y_mapped != -1
                if mask.sum() == 0:
                    continue

                # Convert predictions to same space
                out_mapped = torch.tensor([label_map.get(pred.item(), -1) for pred in out], device=out.device)
                out_mapped = out_mapped[mask]
                y_mapped = y_mapped[mask]

                # Compute loss
                loss = criterion(out_mapped.float(), y_mapped.long())
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                epoch_train_acc += (out_mapped == y_mapped).sum().item() / y_mapped.shape[0]
                count += 1

        if count > 0:
            train_loss.append(epoch_train_loss / count)
            train_acc.append(epoch_train_acc / count)
        else:
            train_loss.append(0)
            train_acc.append(0)

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, list):
                    # Torch ragged lane
                    Xs, ys = batch
                    Xs = [x.to(device) for x in Xs]
                    ys = [y.to(device) for y in ys]

                    outs = model(Xs)

                    loss = 0
                    correct = 0
                    total = 0

                    for out, y in zip(outs, ys):
                        # Convert to classification problem
                        unique_labels = torch.unique(y[y != -1])
                        if len(unique_labels) == 0:
                            continue

                        # Create mapping from true labels to 0..K-1
                        label_map = {label.item(): i for i, label in enumerate(unique_labels)}
                        y_mapped = torch.tensor([label_map.get(label.item(), -1) for label in y], device=y.device)

                        # Filter out noise
                        mask = y_mapped != -1
                        if mask.sum() == 0:
                            continue

                        # Convert predictions to same space
                        out_mapped = torch.tensor([label_map.get(pred.item(), -1) for pred in out], device=out.device)
                        out_mapped = out_mapped[mask]
                        y_mapped = y_mapped[mask]

                        # Compute loss
                        loss += criterion(out_mapped.float(), y_mapped.long())
                        correct += (out_mapped == y_mapped).sum().item()
                        total += y_mapped.shape[0]

                    if total > 0:
                        epoch_val_loss += loss.item()
                        epoch_val_acc += correct / total
                        count += 1

                else:
                    # PyG lane
                    batch = batch.to(device)
                    out = model(batch)

                    # Convert to classification problem
                    unique_labels = torch.unique(batch.y[batch.y != -1])
                    if len(unique_labels) == 0:
                        continue

                    # Create mapping from true labels to 0..K-1
                    label_map = {label.item(): i for i, label in enumerate(unique_labels)}
                    y_mapped = torch.tensor([label_map.get(label.item(), -1) for label in batch.y], device=batch.y.device)

                    # Filter out noise
                    mask = y_mapped != -1
                    if mask.sum() == 0:
                        continue

                    # Convert predictions to same space
                    out_mapped = torch.tensor([label_map.get(pred.item(), -1) for pred in out], device=out.device)
                    out_mapped = out_mapped[mask]
                    y_mapped = y_mapped[mask]

                    # Compute loss
                    loss = criterion(out_mapped.float(), y_mapped.long())

                    epoch_val_loss += loss.item()
                    epoch_val_acc += (out_mapped == y_mapped).sum().item() / y_mapped.shape[0]
                    count += 1

        if count > 0:
            val_loss.append(epoch_val_loss / count)
            val_acc.append(epoch_val_acc / count)
        else:
            val_loss.append(0)
            val_acc.append(0)

        # Early stopping and learning rate scheduling
        current_val_loss = val_loss[-1]
        scheduler.step(current_val_loss)

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

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
    if mode == "torch_ragged_xy":
        Xs, ys = first_batch
        Xs = [x.to(device) for x in Xs]
        ys = [y.to(device) for y in ys]
        model = make_model(Xs).to(device)
        out = model(Xs)
    elif mode == "pyg_batch":
        G = first_batch.to(device)
        model = make_model(G).to(device)
        out = model(G)

    assert_label_output_by_lane(mode, first_batch, out, allow_noise_label=True)

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
                        out = trained_model(Xs)
                    elif mode == "pyg_batch":
                        G = batch.to(device)
                        out = trained_model(G)
                    else:
                        raise RuntimeError(f"Unknown lane mode: {mode}")
                    assert_label_output_by_lane(mode, batch, out, allow_noise_label=True)
                    if i >= 3:  # 4 batches
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

