
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
import networkx as nx

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

        # Compute layer statistics
        layer_ids = all_hits[:, 3]
        unique_layers = np.unique(layer_ids)
        self.layer_means = {}
        self.layer_stds = {}
        for layer in unique_layers:
            mask = (layer_ids == layer)
            layer_hits = all_hits[mask, :3]
            self.layer_means[layer] = np.mean(layer_hits, axis=0)
            self.layer_stds[layer] = np.std(layer_hits, axis=0)
        return self

    def transform(self, X):
        # X: [N_hits, 4] (r, theta, z, layer_id)
        X_np = X.numpy()
        scaled = self.scaler.transform(X_np[:, :3])
        layer_id = X_np[:, 3].reshape(-1, 1)

        # Add layer-relative features
        layer_features = np.zeros((len(X), 3))
        for i, lid in enumerate(layer_id.flatten()):
            if lid in self.layer_means:
                layer_features[i] = (scaled[i] - self.layer_means[lid]) / (self.layer_stds[lid] + 1e-6)

        # Combine features: scaled + layer_relative + layer_id
        X_out = np.hstack([scaled, layer_features, layer_id])
        return torch.from_numpy(X_out).float()

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
            self.lane = "torch_ragged_xy"
        else:
            # PyG lane
            self.input_dim = example_batch_x.x.shape[1]
            self.lane = "pyg_batch"

        # GNN architecture
        self.conv1 = GCNConv(self.input_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.fc = nn.Linear(16, 8)

        # Clustering head
        self.cluster_head = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, batch_x):
        if self.lane == "torch_ragged_xy":
            # Convert ragged batch to PyG format
            graphs = []
            for i, x in enumerate(batch_x):
                # Create edge indices based on spatial proximity
                dist = torch.cdist(x[:, :3], x[:, :3])
                edges = (dist < 0.5).nonzero().t()
                graphs.append(Data(x=x, edge_index=edges))
            batch = Batch.from_data_list(graphs)
        else:
            batch = batch_x

        # GNN forward
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = F.relu(self.conv2(x, batch.edge_index))
        x = F.relu(self.conv3(x, batch.edge_index))
        x = global_mean_pool(x, batch.batch)
        x = self.fc(x)

        # Get node embeddings
        node_emb = self.cluster_head(x[batch.batch])
        return node_emb

    def predict_labels(self, batch_x):
        with torch.no_grad():
            embeddings = self.forward(batch_x)

            if self.lane == "torch_ragged_xy":
                # Process each event separately
                labels = []
                for i, x in enumerate(batch_x):
                    # Get embeddings for this event
                    event_emb = embeddings[i * len(x): (i+1) * len(x)]

                    # Convert to numpy for DBSCAN
                    emb_np = event_emb.cpu().numpy()

                    # DBSCAN clustering
                    clustering = DBSCAN(eps=0.5, min_samples=4, metric='cosine').fit(emb_np)
                    pred_labels = torch.from_numpy(clustering.labels_).to(device)

                    # Map cluster labels to track IDs (arbitrary but consistent)
                    unique_labels = torch.unique(pred_labels)
                    label_map = {old: new for new, old in enumerate(unique_labels)}
                    pred_labels = torch.tensor([label_map[l.item()] if l.item() != -1 else -1
                                              for l in pred_labels])

                    labels.append(pred_labels)
                return labels
            else:
                # PyG batch processing
                emb_np = embeddings.cpu().numpy()
                clustering = DBSCAN(eps=0.5, min_samples=4, metric='cosine').fit(emb_np)
                pred_labels = torch.from_numpy(clustering.labels_).to(device)

                # Map cluster labels
                unique_labels = torch.unique(pred_labels)
                label_map = {old: new for new, old in enumerate(unique_labels)}
                pred_labels = torch.tensor([label_map[l.item()] if l.item() != -1 else -1
                                          for l in pred_labels])
                return pred_labels

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

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        count = 0

        for batch in train_loader:
            if model.lane == "torch_ragged_xy":
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                # Create PyG batch for training
                graphs = []
                for i, x in enumerate(Xs):
                    dist = torch.cdist(x[:, :3], x[:, :3])
                    edges = (dist < 0.5).nonzero().t()
                    graphs.append(Data(x=x, edge_index=edges, y=ys[i]))
                batch_data = Batch.from_data_list(graphs)
            else:
                batch_data = batch.to(device)

            optimizer.zero_grad()
            embeddings = model(batch_data)

            # Create pseudo-labels for training (using truth)
            if model.lane == "torch_ragged_xy":
                # For ragged, we need to handle each event separately
                loss = 0
                for i, x in enumerate(Xs):
                    event_emb = embeddings[i * len(x): (i+1) * len(x)]
                    unique_labels = torch.unique(ys[i])
                    label_map = {old: new for new, old in enumerate(unique_labels)}
                    mapped_labels = torch.tensor([label_map[l.item()] if l.item() != -1 else -1
                                                for l in ys[i]]).to(device)

                    # Only compute loss for non-noise hits
                    mask = mapped_labels != -1
                    if mask.sum() > 0:
                        loss += criterion(event_emb[mask], mapped_labels[mask])
                loss /= len(Xs)
            else:
                # For PyG batch
                unique_labels = torch.unique(batch_data.y)
                label_map = {old: new for new, old in enumerate(unique_labels)}
                mapped_labels = torch.tensor([label_map[l.item()] if l.item() != -1 else -1
                                            for l in batch_data.y]).to(device)

                mask = mapped_labels != -1
                if mask.sum() > 0:
                    loss = criterion(embeddings[mask], mapped_labels[mask])
                else:
                    loss = torch.tensor(0.0, device=device)

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            # Compute accuracy
            with torch.no_grad():
                pred_labels = model.predict_labels(batch)
                if model.lane == "torch_ragged_xy":
                    acc = 0
                    for i in range(len(Xs)):
                        mask = ys[i] != 0  # Non-noise hits
                        if mask.sum() > 0:
                            # Match predicted clusters to truth
                            pred = pred_labels[i][mask]
                            truth = ys[i][mask]
                            acc += (pred == truth).float().mean().item()
                    acc /= len(Xs)
                else:
                    mask = batch_data.y != 0
                    if mask.sum() > 0:
                        pred = pred_labels[mask]
                        truth = batch_data.y[mask]
                        acc = (pred == truth).float().mean().item()
                    else:
                        acc = 0
                epoch_train_acc += acc
            count += 1

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                if model.lane == "torch_ragged_xy":
                    Xs, ys = batch
                    Xs = [x.to(device) for x in Xs]
                    ys = [y.to(device) for y in ys]

                    # Create PyG batch for validation
                    graphs = []
                    for i, x in enumerate(Xs):
                        dist = torch.cdist(x[:, :3], x[:, :3])
                        edges = (dist < 0.5).nonzero().t()
                        graphs.append(Data(x=x, edge_index=edges, y=ys[i]))
                    batch_data = Batch.from_data_list(graphs)
                else:
                    batch_data = batch.to(device)

                embeddings = model(batch_data)

                # Compute loss
                if model.lane == "torch_ragged_xy":
                    loss = 0
                    for i, x in enumerate(Xs):
                        event_emb = embeddings[i * len(x): (i+1) * len(x)]
                        unique_labels = torch.unique(ys[i])
                        label_map = {old: new for new, old in enumerate(unique_labels)}
                        mapped_labels = torch.tensor([label_map[l.item()] if l.item() != -1 else -1
                                                    for l in ys[i]]).to(device)

                        mask = mapped_labels != -1
                        if mask.sum() > 0:
                            loss += criterion(event_emb[mask], mapped_labels[mask])
                    loss /= len(Xs)
                else:
                    unique_labels = torch.unique(batch_data.y)
                    label_map = {old: new for new, old in enumerate(unique_labels)}
                    mapped_labels = torch.tensor([label_map[l.item()] if l.item() != -1 else -1
                                                for l in batch_data.y]).to(device)

                    mask = mapped_labels != -1
                    if mask.sum() > 0:
                        loss = criterion(embeddings[mask], mapped_labels[mask])
                    else:
                        loss = torch.tensor(0.0, device=device)

                epoch_val_loss += loss.item()

                # Compute accuracy
                pred_labels = model.predict_labels(batch)
                if model.lane == "torch_ragged_xy":
                    acc = 0
                    for i in range(len(Xs)):
                        mask = ys[i] != 0
                        if mask.sum() > 0:
                            pred = pred_labels[i][mask]
                            truth = ys[i][mask]
                            acc += (pred == truth).float().mean().item()
                    acc /= len(Xs)
                else:
                    mask = batch_data.y != 0
                    if mask.sum() > 0:
                        pred = pred_labels[mask]
                        truth = batch_data.y[mask]
                        acc = (pred == truth).float().mean().item()
                    else:
                        acc = 0
                epoch_val_acc += acc
                val_count += 1

        # Average metrics
        train_loss.append(epoch_train_loss / count)
        train_acc.append(epoch_train_acc / count)
        val_loss.append(epoch_val_loss / val_count)
        val_acc.append(epoch_val_acc / val_count)

        scheduler.step(epoch_val_loss / val_count)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, "
              f"Train Acc: {train_acc[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}")

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

