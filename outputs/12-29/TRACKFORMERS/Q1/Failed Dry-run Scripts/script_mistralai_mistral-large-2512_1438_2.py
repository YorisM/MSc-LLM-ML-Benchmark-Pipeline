
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
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean, scatter_add
from sklearn.preprocessing import StandardScaler
import hdbscan
import numpy as np

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

            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,

            "collate": None,

            "extra_loader_kwargs": {"follow_batch": ["x"]},

            "eval_overrides": {"shuffle": False, "batch_size": 64}
        }

    def fit(self, Xs):
        # Compute global statistics for normalization
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Only scale r, theta, z

        # Compute layer statistics
        layers = all_X[:, 3]
        self.layer_mean = np.mean(layers)
        self.layer_std = np.std(layers)

        return self

    def transform(self, X):
        # X: [N_hits, 4] - r, theta, z, layer_id
        X = X.clone().numpy() if isinstance(X, torch.Tensor) else X

        # Normalize r, theta, z
        X[:, :3] = self.scaler.transform(X[:, :3])

        # Normalize layer_id
        if self.layer_std > 0:
            X[:, 3] = (X[:, 3] - self.layer_mean) / self.layer_std

        return torch.from_numpy(X).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()

        # Determine input feature dimension
        if isinstance(example_batch_x, list):
            input_dim = example_batch_x[0].shape[1]
        else:
            input_dim = example_batch_x.shape[1]

        # Graph neural network layers
        self.conv1 = GATv2Conv(input_dim, 64, heads=4, concat=True, dropout=0.1)
        self.conv2 = GATv2Conv(64*4, 64, heads=4, concat=True, dropout=0.1)
        self.conv3 = GATv2Conv(64*4, 32, heads=2, concat=False, dropout=0.1)

        # Edge prediction network
        self.edge_mlp = nn.Sequential(
            nn.Linear(64*4*2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Track classification head
        self.classifier = nn.Sequential(
            nn.Linear(32 + 4, 128),  # 32 from GNN, 4 original features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Will use sigmoid for track probability
        )

        # Track embedding for clustering
        self.track_embed = nn.Linear(32, 16)

        self.dropout = nn.Dropout(0.2)

    def build_graph(self, x, batch_idx):
        # Create complete graph (fully connected within each event)
        num_nodes = x.size(0)
        edge_index = []

        # Create edges within each event
        for i in range(batch_idx.max() + 1):
            mask = (batch_idx == i)
            nodes = torch.where(mask)[0]
            n = nodes.size(0)

            # Create complete graph for this event
            if n > 1:
                src = nodes.repeat(n)
                dst = nodes.repeat_interleave(n)
                edge_index.append(torch.stack([src, dst]))

        if len(edge_index) > 0:
            edge_index = torch.cat(edge_index, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)

        return edge_index

    def forward(self, batch_x):
        # Handle ragged input
        if isinstance(batch_x, list):
            # Convert to PyG Batch
            data_list = []
            for i, x in enumerate(batch_x):
                # Create complete graph for this event
                edge_index = self.build_graph(x, torch.full((x.size(0),), i, device=x.device))

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    batch=torch.full((x.size(0),), i, device=x.device)
                )
                data_list.append(data)

            batch = Batch.from_data_list(data_list)
            x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        else:
            # Assume it's already a PyG Batch
            x, edge_index, batch_idx = batch_x.x, batch_x.edge_index, batch_x.batch

        # Graph neural network
        x = self.dropout(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x_gnn = F.relu(self.conv3(x, edge_index))

        # Combine with original features
        x_combined = torch.cat([x_gnn, batch_x.x if isinstance(batch_x, Batch) else batch_x], dim=1)

        # Predict track probabilities
        track_probs = torch.sigmoid(self.classifier(x_combined)).squeeze(1)

        # Create track embeddings for clustering
        track_embeddings = self.track_embed(x_gnn)

        # Cluster hits into tracks
        pred_labels = self.cluster_hits(track_embeddings, track_probs, batch_idx)

        return pred_labels

    def cluster_hits(self, embeddings, track_probs, batch_idx):
        # Convert to numpy for HDBSCAN
        device = embeddings.device
        embeddings_np = embeddings.detach().cpu().numpy()
        track_probs_np = track_probs.detach().cpu().numpy()
        batch_idx_np = batch_idx.detach().cpu().numpy()

        all_labels = []
        min_cluster_size = 4  # Minimum hits per track

        for i in range(batch_idx_np.max() + 1):
            mask = (batch_idx_np == i)
            if mask.sum() < min_cluster_size:
                # Not enough hits for clustering
                all_labels.append(torch.zeros(mask.sum(), dtype=torch.long, device=device) - 1)
                continue

            # Get embeddings and probabilities for this event
            event_embeddings = embeddings_np[mask]
            event_probs = track_probs_np[mask]

            # Combine embeddings with track probabilities
            combined = np.hstack([event_embeddings, event_probs.reshape(-1, 1)])

            # Cluster using HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=2,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(combined)

            # Convert to torch tensor and handle noise (-1)
            labels = torch.from_numpy(labels).to(device)
            labels[labels == -1] = -1  # Keep noise as -1

            all_labels.append(labels)

        # Combine all events
        return torch.cat(all_labels)

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()

            # Forward pass
            pred_labels = model(xb)

            # Compute loss
            loss = self_supervised_loss(pred_labels, yb, xb)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * yb[0].size(0) if isinstance(yb, list) else loss.item() * yb.size(0)

            # Simple accuracy metric (not the evaluation metric)
            if isinstance(yb, list):
                for pred, true in zip(pred_labels, yb):
                    correct = (pred == true).sum().item()
                    total_correct += correct
                    total_samples += true.size(0)
            else:
                correct = (pred_labels == yb).sum().item()
                total_correct += correct
                total_samples += yb.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                pred_labels = model(xb)
                loss = self_supervised_loss(pred_labels, yb, xb)
                val_loss += loss.item() * yb[0].size(0) if isinstance(yb, list) else loss.item() * yb.size(0)

                if isinstance(yb, list):
                    for pred, true in zip(pred_labels, yb):
                        correct = (pred == true).sum().item()
                        val_correct += correct
                        val_samples += true.size(0)
                else:
                    correct = (pred_labels == yb).sum().item()
                    val_correct += correct
                    val_samples += yb.size(0)

        # Calculate metrics
        train_loss = total_loss / total_samples
        val_loss = val_loss / val_samples
        train_acc = total_correct / total_samples
        val_acc = val_correct / val_samples

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
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

    return model, train_losses, val_losses, train_accs, val_accs

def self_supervised_loss(pred_labels, true_labels, batch_x):
    # Convert to consistent format
    if isinstance(pred_labels, list):
        pred_labels = torch.cat(pred_labels)
        true_labels = torch.cat(true_labels)
        batch_x = Batch.from_data_list(batch_x) if not isinstance(batch_x, Batch) else batch_x

    # Create mask for non-noise hits
    non_noise_mask = (true_labels > 0)

    # Only compute loss for non-noise hits
    if non_noise_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_labels.device)

    pred = pred_labels[non_noise_mask]
    true = true_labels[non_noise_mask]

    # Create a mapping from true track IDs to cluster IDs
    unique_true = torch.unique(true)
    cluster_mapping = torch.zeros(unique_true.max() + 1, dtype=torch.long, device=pred.device) - 1

    for i, track_id in enumerate(unique_true):
        # Find the most common predicted cluster for this track
        mask = (true == track_id)
        if mask.sum() > 0:
            pred_for_track = pred[mask]
            if pred_for_track.numel() > 0:
                cluster_id = torch.mode(pred_for_track).values
                cluster_mapping[track_id] = cluster_id

    # Create target labels based on the mapping
    target = cluster_mapping[true]
    valid_mask = (target != -1)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    # Cross entropy loss
    loss = F.cross_entropy(
        pred[valid_mask].unsqueeze(1),
        target[valid_mask].unsqueeze(1),
        reduction='mean'
    )

    # Add regularization to encourage larger clusters
    cluster_sizes = scatter_add(torch.ones_like(pred), pred)
    size_reg = -torch.log(cluster_sizes.float() + 1e-6).mean()

    return loss + 0.1 * size_reg

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

