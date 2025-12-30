
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.llm_io import normalise_batch

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

def _split_X_y(evt):
    X = np.column_stack((evt["hit_r"],
                        evt["hit_theta"],
                        evt["hit_z"],
                        evt["layer_id"]))
    y = evt["track_id"].astype(np.int32)
    return (torch.from_numpy(X),torch.from_numpy(y))

def _make_dataset(events, pre, *, train: bool):
    custom = globals().get("make_dataset", None)
    if callable(custom):
        ds = custom(events, pre, train=train)
        if ds is not None:
            return ds
    return EventDataset(events, pre, train=train)

def make_loaders(raw_train, raw_val, pre, *, batch=512,
                 collate_fn=None, loader_cls=None, workers=0):
    train_ds  = _make_dataset(raw_train, pre, train=True)
    val_ds    = _make_dataset(raw_val,  pre, train=False)

    if loader_cls is None:
        loader_cls = DataLoader

    pin = (device.type == "cuda")
    train_ld = loader_cls(train_ds, batch_size=batch, shuffle=True,
                        num_workers=workers, collate_fn=collate_fn,
                        pin_memory=pin, persistent_workers=(workers > 0))
    val_ld   = loader_cls(val_ds,   batch_size=batch, shuffle=False,
                        num_workers=workers, collate_fn=collate_fn,
                        pin_memory=pin, persistent_workers=(workers > 0))
    return train_ld, val_ld
    
class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, track_id = _split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, track_id)

def _ragged(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    # batch[i] = (hits_i, track_id_i)      <- shapes: (N_i, F), (N_i)
    return batch

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.nn import GATv2Conv
from sklearn.cluster import DBSCAN
from collections import defaultdict
import copy

# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit()
    #   - transform()

    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    # TIPS
    #   When modifying data features or feature engineering: annotate tensor size as comments after each tensor operation to reduce dimension mismatches.

    def __init__(self):
        # Stateful components for normalization
        self.mean = None
        self.std = None

    def _raw_reshape(self, data):           
        # No raw data reshaping needed.
        return data # Returns identity by default

    def make_loader_cfg(self):
        # Configure the data loader for PyTorch Geometric
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 1, # Process one event graph at a time
        }

    def fit(self, data):
        # data is a list of event dictionaries
        # Extract r, theta, z and compute normalization statistics
        all_coords = []
        for evt in data:
            coords = np.column_stack((evt["hit_r"], evt["hit_theta"], evt["hit_z"]))
            all_coords.append(coords)

        all_coords_np = np.vstack(all_coords) # [Total_hits, 3]
        self.mean = torch.from_numpy(all_coords_np.mean(axis=0, dtype=np.float32))
        self.std = torch.from_numpy(all_coords_np.std(axis=0, dtype=np.float32))
        self.std[self.std == 0] = 1.0 # Avoid division by zero for stable transformation
        return self

    def transform(self, data):
        # data is a torch.Tensor of shape (N_hits, 4) from _split_X_y
        # We only normalize the first 3 features (r, theta, z)
        coords = data[:, :3] # (N_hits, 3)
        coords = (coords - self.mean) / self.std # (N_hits, 3)
        # Re-combine with the un-normalized layer_id
        return torch.cat([coords, data[:, 3:]], dim=1) # (N_hits, 4)

class EventGraphDataset(PyGDataset):
    """Custom PyTorch Geometric Dataset to handle event graph creation."""
    def __init__(self, events, pre, train=True):
        super().__init__()
        self.events = events
        self.pre = pre
        self.is_train = train
        self.graphs = self._process_events()

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    def _process_events(self):
        graphs = []
        for event in self.events:
            X, y = _split_X_y(event)
            # X shape: [N_hits, 4], y shape: [N_hits]

            X_transformed = self.pre.transform(X)

            # Build graph edges based on adjacency in layers
            layer_ids = X[:, 3]

            src, dst = [], []
            # Group hits by layer ID for efficient edge construction
            hits_by_layer = defaultdict(list)
            for i, lid in enumerate(layer_ids):
                hits_by_layer[lid.item()].append(i)

            sorted_layers = sorted(hits_by_layer.keys())

            # Connect hits in physically adjacent layers
            for i, layer_id in enumerate(sorted_layers[:-1]):
                next_layer_id = sorted_layers[i+1]

                # A simple heuristic for layer adjacency, assuming layer_ids are ordered
                # Note: A more robust approach would use geometry info if available
                # For this dataset, a simple difference check captures most of the structure.
                if next_layer_id - layer_id < 3.0: # Connect if layers are "close"
                    current_nodes = hits_by_layer[layer_id]
                    next_nodes = hits_by_layer[next_layer_id]

                    # Create directed edges from current to next layer
                    for u in current_nodes:
                        for v in next_nodes:
                            src.append(u)
                            dst.append(v)

            # Make the graph undirected
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

            # Use normalized (r, theta, z) as node features
            node_features = X_transformed[:, :3] # [N_hits, 3]

            data = Data(x=node_features, edge_index=edge_index, y=y)
            graphs.append(data)
        return graphs

def make_dataset(events, pre, *, train: bool):
    """Factory function to create the dataset."""
    return EventGraphDataset(events, pre, train=train)


def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, hidden_channels=64, embedding_dim=16):
        super().__init__()

        # GNN with Graph Attention v2 layers for message passing
        self.gcn_layers = nn.ModuleList([
            GATv2Conv(in_features, hidden_channels, heads=4, dropout=0.1),
            GATv2Conv(hidden_channels * 4, hidden_channels, heads=4, dropout=0.1),
            GATv2Conv(hidden_channels * 4, embedding_dim, heads=1, concat=False, dropout=0.1)
        ])

        # DBSCAN for clustering in inference mode. Epsilon is the key hyperparameter.
        self.dbscan = DBSCAN(eps=0.25, min_samples=2, metric='euclidean', n_jobs=-1)

    def forward(self, data):
        # data is a torch_geometric.data.Batch object
        x, edge_index, batch = data.x, data.edge_index, data.batch # x: [N_total_hits, 3]

        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, edge_index)
            if i < len(self.gcn_layers) - 1: # No activation on the last layer
                x = F.leaky_relu(x)

        # L2-normalize embeddings to live on a hypersphere.
        # This helps stabilize the contrastive loss and makes clustering robust.
        embeddings = F.normalize(x, p=2, dim=1) # [N_total_hits, embedding_dim]

        if self.training:
            # During training, return embeddings for the loss function
            return embeddings
        else:
            # Inference mode: perform clustering to predict track IDs
            pred_ids = torch.zeros(data.num_nodes, dtype=torch.long, device=x.device)
            # Process one event at a time from the batch
            node_counts = torch.bincount(batch)
            start_idx = 0
            track_id_offset = 1 # Start track IDs from 1, use 0 for noise

            for i in range(batch.max().item() + 1): # Iterate through events in the batch
                num_nodes_in_event = node_counts[i]
                if num_nodes_in_event > 0:
                    event_embeddings = embeddings[start_idx : start_idx + num_nodes_in_event]
                    event_embeddings_np = event_embeddings.cpu().detach().numpy()

                    # Run DBSCAN on the embeddings of the current event
                    cluster_labels = self.dbscan.fit_predict(event_embeddings_np)

                    # Convert labels to a tensor
                    cluster_labels = torch.from_numpy(cluster_labels).to(x.device)
                    noise_mask = cluster_labels == -1

                    # Shift valid cluster IDs to be unique across events
                    cluster_labels[~noise_mask] += track_id_offset

                    # Assign a track ID of 0 to all noise points
                    cluster_labels[noise_mask] = 0

                    pred_ids[start_idx : start_idx + num_nodes_in_event] = cluster_labels

                    # Update offset for the next event's track IDs
                    if (~noise_mask).any():
                        track_id_offset = cluster_labels.max() + 1

                    start_idx += num_nodes_in_event

            return pred_ids


def make_model(in_features):
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
class ContrastiveLoss(nn.Module):
    """Contrastive loss function for metric learning."""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, edge_index, y):
        # Get embeddings for source and destination nodes of each edge
        start_nodes_emb = embeddings[edge_index[0]] # [N_edges, emb_dim]
        end_nodes_emb = embeddings[edge_index[1]] # [N_edges, emb_dim]

        # Get true labels: 1 if from the same track, 0 if different
        y_start = y[edge_index[0]]
        y_end = y[edge_index[1]]
        # Ignore edges connecting hits with track_id 0 (noise)
        valid_edge_mask = (y_start > 0) & (y_end > 0)
        true_edge = (y_start == y_end).float()[valid_edge_mask]

        # Calculate squared Euclidean distance for valid edges
        dist_sq = torch.sum((start_nodes_emb - end_nodes_emb) ** 2, dim=1)[valid_edge_mask]

        # Contrastive loss: pull same-track hits together, push different-track hits apart
        loss_contrastive = (true_edge * dist_sq) + \
                           ((1 - true_edge) * F.relu(self.margin - dist_sq))

        return loss_contrastive.mean()

EPOCHS = 25 
def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)
    criterion = ContrastiveLoss(margin=1.0)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None

    train_loss_history, val_loss_history = [], []
    # "Accuracy" will be a proxy metric: edge classification accuracy
    train_acc_history, val_acc_history = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss, total_train_acc, train_batches = 0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            embeddings = model(batch)
            loss = criterion(embeddings, batch.edge_index, batch.y)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Calculate proxy accuracy for monitoring
            with torch.no_grad():
                dist_sq = torch.sum((embeddings[batch.edge_index[0]] - embeddings[batch.edge_index[1]]) ** 2, dim=1)
                true_edge = (batch.y[batch.edge_index[0]] == batch.y[batch.edge_index[1]])
                # Predict edge if distance is less than a threshold (e.g., half the margin)
                pred_edge = (dist_sq < criterion.margin / 2.0)
                total_train_acc += (pred_edge == true_edge).float().mean().item()

            train_batches += 1

        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        avg_train_acc = total_train_acc / train_batches if train_batches > 0 else 0
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(avg_train_acc)

        # Validation loop
        model.eval()
        total_val_loss, total_val_acc, val_batches = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                embeddings = model(batch)
                loss = criterion(embeddings, batch.edge_index, batch.y)
                total_val_loss += loss.item()

                dist_sq = torch.sum((embeddings[batch.edge_index[0]] - embeddings[batch.edge_index[1]]) ** 2, dim=1)
                true_edge = (batch.y[batch.edge_index[0]] == batch.y[batch.edge_index[1]])
                pred_edge = (dist_sq < criterion.margin / 2.0)
                total_val_acc += (pred_edge == true_edge).float().mean().item()

                val_batches += 1

        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
        avg_val_acc = total_val_acc / val_batches if val_batches > 0 else 0
        val_loss_history.append(avg_val_loss)
        val_acc_history.append(avg_val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Acc: {avg_val_acc:.4f}")

        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Deepcopy is essential to save the state at this point
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break

    # Load the best model state found during training
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

def _import_dotted(path: str):
    mod, name = path.rsplit(".", 1)
    module = importlib.import_module(mod)
    return getattr(module, name)

def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    pre = make_preprocessor().fit(raw_train)
    collate = getattr(pre, "_collate_fn", None)
    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None

    train_loader, val_loader = make_loaders(raw_train, raw_val, pre,
                                            batch = cfg.get("batch_size", 128),
                                            collate_fn = collate or _ragged,
                                            loader_cls = loader_cls,
                                            workers    = cfg.get("num_workers", 0))

    # 2. Build model
    first_batch = next(iter(train_loader))
    view        = normalise_batch(first_batch, device=device)
    model       = make_model(view.batch_x)
    model       = model.to(device)


    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single reduced forward pass
    if dryrun:
        try:
            batch = first_batch
            view  = normalise_batch(batch, device=device)
            with torch.no_grad():
                _ = trained_model(view.batch_x)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

        pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
        pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
        pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

        torch.save(trained_model.state_dict(), pth_state)
        with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
        with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

        # 6. Save plots
        _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
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


