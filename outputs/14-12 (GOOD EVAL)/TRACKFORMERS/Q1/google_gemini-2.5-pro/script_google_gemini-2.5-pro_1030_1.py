
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
    X = np.column_stack((evt["hit_r"].astype(np.float32),
                        evt["hit_theta"].astype(np.float32),
                        evt["hit_z"].astype(np.float32),
                        evt["layer_id"].astype(np.float32)))
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

# <start code template>
# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import DynamicEdgeConv, GATv2Conv
from torch.nn import Sequential, Linear, ReLU
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import torch.nn.functional as F
import copy

# 1.1 -------- OPTIONAL: CUSTOM DATASET / DATA-CLASS --------
class GraphEventDataset(PyGDataset):
    def __init__(self, events, pre, train=True):
        super().__init__()
        self.events, self.pre, self.train = events, pre, train

        self.data_list = []
        for i in range(len(self.events)):
            X_raw, y_raw = _split_X_y(self.events[i])
            X = self.pre.transform(X_raw)
            # Create a PyG Data object for each event.
            # No edges are stored; they will be computed dynamically.
            data = Data(x=X, y=y_raw)
            self.data_list.append(data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def make_dataset(events, pre, train: bool):
    # Use a custom PyG Dataset to build graph-structured data on the fly.
    return GraphEventDataset(events, pre, train=train)

# 1.2 ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # Store mean and std for feature normalization
        self.mean_ = None
        self.std_ = None

    def _raw_reshape(self, data):           
        return data

    def make_loader_cfg(self):
        # Use PyG's DataLoader to handle batching of graph Data objects
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 32, # Smaller batch size for GNNs
        }

    def fit(self, events):
        # Calculate mean and std for r, theta, z from all training events
        rs, thetas, zs = [], [], []
        for evt in events:
            rs.append(evt["hit_r"])
            thetas.append(evt["hit_theta"])
            zs.append(evt["hit_z"])

        r_all = np.concatenate(rs)
        theta_all = np.concatenate(thetas)
        z_all = np.concatenate(zs)

        # Convert to Cartesian to find stats
        x_all = r_all * np.cos(theta_all)
        y_all = r_all * np.sin(theta_all)

        coords = np.stack([x_all, y_all, z_all], axis=1)
        self.mean_ = torch.from_numpy(coords.mean(axis=0, dtype=np.float32))
        self.std_ = torch.from_numpy(coords.std(axis=0, dtype=np.float32))

        # Add a small epsilon to std to avoid division by zero
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, data):
        # data is a tensor of shape [N_hits, 4] with (r, theta, z, layer_id)

        # Convert to Cartesian coordinates
        r, theta, z = data[:, 0], data[:, 1], data[:, 2]
        x = r * torch.cos(theta) # [N_hits,]
        y = r * torch.sin(theta) # [N_hits,]

        coords = torch.stack([x, y, z], dim=1) # [N_hits, 3]

        # Normalize Cartesian coordinates
        coords = (coords - self.mean_) / self.std_

        # Keep layer_id as is, it's a categorical-like feature
        # Concatenate features
        return torch.cat([coords, data[:, 3:]], dim=1) # [N_hits, 4]

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class GNN_Embedder(nn.Module):
    def __init__(self, in_channels, k=20, emb_dim=16):
        super().__init__()
        # DynamicEdgeConv is a good choice for point-cloud like data
        self.conv1 = DynamicEdgeConv(
            nn=Sequential(Linear(2 * in_channels, 64), ReLU(), Linear(64, 64)),
            k=k
        )
        self.conv2 = DynamicEdgeConv(
            nn=Sequential(Linear(2 * 64, 128), ReLU(), Linear(128, 128)),
            k=k
        )
        # Final embedding layer
        self.final_mlp = Sequential(Linear(128, emb_dim))

    def forward(self, batch):
        # batch is a torch_geometric.data.Batch object
        x, batch_idx = batch.x, batch.batch
        x1 = self.conv1(x, batch_idx) # [N_total_hits, 64]
        x2 = self.conv2(x1, batch_idx) # [N_total_hits, 128]
        emb = self.final_mlp(x2) # [N_total_hits, emb_dim]
        # Normalize embeddings to lie on a unit hypersphere
        emb = F.normalize(emb, p=2, dim=1)
        return emb

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # example_batch_x is a PyG Batch object from our custom loader
        in_features = example_batch_x.num_node_features

        self.gnn = GNN_Embedder(in_channels=in_features, k=20, emb_dim=16)

        # DBSCAN parameters for clustering during inference
        self.dbscan_eps = 0.4
        self.dbscan_min_samples = 4

    def forward(self, batch_x):
        # Get embeddings from the GNN submodule
        embeddings = self.gnn(batch_x) # [N_total_hits, D]

        if self.training:
            # During training, return embeddings for the loss function
            return embeddings
        else:
            # During evaluation, perform clustering to predict track IDs
            pred_labels = torch.full((embeddings.size(0),), -1, dtype=torch.long, device=embeddings.device)
            # Process each event in the batch separately
            for event_id in torch.unique(batch_x.batch):
                event_mask = (batch_x.batch == event_id)
                event_embeddings = embeddings[event_mask].cpu().numpy()

                # Run DBSCAN
                if event_embeddings.shape[0] < self.dbscan_min_samples:
                    # Not enough hits to form a cluster, all are noise (-1)
                    continue

                clusterer = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric='euclidean')
                event_labels = clusterer.fit_predict(event_embeddings) # np.array

                # Assign labels back to the correct positions in the full tensor
                # Note: DBSCAN labels start at -1 (noise). We can keep this.
                pred_labels[event_mask] = torch.from_numpy(event_labels).to(pred_labels.device)

            return pred_labels


def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 40  

def _mine_triplets(labels, batch_idx):
    """Randomly samples triplets (anchor, positive, negative) for a batch."""
    anchors, positives, negatives = [], [], []
    device = labels.device
    indices = torch.arange(len(labels), device=device)

    for event_id in torch.unique(batch_idx):
        event_mask = (batch_idx == event_id)
        event_indices = indices[event_mask]
        event_labels = labels[event_mask]

        unique_labels, counts = torch.unique(event_labels, return_counts=True)
        # Filter out labels with less than 2 hits
        valid_labels = unique_labels[counts >= 2]

        if len(valid_labels) == 0 or len(unique_labels) < 2:
            continue

        for label in valid_labels:
            positive_mask = (event_labels == label)
            negative_mask = ~positive_mask

            positive_indices = event_indices[positive_mask]
            negative_indices = event_indices[negative_mask]

            if len(negative_indices) == 0:
                continue

            # Sample n_pos anchor-positive pairs
            n_pos = len(positive_indices)
            ap_pairs = torch.randperm(n_pos, device=device)

            anchor_indices = positive_indices[ap_pairs % n_pos]
            positive_indices_perm = positive_indices[(ap_pairs + 1 + torch.randint(1, n_pos, (n_pos,), device=device)) % n_pos]

            # Sample negatives
            neg_samples = torch.randint(0, len(negative_indices), (n_pos,), device=device)
            negative_indices_sampled = negative_indices[neg_samples]

            anchors.append(anchor_indices)
            positives.append(positive_indices_perm)
            negatives.append(negative_indices_sampled)

    if not anchors:
        return None, None, None

    return torch.cat(anchors), torch.cat(positives), torch.cat(negatives)

def _clustering_accuracy(true_labels, pred_labels):
    """Computes clustering accuracy using the Hungarian algorithm."""

    # Create a contingency matrix
    contingency = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(true_labels.tolist(), pred_labels.tolist()):
        if pred == -1: continue # Ignore noise points in prediction
        if true == 0: continue # Ignore noise points in truth
        contingency[true][pred] += 1

    if not contingency:
        return 0.0

    # Build cost matrix for Hungarian algorithm
    true_ids = sorted(contingency.keys())
    pred_ids = sorted(list(set(p for T in contingency.values() for p in T.keys())))

    cost_matrix = np.zeros((len(true_ids), len(pred_ids)))
    for i, true_id in enumerate(true_ids):
        for j, pred_id in enumerate(pred_ids):
            cost_matrix[i, j] = -contingency[true_id][pred_id]

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Calculate accuracy
    correctly_classified = -cost_matrix[row_ind, col_ind].sum()
    total = len(true_labels[true_labels != 0])

    return correctly_classified / total if total > 0 else 0.0

def train_model(model, train_loader, val_loader, epochs):
    """Main training loop."""
    optimizer = torch.optim.AdamW(model.gnn.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Use TripletMarginLoss for metric learning
    loss_fn = nn.TripletMarginLoss(margin=0.5, p=2)

    best_val_loss = float('inf')
    best_model_state = None
    patience, max_patience = 0, 8

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Model returns embeddings in training mode
            embeddings = model(batch)

            # Sample triplets for the loss function
            anchors_idx, positives_idx, negatives_idx = _mine_triplets(batch.y, batch.batch)

            if anchors_idx is None:
                continue

            # Compute loss
            loss = loss_fn(embeddings[anchors_idx], embeddings[positives_idx], embeddings[negatives_idx])

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        train_acc.append(1 - avg_train_loss) # Proxy accuracy

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        all_val_acc = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # Get embeddings from the gnn submodule directly
                embeddings = model.gnn(batch)

                # Get triplet samples for consistent loss calculation
                anchors_idx, positives_idx, negatives_idx = _mine_triplets(batch.y, batch.batch)

                if anchors_idx is not None:
                    loss = loss_fn(embeddings[anchors_idx], embeddings[positives_idx], embeddings[negatives_idx])
                    total_val_loss += loss.item()

                # Get cluster predictions by calling the full model in eval mode
                pred_labels = model(batch)

                # Calculate clustering accuracy per event
                for event_id in torch.unique(batch.batch):
                    event_mask = batch.batch == event_id
                    true = batch.y[event_mask]
                    pred = pred_labels[event_mask]
                    all_val_acc.append(_clustering_accuracy(true, pred))

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_acc = np.mean(all_val_acc) if all_val_acc else 0

        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {avg_val_acc:.4f}")

        # Scheduler and Early Stopping
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    trained_model = model
    return trained_model, train_loss, val_loss, train_acc, val_acc

# <end code template>

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

    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None

    train_loader, val_loader = make_loaders(raw_train, raw_val, pre,
                                            batch = cfg.get("batch_size", 128),
                                            collate_fn = _ragged,
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

