
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data"
TAG      = "10_50_linear_frac0.05"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"REDVID_{TAG}_{split}.pkl.gz")
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

# <start code template>
# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import knn_graph
from sklearn.metrics import roc_auc_score
import copy

# This custom dataset needs to be at the top-level for pickling and multiprocessing
class GeometricEventDataset(Dataset):
    """
    Custom PyTorch Dataset that transforms raw event data into
    torch_geometric.data.Data objects (i.e., graphs).
    """
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        # Load raw data for one event
        X_raw, track_id = _split_X_y(self.events[idx])

        # Apply feature engineering and normalization
        node_features = self.pre.transform(X_raw) # (N_hits, F)

        # Build a k-Nearest Neighbors graph on the normalized xyz coordinates.
        # This connects hits that are close in space, a strong prior for track segments.
        # We only use the first 3 features (x_norm, y_norm, z_norm) for building the graph.
        edge_index = knn_graph(node_features[:, :3], k=10, loop=False, flow='source_to_target')

        # Return a PyG Data object
        return Data(x=node_features, edge_index=edge_index, y=track_id)

def make_dataset(events, pre, *, train: bool):
    """Factory function for creating the custom dataset."""
    return GeometricEventDataset(events, pre, train=train)


# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    """
    A preprocessor for standardizing hit features and configuring the data loading
    for graph-based learning.
    """
    def __init__(self):
        # This will store the normalization statistics (mean, std)
        self.stats = {}

    @staticmethod
    def _collate_fn(batch: list[Data]):
        """
        A static collate function to batch a list of PyG Data objects into a
        single PyG Batch object. This is necessary for using a standard
        torch.utils.data.DataLoader with PyG data.
        """
        return Batch.from_data_list(batch)

    def make_loader_cfg(self):
        """
        Specifies the configuration for the DataLoader. We provide our custom
        _collate_fn to handle the batching of graph data.
        """
        return {
           "collate_fn": "self._collate_fn",
           "batch_size": 32,
        }

    def fit(self, data: list[dict]):
        """
        Calculates normalization statistics from the training data.
        It computes the mean and standard deviation of the hits' Cartesian
        coordinates (x, y, z) across all training events.
        """
        # Stack all hits from all events to compute global statistics
        all_hits = np.vstack([
            np.column_stack((evt["hit_r"], evt["hit_theta"], evt["hit_z"])) for evt in data
        ])
        r, theta, z = all_hits[:, 0], all_hits[:, 1], all_hits[:, 2]

        # Convert to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Store statistics for later use in transform()
        self.stats = {
            'x': {'mean': np.mean(x), 'std': np.std(x)},
            'y': {'mean': np.mean(y), 'std': np.std(y)},
            'z': {'mean': np.mean(z), 'std': np.std(z)},
        }
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Applies pre-calculated transformations and feature engineering to the raw
        hit data of a single event.
        """
        r, theta, z, layer = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        # Convert to Cartesian coordinates
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # Apply Z-score normalization using stats from fit()
        x_norm = (x - self.stats['x']['mean']) / self.stats['x']['std']
        y_norm = (y - self.stats['y']['mean']) / self.stats['y']['std']
        z_norm = (z - self.stats['z']['mean']) / self.stats['z']['std']

        # Feature engineering: Add normalized radius and scaled raw features
        r_norm = torch.sqrt(x_norm**2 + y_norm**2)

        # Combine features for the GNN node inputs.
        return torch.stack([
            x_norm,
            y_norm,
            z_norm,
            r_norm,
            r / 1000.0, # Scale radius
            layer / 10.0   # Scale layer id
        ], dim=1) # (N_hits, 6)

def make_preprocessor():
    """Factory function for creating the preprocessor."""
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    """
    A Graph Attention Network (GAT) model to learn embeddings for each hit.
    Hits from the same particle track should have similar embeddings.
    """
    def __init__(self, in_features: int, hidden_dim: int = 128, embed_dim: int = 8, n_layers: int = 4):
        super().__init__()

        # Initial MLP to encode input features into the hidden space
        self.node_encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # A stack of GATv2 layers to propagate information through the graph
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            # Using concat=False to allow for residual connections
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False))

        # Final MLP to project the learned representations into the embedding space
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, data: Batch) -> torch.Tensor:
        """The forward pass of the model."""
        x, edge_index = data.x, data.edge_index

        x_encoded = self.node_encoder(x)

        x_out = x_encoded
        for conv in self.convs:
            # Add a residual connection to help with training deep GNNs
            x_out = x_out + conv(x_out, edge_index)

        embedding = self.output_proj(x_out)

        # L2-normalize the final embeddings. This is crucial for contrastive loss,
        # as it forces embeddings onto a hypersphere, making distances comparable.
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1) # (N_hits, embed_dim)
        return embedding

def make_model(example_sample):
    """
    Factory function for the model. It inspects the first data sample to
    determine the input feature dimension.
    """
    if isinstance(example_sample, (Data, Batch)):
        in_features = example_sample.num_node_features
    else: # Fallback for unexpected sample type
        in_features = example_sample.shape[1]
    return HitClassifier(in_features=in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25   # Increased epochs for better convergence, with early stopping

def contrastive_loss_fn(embeddings, track_ids, ptr, margin=1.0, device='cpu'):
    """
    Computes the contrastive loss for a batch of events.
    It pushes embeddings of hits from the same track together and pulls
    embeddings from different tracks apart.
    """
    total_loss, num_events = 0.0, 0

    # Process each event in the batch separately
    for i in range(len(ptr) - 1):
        start, end = ptr[i], ptr[i+1]
        if end - start < 2: continue

        event_embeds = embeddings[start:end]
        event_y = track_ids[start:end]

        y_i, y_j = event_y.unsqueeze(1), event_y.unsqueeze(0)

        # Masks to identify positive and negative pairs
        is_same_track = (y_i == y_j)
        is_not_self = ~torch.eye(is_same_track.shape[0], dtype=torch.bool, device=device)
        valid_track_mask = (y_i > 0) # Ignore noise hits (track_id=0)

        pos_mask = is_same_track & is_not_self & valid_track_mask
        neg_mask = ~is_same_track

        pos_indices = torch.where(pos_mask)
        neg_indices = torch.where(neg_mask)

        if pos_indices[0].shape[0] == 0 or neg_indices[0].shape[0] == 0:
            continue

        # Sub-sample negative pairs to balance the loss and prevent bias
        num_pos = pos_indices[0].shape[0]
        num_neg = neg_indices[0].shape[0]
        if num_neg > num_pos * 2:
             rand_perm = torch.randperm(num_neg, device=device)[:num_pos*2]
             neg_indices = (neg_indices[0][rand_perm], neg_indices[1][rand_perm])

        # Calculate squared Euclidean distances
        pos_embeds1 = event_embeds[pos_indices[0]]
        pos_embeds2 = event_embeds[pos_indices[1]]
        pos_dists = torch.sum((pos_embeds1 - pos_embeds2)**2, dim=1)

        neg_embeds1 = event_embeds[neg_indices[0]]
        neg_embeds2 = event_embeds[neg_indices[1]]
        neg_dists = torch.sum((neg_embeds1 - neg_embeds2)**2, dim=1)

        # Loss components
        loss_pos = pos_dists.mean()
        loss_neg = torch.clamp(margin - neg_dists, min=0).mean()

        total_loss += loss_pos + loss_neg
        num_events += 1

    return total_loss / num_events if num_events > 0 else torch.tensor(0.0, device=device)

def train_model(model, train_loader, val_loader, epochs):
    """
    Main training loop for the GNN model. Implements a standard training and
    validation cycle with early stopping.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], [] # Using 'acc' to store AUC proxy metric

    best_val_loss = float('inf')
    best_model_state = None
    patience, patience_counter = 3, 0

    def calculate_auc(embeddings, track_ids, ptr, device='cpu'):
        """Calculates a proxy metric for embedding quality: the AUC for
        distinguishing same-track vs. different-track pairs."""
        aucs = []
        for i in range(len(ptr) - 1):
            start, end = ptr[i], ptr[i+1]
            if end - start < 2: continue

            event_embeds = embeddings[start:end]
            event_y = track_ids[start:end]

            y_i, y_j = event_y.unsqueeze(1), event_y.unsqueeze(0)
            is_same = (y_i == y_j) & (y_i > 0)
            is_not_self = ~torch.eye(is_same.shape[0], dtype=torch.bool, device=device)

            pos_mask = is_same & is_not_self
            neg_mask = ~is_same & (y_i > 0) & (y_j > 0) # Only compare valid tracks

            if not pos_mask.any() or not neg_mask.any(): continue

            dists = torch.cdist(event_embeds, event_embeds)

            pos_dists = dists[pos_mask]
            neg_dists = dists[neg_mask]

            # Higher distance means less likely to be same track, so use -dists
            scores = torch.cat([-pos_dists, -neg_dists]).flatten().cpu().numpy()
            labels = torch.cat([torch.ones_like(pos_dists), torch.zeros_like(neg_dists)]).flatten().cpu().numpy()

            try:
                aucs.append(roc_auc_score(labels, scores))
            except ValueError:
                pass # Happens if only one class is present
        return np.mean(aucs) if aucs else 0.5

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss, running_auc = 0.0, 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            embeddings = model(batch)
            loss = contrastive_loss_fn(embeddings, batch.y, batch.ptr, device=device)

            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            with torch.no_grad():
                running_auc += calculate_auc(embeddings, batch.y, batch.ptr, device=device)

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(running_auc / len(train_loader))

        # --- Validation Phase ---
        model.eval()
        running_vloss, running_vauc = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embeddings = model(batch)
                vloss = contrastive_loss_fn(embeddings, batch.y, batch.ptr, device=device)
                vauc = calculate_auc(embeddings, batch.y, batch.ptr, device=device)
                running_vloss += vloss.item()
                running_vauc += vauc

        epoch_vloss = running_vloss / len(val_loader)
        val_loss.append(epoch_vloss)
        val_acc.append(running_vauc / len(val_loader))

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss[-1]:.4f}, Train AUC: {train_acc[-1]:.4f}, "
              f"Val Loss: {val_loss[-1]:.4f}, Val AUC: {val_acc[-1]:.4f}")

        # --- Early Stopping ---
        if epoch_vloss < best_val_loss:
            best_val_loss = epoch_vloss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore the best model found during training
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_loss, val_loss, train_acc, val_acc

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
    collate = getattr(pre, "_collate_fn", None)
    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None

    train_loader, val_loader = make_loaders(raw_train, raw_val, pre,
                                            batch = cfg.get("batch_size", 128),
                                            collate_fn = collate or _ragged,
                                            loader_cls = loader_cls,
                                            workers    = cfg.get("num_workers", 0))

    # 2. Build model
    first_batch    = next(iter(train_loader))
    example_sample = first_batch[0]
    model          = make_model(example_sample)
    model          = model.to(device)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single toy forward pass
    if dryrun:
        try:
            _ = trained_model(first_batch)
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


