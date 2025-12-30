
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

# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, knn_graph
from torch_geometric.utils import to_dense_batch

# This function must be in the global scope for the harness to find it.
# It defines a custom PyTorch Geometric dataset that creates a graph for each event.
class GraphDataset(Dataset):
    """
    Custom PyTorch Geometric dataset.
    For each event, it creates a graph where hits are nodes and edges
    are constructed using k-Nearest Neighbors in coordinate space.
    """
    def __init__(self, events, preprocessor, k=8, train=True):
        self.events = events
        self.preprocessor = preprocessor
        self.k = k
        self.is_train = train

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        # Load raw hit and track ID data for one event
        X_raw, track_id = _split_X_y(self.events[idx])

        # Apply feature engineering and normalization
        X_processed = self.preprocessor.transform(X_raw) # (N_hits, 6)

        # Use normalized Cartesian coordinates (x,y,z) for k-NN.
        # These are at indices 4, 5, 2 in the processed tensor.
        pos = X_processed[:, [4, 5, 2]]

        # Build graph using k-NN. loop=False to avoid self-loops.
        edge_index = knn_graph(pos, self.k, loop=False, cosine=False)

        # Return a PyG Data object encapsulating the graph
        return Data(
            x=X_processed,
            y=track_id,
            edge_index=edge_index,
            pos=pos
        )

def make_dataset(events, pre, *, train: bool):
    """
    This factory function is called by the harness to create the dataset.
    It returns an instance of our custom GraphDataset.
    """
    return GraphDataset(events, pre, train=train, k=8)


# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    """
    Preprocessor for track-hit data. It performs two main functions:
    1. Feature Engineering: Converts cylindrical (r, theta) coordinates to Cartesian (x, y).
    2. Normalization: Calculates and applies z-score normalization to geometric features.
    """
    def __init__(self):
        self.scalers = {}
        # Features to be normalized
        self.feature_names = ['hit_r', 'hit_theta', 'hit_z', 'x', 'y']

    def fit(self, events):
        """
        Calculates normalization statistics (mean, std) from the training data.
        """
        all_features_list = []
        for evt in events:
            r = torch.from_numpy(evt["hit_r"])
            theta = torch.from_numpy(evt["hit_theta"])
            z = torch.from_numpy(evt["hit_z"])
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            all_features_list.append(torch.stack([r, theta, z, x, y], dim=1))

        all_features_tensor = torch.cat(all_features_list, dim=0)

        means = torch.mean(all_features_tensor, dim=0)
        stds = torch.std(all_features_tensor, dim=0)
        stds[stds == 0] = 1.0 # Avoid division by zero

        for i, name in enumerate(self.feature_names):
            self.scalers[name] = {'mean': means[i], 'std': stds[i]}

        return self

    def transform(self, data):
        """
        Applies the fitted transformation to input data.
        """
        # data is a tensor of shape (N_hits, 4) with columns: r, theta, z, layer_id
        r, theta, z, layer_id = data.T
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # Re-create a dictionary of features for easy access
        features = {'hit_r': r, 'hit_theta': theta, 'hit_z': z, 'x': x, 'y': y}

        # Normalize the specified features
        for name in self.feature_names:
            mean = self.scalers[name]['mean']
            std = self.scalers[name]['std']
            features[name] = (features[name] - mean) / std

        # Return a new tensor with original and new/normalized features
        # Shape: (N_hits, 6) -> r, theta, z, layer_id, x, y
        return torch.stack([
            features['hit_r'], features['hit_theta'], features['hit_z'],
            layer_id, # Keep layer_id unnormalized as it's a categorical feature
            features['x'], features['y']
        ], dim=1).to(torch.float32)

    def make_loader_cfg(self):
        """
        Configures the DataLoader. We use PyG's DataLoader to handle graph batches.
        """
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 32,
        }

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    """
    A Graph Neural Network model to learn embeddings for hits.
    - Architecture: Stacked EdgeConv layers to perform message passing on the hit graph.
    - Training: Learns embeddings via a contrastive loss.
    - Inference: Applies DBSCAN clustering on the embeddings to reconstruct tracks.
    """
    def __init__(self, in_features_or_data_obj):
        super().__init__()

        # Handle flexible input from harness
        if isinstance(in_features_or_data_obj, Data):
            in_features = in_features_or_data_obj.num_node_features
        else: 
            in_features = in_features_or_data_obj

        embedding_dim = 64

        # GNN architecture using EdgeConv for message passing
        mlp1 = nn.Sequential(nn.Linear(2 * in_features, 64), nn.ReLU(), nn.Linear(64, 64))
        self.gnn1 = EdgeConv(mlp1, aggr='max')

        mlp2 = nn.Sequential(nn.Linear(2 * 64, 128), nn.ReLU(), nn.Linear(128, 128))
        self.gnn2 = EdgeConv(mlp2, aggr='max')

        mlp3 = nn.Sequential(nn.Linear(2 * 128, 256), nn.ReLU(), nn.Linear(256, 256))
        self.gnn3 = EdgeConv(mlp3, aggr='max')

        # MLP to process concatenated GNN features into a final embedding
        self.post_mlp = nn.Sequential(
            nn.Linear(64 + 128 + 256, 256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim)
        )

        # Hyperparameters for DBSCAN clustering during inference
        self.dbscan_eps = 0.5
        self.dbscan_min_samples = 4 # Match metric requirement for >= 4 hits

    def forward(self, batch):
        x, edge_index, batch_map = batch.x, batch.edge_index, batch.batch

        # Apply GNN layers, saving intermediate representations
        x1 = self.gnn1(x, edge_index) # (N_hits, 64)
        x2 = self.gnn2(x1, edge_index) # (N_hits, 128)
        x3 = self.gnn3(x2, edge_index) # (N_hits, 256)

        # Concatenate features from all layers (like a DenseNet)
        x_combined = torch.cat([x1, x2, x3], dim=-1) # (N_hits, 448)

        embeddings = self.post_mlp(x_combined) # (N_hits, 64)

        # L2-normalize embeddings to project them onto a hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=1)

        if self.training:
            # During training, return embeddings for the loss function
            return embeddings
        else:
            # During inference, perform clustering to get track labels
            from sklearn.cluster import DBSCAN

            # Split batched embeddings back into a list of per-event tensors
            embeddings_list, mask = to_dense_batch(embeddings, batch_map)

            all_labels = []
            for i in range(embeddings_list.shape[0]):
                # Get embeddings for the current event, removing padding
                event_embeds = embeddings_list[i, :mask[i].sum(), :]

                # Check if there are enough hits to form a track
                if event_embeds.shape[0] < self.dbscan_min_samples:
                    labels = torch.full((event_embeds.shape[0],), -1, dtype=torch.long)
                else:
                    # Run DBSCAN on CPU. detach().cpu() is important.
                    db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(event_embeds.detach().cpu().numpy())
                    labels = torch.from_numpy(db.labels_).to(x.device)

                all_labels.append(labels)

            # Concatenate labels from all events in the batch into a single tensor
            return torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long, device=x.device)

def make_model(in_features_or_data_obj):
    return HitClassifier(in_features_or_data_obj)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50   

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss, total_train_acc, train_batches = 0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            embeddings = model(batch)
            y, batch_map = batch.y, batch.batch

            # Contrastive loss with random pair sampling
            n_hits = embeddings.shape[0]
            n_pairs = min(n_hits * 4, 16384) 

            idx1 = torch.randint(0, n_hits, (n_pairs,), device=device)
            idx2 = torch.randint(0, n_hits, (n_pairs,), device=device)

            mask = idx1 != idx2
            idx1, idx2 = idx1[mask], idx2[mask]

            is_positive = (y[idx1] == y[idx2]) & (batch_map[idx1] == batch_map[idx2])

            dist_sq = torch.sum((embeddings[idx1] - embeddings[idx2])**2, dim=-1)

            margin = 1.0
            loss = (is_positive.float() * dist_sq + 
                    (1 - is_positive.float()) * F.relu(margin - dist_sq)).mean()

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            with torch.no_grad():
                pred_same = dist_sq < (margin / 2.0)
                acc = (pred_same == is_positive).float().mean()
                total_train_acc += acc.item()
            train_batches += 1

        avg_train_loss = total_train_loss / train_batches
        avg_train_acc = total_train_acc / train_batches
        train_loss_hist.append(avg_train_loss)
        train_acc_hist.append(avg_train_acc)

        # Validation
        model.eval()
        total_val_loss, total_val_acc, val_batches = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                embeddings = model(batch)
                y, batch_map = batch.y, batch.batch

                n_hits = embeddings.shape[0]
                n_pairs = min(n_hits * 4, 16384)

                idx1 = torch.randint(0, n_hits, (n_pairs,), device=device)
                idx2 = torch.randint(0, n_hits, (n_pairs,), device=device)
                mask = idx1 != idx2
                idx1, idx2 = idx1[mask], idx2[mask]

                is_positive = (y[idx1] == y[idx2]) & (batch_map[idx1] == batch_map[idx2])
                dist_sq = torch.sum((embeddings[idx1] - embeddings[idx2])**2, dim=-1)

                margin = 1.0
                loss = (is_positive.float() * dist_sq + 
                        (1 - is_positive.float()) * F.relu(margin - dist_sq)).mean()

                total_val_loss += loss.item()
                pred_same = dist_sq < (margin / 2.0)
                acc = (pred_same == is_positive).float().mean()
                total_val_acc += acc.item()
                val_batches += 1

        avg_val_loss = total_val_loss / val_batches
        avg_val_acc = total_val_acc / val_batches
        val_loss_hist.append(avg_val_loss)
        val_acc_hist.append(avg_val_acc)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

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


