
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.7.1, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

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

    train_ld = loader_cls(train_ds, batch_size=batch, shuffle=True,
                          num_workers=workers, collate_fn=collate_fn)
    val_ld   = loader_cls(val_ds,   batch_size=batch, shuffle=False,
                          num_workers=workers, collate_fn=collate_fn)
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# Import torch_geometric, which is available in the environment
try:
    import torch_geometric
    import torch_geometric.data
    import torch_geometric.nn
    import torch_geometric.loader
except ImportError:
    # This is a fallback, but the environment is guaranteed to have the libraries.
    # In a local setup one might need to install them.
    print("PyTorch Geometric not found. Please install it.")
    pass

# We define a custom Dataset class that will be instantiated via the `make_dataset` function below.
# This allows us to correctly package the preprocessed data (a PyG `Data` object) with its labels.
class MyGraphDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events = events
        self.pre = pre
        self.is_train = train
        self.n_events = len(events)

    def __len__(self):
        return self.n_events

    def __getitem__(self, idx):
        # Extract raw data (X: features, y: track_ids) for one event
        X, track_id = _split_X_y(self.events[idx])

        # The preprocessor's transform method will create a graph structure (a PyG Data object)
        # from the hit features X, but it doesn't have access to the track_id.
        data_obj = self.pre.transform(X)

        # We attach the ground truth labels to the graph object here.
        # This is the standard way to handle labels for graphs in PyG.
        data_obj.y = track_id

        return data_obj

def make_dataset(events, pre, train=True):
    """
    This factory function is detected by the evaluation harness.
    It returns an instance of our custom Dataset.
    """
    return MyGraphDataset(events, pre, train=train)

# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # This preprocessor transforms hit data into graph structures suitable for a GNN.
    # It engineers features, normalizes them, and builds a graph by connecting
    # hits in adjacent detector layers.

    def __init__(self):
        # We'll use a standard scaler for the Cartesian coordinates.
        self.scaler = StandardScaler()
        # We'll also store the maximum layer ID for normalization.
        self.max_layer_id = 1.0

    def _cartesian(self, hits):
        # Convert cylindrical (r, theta, z) to Cartesian (x, y, z) coordinates.
        # `hits` is a tensor of shape [N_hits, 4] with columns (r, theta, z, layer_id).
        r, theta, z = hits[:, 0], hits[:, 1], hits[:, 2]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y, z], dim=1) # Returns tensor of shape [N_hits, 3]

    def make_loader_cfg(self):
        # This configures the DataLoader. We need PyG's DataLoader to handle batches of graphs.
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 32, # Batch multiple graphs (events) together
           "shuffle": True
        }

    def fit(self, events):
        # The `fit` method learns statistics from the training data.
        # Here, we learn the mean and std deviation of the Cartesian coordinates
        # and the maximum layer ID across all training events.

        all_coords = []
        max_layer = 0
        for evt in events:
            X, _ = _split_X_y(evt)
            xyz = self._cartesian(X)
            all_coords.append(xyz.numpy())
            if X.shape[0] > 0:
                max_layer = max(max_layer, X[:, 3].max().item())

        if all_coords:
            all_coords_np = np.vstack(all_coords)
            self.scaler.fit(all_coords_np)

        self.max_layer_id = float(max_layer) if max_layer > 0 else 1.0
        return self

    def transform(self, data):
        # The `transform` method applies the learned preprocessing to a single event's data.
        # It returns a `torch_geometric.data.Data` object, representing a graph.

        if data.shape[0] == 0:
            # Handle empty events
            return torch_geometric.data.Data(x=torch.empty((0, 4)), edge_index=torch.empty((2, 0), dtype=torch.long))

        # 1. Feature Engineering and Normalization
        coords_xyz = self._cartesian(data)
        coords_xyz_scaled = self.scaler.transform(coords_xyz.numpy())
        layer_ids_scaled = data[:, 3] / self.max_layer_id
        node_features = torch.cat([
            torch.from_numpy(coords_xyz_scaled).float(),
            layer_ids_scaled.unsqueeze(1)
        ], dim=1) # [N_hits, 4]

        # 2. Graph Construction
        # We build the graph by connecting every hit in a layer to every hit in the next layer.
        layers = data[:, 3].int().numpy()
        hit_indices = np.arange(len(layers))
        unique_layers = np.unique(layers)
        layer_map = {l: hit_indices[layers == l] for l in unique_layers}

        src_list, dst_list = [], []
        for l_id in unique_layers:
            if l_id + 1 in layer_map:
                src_indices = layer_map[l_id]
                dst_indices = layer_map[l_id + 1]
                if len(src_indices) > 0 and len(dst_indices) > 0:
                    src = torch.from_numpy(src_indices).repeat_interleave(len(dst_indices))
                    dst = torch.from_numpy(dst_indices).repeat(len(src_indices))
                    src_list.append(src)
                    dst_list.append(dst)

        if src_list:
            edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0).long()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # 3. Return the graph as a PyG Data object
        return torch_geometric.data.Data(x=node_features, edge_index=edge_index)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    # This is a Graph Neural Network model that learns an embedding for each hit.
    # The goal is that hits from the same track will have similar embeddings.
    def __init__(self, in_features, hidden_dim=128, embedding_dim=16):
        super().__init__()

        # An initial MLP to project input features into a higher-dimensional space.
        self.input_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # A stack of EdgeConv layers. EdgeConv is a powerful GNN layer for point cloud tasks.
        # It performs message passing on a k-NN graph defined in feature space at each layer.
        self.conv1 = torch_geometric.nn.EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            ), aggr='mean'
        )
        self.conv2 = torch_geometric.nn.EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            ), aggr='mean'
        )
        self.conv3 = torch_geometric.nn.EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            ), aggr='mean'
        )

        # A final MLP (the "head") to project the learned representations into the final embedding space.
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, batch):
        # The forward pass processes a batch of graphs.
        # `batch` is a `torch_geometric.data.Batch` object.
        x, edge_index = batch.x, batch.edge_index

        x_in = self.input_net(x) # Shape: [N_total_hits, hidden_dim]

        # Apply the EdgeConv layers sequentially.
        x1 = self.conv1(x_in, edge_index)   # Shape: [N_total_hits, hidden_dim]
        x2 = self.conv2(x1, edge_index)     # Shape: [N_total_hits, hidden_dim]
        x3 = self.conv3(x2, edge_index)     # Shape: [N_total_hits, hidden_dim]

        embeddings = self.output_head(x3)   # Shape: [N_total_hits, embedding_dim]

        # L2-normalize the output embeddings. This is crucial for metric learning,
        # as it forces embeddings onto a hypersphere, making distance comparisons stable.
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

def make_model(in_features):
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50   # We train for more epochs as GNNs can take longer to converge.
MARGIN = 1.0  # Margin for the contrastive loss function.

def weighted_contrastive_loss(embeddings, edge_index, track_ids, margin):
    # This custom loss function implements a weighted contrastive loss.
    # It operates on the pairs of hits defined by the graph edges.

    start_nodes, end_nodes = edge_index
    emb1 = embeddings[start_nodes] # Embeddings of source nodes
    emb2 = embeddings[end_nodes]   # Embeddings of destination nodes

    # Ground truth: 1 if the two hits are from the same track, 0 otherwise.
    labels = (track_ids[start_nodes] == track_ids[end_nodes]).float()

    # Filter out noise hits (which have track_id == 0) from the loss calculation.
    valid_mask = (track_ids[start_nodes] > 0) & (track_ids[end_nodes] > 0)
    if not valid_mask.any():
        return torch.tensor(0.0), torch.tensor(0.0)

    emb1, emb2, labels, valid_mask = emb1[valid_mask], emb2[valid_mask], labels[valid_mask], valid_mask[valid_mask]

    # Calculate Euclidean distance between embedding pairs.
    dists = torch.norm(emb1 - emb2, p=2, dim=1)

    pos_mask = labels == 1
    neg_mask = labels == 0

    # Loss for "positive" pairs (same track): pull them together (minimize distance).
    loss_pos = torch.pow(dists[pos_mask], 2)
    # Loss for "negative" pairs (different tracks): push them apart by at least `margin`.
    loss_neg = torch.pow(F.relu(margin - dists[neg_mask]), 2)

    num_pos = pos_mask.sum().item()
    num_neg = neg_mask.sum().item()

    # We weight the positive loss to counteract the class imbalance (many more negative pairs).
    total_loss = 0
    if num_pos > 0:
        pos_weight = num_neg / num_pos if num_neg > 0 else 1.0
        total_loss += pos_weight * loss_pos.mean()
    if num_neg > 0:
        total_loss += loss_neg.mean()

    # Calculate a proxy accuracy for monitoring: is the distance below a threshold for positive pairs and above for negative?
    with torch.no_grad():
        preds = (dists < (margin / 2.0)).float()
        accuracy = (preds[valid_mask] == labels).float().mean() if len(labels) > 0 else 0.0

    return total_loss, accuracy

def train_model(model, train_loader, val_loader, epochs):
    # This function implements the training loop with early stopping.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7  # Early stopping patience

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss, running_acc, n_batches = 0.0, 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            embeddings = model(batch)
            if batch.edge_index.shape[1] == 0: continue

            loss, acc = weighted_contrastive_loss(embeddings, batch.edge_index, batch.y, MARGIN)
            if torch.isnan(loss) or loss.item() == 0.0: continue

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc.item()
            n_batches += 1

        epoch_train_loss = running_loss / n_batches if n_batches > 0 else 0
        epoch_train_acc = running_acc / n_batches if n_batches > 0 else 0
        train_loss_hist.append(epoch_train_loss)
        train_acc_hist.append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        running_val_loss, running_val_acc, n_val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embeddings = model(batch)

                if batch.edge_index.shape[1] == 0: continue

                v_loss, v_acc = weighted_contrastive_loss(embeddings, batch.edge_index, batch.y, MARGIN)
                if torch.isnan(v_loss): continue

                running_val_loss += v_loss.item()
                running_val_acc += v_acc.item()
                n_val_batches += 1

        epoch_val_loss = running_val_loss / n_val_batches if n_val_batches > 0 else 0
        epoch_val_acc = running_val_acc / n_val_batches if n_val_batches > 0 else 0
        val_loss_hist.append(epoch_val_loss)
        val_acc_hist.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")

        scheduler.step(epoch_val_loss)

        # --- Early Stopping ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # A real implementation would save the best model state here,
            # but for the challenge we just track it.
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist
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
    hits0, _       = first_batch[0]
    in_features    = hits0.shape[-1]                   
    model          = make_model(in_features)

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


