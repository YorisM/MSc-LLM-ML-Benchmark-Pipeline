
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
import copy
from torch.nn import Sequential, Linear, ReLU, ModuleList
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MetaLayer


# Helper function and custom dataset that will be used by the solution components.
# This approach encapsulates the complex data transformation logic from raw event hits
# to graph structures suitable for a Graph Neural Network.

def build_graph(X, track_id, pre, apply_scaling=True, max_layer_skip=2):
    """
    Builds a graph representation for a single particle physics event.

    Args:
        X (torch.Tensor): Tensor of hit features [r, theta, z, layer_id].
        track_id (torch.Tensor): Ground truth track ID for each hit.
        pre (MyPreprocessor): The preprocessor instance containing scaling parameters.
        apply_scaling (bool): Flag to apply feature normalization.
        max_layer_skip (int): Maximum number of layers to skip when forming edges.

    Returns:
        torch_geometric.data.Data: A graph object for the event.
    """
    r, theta, z, layer_id = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    # Node features: Convert cylindrical to Cartesian coordinates + r
    x_coord = r * torch.cos(theta)
    y_coord = r * torch.sin(theta)
    node_features = torch.stack([x_coord, y_coord, z, r], dim=-1) # (N_hits, 4)

    # Edge construction: Connect hits in nearby layers
    unique_layers, inverse_indices = torch.unique(layer_id, return_inverse=True)
    layer_indices = [torch.where(inverse_indices == i)[0] for i in range(len(unique_layers))]

    edge_list = []
    for i in range(len(unique_layers)):
        for j in range(i + 1, min(i + 1 + max_layer_skip, len(unique_layers))):
            indices_from = layer_indices[i]
            indices_to = layer_indices[j]
            if len(indices_from) > 0 and len(indices_to) > 0:
                edge_list.append(torch.cartesian_prod(indices_from, indices_to))

    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.cat(edge_list, dim=0).T # (2, N_edges)

    row, col = edge_index[0], edge_index[1]

    # Edge features: Displacement vector between connected hits
    if edge_index.shape[1] > 0:
        edge_features = node_features[col] - node_features[row] # (N_edges, 4) -> dx, dy, dz, dr
    else:
        edge_features = torch.empty((0, 4), dtype=torch.float32)

    # Edge labels (ground truth for training)
    y_edges = (track_id[row] == track_id[col]) & (track_id[row] != 0)
    y_edges = y_edges.to(torch.float32)

    # Apply feature scaling
    if apply_scaling and pre.is_fitted():
        node_features = (node_features - pre.node_mean) / (pre.node_std + 1e-8)
        if edge_features.shape[0] > 0:
            edge_features = (edge_features - pre.edge_mean) / (pre.edge_std + 1e-8)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=y_edges)


class GraphEventDataset(Dataset):
    """Custom PyTorch Dataset to convert events into graphs on-the-fly."""
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        X, track_id = _split_X_y(self.events[idx])
        return build_graph(X, track_id, self.pre, apply_scaling=True)

def make_dataset(events, pre, *, train: bool):
    """Factory function for creating the graph dataset."""
    return GraphEventDataset(events, pre, train=train)


# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    """
    A preprocessor that calculates normalization statistics (mean, std) for node and
    edge features. These stats are computed from the training data and applied during
    graph construction.
    """
    def __init__(self):
        self.node_mean, self.node_std = None, None
        self.edge_mean, self.edge_std = None, None

    def is_fitted(self):
        """Checks if the preprocessor has been fitted."""
        return self.node_mean is not None

    @staticmethod
    def _collate_fn(batch: list):
        """Custom collate function to batch multiple graph Data objects."""
        return Batch.from_data_list(batch)

    def make_loader_cfg(self):
        """Configuration for the DataLoader, specifying the custom collate function."""
        return {"collate_fn": "self._collate_fn", "batch_size": 32}

    def fit(self, data):
        """Computes mean and std for feature normalization from a subset of events."""
        print("Fitting preprocessor...")
        all_nodes, all_edges = [], []
        num_fit_events = min(len(data), 500)
        for i in range(num_fit_events):
            X, track_id = _split_X_y(data[i])
            graph = build_graph(X, track_id, self, apply_scaling=False)
            if graph.num_nodes > 0:
                all_nodes.append(graph.x)
            if graph.num_edges > 0:
                all_edges.append(graph.edge_attr)

        if all_nodes:
            all_nodes_tensor = torch.cat(all_nodes, dim=0)
            self.node_mean = all_nodes_tensor.mean(dim=0)
            self.node_std = all_nodes_tensor.std(dim=0)

        if all_edges:
            all_edges_tensor = torch.cat(all_edges, dim=0)
            self.edge_mean = all_edges_tensor.mean(dim=0)
            self.edge_std = all_edges_tensor.std(dim=0)

        print("Preprocessor fitted.")
        return self

    def transform(self, data):
        """
        Identity transform, as feature transformation is handled inside the
        GraphEventDataset.
        """
        return data

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class EdgeModel(nn.Module):
    """A model to update edge features in the GNN."""
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.net = Sequential(
            Linear(2 * node_dim + edge_dim, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.net(out)

class NodeModel(nn.Module):
    """A model to update node features in the GNN."""
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.net = Sequential(
            Linear(node_dim + edge_dim, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        # Aggregate edge features for each node
        agg = torch_geometric.utils.scatter(edge_attr, row, dim=0, reduce='mean')
        out = torch.cat([x, agg], dim=1)
        return self.net(out)

class HitClassifier(nn.Module):
    """
    Graph Neural Network for classifying edges between hits.
    The architecture uses encoders, a series of message passing layers (MetaLayer),
    and a final classifier.
    """
    def __init__(self, data_sample, hidden_dim=128, n_graph_iters=3):
        super().__init__()

        node_in_dim = data_sample.num_node_features
        edge_in_dim = data_sample.num_edge_features

        self.node_encoder = Sequential(Linear(node_in_dim, hidden_dim), ReLU())
        self.edge_encoder = Sequential(Linear(edge_in_dim, hidden_dim), ReLU())

        self.gnn_layers = ModuleList()
        for _ in range(n_graph_iters):
            self.gnn_layers.append(
                MetaLayer(
                    EdgeModel(hidden_dim, hidden_dim, hidden_dim),
                    NodeModel(hidden_dim, hidden_dim, hidden_dim)
                )
            )

        self.edge_classifier = Sequential(
            Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for gnn in self.gnn_layers:
            x_new, edge_attr_new, _ = gnn(x, edge_index, edge_attr)
            # Residual connections for stability
            x = x + x_new
            edge_attr = edge_attr + edge_attr_new

        return self.edge_classifier(edge_attr).squeeze(-1)

def make_model(in_features):
    # in_features is a torch_geometric.data.Data object (the first sample)
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50
def train_model(model, train_loader, val_loader, epochs):
    """
    Trains the GNN model using a standard training loop with validation,
    learning rate scheduling, and early stopping.
    """
    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    print("Calculating positive weight for BCE loss...")
    num_pos, num_total = 0, 0
    for batch in train_loader:
      if batch.y is not None:
        num_pos += batch.y.sum()
        num_total += len(batch.y)
    num_neg = num_total - num_pos
    pos_weight = num_neg / num_pos if num_pos > 0 else torch.tensor(1.0)
    print(f"Positive weight: {pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', factor=0.5, patience=5, min_lr=1e-6)

    best_val_acc, epochs_no_improve, patience = -1, 0, 10
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_count = 0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            if batch.edge_index.shape[1] == 0: continue

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            preds = (output > 0).float()
            total_correct += (preds == batch.y).sum().item()
            total_count += len(batch.y)

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_train_acc = total_correct / total_count if total_count > 0 else 0
        train_loss.append(avg_train_loss); train_acc.append(avg_train_acc)

        model.eval()
        total_loss, total_correct, total_count = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if batch.edge_index.shape[1] == 0: continue
                output = model(batch)

                loss = criterion(output, batch.y)
                total_loss += loss.item() * batch.num_graphs
                preds = (output > 0).float()
                total_correct += (preds == batch.y).sum().item()
                total_count += len(batch.y)

        avg_val_loss = total_loss / len(val_loader.dataset)
        avg_val_acc = total_correct / total_count if total_count > 0 else 0
        val_loss.append(avg_val_loss); val_acc.append(avg_val_acc)

        scheduler.step(avg_val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    model.load_state_dict(best_model_state)
    return model, train_loss, val_loss, train_acc, val_acc

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


