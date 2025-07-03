
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

# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import EdgeConv


# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
# Custom Dataset for Graph Representation
class GraphEventDataset(Dataset):
    """
    Custom PyTorch Dataset that transforms each event into a graph structure
    compatible with PyTorch Geometric.
    """
    def __init__(self, events, preprocessor, train=True):
        self.events = events
        self.pre = preprocessor
        self.train = train

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        event_data = self.events[idx]
        X, track_id = _split_X_y(event_data)

        # 1. Preprocess features using the provided preprocessor
        node_features = self.pre.transform(X) # (N_hits, num_features)

        # 2. Build graph edges by connecting hits in adjacent layers
        layer_ids = X[:, 3]
        unique_layers = torch.unique(layer_ids).sort()[0]

        edge_list = []
        for i in range(len(unique_layers) - 1):
            layer1_mask = (layer_ids == unique_layers[i])
            layer2_mask = (layer_ids == unique_layers[i+1])

            idx1 = torch.where(layer1_mask)[0]
            idx2 = torch.where(layer2_mask)[0]

            if len(idx1) > 0 and len(idx2) > 0:
                # Create edges between all hits in layer i and all hits in layer i+1
                edge_list.append(torch.cartesian_prod(idx1, idx2))

        if not edge_list:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.cat(edge_list).T # Shape: (2, E)

        # 3. Create edge labels for training. An edge is "true" if the two
        #    connected hits belong to the same particle track.
        if edge_index.shape[1] > 0:
            start_nodes_track_id = track_id[edge_index[0]]
            end_nodes_track_id = track_id[edge_index[1]]

            # A true edge connects hits from the same track_id (and not noise, where track_id=0)
            edge_y = (start_nodes_track_id == end_nodes_track_id) & (start_nodes_track_id != 0)
        else:
            # Handle cases with no edges
            edge_y = torch.empty(0, dtype=torch.bool)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_y=edge_y.long(),
            num_nodes=len(X)
        )

def make_dataset(events, pre, *, train: bool):
    """
    This factory function is called by the harness to create the dataset.
    We return our custom graph dataset.
    """
    return GraphEventDataset(events, pre, train=train)


class MyPreprocessor:
    """
    A preprocessor that standardizes hit features and prepares them for the GNN.
    It calculates cartesian coordinates and normalizes them along with r, theta, z.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def _raw_reshape(self, data):           
        return data

    def make_loader_cfg(self):
        """
        Specifies that PyTorch Geometric's DataLoader should be used, which can
        natively handle batching of graph-structured data.
        """
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 32, # A sensible default
        }

    def _get_features(self, X):
        """Extracts and computes features to be scaled."""
        r, theta, z = X[:, 0], X[:, 1], X[:, 2]
        cart_x = r * torch.cos(theta)
        cart_y = r * torch.sin(theta)

        return torch.stack([r, theta, z, cart_x, cart_y], dim=1) # (N_hits, 5)

    def fit(self, events):
        """Fits the preprocessor by computing normalization statistics from the training data."""
        all_features = []
        for evt in events:
            X, _ = _split_X_y(evt)
            features = self._get_features(X)
            all_features.append(features)

        all_features_tensor = torch.cat(all_features, dim=0)
        self.mean = all_features_tensor.mean(dim=0)
        self.std = all_features_tensor.std(dim=0)
        # Avoid division by zero for features with no variance
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        """Applies the preprocessing to a given set of hits."""
        # X is (N_hits, 4) tensor: [hit_r, hit_theta, hit_z, layer_id]

        features_to_scale = self._get_features(X)
        scaled_features = (features_to_scale - self.mean) / self.std

        # Combine scaled geometric features with the unscaled layer_id
        layer_ids = X[:, 3].unsqueeze(1)

        # Final feature vector for each node (hit)
        return torch.cat([scaled_features, layer_ids], dim=1) # (N_hits, 6)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    """
    A Graph Neural Network model for edge classification.
    It uses EdgeConv layers to learn representations of hits in the context
    of their neighbors, and then predicts if an edge connects two hits from the
    same track.
    """
    def __init__(self, in_features, hidden_dim=128, num_layers=3):
        super().__init__()

        # Initial MLP to project node features into a higher-dimensional space
        self.node_encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Stack of EdgeConv layers for message passing
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(
                EdgeConv(nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim)
                ))
            )

        # Final MLP to classify edges
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        """Defines the forward pass of the model."""
        # The input is a PyG Batch object, containing a batch of graphs
        x, edge_index = batch.x, batch.edge_index # x: (N_total_hits, F), edge_index: (2, N_total_edges)

        # 1. Encode node features
        x = self.node_encoder(x)

        # 2. Propagate information through GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)

        # 3. Predict edge existence
        # For each edge, concatenate features of its start and end nodes
        start_node_features = x[edge_index[0]] # (N_total_edges, H)
        end_node_features = x[edge_index[1]] # (N_total_edges, H)
        edge_features = torch.cat([start_node_features, end_node_features], dim=-1) # (N_total_edges, 2*H)

        # Classify each edge using the final MLP
        return self.edge_classifier(edge_features) # (N_total_edges, 1)

def make_model(in_features):
    # in_features will be 6, as defined by our preprocessor
    return HitClassifier(in_features=in_features, hidden_dim=128, num_layers=3)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25   
def train_model(model, train_loader, val_loader, epochs):
    """
    Main training loop for the GNN model, including validation,
    learning rate scheduling, and early stopping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=False)
    loss_fn = nn.BCEWithLogitsLoss()

    # Early stopping state
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = model.state_dict()

    # History lists for plotting
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss, total_train_correct, total_train_edges = 0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            target = batch.edge_y.float().unsqueeze(1) # Target labels, shape (E, 1)

            # Skip batches with no edges to classify
            if target.numel() == 0:
                continue

            optimizer.zero_grad()
            logits = model(batch) # Predictions, shape (E, 1)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()

            # Accumulate stats
            total_train_loss += loss.item() * batch.num_graphs
            preds = (torch.sigmoid(logits) > 0.5).long()
            total_train_correct += (preds == target).sum().item()
            total_train_edges += len(target)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_train_acc = total_train_correct / total_train_edges if total_train_edges > 0 else 0
        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)

        # --- Validation Phase ---
        model.eval()
        total_val_loss, total_val_correct, total_val_edges = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                target = batch.edge_y.float().unsqueeze(1)

                if target.numel() == 0:
                    continue

                logits = model(batch)
                loss = loss_fn(logits, target)
                total_val_loss += loss.item() * batch.num_graphs
                preds = (torch.sigmoid(logits) > 0.5).long()
                total_val_correct += (preds == target).sum().item()
                total_val_edges += len(target)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_acc = total_val_correct / total_val_edges if total_val_edges > 0 else 0
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Restore the best model found during training
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


