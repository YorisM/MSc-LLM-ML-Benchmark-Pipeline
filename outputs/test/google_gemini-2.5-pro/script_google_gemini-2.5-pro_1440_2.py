
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
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# This solution uses PyTorch Geometric (PyG) for its Graph Neural Network.
# These modules are available in the execution environment.
import torch_geometric.data
import torch_geometric.loader
from torch_geometric.nn import EdgeConv


# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # This preprocessor standardizes the hit features and prepares them for
    # graph construction. It uses Cartesian coordinates to avoid the
    # periodicity issues of cylindrical coordinates.

    def __init__(self):
        # Using StandardScaler for feature normalization
        self.scaler = StandardScaler()
        self.fitted = False
        # The number of features created by the preprocessor's transform method.
        self.n_features = 5

    def _events_to_feature_tensor(self, events: list[dict]):
        # Helper to extract and convert all features from a list of events for fitting the scaler.
        all_features = []
        for evt in events:
            # Using cartesian coordinates to avoid periodicity issues with theta
            x = evt["hit_r"] * np.cos(evt["hit_theta"])
            y = evt["hit_r"] * np.sin(evt["hit_theta"])
            z = evt["hit_z"]
            r = evt["hit_r"]
            # Features for scaling: x, y, z, r
            features = np.stack([x, y, z, r], axis=-1)
            all_features.append(features)
        return np.vstack(all_features)

    def make_loader_cfg(self):
        # Configure the data loader. We use PyG's DataLoader to automatically
        # batch multiple graph (event) objects into a single large graph.
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 16, # No. of graphs (events) per batch
           "shuffle": True,
        }

    def fit(self, data: list[dict]):
        # Fit scalers to the training data.
        feature_matrix = self._events_to_feature_tensor(data) # [N_total_hits, 4]
        self.scaler.fit(feature_matrix)
        self.fitted = True
        return self

    def transform(self, data: torch.Tensor):
        # Transform a single event's feature tensor.
        # Input `data` is a torch.Tensor of shape [N_hits, 4]
        # with columns: hit_r, hit_theta, hit_z, layer_id
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before use.")

        r, theta, z, layer_id = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        # Convert to Cartesian coordinates
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # Scale features using the fitted scaler
        cartesian_features = torch.stack([x, y, z, r], dim=-1) # [N_hits, 4]
        scaled_features = self.scaler.transform(cartesian_features.numpy())
        scaled_features = torch.from_numpy(scaled_features).float() # [N_hits, 4]

        # Append raw layer_id as a fifth feature
        final_features = torch.cat(
            [scaled_features, layer_id.unsqueeze(1)], dim=1
        ) # [N_hits, 5]
        return final_features

def make_preprocessor():
    return MyPreprocessor()

# By defining this global function, we override the default dataset creation logic in the harness.
def make_dataset(events, pre, *, train: bool):
    # This custom Dataset class processes each event into a graph structure
    # compatible with PyTorch Geometric.
    class GraphDataset(torch_geometric.data.Dataset):
        def __init__(self, events, preprocessor):
            super().__init__()
            self.events = events
            self.pre = preprocessor

        def len(self):
            return len(self.events)

        def get(self, idx):
            # 1. Load event data and apply feature transformation
            X_raw, track_ids = _split_X_y(self.events[idx])
            node_features = self.pre.transform(X_raw)

            # 2. Build graph edges by connecting hits in adjacent layers
            layer_ids = X_raw[:, 3]
            unique_layers = torch.unique(layer_ids).sort()[0]

            edge_list_src, edge_list_dst = [], []

            for i in range(len(unique_layers) - 1):
                nodes_in_layer1 = (layer_ids == unique_layers[i]).nonzero(as_tuple=True)[0]
                nodes_in_layer2 = (layer_ids == unique_layers[i+1]).nonzero(as_tuple=True)[0]

                # Create all-to-all directed edges from layer i to i+1
                if len(nodes_in_layer1) > 0 and len(nodes_in_layer2) > 0:
                    src = nodes_in_layer1.repeat_interleave(len(nodes_in_layer2))
                    dst = nodes_in_layer2.repeat(len(nodes_in_layer1))
                    edge_list_src.append(src)
                    edge_list_dst.append(dst)

            if not edge_list_src: # Handle cases with no valid layer pairs
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.stack([torch.cat(edge_list_src), torch.cat(edge_list_dst)], dim=0)

            # 3. Create edge labels: 1 if hits belong to the same track (and not noise), else 0.
            track_ids_src = track_ids[edge_index[0]]
            track_ids_dst = track_ids[edge_index[1]]
            edge_y = ((track_ids_src == track_ids_dst) & (track_ids_src != 0)).float()

            return torch_geometric.data.Data(
                x=node_features, edge_index=edge_index, y=edge_y
            )

    return GraphDataset(events, pre)

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    # This model uses a Graph Neural Network (GNN) to classify edges between hits.
    # An edge is 'true' if the two connected hits belong to the same particle track.
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()

        # A series of EdgeConv layers to learn representations of hits (nodes)
        # by aggregating information from their neighbors.
        mlp1 = nn.Sequential(
            nn.Linear(2 * in_features, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        mlp2 = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
        )
        mlp3 = nn.Sequential(
            nn.Linear(2 * (hidden_dim * 2), hidden_dim * 4), nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4), nn.ReLU(),
        )

        self.edge_conv1 = EdgeConv(nn=mlp1, aggr='mean')
        self.edge_conv2 = EdgeConv(nn=mlp2, aggr='mean')
        self.edge_conv3 = EdgeConv(nn=mlp3, aggr='mean')

        # The final node embedding dimension includes skip connections from all layers
        node_feature_dim = in_features + hidden_dim + (hidden_dim*2) + (hidden_dim*4)

        # An MLP head to classify edges based on the concatenated features of their endpoint nodes.
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * node_feature_dim, hidden_dim * 4), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, batch: torch_geometric.data.Batch):
        # The input is a PyG Batch object, containing a single large graph
        x, edge_index = batch.x, batch.edge_index

        # Message passing through GNN layers, with skip connections
        x1 = self.edge_conv1(x, edge_index)
        x2 = self.edge_conv2(x1, edge_index)
        x3 = self.edge_conv3(x2, edge_index)

        # Concatenate node features from all layers for a rich representation
        x_all = torch.cat([x, x1, x2, x3], dim=-1) # [N_nodes, node_feature_dim]

        # Construct edge features from the final node features
        edge_features = torch.cat([x_all[edge_index[0]], x_all[edge_index[1]]], dim=-1) # [N_edges, 2*node_feature_dim]

        # Predict a score for each edge
        return self.edge_classifier(edge_features)

def make_model(in_features):
    # The harness determines in_features, which may be unreliable for our PyG data structure.
    # We use the known feature dimension (5) from our preprocessor for robustness.
    return HitClassifier(in_features=5, hidden_dim=64)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50   # Use a higher epoch count with early stopping
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Calculate pos_weight for BCEWithLogitsLoss to handle the imbalanced
    # nature of edge classification (many more false edges than true ones).
    num_pos, num_total = 0, 0
    with torch.no_grad():
        for batch in train_loader:
            num_pos += batch.y.sum().item()
            num_total += len(batch.y)
    # Handle case where num_pos might be 0
    pos_weight = torch.tensor((num_total - num_pos) / num_pos if num_pos > 0 else 1.0, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10
    best_model_state = None

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, n_samples = 0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).squeeze(-1) # [N_edges]
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            preds = (out > 0).float()
            total_correct += (preds == batch.y).sum().item()
            n_samples += len(batch.y)

        train_loss.append(total_loss / len(train_loader.dataset))
        train_acc.append(total_correct / n_samples if n_samples > 0 else 0)

        # Validation loop
        model.eval()
        total_loss, total_correct, n_samples = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch).squeeze(-1)
                loss = criterion(out, batch.y)

                total_loss += loss.item() * batch.num_graphs
                preds = (out > 0).float()
                total_correct += (preds == batch.y).sum().item()
                n_samples += len(batch.y)

        avg_val_loss = total_loss / len(val_loader.dataset)
        val_loss.append(avg_val_loss)
        val_acc.append(total_correct / n_samples if n_samples > 0 else 0)

        scheduler.step(avg_val_loss)

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Restore the best model state found during training
    if best_model_state:
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


