
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

# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MetaLayer
import torch_geometric.utils
from sklearn.preprocessing import StandardScaler

# 1.1 -------- OPTIONAL: CUSTOM DATASET / DATA-CLASS --------
class GraphEventDataset(Dataset):
    """
    Custom Dataset to transform raw events into a graph representation.
    Each event becomes a graph where hits are nodes and potential connections
    between hits in nearby layers are edges.
    """
    def __init__(self, events, pre, train=True):
        self.events = events
        self.pre = pre
        self.train = train
        self.graphs = self._build_graphs()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def _build_graphs(self):
        """Processes all events into a list of torch_geometric.data.Data objects."""
        graphs = []
        for evt in self.events:
            # Split features and labels from the raw event dictionary
            X, track_id = _split_X_y(evt)

            # Preprocess node features using the provided preprocessor
            node_features = self.pre.transform(X) # (N_hits, F_node)

            # Construct graph edges, edge features, and edge labels
            edge_index, y_edge, edge_features = self._build_edges(X, track_id)

            # Create the graph data object
            graph = Data(x=node_features,
                         edge_index=edge_index,
                         edge_attr=edge_features,
                         y=y_edge,  # Edge-level labels for training
                         track_id=track_id)  # Node-level truth for validation/analysis
            graphs.append(graph)
        return graphs

    def _build_edges(self, X_raw, track_id):
        """Constructs edges between hits in different layers."""
        r, theta, z, layers = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2], X_raw[:, 3]

        # Convert to Cartesian coordinates for distance-based features
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        cartesian_coords = torch.stack([x, y, z], dim=-1) # (N_hits, 3)

        # Group hits by their layer ID
        unique_layers, inverse_indices = torch.unique(layers, return_inverse=True)
        hits_by_layer = [torch.where(inverse_indices == i)[0] for i in range(len(unique_layers))]
        layer_map = {int(l.item()): i for i, l in enumerate(unique_layers)}

        edge_list = []
        max_layer_skip = 2  # Connect hits up to 2 layers apart

        # Create edges between hits in successive layers
        for i, l_i_val in enumerate(unique_layers):
            for j in range(1, max_layer_skip + 1):
                l_j_val = l_i_val + j
                if int(l_j_val.item()) not in layer_map:
                    continue

                idx_i = hits_by_layer[i]
                idx_j = hits_by_layer[layer_map[int(l_j_val.item())]]

                # Create a full bipartite graph between the two sets of hits
                edge_product = torch.cartesian_prod(idx_i, idx_j)
                edge_list.append(edge_product)

        if not edge_list:
             return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float), torch.empty((0, 3), dtype=torch.float)

        edge_index = torch.cat(edge_list, dim=0).T.contiguous() # (2, N_edges)

        # Edge features are the displacement vector between connected hits
        start_nodes, end_nodes = edge_index[0], edge_index[1]
        edge_features = cartesian_coords[end_nodes] - cartesian_coords[start_nodes] # (N_edges, 3)

        # Edge labels: 1 if hits belong to the same track, 0 otherwise
        y_edge = (track_id[start_nodes] == track_id[end_nodes]).float()
        # Exclude pairs involving noise hits (track_id == 0)
        y_edge[(track_id[start_nodes] == 0) | (track_id[end_nodes] == 0)] = 0

        return edge_index, y_edge, edge_features


def make_dataset(events, pre, train: bool):
    """Factory function for creating the custom dataset."""
    return GraphEventDataset(events, pre, train=train)


# 1.2 ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        """Initializes the preprocessor, which will learn scaling parameters."""
        self.scaler = StandardScaler()

    def _raw_reshape(self, data):
        return data

    def make_loader_cfg(self):
        """Specifies configuration for the DataLoader.
        We use PyTorch Geometric's DataLoader, which handles batches of graphs.
        """
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 1, # Each graph is an event; PyG handles batching internally
           "shuffle": True, # Shuffle for training
           "num_workers": 0,
        }

    def fit(self, events):
        """Fits the StandardScaler on the training data."""
        all_features_list = []
        for evt in events:
            X_tensor, _ = _split_X_y(evt)
            r, theta, z = X_tensor[:, 0], X_tensor[:, 1], X_tensor[:, 2]
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)

            # Features to scale: r, z, x, y
            features_to_scale = torch.stack([r, z, x, y], dim=-1)
            all_features_list.append(features_to_scale)

        all_features = torch.cat(all_features_list, dim=0).numpy()
        self.scaler.fit(all_features)
        return self

    def transform(self, data):
        """Applies the learned transformation to the node features."""
        r, theta, z, layer_id = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        features_to_scale = torch.stack([r, z, x, y], dim=-1) # (N_hits, 4)
        scaled_features = torch.from_numpy(self.scaler.transform(features_to_scale.numpy())).float()

        # Final node features: scaled coordinates, plus unscaled theta and layer_id
        all_node_features = torch.cat([scaled_features, theta.unsqueeze(-1), layer_id.unsqueeze(-1)], dim=-1)
        # Shape: (N_hits, 6)

        return all_node_features

def make_preprocessor():
    """Factory function for the preprocessor."""
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class EdgeModel(torch.nn.Module):
    """MLP for updating edge features in the GNN."""
    def __init__(self, node_in, edge_in, hidden_dim, out_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_in * 2 + edge_in, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], dim=1) # (E, 2*F_node + F_edge)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    """MLP for updating node features in the GNN."""
    def __init__(self, node_in, edge_in, hidden_dim, out_dim):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in + edge_in, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        _, dest = edge_index
        # Aggregate messages (edge features) for each node
        agg_msg = torch_geometric.utils.scatter(edge_attr, dest, dim=0, dim_size=x.size(0), reduce='mean')
        out = torch.cat([x, agg_msg], dim=1) # (N, F_node + F_edge_aggregated)
        return self.node_mlp(out)

class HitClassifier(nn.Module):
    """
    Graph Neural Network for edge classification.
    Predicts if two connected hits belong to the same particle track.
    """
    def __init__(self, in_features):
        super().__init__()
        node_input_dim = in_features # from preprocessor
        edge_input_dim = 3 # dx, dy, dz
        hidden_dim = 128

        # 1. Encoders for initial node and edge features
        self.node_encoder = nn.Sequential(nn.Linear(node_input_dim, hidden_dim))
        self.edge_encoder = nn.Sequential(nn.Linear(edge_input_dim, hidden_dim))

        # 2. Stack of interaction layers (MetaLayer)
        self.interaction_layers = nn.ModuleList()
        for _ in range(4): # 4 interaction layers
             op = MetaLayer(
                edge_model=EdgeModel(hidden_dim, hidden_dim, hidden_dim, hidden_dim),
                node_model=NodeModel(hidden_dim, hidden_dim, hidden_dim, hidden_dim),
             )
             self.interaction_layers.append(op)

        # 3. Final classifier to predict edge truth
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch_x):
        # batch_x is a torch_geometric.data.Batch object
        x, edge_index, edge_attr = batch_x.x, batch_x.edge_index, batch_x.edge_attr

        # Encode node and edge features into latent space
        x = self.node_encoder(x) # (N_total, hidden_dim)
        edge_attr = self.edge_encoder(edge_attr) # (E_total, hidden_dim)

        # Propagate information through the graph with interaction layers
        for layer in self.interaction_layers:
            x_res, edge_attr_res, _ = layer(x, edge_index, edge_attr)
            # Add residual connections
            x = x + x_res
            edge_attr = edge_attr + edge_attr_res

        # Classify edges based on the final node embeddings
        start_nodes, end_nodes = edge_index
        classifier_input = torch.cat([x[start_nodes], x[end_nodes]], dim=1) # (E_total, 2*hidden_dim)
        edge_scores = self.edge_classifier(classifier_input).squeeze(-1) # (E_total,)

        return edge_scores

def make_model(input_features):
    """Factory function for the model."""
    return HitClassifier(input_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25 

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.3, patience=3, verbose=False)

    # Estimate pos_weight for BCE loss to handle class imbalance
    # A large weight for the positive class (true edges) is needed as they are rare
    pos_count, neg_count = 0, 0
    for i, data in enumerate(train_loader):
        if i >= 10: break # Use 10 batches to estimate
        y = data.y
        pos_count += y.sum().item()
        neg_count += len(y) - y.sum().item()
    pos_weight = torch.tensor([neg_count / pos_count] if pos_count > 0 else 1.0).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_val_acc = -1
    patience_counter = 0
    PATIENCE = 7 # Early stopping patience

    for epoch in range(epochs):
        model.train()
        running_loss, total_correct, total_edges = 0, 0, 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data.y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.num_graphs
            preds = (outputs > 0)
            total_correct += (preds == data.y).sum().item()
            total_edges += len(data.y)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = total_correct / total_edges if total_edges > 0 else 0
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        model.eval()
        running_vloss, total_vcorrect, total_vedges = 0, 0, 0
        with torch.no_grad():
            for vdata in val_loader:
                vdata = vdata.to(device)
                voutputs = model(vdata)
                vloss = criterion(voutputs, vdata.y)

                running_vloss += vloss.item() * vdata.num_graphs
                vpreds = (voutputs > 0)
                total_vcorrect += (vpreds == vdata.y).sum().item()
                total_vedges += len(vdata.y)

        epoch_val_loss = running_vloss / len(val_loader.dataset)
        epoch_val_acc = total_vcorrect / total_vedges if total_vedges > 0 else 0
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        scheduler.step(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            # print(f"Early stopping at epoch {epoch+1}")
            break

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


