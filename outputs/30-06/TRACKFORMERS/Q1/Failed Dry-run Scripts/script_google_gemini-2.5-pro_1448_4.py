
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
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU, LayerNorm
import torch_geometric.utils


# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit()
    #   - transform()

    # REQUIREMENTS
    #   IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   May allocate NumPy arrays or Torch tensors internally, but:
    #   transform() must be deterministic.
    #   fit(events) receives the *raw* event dicts list, not a tensor batch.
    #   Store only derived parameters needed for transform i.e. do not store the raw data
    #    itself in the preprocessor object.

    # TIPS
    #   When modifying data features or feature engineering: annotate tensor size as comments after 
    #   each tensor operation to reduce dimension mismatches.

    def __init__(self):
        # This will be a StandardScaler for the x, y, z coordinates of the hits.
        self.scaler = StandardScaler()
        # This will hold the weight for the positive class in the loss function,
        # calculated from the training set to handle class imbalance.
        self.pos_weight = torch.tensor(1.0)

    def _raw_reshape(self, data):           
        # No raw data reshaping is done.
        return data # Returns identity by default

    def make_loader_cfg(self):
        # Return dict or None.  If dict, evaluator uses it to rebuild loader:
        # We use torch_geometric's DataLoader.
        # Batch size 1 is safer for graph data, as graphs can be large.
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 1,
        }

    def fit(self, events):
        # We fit the scaler on the cartesian coordinates of all hits in the training set.
        all_x, all_y, all_z = [], [], []
        for evt in events:
            r, theta, z = evt["hit_r"], evt["hit_theta"], evt["hit_z"]
            all_x.append(r * np.cos(theta))
            all_y.append(r * np.sin(theta))
            all_z.append(z)

        # Concatenate all hits into a single numpy array for fitting the scaler.
        features = np.column_stack((np.concatenate(all_x), 
                                    np.concatenate(all_y), 
                                    np.concatenate(all_z))) # (Total_hits, 3)
        self.scaler.fit(features)
        return self

    def transform(self, data):
        # The input data is a tensor of shape (N_hits, 4) with columns (r, theta, z, layer_id).
        # We transform it to scaled cartesian coordinates (x, y, z) for the GNN node features.
        r, theta, z = data[:, 0], data[:, 1], data[:, 2] # (N_hits,) each
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # Stack coordinates into a (N_hits, 3) tensor.
        features_np = torch.stack([x, y, z], dim=1).numpy()

        # Apply the fitted scaler.
        scaled_features = self.scaler.transform(features_np)

        # Return the scaled features as a float tensor.
        return torch.from_numpy(scaled_features).float() # (N_hits, 3)

# This custom dataset processes events into graph structures for torch_geometric.
class GraphDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events = events
        self.pre = pre
        self.train = train
        # Process all events into graphs upon initialization.
        self.graphs = self._process_events()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def _process_events(self):
        graphs = []
        total_pos_edges = 0
        total_neg_edges = 0

        for evt in self.events:
            X, track_id = _split_X_y(evt)

            # Get scaled (x,y,z) node features from the preprocessor.
            node_features = self.pre.transform(X) # (N_hits, 3)

            # Use original, unscaled coordinates for geometric graph construction.
            r, theta, z, layer_id_ = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
            coords = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=1) # (N_hits, 3)

            unique_layers = torch.unique(layer_id_)
            unique_layers = torch.sort(unique_layers)[0]

            # Group hit indices by their layer.
            layer_map = {layer.item(): i for i, layer in enumerate(unique_layers)}
            layer_indices = [[] for _ in range(len(unique_layers))]
            for i, lid in enumerate(layer_id_):
                layer_indices[layer_map[lid.item()]].append(i)

            # Construct edges by connecting hits in adjacent layers.
            edge_list = []
            for i in range(len(unique_layers) - 1):
                current_layer_idxs_list = layer_indices[i]
                next_layer_idxs_list = layer_indices[i+1]

                if not current_layer_idxs_list or not next_layer_idxs_list:
                    continue

                current_layer_idxs = torch.tensor(current_layer_idxs_list, dtype=torch.long)
                next_layer_idxs = torch.tensor(next_layer_idxs_list, dtype=torch.long)

                # Use a k-D tree for efficient k-NN search.
                tree = cKDTree(coords[next_layer_idxs].numpy())
                k = min(5, len(next_layer_idxs))
                dist, nn_indices = tree.query(coords[current_layer_idxs].numpy(), k=k)

                for j, src_idx in enumerate(current_layer_idxs):
                    dest_indices = nn_indices[j]
                    if np.isscalar(dest_indices): dest_indices = [dest_indices]
                    for dest_node_local_idx in dest_indices:
                        if dest_node_local_idx < len(next_layer_idxs):
                             dest_idx = next_layer_idxs[dest_node_local_idx]
                             edge_list.append([src_idx.item(), dest_idx.item()])

            if not edge_list:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_y = torch.empty((0,), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_list, dtype=torch.long).T
                # Make the graph undirected
                edge_index, _ = torch_geometric.utils.to_undirected(edge_index, num_nodes=node_features.shape[0])

                # Calculate edge labels: 1 if hits are from the same track, 0 otherwise.
                y_true = (track_id[edge_index[0]] == track_id[edge_index[1]])
                # Noise hits (track_id=0) cannot form true edges.
                not_noise = (track_id[edge_index[0]] != 0) & (track_id[edge_index[1]] != 0)
                edge_y = (y_true & not_noise).long()

                if self.train:
                    total_pos_edges += torch.sum(edge_y == 1)
                    total_neg_edges += torch.sum(edge_y == 0)

            graph = Data(x=node_features, edge_index=edge_index, edge_y=edge_y)
            graphs.append(graph)

        # Calculate and store pos_weight in the preprocessor instance.
        if self.train and total_pos_edges > 0:
            self.pre.pos_weight = torch.tensor(total_neg_edges / total_pos_edges, dtype=torch.float32)

        return graphs

# This hook is called by the harness to create the dataset.
def make_dataset(events, pre, *, train: bool):
    return GraphDataset(events, pre, train=train)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim=128, n_layers=4):
        super().__init__()

        # Node encoder: MLP to embed initial node features into a higher-dimensional space.
        self.node_encoder = Sequential(
            Linear(in_features, hidden_dim),
            ReLU(),
            LayerNorm(hidden_dim),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            LayerNorm(hidden_dim)
        )

        # GNN layers: A stack of SAGEConv layers for message passing.
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))

        # Edge classifier: MLP to predict if an edge connects two hits from the same track.
        self.edge_classifier = Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            ReLU(),
            LayerNorm(hidden_dim),
            Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index # (N_total_hits,_in_features), (2, N_total_edges)

        # 1. Encode node features.
        h = self.node_encoder(x) # (N_total_hits, D_hid)

        # 2. Apply GNN layers. Residual connections could be added for deeper models.
        for layer in self.gnn_layers:
            h = layer(h, edge_index) + h # Residual connection
            h = F.relu(h)

        # 3. Extract node embeddings for each edge.
        src, dst = edge_index
        h_src = h[src] # (N_total_edges, D_hid)
        h_dst = h[dst] # (N_total_edges, D_hid)

        # 4. Concatenate embeddings and classify edges.
        edge_features = torch.cat([h_src, h_dst], dim=1) # (N_total_edges, 2*D_hid)
        edge_logits = self.edge_classifier(edge_features) # (N_total_edges, 1)

        return edge_logits.squeeze(-1) # (N_total_edges,)

def make_model(in_features):
    # The number of input features is determined by the preprocessor's output.
    # Here, it's 3 for scaled (x, y, z).
    return HitClassifier(in_features=in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30
def train_model(model, train_loader, val_loader, epochs):
    # REQUIREMENTS 
    #   Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #   Must return trained_model, train_loss, val_loss, train_acc, val_acc
    #   Implement early-stopping.
    #   Forward signature must match.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    # The pos_weight is retrieved from the preprocessor via the dataset.
    pos_weight = train_loader.dataset.pre.pos_weight.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct_preds, total_preds = 0.0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch)
            target = batch.edge_y.float()

            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.num_graphs
            preds = (logits > 0).long()
            correct_preds += (preds == target.long()).sum().item()
            total_preds += len(target)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_preds / total_preds if total_preds > 0 else 0
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        model.eval()
        running_val_loss, correct_val_preds, total_val_preds = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                target = batch.edge_y.float()

                loss = criterion(logits, target)
                running_val_loss += loss.item() * batch.num_graphs
                preds = (logits > 0).long()
                correct_val_preds += (preds == target.long()).sum().item()
                total_val_preds += len(target)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val_preds / total_val_preds if total_val_preds > 0 else 0
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
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


