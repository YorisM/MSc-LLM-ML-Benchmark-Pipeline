
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
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.nn import knn_graph
from torch_geometric.utils import scatter
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

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

    def __init__(self, k_neighbors=10):
        # We will use a k-NN graph and an edge classifier. k is a key hyperparameter.
        self.k_neighbors = k_neighbors
        # Scaler for input features
        self.scaler = StandardScaler()
        self.fitted = False

    def _raw_reshape(self, data: torch.Tensor):           
        # data has shape [N_hits, 4] with columns (r, theta, z, layer_id)
        r, theta, z, layer_id = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        # Convert to Cartesian coordinates, which are better for distance calculations
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # We will use x, y, z for graph building and (x, y, z, r, layer_id) as node features
        features = torch.stack([x, y, z, r, layer_id], dim=-1) # [N_hits, 5]
        return features

    # This custom collate function is not used since PyG's DataLoader has its own
    # @staticmethod
    # def _collate_fn(batch: list):
    #    return None

    def make_loader_cfg(self):
        # Return dict or None.  If dict, evaluator uses it to rebuild loader:
        # We need torch_geometric's DataLoader to correctly batch graph data.
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 32,
        }

    def fit(self, data):
        # data is a list of raw event dictionaries
        all_features = []
        for event_dict in data:
            X, _ = _split_X_y(event_dict) # X is a torch.Tensor [N_hits, 4]
            features = self._raw_reshape(X) # features is a torch.Tensor [N_hits, 5]
            all_features.append(features.numpy())

        # Concatenate all hits from all events to fit the scaler
        all_features_cat = np.concatenate(all_features, axis=0)
        self.scaler.fit(all_features_cat)
        self.fitted = True
        return self

    def transform(self, data):
        # data is a torch.Tensor of one event's hits [N_hits, 4]
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform is called.")

        features = self._raw_reshape(data) # [N_hits, 5]
        scaled_features = self.scaler.transform(features.numpy())

        # Return as a float32 tensor
        return torch.from_numpy(scaled_features).float()

def make_preprocessor():
    # This factory function is called by the harness to get a preprocessor instance.
    return MyPreprocessor(k_neighbors=10)

# We provide a custom `make_dataset` function to build graph-structured data
# for each event, which is the most natural representation for this problem.
def make_dataset(events: list, pre: MyPreprocessor, *, train: bool) -> PyGDataset:

    class GraphEventDataset(PyGDataset):
        def __init__(self, events, preprocessor, is_train_split):
            super().__init__()
            self.events = events
            self.pre = preprocessor
            self.is_train = is_train_split
            self.k = self.pre.k_neighbors

            # Process all events into a list of graph Data objects at initialization
            self.graphs = self._build_graphs()

        def len(self) -> int:
            return len(self.graphs)

        def get(self, idx: int) -> Data:
            return self.graphs[idx]

        def _build_graphs(self) -> list[Data]:
            graph_list = []
            for event_dict in self.events:
                # 1. Get raw data and ground truth
                X, track_id = _split_X_y(event_dict) # [N_hits, 4], [N_hits,]

                # 2. Preprocess to get node features
                node_features = self.pre.transform(X) # [N_hits, 5]

                # 3. Build graph edges using k-Nearest-Neighbors on scaled (x,y,z) coordinates.
                # This connects hits that are geometrically close.
                edge_index = knn_graph(node_features[:, :3], k=self.k, loop=False) # [2, E]

                # 4. Create ground truth labels for edges.
                # An edge is "true" if its two nodes belong to the same particle track.
                row, col = edge_index[0], edge_index[1]
                y_true_row = track_id[row]
                y_true_col = track_id[col]

                # We only want to connect true track hits (track_id > 0, as 0 is noise)
                is_track_row = y_true_row > 0
                is_track_col = y_true_col > 0
                same_track = (y_true_row == y_true_col)

                edge_y = (same_track & is_track_row & is_track_col).float().unsqueeze(-1) # [E, 1]

                # 5. Create a torch_geometric Data object
                graph = Data(x=node_features, edge_index=edge_index, y=edge_y)
                graph_list.append(graph)

            return graph_list

    return GraphEventDataset(events, pre, train)

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim=128, n_layers=4):
        super().__init__()

        # 1. Encodes raw node features into a higher-dimensional space
        self.node_encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # 2. A series of graph neural network layers for message passing
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            # Each layer has an "edge model" to compute messages and a "node model" to update nodes
            edge_model = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
            )
            node_model = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
            )
            self.gnn_layers.append(nn.ModuleDict({'edge': edge_model, 'node': node_model}))

        # 3. A final classifier that predicts if an edge connects two hits from the same track
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        # batch is a torch_geometric.data.Batch object, which combines multiple graphs
        x, edge_index = batch.x, batch.edge_index # x: [N_total, F], edge_index: [2, E_total]

        # 1. Initial node embedding
        h = self.node_encoder(x) # [N_total, D_hidden]

        # 2. Message passing loop
        for layer in self.gnn_layers:
            row, col = edge_index

            # Message computation (on edges)
            edge_input = torch.cat([h[row], h[col]], dim=1) # [E_total, 2*D_hidden]
            messages = layer['edge'](edge_input) # [E_total, D_hidden]

            # Message aggregation (for nodes)
            agg_messages = scatter(messages, row, dim=0, reduce='mean', dim_size=h.size(0)) # [N_total, D_hidden]

            # Node update
            node_update_input = torch.cat([h, agg_messages], dim=1) # [N_total, 2*D_hidden]
            node_updates = layer['node'](node_update_input) # [N_total, D_hidden]

            # Residual connection for stability and deeper models
            h = h + node_updates 

        # 3. Final edge classification
        row, col = edge_index
        edge_clf_input = torch.cat([h[row], h[col]], dim=1) # [E_total, 2*D_hidden]

        return self.edge_classifier(edge_clf_input) # [E_total, 1]

def make_model(example_sample):
    # example_sample will be a torch_geometric.data.Data object (the first graph).
    # We can infer the number of input features from it.
    in_features = example_sample.num_node_features
    model = HitClassifier(in_features=in_features, hidden_dim=128, n_layers=4)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 35 # GNNs can take a bit longer to converge
def train_model(model, train_loader, val_loader, epochs):
    # If your method is non-parametric, train_model may be a no-op that returns the 
    # unmodified model and empty metric lists, otherwise:

    # REQUIREMENTS 
    #   Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #   Must return trained_model, train_loss, val_loss, train_acc, val_acc
    #   Implement early-stopping.
    #   Use CUDA - torch.cuda.is_available()
    #   Forward signature must match.

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    # Binary cross-entropy with logits is suitable for our binary edge classification task
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    early_stop_patience = 7
    patience_counter = 0
    best_model_state = None

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss, running_acc, n_batches = 0.0, 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            edge_scores = model(batch) # [E, 1]
            loss = criterion(edge_scores, batch.y) # batch.y is [E, 1]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Accuracy is the fraction of correctly classified edges
            preds = (edge_scores > 0).float()
            acc = (preds == batch.y).float().mean()
            running_acc += acc.item()
            n_batches += 1

        epoch_train_loss = running_loss / n_batches
        epoch_train_acc = running_acc / n_batches
        train_loss_hist.append(epoch_train_loss)
        train_acc_hist.append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        running_val_loss, running_val_acc, n_val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                edge_scores = model(batch)

                loss = criterion(edge_scores, batch.y)
                running_val_loss += loss.item()

                preds = (edge_scores > 0).float()
                acc = (preds == batch.y).float().mean()
                running_val_acc += acc.item()
                n_val_batches += 1

        epoch_val_loss = running_val_loss / n_val_batches
        epoch_val_acc = running_val_acc / n_val_batches
        val_loss_hist.append(epoch_val_loss)
        val_acc_hist.append(epoch_val_acc)

        # print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")

        scheduler.step(epoch_val_loss)

        # --- Early Stopping Logic ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            # print(f"Early stopping at epoch {epoch+1}")
            break

    # Load the best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
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


