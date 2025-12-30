
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
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, knn_graph
import copy


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
        # We will use a standard scaler for some input features
        self.scaler = StandardScaler()
        # This will hold the weight for the positive class in the loss function
        self.pos_weight = torch.tensor(1.0, dtype=torch.float32)
        # Number of neighbors for graph construction
        self.k = k_neighbors
        self.feature_dim = 6 # r, theta, z, layer_id, x, y

    def _get_cartesian(self, X):
        # X has columns [r, theta, z, layer_id]
        r, theta = X[:, 0], X[:, 1]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y], dim=1) # [N_hits, 2]

    def fit(self, events):
        # 1. Fit the StandardScaler
        # We scale r, theta, z and the derived x, y coordinates
        all_features_to_scale = []
        for evt in events:
            # The harness provides a helper to split the raw event dict
            X, _ = _split_X_y(evt)
            cartesian_coords = self._get_cartesian(X) # [N_hits, 2]
            # Features to be scaled: [r, theta, z, x, y]
            features_to_scale = torch.cat([X[:, :3], cartesian_coords], dim=1) # [N_hits, 5]
            all_features_to_scale.append(features_to_scale)

        all_features_to_scale_np = torch.cat(all_features_to_scale, dim=0).numpy()
        self.scaler.fit(all_features_to_scale_np)

        # 2. Calculate the pos_weight for the imbalanced edge classification task
        n_pos_edges, n_neg_edges = 0, 0
        for evt in events:
            X, y = _split_X_y(evt)

            # Use original coordinates for KNN graph construction
            edge_index = knn_graph(X[:, :3], k=self.k, loop=False) # [2, N_edges]
            row, col = edge_index

            # An edge is positive if hits are from the same track (and not noise)
            y_row, y_col = y[row], y[col]
            positive_mask = (y_row == y_col) & (y_row != 0)

            n_pos_edges += positive_mask.sum().item()
            n_neg_edges += (~positive_mask).sum().item()

        if n_pos_edges > 0:
            self.pos_weight = torch.tensor(n_neg_edges / n_pos_edges, dtype=torch.float32)

        return self

    def transform(self, data):
        # This function is called by the default EventDataset on each item.
        # We will make it an identity operation and handle all processing
        # in the custom collate_fn, which has access to the whole batch
        # and our fitted preprocessor state.
        return data

    def _collate_fn(self, batch: list):
        # This function receives a list of (X, y) tuples, where X is a tensor of hits.
        data_list = []
        for X_orig, y_node in batch:
            # 1. Feature Engineering
            cartesian_coords = self._get_cartesian(X_orig)
            features_to_scale = torch.cat([X_orig[:, :3], cartesian_coords], dim=1)
            scaled_part = self.scaler.transform(features_to_scale.numpy())

            # Final node features: scaled(r,th,z,x,y) and unscaled(layer_id)
            node_features = torch.cat([
                torch.from_numpy(scaled_part).float(),
                X_orig[:, 3].unsqueeze(1).float()
            ], dim=1) # [N_hits, 6]

            # 2. Graph Construction using original coordinates for geometric accuracy
            edge_index = knn_graph(X_orig[:, :3], k=self.k, loop=False) # [2, N_edges]

            # 3. Edge Label Generation
            row, col = edge_index
            y_row, y_col = y_node[row], y_node[col]
            edge_y = (y_row == y_col) & (y_row != 0) # [N_edges]

            data_list.append(Data(x=node_features, edge_index=edge_index, y=edge_y.long()))

        # Use PyG's Batch to collate the list of Data objects into a single graph
        return Batch.from_data_list(data_list)

    def make_loader_cfg(self):
        # Return dict or None.  If dict, evaluator uses it to rebuild loader.
        return {
           # We don't need a custom loader class, just a custom collate function.
           "collate_fn": "self._collate_fn",
           "batch_size": 32,
           "shuffle": True,
        }

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class MLP(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i+1]))
            if i < len(channels) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class HitClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim=128, n_graph_layers=4, heads=4):
        super().__init__()

        # 1. Node Encoder: a simple MLP to project input features to the hidden dimension
        self.node_encoder = MLP([in_features, hidden_dim, hidden_dim])

        # 2. Graph Neural Network Layers for message passing
        self.gnn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for _ in range(n_graph_layers):
            conv = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=0.1)
            self.gnn_layers.append(conv)
            self.norm_layers.append(nn.LayerNorm(hidden_dim))

        # 3. Edge Classifier: an MLP to predict edge connectivity from node embeddings
        self.edge_classifier = MLP([2 * hidden_dim, hidden_dim, hidden_dim, 1])

    def forward(self, batch):
        # batch is a torch_geometric.data.Batch object from our collate_fn
        x, edge_index = batch.x, batch.edge_index # x: [N_total_hits, F], edge_index: [2, N_total_edges]

        # 1. Encode node features
        x = self.node_encoder(x) # [N_total_hits, H]

        # 2. Apply GNN layers with residual connections and layer normalization
        for i in range(len(self.gnn_layers)):
            x_res = x
            x = self.gnn_layers[i](x, edge_index)
            x = self.norm_layers[i](x)
            x = F.relu(x)
            x = x + x_res # Residual connection

        # 3. Predict edge scores
        row, col = edge_index
        x_row, x_col = x[row], x[col] # [N_total_edges, H]

        # Concatenate features of adjacent nodes to form edge features
        edge_features = torch.cat([x_row, x_col], dim=-1) # [N_total_edges, 2*H]

        # Return logits for a binary classification on each edge
        return self.edge_classifier(edge_features) # [N_total_edges, 1]

def make_model(sample):
    # This function needs to be robust to the type of `sample`
    if hasattr(sample, 'num_node_features'): # It's a PyG Data/Batch object
        in_features = sample.num_node_features
    elif isinstance(sample, tuple): # It's a tuple (X, y) from the default dataset
        in_features = sample[0].shape[1]
    else: # It's a tensor
        in_features = sample.shape[1]

    # We know the preprocessor creates 6 features
    preproc = make_preprocessor()
    return HitClassifier(in_features=preproc.feature_dim)


# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30
def train_model(model, train_loader, val_loader, epochs):

    # REQUIREMENTS
    #   Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #   Must return trained_model, train_loss, val_loss, train_acc, val_acc
    #   Implement early-stopping.
    #   Use CUDA - torch.cuda.is_available()
    #   Forward signature must match.

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Scheduler to reduce learning rate on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Loss function with weight for the positive class to handle imbalance
    # We retrieve the pos_weight calculated in the preprocessor
    pos_weight = train_loader.dataset.pre.pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Metrics storage
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    # Early stopping parameters
    best_val_loss = float('inf')
    patience, best_model_state = 5, None
    patience_counter = 0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss, total_correct, total_edges = 0.0, 0, 0
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            edge_logits = model(batch).squeeze() # [N_edges]
            loss = criterion(edge_logits, batch.y.float())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.num_graphs
            preds = (edge_logits > 0)
            total_correct += (preds == batch.y).sum().item()
            total_edges += len(batch.y)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = total_correct / total_edges if total_edges > 0 else 0
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # --- Validation ---
        model.eval()
        running_loss, total_correct, total_edges = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                edge_logits = model(batch).squeeze()
                loss = criterion(edge_logits, batch.y.float())

                running_loss += loss.item() * batch.num_graphs
                preds = (edge_logits > 0)
                total_correct += (preds == batch.y).sum().item()
                total_edges += len(batch.y)

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = total_correct / total_edges if total_edges > 0 else 0
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        scheduler.step(epoch_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # --- Early Stopping ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model state before returning
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


