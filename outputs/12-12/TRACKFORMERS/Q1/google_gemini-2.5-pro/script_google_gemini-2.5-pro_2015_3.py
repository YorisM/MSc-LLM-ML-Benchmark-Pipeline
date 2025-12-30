
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

# The harness expects an installation of torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph, EdgeConv
import torch_geometric.loader


# ----------- (CUSTOM) DATASET HANDLING ----------
# This part is necessary because the preprocessor will return a tuple (features, positions),
# and we need a custom Dataset to handle this and pass it to the collate function.
class MyEventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        # The base harness splits X and y. We need the raw X to pass to our preprocessor.
        X_raw, track_id = _split_X_y(self.events[idx])
        # Our preprocessor's transform method returns a tuple
        X_features, pos_for_graph = self.pre.transform(X_raw)
        return (X_features, pos_for_graph, track_id)

def make_dataset(events, pre, *, train: bool):
    """
    This factory function is picked up by the harness to create our custom dataset.
    """
    return MyEventDataset(events, pre, train=train)


# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit()
    #   - transform()

    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    # TIPS
    #   When modifying data features or feature engineering: annotate tensor size as comments after each tensor operation to reduce dimension mismatches.

    def __init__(self):
        # We will scale the geometric features
        self.scaler = StandardScaler()
        # K for the K-Nearest-Neighbors graph construction
        # This is stored here but used in the static collate_fn for consistency.
        self.k = 8

    def _raw_reshape(self, data):           
        return data # Returns identity by default

    @staticmethod
    def _collate_fn(batch: list):
        # This function is the core of the graph-based approach. It converts a list of
        # events into a single large batched graph for efficient processing with PyTorch Geometric.
        # batch: a list of (X_features, pos_for_graph, track_id) tuples from MyEventDataset
        data_list = []
        k = 8 # K for k-NN graph
        for X_features, pos, y_truth in batch:
            # Construct a k-NN graph for the current event using the physical positions.
            edge_index = knn_graph(pos, k=k, loop=False) # Shape: [2, num_nodes * k]

            # For training, we create edge-level labels: 1 if two connected hits
            # belong to the same track, 0 otherwise. We ignore noise hits (track_id == 0).
            row, col = edge_index[0], edge_index[1]
            y_edge = ((y_truth[row] == y_truth[col]) & (y_truth[row] != 0)).float()

            data = Data(x=X_features, edge_index=edge_index, y=y_edge)
            data_list.append(data)

        # `Batch.from_data_list` combines the individual graphs into a single
        # large graph with a `batch` attribute to distinguish nodes from different events.
        return Batch.from_data_list(data_list)

    def make_loader_cfg(self):
        # This configuration tells the harness to use the PyTorch Geometric DataLoader,
        # which is necessary for handling batched graph data, and our custom collate function.
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "collate_fn": "MyPreprocessor._collate_fn",
           "batch_size": 32, # GNNs are memory-intensive, so a smaller batch size is preferred
           "shuffle": True,
        }

    def fit(self, events):
        # We fit the scaler on a representative sample of the data.
        # 1. Combine all events into a single large array.
        all_X = np.vstack([_split_X_y(e)[0].numpy() for e in events])
        # 2. Engineer Cartesian coordinates, which are useful geometric features.
        r, theta = all_X[:, 0], all_X[:, 1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = all_X[:, 2]
        # 3. Create the feature matrix for scaling: [r, theta, z, x, y]
        features_to_scale = np.column_stack((r, theta, z, x, y))
        self.scaler.fit(features_to_scale)
        return self

    def transform(self, X_tensor: torch.Tensor):
        # This method is called for each event.
        X_np = X_tensor.numpy()
        r, theta, z, layer_id = X_np[:, 0], X_np[:, 1], X_np[:, 2], X_np[:, 3]

        # 1. Generate Cartesian coordinates for graph construction. These remain unscaled
        #    to preserve the true physical geometry.
        x_pos = r * np.cos(theta)
        y_pos = r * np.sin(theta)
        pos_for_graph = torch.from_numpy(np.column_stack((x_pos, y_pos, z))).float() # [N_hits, 3]

        # 2. Prepare the node feature matrix for the GNN.
        features_to_scale = np.column_stack((r, theta, z, x_pos, y_pos))
        scaled_features = self.scaler.transform(features_to_scale)

        # The final node feature vector includes scaled geometric features and the unscaled layer_id.
        final_features = np.column_stack((scaled_features, layer_id)) # [N_hits, 6]

        return torch.from_numpy(final_features).float(), pos_for_graph


def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        # The model is a Graph Neural Network that performs edge classification.
        # It learns to predict whether any two connected hits belong to the same track.

        node_dim = 128
        edge_mlp_dim = 256

        # 1. An MLP to encode the raw input node features into a higher-dimensional space.
        self.node_encoder = nn.Sequential(
            nn.Linear(in_features, node_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(node_dim)
        )

        # 2. A series of EdgeConv layers for message passing. EdgeConv is powerful as it
        #    operates on edge features `(x_i, x_j - x_i)`, effectively learning from local neighborhoods.
        #    We use residual connections to aid gradient flow and stabilize training.
        self.edge_conv1 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * node_dim, edge_mlp_dim), nn.LeakyReLU(), nn.LayerNorm(edge_mlp_dim),
                nn.Linear(edge_mlp_dim, node_dim), nn.LeakyReLU(), nn.LayerNorm(node_dim)
            ), aggr='mean'
        )
        self.edge_conv2 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * node_dim, edge_mlp_dim), nn.LeakyReLU(), nn.LayerNorm(edge_mlp_dim),
                nn.Linear(edge_mlp_dim, node_dim), nn.LeakyReLU(), nn.LayerNorm(node_dim)
            ), aggr='mean'
        )
        self.edge_conv3 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * node_dim, edge_mlp_dim), nn.LeakyReLU(), nn.LayerNorm(edge_mlp_dim),
                nn.Linear(edge_mlp_dim, node_dim), nn.LeakyReLU(), nn.LayerNorm(node_dim)
            ), aggr='mean'
        )

        # 3. An MLP to predict edge scores. It takes the concatenated embeddings of two nodes
        #    and outputs a single logit representing the probability of them being in the same track.
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * node_dim, edge_mlp_dim),
            nn.LeakyReLU(),
            nn.Linear(edge_mlp_dim, node_dim),
            nn.LeakyReLU(),
            nn.Linear(node_dim, 1)
        )

    def forward(self, batch_data: Batch):
        # `batch_data` is a PyG `Batch` object from our collate function.
        x, edge_index = batch_data.x, batch_data.edge_index # x: [N_total_hits, F], edge_index: [2, N_total_edges]

        # 1. Encode node features.
        x_encoded = self.node_encoder(x) # [N_total_hits, node_dim]

        # 2. Apply message passing layers with residual connections.
        x1 = self.edge_conv1(x_encoded, edge_index) + x_encoded
        x2 = self.edge_conv2(x1, edge_index) + x1
        x3 = self.edge_conv3(x2, edge_index) + x2

        # 3. Predict scores for each edge in the graph.
        row, col = edge_index[0], edge_index[1]
        edge_features = torch.cat([x3[row], x3[col]], dim=1) # [N_total_edges, 2 * node_dim]

        return self.edge_predictor(edge_features).squeeze(-1) # [N_total_edges]


def make_model(in_features):
    # The harness will determine in_features from the data.
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 40
def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

    # Calculate pos_weight for BCEWithLogitsLoss to handle the severe class imbalance
    # (many more false edges than true track edges).
    pos_count, neg_count = 0, 0
    for i, data in enumerate(train_loader):
        if i >= 10: break # Estimate from first 10 batches
        y_edge = data.y
        pos_count += y_edge.sum().item()
        neg_count += len(y_edge) - pos_count
    pos_weight_val = (neg_count / pos_count) if pos_count > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 8 # For early stopping

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct_preds, total_preds = 0.0, 0, 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            edge_logits = model(data)
            loss = criterion(edge_logits, data.y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.num_graphs
            preds = (edge_logits > 0).float()
            correct_preds += (preds == data.y).sum().item()
            total_preds += len(data.y)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_preds / total_preds if total_preds > 0 else 0
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        model.eval()
        running_val_loss, correct_val_preds, total_val_preds = 0.0, 0, 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                edge_logits = model(data)
                loss = criterion(edge_logits, data.y)

                running_val_loss += loss.item() * data.num_graphs
                preds = (edge_logits > 0).float()
                correct_val_preds += (preds == data.y).sum().item()
                total_val_preds += len(data.y)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val_preds / total_val_preds if total_val_preds > 0 else 0
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} "
              f"| Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}", flush=True)

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # A mechanism to save the best model could be implemented here,
            # but the harness saves the final model state.
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return model, train_loss, val_loss, train_acc, val_acc

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


