
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

# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATConv, knn_graph


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
        self.means = None
        self.stds = None

    def _raw_reshape(self, data):           
        # No raw data reshaping is needed for this approach.
        return data # Returns identity by default

    @staticmethod
    def _collate_fn(batch: list):
      # This function converts a list of events into a single batched graph.
      # REQUIREMENT: Must output one of the following formats:
      #   A) ragged list: [(X_i, y_i), ...] or [X_i, ...]
      #   B) tuple: (X, y) where X can be Tensor or list/tuple
      #   C) plain Tensor: X
      #   D) dict: {"x": X, "y": y} or {"inputs": X, "labels": y}
      k = 10 # k for k-NN graph construction

      data_list = []
      for x, y in batch:
          # x: (N_hits, F) tensor of normalized node features, F=3 for (x, y, z)
          # y: (N_hits,) tensor of ground truth track_ids

          # Construct a k-Nearest-Neighbors graph.
          # Edges connect hits that are spatially close.
          # This is done on CPU and might be slow for very large events.
          edge_index = knn_graph(x, k=k, loop=False)

          # Generate edge labels: 1 if two connected hits are from the same track, 0 otherwise.
          row, col = edge_index[0], edge_index[1]

          # Noise hits (track_id <= 0) should not form positive edges.
          # An edge is positive only if both hits belong to the same valid track.
          is_valid_track = (y > 0)
          y_row, y_col = y[row], y[col]

          same_track = (y_row == y_col)
          both_valid = is_valid_track[row] & is_valid_track[col]
          edge_y = (same_track & both_valid).float()

          # Create a torch_geometric.data.Data object for the event.
          data = Data(x=x, edge_index=edge_index, y=edge_y, num_nodes=x.size(0))
          data_list.append(data)

      # Batch all Data objects into a single large graph.
      return Batch.from_data_list(data_list)

    def make_loader_cfg(self):
        # Configure the DataLoader for graph data.
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "collate_fn": "MyPreprocessor._collate_fn",
           "batch_size": 32, # Batch size for graph learning is typically smaller
        }

    def fit(self, data):
        # data is the raw list of event dictionaries.
        # This method calculates normalization statistics (mean, std) for coordinate transformation.
        all_hits_coords = []
        for evt in data:
            # Replicate the logic of _split_X_y to get tensors
            X_np = np.column_stack((evt["hit_r"], evt["hit_theta"], evt["hit_z"], evt["layer_id"]))
            X = torch.from_numpy(X_np).float()

            r, theta = X[:, 0], X[:, 1]
            x_coord = r * torch.cos(theta)
            y_coord = r * torch.sin(theta)
            z_coord = X[:, 2]
            coords = torch.stack([x_coord, y_coord, z_coord], dim=-1) # (N_hits, 3)
            all_hits_coords.append(coords)

        all_hits_coords = torch.cat(all_hits_coords, dim=0) # (Total_hits, 3)
        self.means = all_hits_coords.mean(dim=0)
        self.stds = all_hits_coords.std(dim=0)
        self.stds[self.stds == 0] = 1.0  # Avoid division by zero
        return self

    def transform(self, data):
        # `data` is a (N_hits, 4) tensor from EventDataset.
        # This method transforms raw hit features into normalized Cartesian coordinates.
        r, theta, z = data[:, 0], data[:, 1], data[:, 2]
        x_coord = r * torch.cos(theta)
        y_coord = r * torch.sin(theta)
        coords = torch.stack([x_coord, y_coord, z], dim=-1) # (N_hits, 3)

        if self.means is not None:
            # Apply z-score normalization
            coords = (coords - self.means.to(data.device)) / self.stds.to(data.device)

        return coords # (N_hits, 3)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_heads=4):
        super().__init__()
        # This model uses Graph Attention Networks (GAT) to learn node embeddings,
        # followed by an MLP to classify edges.

        # GNN layers for learning node (hit) representations
        self.gnn1 = GATConv(in_features, hidden_dim, heads=num_heads, concat=True, dropout=0.2)
        self.gnn2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True, dropout=0.2)
        self.gnn3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=0.2)

        # MLP for edge classification
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch_x):
        # We assume batch_x is a torch_geometric.data.Batch object,
        # containing the batched graph data from our collate function.
        x, edge_index = batch_x.x, batch_x.edge_index

        # Pass through GNN layers to get node embeddings
        x = F.relu(self.gnn1(x, edge_index)) # (N_total_hits, H*D_hid)
        x = F.relu(self.gnn2(x, edge_index)) # (N_total_hits, H*D_hid)
        x = self.gnn3(x, edge_index)           # (N_total_hits, D_hid)

        # Use node embeddings to predict edge properties
        row, col = edge_index[0], edge_index[1]
        edge_features = torch.cat([x[row], x[col]], dim=1) # (N_edges, 2*D_hid)
        edge_logits = self.edge_mlp(edge_features).squeeze(-1) # (N_edges,)

        return edge_logits

def make_model(in_features):
    # This function is called by the harness to create the model.
    # `in_features` will be the number of features per node.
    return HitClassifier(in_features=in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30
def train_model(model, train_loader, val_loader, epochs):
    # If your method is non-parametric, train_model may be a no-op that returns the unmodified model and empty metric lists, otherwise:

    # REQUIREMENTS 
    #   Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #   Must return trained_model, train_loss, val_loss, train_acc, val_acc
    #   Implement early-stopping.
    #   Use CUDA - torch.cuda.is_available()
    #   Forward signature must match.

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    # Early stopping parameters
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        epoch_train_loss, epoch_train_correct, epoch_train_total = 0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch)
            targets = batch.y

            # Use weighted BCE loss to handle class imbalance (many more negative than positive edges)
            num_pos = targets.sum()
            pos_weight = (targets.numel() - num_pos) / (num_pos + 1e-6)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * batch.num_graphs
            preds = (logits > 0).float()
            epoch_train_correct += (preds == targets).sum().item()
            epoch_train_total += len(targets)

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_train_acc = epoch_train_correct / epoch_train_total
        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)

        # --- Validation Phase ---
        model.eval()
        epoch_val_loss, epoch_val_correct, epoch_val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                targets = batch.y

                num_pos = targets.sum()
                pos_weight = (targets.numel() - num_pos) / (num_pos + 1e-6)
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                loss = loss_fn(logits, targets)

                epoch_val_loss += loss.item() * batch.num_graphs
                preds = (logits > 0).float()
                epoch_val_correct += (preds == targets).sum().item()
                epoch_val_total += len(targets)

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        avg_val_acc = epoch_val_correct / epoch_val_total
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Load the best performing model state
    if best_model_state is not None:
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


