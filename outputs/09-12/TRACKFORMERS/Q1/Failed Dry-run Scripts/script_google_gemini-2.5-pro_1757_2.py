
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
import collections

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
        self.scaler = StandardScaler()
        self.fitted = False
        # Features in order: r, z, x, y
        # We engineer x, y and scale r, z, x, y.
        # We also pass through sin(theta), cos(theta), and a scaled layer_id.
        self.n_features_out = 7

    def _get_features_from_dict(self, data_dict):
        r = data_dict["hit_r"]
        theta = data_dict["hit_theta"]
        z = data_dict["hit_z"]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        # Features to be scaled
        return np.stack([r, z, x, y], axis=1)

    def fit(self, events):
        # Concatenate features from all events
        all_features = np.vstack([self._get_features_from_dict(evt) for evt in events])
        self.scaler.fit(all_features)
        self.fitted = True
        return self

    def transform(self, data_tensor):
        # data_tensor is (N_hits, 4) from _split_X_y: r, theta, z, layer_id
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fit before transform is called.")

        r, theta, z, layer_id = data_tensor.T.float()

        # Feature engineering
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # Apply scaling
        features_to_scale = torch.stack([r, z, x, y], dim=1) # [N_hits, 4]
        scaled_numpy = self.scaler.transform(features_to_scale.cpu().numpy())
        scaled_features = torch.from_numpy(scaled_numpy).float().to(data_tensor.device) # [N_hits, 4]

        # Combine into final feature vector for GNN nodes
        final_features = torch.cat([
            scaled_features,                # r, z, x, y (scaled) [N_hits, 4]
            torch.sin(theta.unsqueeze(1)),  # [N_hits, 1]
            torch.cos(theta.unsqueeze(1)),  # [N_hits, 1]
            layer_id.unsqueeze(1) / 28.0    # Heuristic scaling for layer_id [N_hits, 1]
        ], dim=1)                           # [N_hits, 7]
        return final_features

    def make_loader_cfg(self):
        # Use PyG's DataLoader, which correctly handles `Data` objects.
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 16, # Number of events per batch
           "shuffle": True,  # Shuffle for training
           "num_workers": 0, # For compatibility
        }

# Custom Dataset to build graphs on the fly.
# This will be injected into the harness via the `make_dataset` function.
class MyEventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events = events
        self.pre = pre
        self.train = train

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        X_raw, track_id = _split_X_y(self.events[idx])

        # 1. Transform features using the preprocessor
        X_transformed = self.pre.transform(X_raw) # [N_hits, F_out]

        # 2. Build graph edges
        layer_ids = X_raw[:, 3] # Use original, unscaled layer IDs
        num_hits = X_raw.shape[0]

        nodes_by_layer = collections.defaultdict(list)
        for i in range(num_hits):
            nodes_by_layer[int(layer_ids[i].item())].append(i)

        edge_list = []
        sorted_layers = sorted(nodes_by_layer.keys())

        for i, layer1_id in enumerate(sorted_layers):
            # Connect to subsequent layers within a window
            for j in range(i, len(sorted_layers)):
                layer2_id = sorted_layers[j]
                delta_layer = layer2_id - layer1_id
                if 0 < delta_layer <= 2:
                    for node1 in nodes_by_layer[layer1_id]:
                        for node2 in nodes_by_layer[layer2_id]:
                            edge_list.append([node1, node2])
                            edge_list.append([node2, node1])

        if not edge_list:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # 3. Create edge-level ground truth
        # An edge is "true" if its nodes are from the same track and not noise.
        # Noise hits have track_id == 0.
        if edge_index.shape[1] > 0:
            is_track_node = (track_id > 0)
            y_edge = (track_id[edge_index[0]] == track_id[edge_index[1]]) & \
                     is_track_node[edge_index[0]] & is_track_node[edge_index[1]]
        else:
            y_edge = torch.empty(0, dtype=torch.bool)


        # Create PyG Data object for this event
        data = Data(
            x=X_transformed,
            edge_index=edge_index,
            y=y_edge.float()
        )
        return data

def make_dataset(events, pre, *, train: bool):
    """This function is found by the harness and overrides the default dataset creation."""
    return MyEventDataset(events, pre, train=train)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim=128, n_layers=4, heads=4):
        super().__init__()
        # Initial projection of node features
        self.node_encoder = nn.Linear(in_features, hidden_dim)

        # Stack of GAT layers for message passing
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            conv = GATv2Conv(hidden_dim, hidden_dim, heads=heads,
                           concat=False, dropout=0.1, add_self_loops=False)
            self.gnn_layers.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Final classifier for edges
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        # `batch` is a torch_geometric.data.Batch object
        x, edge_index = batch.x, batch.edge_index

        # 1. Encode node features
        h = self.node_encoder(x) # [N_total_hits, hidden_dim]

        # 2. Propagate information through GNN layers
        for layer, norm in zip(self.gnn_layers, self.norms):
            h_res = h
            h = layer(h, edge_index)
            h = F.leaky_relu(norm(h + h_res)) # Residual connection + LayerNorm

        # 3. Predict edge truth
        h_src = h[edge_index[0]] # [N_edges, hidden_dim]
        h_dst = h[edge_index[1]] # [N_edges, hidden_dim]
        edge_attr = torch.cat([h_src, h_dst], dim=-1) # [N_edges, 2 * hidden_dim]
        edge_logits = self.edge_classifier(edge_attr) # [N_edges, 1]

        return edge_logits.squeeze(-1) # [N_edges]

def make_model(example_sample):
    # example_sample is a torch_geometric.data.Data object
    in_features = example_sample.num_node_features
    return HitClassifier(in_features=in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 40
def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7
    best_model_state = None

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.
        total_train_correct = 0
        total_train_edges = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            edge_logits = model(batch)
            y_edge = batch.y

            # Handle class imbalance for BCE loss
            if y_edge.numel() > 0:
                num_pos = y_edge.sum()
                num_neg = y_edge.numel() - num_pos
                pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=100.0) # Avoid infinite weights
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss = loss_fn(edge_logits, y_edge)

                if torch.isfinite(loss):
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item() * batch.num_graphs

                # Accuracy calculation
                preds = (torch.sigmoid(edge_logits) > 0.5)
                total_train_correct += (preds == y_edge).sum().item()
                total_train_edges += y_edge.numel()

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_train_acc = total_train_correct / (total_train_edges + 1e-6)
        train_loss_hist.append(avg_train_loss)
        train_acc_hist.append(avg_train_acc)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.
        total_val_correct = 0
        total_val_edges = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                edge_logits = model(batch)
                y_edge = batch.y

                if y_edge.numel() > 0:
                    num_pos = y_edge.sum()
                    num_neg = y_edge.numel() - num_pos
                    pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=100.0)
                    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    loss = loss_fn(edge_logits, y_edge)
                    total_val_loss += loss.item() * batch.num_graphs

                    preds = (torch.sigmoid(edge_logits) > 0.5)
                    total_val_correct += (preds == y_edge).sum().item()
                    total_val_edges += y_edge.numel()

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_acc = total_val_correct / (total_val_edges + 1e-6)
        val_loss_hist.append(avg_val_loss)
        val_acc_hist.append(avg_val_acc)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break

    # Load best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

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


