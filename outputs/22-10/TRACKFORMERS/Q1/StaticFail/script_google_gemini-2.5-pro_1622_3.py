
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
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, BatchNorm
from itertools import product


# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------

# Custom Dataset to handle graph-structured events
class GraphEventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        super().__init__()
        self.events = events
        self.pre = pre
        self.train = train
        self.graphs = self._process_events()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def _process_events(self):
        """
        Processes raw events into a list of torch_geometric.data.Data objects.
        """
        graphs = []
        for evt in self.events:
            X_raw, y = _split_X_y(evt)

            if X_raw.shape[0] == 0:
                continue

            # Use preprocessor to get standardized node features
            node_features = self.pre.transform(X_raw) # (N_hits, F_nodes)

            # --- Graph construction ---
            # We connect hits in adjacent detector layers
            layers = X_raw[:, 3]
            unique_layers, layer_indices = torch.unique(layers, return_inverse=True, sorted=True)

            hits_by_layer = [[] for _ in range(len(unique_layers))]
            for i, layer_idx in enumerate(layer_indices):
                hits_by_layer[layer_idx].append(i)

            edge_list = []
            # Connect each hit in a layer to all hits in the next layer
            for i in range(len(unique_layers) - 1):
                current_hits = hits_by_layer[i]
                next_hits = hits_by_layer[i+1]
                edge_list.extend(product(current_hits, next_hits))

            if not edge_list: # Handle events with hits in < 2 layers
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

            # Make the graph undirected for EdgeConv message passing
            if edge_index.numel() > 0:
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

            # Create PyG Data object for the event
            data = Data(x=node_features, edge_index=edge_index, y=y)
            graphs.append(data)

        return graphs

# This function will be found and used by the harness to create the dataset
def make_dataset(events, pre, *, train: bool):
    """
    Factory function for creating the custom GraphEventDataset.
    """
    return GraphEventDataset(events, pre, train=train)

class MyPreprocessor:
    """
    A preprocessor that converts cylindrical coordinates to Cartesian,
    standardizes the features, and configures the data loader for graph data.
    """
    def __init__(self):
        # These will be learned from the training data in fit()
        self.mean = None
        self.std = None

    def fit(self, events):
        """
        Calculates the mean and standard deviation of features from the training data.
        """
        all_features = []
        for evt in events:
            X, _ = _split_X_y(evt) # X: (r, theta, z, layer_id)
            if X.shape[0] == 0:
                continue

            r, theta, z = X[:, 0], X[:, 1], X[:, 2]

            # Convert to Cartesian and also keep 'r' as a feature
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)

            features = torch.stack([x, y, z, r], dim=1) # (N_hits, 4)
            all_features.append(features)

        all_features = torch.cat(all_features, dim=0) # (Total_hits, 4)
        self.mean = all_features.mean(dim=0)
        self.std = all_features.std(dim=0)
        # Add a small epsilon to std to prevent division by zero
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, data: torch.Tensor):
        """
        Applies the learned transformation to the data.
        data is a torch.Tensor: [N_hits, 4] with columns (r, theta, z, layer_id)
        """
        r, theta, z = data[:, 0], data[:, 1], data[:, 2]

        # Convert to Cartesian and keep 'r'
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        features = torch.stack([x, y, z, r], dim=1) # (N_hits, 4)

        # Apply standardization
        if self.mean is not None and self.std is not None:
            features = (features - self.mean) / self.std

        return features

    def make_loader_cfg(self):
        """
        Returns a configuration dictionary for the DataLoader.
        We specify torch_geometric's DataLoader to handle graph batches.
        """
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 32, # GNNs often benefit from smaller batch sizes
        }

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    """
    A Graph Neural Network model to learn embeddings for each hit.
    It uses multiple EdgeConv layers to pass information between connected hits.
    """
    def __init__(self, in_features, hidden_dim=64, embedding_dim=16):
        super().__init__()

        # EdgeConv layers use an MLP to process messages between nodes.
        # The input to the MLP is 2 * node_feature_dim.

        mlp1 = self.build_mlp(2 * in_features, hidden_dim)
        self.conv1 = EdgeConv(nn=mlp1, aggr='mean')
        self.bn1 = BatchNorm(hidden_dim)

        mlp2 = self.build_mlp(2 * hidden_dim, hidden_dim)
        self.conv2 = EdgeConv(nn=mlp2, aggr='mean')
        self.bn2 = BatchNorm(hidden_dim)

        mlp3 = self.build_mlp(2 * hidden_dim, hidden_dim)
        self.conv3 = EdgeConv(nn=mlp3, aggr='mean')
        self.bn3 = BatchNorm(hidden_dim)

        # Final linear layer to project to the embedding space
        self.final_proj = nn.Linear(hidden_dim, embedding_dim)

    def build_mlp(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, batch):
        """
        Forward pass of the GNN.
        Input: a torch_geometric.data.Batch object.
        Output: embeddings for all hits in the batch.
        """
        x, edge_index = batch.x, batch.edge_index # x: [N_total_hits, F_in], edge_index: [2, E_total]

        # Layer 1
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(F.relu(x1))

        # Layer 2
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(F.relu(x2))

        # Layer 3 with residual connection
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(F.relu(x3 + x2)) # Add residual connection

        # Final embedding projection
        embeddings = self.final_proj(x3) # [N_total_hits, embedding_dim]

        return embeddings

def make_model(in_features):
    """
    Factory function for creating the model.
    Handles case where `in_features` is passed as a Data object.
    """
    if isinstance(in_features, Data):
        num_features = in_features.num_node_features
    elif isinstance(in_features, tuple): # from default loader
        num_features = in_features[0].shape[1]
    else: # from harness
        num_features = 4 # Based on our preprocessor

    return HitClassifier(in_features=num_features, hidden_dim=64, embedding_dim=16)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30 

def _contrastive_loss_and_acc(embeddings, true_ids, batch_indices, margin, acc_threshold):
    """
    Helper function to compute contrastive loss and a proxy accuracy.
    Processes each event in the batch separately.
    """
    total_loss = 0.0
    total_acc = 0.0
    num_graphs = batch_indices.max().item() + 1

    for i in range(num_graphs):
        mask = (batch_indices == i)
        event_embeddings = embeddings[mask] # (N_i, D)
        event_ids = true_ids[mask]          # (N_i,)
        n_hits = event_embeddings.shape[0]

        if n_hits < 2:
            continue

        # --- Loss Calculation ---
        pairwise_dist = torch.cdist(event_embeddings, event_embeddings) # (N_i, N_i)

        is_same_track = (event_ids.unsqueeze(1) == event_ids.unsqueeze(0))
        # Positive pairs: same track, not the same hit
        is_pos = is_same_track & ~torch.eye(n_hits, dtype=torch.bool, device=embeddings.device)
        # Negative pairs: different tracks
        is_neg = ~is_same_track

        # Hinge loss: pull positive pairs together, push negative pairs apart by a margin
        pos_loss = pairwise_dist[is_pos].pow(2).mean() if is_pos.any() else 0.0
        neg_loss = F.relu(margin - pairwise_dist[is_neg]).pow(2).mean() if is_neg.any() else 0.0

        total_loss += (pos_loss + neg_loss)

        # --- Accuracy Proxy ---
        with torch.no_grad():
            preds = (pairwise_dist < acc_threshold)
            mask_offdiag = ~torch.eye(n_hits, dtype=torch.bool, device=embeddings.device)
            acc = (preds[mask_offdiag] == is_same_track[mask_offdiag]).float().mean()
            total_acc += acc.item() if not torch.isnan(acc) else 0.0

    return (total_loss / num_graphs), (total_acc / num_graphs)

def train_model(model, train_loader, val_loader, epochs):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 7

    margin = 1.0 # Hyperparameter for contrastive loss
    acc_threshold = 0.7 # Hyperparameter for accuracy proxy

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss, running_acc = 0.0, 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            embeddings = model(batch)
            loss, acc = _contrastive_loss_and_acc(embeddings, batch.y, batch.batch, margin, acc_threshold)

            if torch.isnan(loss): continue
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.num_graphs
            running_acc += acc * batch.num_graphs

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_acc / len(train_loader.dataset)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        running_loss, running_acc = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embeddings = model(batch)
                loss, acc = _contrastive_loss_and_acc(embeddings, batch.y, batch.batch, margin, acc_threshold)

                if torch.isnan(loss): continue
                running_loss += loss.item() * batch.num_graphs
                running_acc += acc * batch.num_graphs

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_acc / len(val_loader.dataset)
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        scheduler.step(epoch_val_loss)

        # --- Early Stopping ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Could save best model state here
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
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


