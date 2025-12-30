
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
from torch_geometric.data import Dataset as GeoDataset, Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

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
        self.feature_means = None
        self.feature_stds = None

    def _raw_reshape(self, data):           
        return data # Returns identity by default

    # Uncomment to implement custom collate function.    
    # @staticmethod
    # def _collate_fn(batch: list):
    #   REQUIREMENT: Must output one of the following formats:
    #     A) ragged list: [(X_i, y_i), ...] or [X_i, ...]
    #     B) tuple: (X, y) where X can be Tensor or list/tuple
    #     C) plain Tensor: X
    #     D) dict: {"x": X, "y": y} or {"inputs": X, "labels": y}

    def make_loader_cfg(self):
        # Return dict or None.  If dict, evaluator uses it to rebuild loader:
        #{
        #   "loader_class": "torch.utils.data.DataLoader",
        #   "collate_fn": "self._collate_fn",
        #   "batch_size": 256,
        #   "shuffle": False,
        #   "num_workers": 0,
        #   "pin_memory": True
        #}
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 32,
        }

    def fit(self, events):
        all_X = np.vstack([
            np.column_stack((evt["hit_r"], evt["hit_theta"], evt["hit_z"]))
            for evt in events
        ])

        r, theta, z = all_X[:, 0], all_X[:, 1], all_X[:, 2]
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        features = np.stack([r, theta, z, x, y], axis=1) # (total_hits, 5)

        self.feature_means = torch.from_numpy(features.mean(axis=0)).float()
        self.feature_stds = torch.from_numpy(features.std(axis=0)).float()
        # Add a small epsilon to std to avoid division by zero
        self.feature_stds[self.feature_stds < 1e-6] = 1e-6

        return self

    def transform(self, data: torch.Tensor):
        # data is a (N_hits, 4) tensor: r, theta, z, layer_id
        r, theta, z = data[:, 0], data[:, 1], data[:, 2]

        x_coord = r * torch.cos(theta)
        y_coord = r * torch.sin(theta)

        features = torch.stack([r, theta, z, x_coord, y_coord], dim=1) # (N_hits, 5)

        # Normalize
        means = self.feature_means.to(data.device)
        stds = self.feature_stds.to(data.device)
        features = (features - means) / stds

        return features # (N_hits, 5)

def make_preprocessor():
    return MyPreprocessor()

# Custom Dataset to be used by the harness's _make_dataset hook
class MyGraphDataset(GeoDataset):
    def __init__(self, events, preprocessor):
        super().__init__()
        self.events = events
        self.pre = preprocessor

    def len(self):
        return len(self.events)

    def get(self, idx):
        evt = self.events[idx]
        X_raw = np.column_stack((evt["hit_r"],
                                 evt["hit_theta"],
                                 evt["hit_z"],
                                 evt["layer_id"]))
        y = evt["track_id"].astype(np.int64)

        X_raw_torch = torch.from_numpy(X_raw).float()
        y_torch = torch.from_numpy(y)

        # Preprocess features to get node features `x`
        x_processed = self.pre.transform(X_raw_torch)

        # Build graph edges based on adjacent layers
        layers = X_raw_torch[:, 3]

        # This is an efficient way to build a directed graph between hits in consecutive layers
        # sort nodes by layer_id to optimize edge creation
        sorted_indices = torch.argsort(layers)

        edge_list = []
        for i_sorted in range(len(sorted_indices)):
            i_raw = sorted_indices[i_sorted]
            i_layer = layers[i_raw]

            # Find hits in the next 2 layers
            for j_sorted in range(i_sorted + 1, len(sorted_indices)):
                j_raw = sorted_indices[j_sorted]
                j_layer = layers[j_raw]

                if j_layer > i_layer and j_layer <= i_layer + 2:
                    edge_list.append([i_raw, j_raw])

                if j_layer > i_layer + 2:
                    break # Since nodes are sorted by layer, we can break early

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x_processed, edge_index=edge_index, y=y_torch)

def make_dataset(events, pre, *, train: bool):
    return MyGraphDataset(events, pre)


# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, embedding_dim=10, hidden_dim=128, dbscan_eps=0.2, dbscan_min_samples=2):
        super().__init__()
        # GNN part to learn embeddings
        self.node_encoder = nn.Linear(in_features, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.output_decoder = nn.Linear(hidden_dim, embedding_dim)

        # Clustering part for inference
        self.clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean', n_jobs=-1)

    def gnn_forward(self, data):
        # This part of the model is what gets trained to produce good embeddings
        x, edge_index = data.x, data.edge_index # (N_hits_batch, F), (2, N_edges_batch)

        x = self.node_encoder(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.output_decoder(x)

        return x # (N_hits_batch, embedding_dim)

    def forward(self, batch_data):
        # The full pipeline for inference, producing track IDs
        # Get embeddings from the GNN
        embeddings = self.gnn_forward(batch_data) # (N_total_hits, embedding_dim)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Move to CPU for sklearn's DBSCAN
        embeddings_np = embeddings.detach().cpu().numpy()

        # Get event boundaries from the torch_geometric.data.Batch object
        ptr = batch_data.ptr.cpu().numpy()

        all_labels = []
        total_max_label = 0

        for i in range(len(ptr) - 1):
            start, end = ptr[i], ptr[i+1]
            event_embeddings = embeddings_np[start:end]

            if event_embeddings.shape[0] == 0:
                continue

            # Cluster hits for this event
            event_labels = self.clusterer.fit_predict(event_embeddings)

            # Make labels unique across the batch
            # Handle noise points (-1) from DBSCAN by giving them unique IDs
            noise_mask = (event_labels == -1)
            event_labels[noise_mask] = -np.arange(1, noise_mask.sum() + 1) # temp negative labels for noise

            # Make track IDs positive and unique across events
            if (event_labels >= 0).sum() > 0:
                event_labels[event_labels >= 0] += total_max_label

            current_max = event_labels.max() if len(event_labels) > 0 and event_labels.max() >= 0 else total_max_label-1

            # Assign new unique positive IDs to noise points
            event_labels[noise_mask] = np.arange(current_max + 1, current_max + 1 + noise_mask.sum())

            all_labels.append(event_labels)

            if len(event_labels) > 0:
                total_max_label = event_labels.max() + 1 if event_labels.max() >= 0 else total_max_label

        if not all_labels:
            return torch.empty(0, dtype=torch.long, device=embeddings.device)

        final_labels = torch.from_numpy(np.concatenate(all_labels)).long()

        return final_labels.to(embeddings.device)

def make_model(in_features):
    # in_features is determined by MyPreprocessor.transform, here 5.
    return HitClassifier(in_features=5)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 40   # Adjusted for better convergence

def compute_loss(embeddings, y, ptr, margin=1.0, alpha=0.5, device='cpu'):
    total_pull_loss = 0.0
    total_push_loss = 0.0
    num_events = 0

    for i in range(len(ptr) - 1):
        start, end = ptr[i], ptr[i+1]
        if end - start < 2: continue

        event_embeddings = embeddings[start:end]
        event_y = y[start:end]

        # Ignore noise hits (track_id == 0) for loss calculation
        valid_indices = (event_y != 0)
        if valid_indices.sum() < 2: continue

        event_embeddings_clean = event_embeddings[valid_indices]
        event_y_clean = event_y[valid_indices]

        unique_y, y_indices = torch.unique(event_y_clean, return_inverse=True)
        num_tracks = len(unique_y)
        if num_tracks < 2: continue

        # Efficiently compute centroids
        sum_embeds = torch.zeros(num_tracks, event_embeddings_clean.size(1), device=device)
        sum_embeds.index_add_(0, y_indices, event_embeddings_clean)
        counts = torch.zeros(num_tracks, device=device, dtype=torch.float)
        counts.index_add_(0, y_indices, torch.ones_like(y_indices, dtype=torch.float))
        centroids = sum_embeds / counts.unsqueeze(1).clamp(min=1e-6)

        # Pull loss: pull embeddings towards their track centroid
        pull_loss_i = (event_embeddings_clean - centroids[y_indices]).pow(2).sum(dim=1).mean()

        # Push loss: push centroids away from each other
        dists = torch.pdist(centroids)
        push_loss_i = torch.clamp(margin - dists, min=0).pow(2).mean()

        total_pull_loss += pull_loss_i
        total_push_loss += push_loss_i
        num_events += 1

    if num_events == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    loss = (total_pull_loss / num_events) + alpha * (total_push_loss / num_events)
    return loss


def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            embeddings = model.gnn_forward(batch)
            loss = compute_loss(embeddings, batch.y, batch.ptr, device=device)

            if torch.isnan(loss) or torch.isinf(loss): continue

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_loss.append(epoch_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_ari = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embeddings = model.gnn_forward(batch)
                loss = compute_loss(embeddings, batch.y, batch.ptr, device=device)
                if torch.isnan(loss) or torch.isinf(loss): continue
                epoch_val_loss += loss.item()

                # Calculate Adjusted Rand Index as an accuracy proxy
                pred_labels = model(batch)
                true_labels = batch.y

                # Filter noise for ARI calculation
                valid_mask = true_labels != 0
                if valid_mask.sum() > 1:
                    score = adjusted_rand_score(true_labels[valid_mask].cpu().numpy(), pred_labels[valid_mask].cpu().numpy())
                    epoch_val_ari += score
                    num_val_batches += 1

        epoch_val_loss /= len(val_loader)
        val_loss.append(epoch_val_loss)

        if num_val_batches > 0:
            epoch_val_ari /= num_val_batches
        val_acc.append(epoch_val_ari)
        train_acc.append(0) # Not computing ARI on train set to save time

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val ARI: {epoch_val_ari:.4f}")

        scheduler.step(epoch_val_loss)

        # Early stopping logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save the best model state
            torch.save(model.state_dict(), "best_model_state.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    # Restore best model found during training
    if os.path.exists("best_model_state.pt"):
        model.load_state_dict(torch.load("best_model_state.pt"))
        os.remove("best_model_state.pt")

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


