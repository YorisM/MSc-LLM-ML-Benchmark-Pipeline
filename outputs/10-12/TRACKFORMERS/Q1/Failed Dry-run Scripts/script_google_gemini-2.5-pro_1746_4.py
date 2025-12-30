
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
from torch_geometric.nn import EdgeConv, knn_graph


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
        # We will store the mean and std for feature normalization.
        self.scalers = {}

    def _raw_reshape(self, data):
        # Returns identity by default
        return data

    def make_loader_cfg(self):
        # We process batches as lists of events inside the model's forward pass
        # and the training loop, so the default loader configuration provided
        # by the harness (with _ragged collate) is what we want.
        return None

    def fit(self, data):
        # data is a list of raw event dicts
        all_hits = []
        for evt in data:
            hits = np.column_stack((evt["hit_r"],
                                    evt["hit_theta"],
                                    evt["hit_z"],
                                    evt["layer_id"]))
            all_hits.append(hits)

        all_hits_np = np.vstack(all_hits)
        r, theta, z, layer_id = all_hits_np.T

        # Add Cartesian coordinates as they are useful for GNNs
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Features: r, theta, z, layer_id, x, y
        features = np.column_stack((r, theta, z, layer_id, x, y))

        # Calculate mean and std for standardization
        self.scalers["mean"] = torch.from_numpy(features.mean(axis=0)).float()
        self.scalers["std"] = torch.from_numpy(features.std(axis=0)).float()

        # Avoid division by zero for features with no variance
        self.scalers["std"][self.scalers["std"] < 1e-6] = 1.0

        return self

    def transform(self, data: torch.Tensor):
        # data is a torch.Tensor of shape [N_hits, 4] with (r, theta, z, layer_id)
        r, theta, z, layer_id = data.T

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # Create the full feature tensor
        features = torch.stack([r, theta, z, layer_id, x, y], dim=1) # [N_hits, 6]

        # Apply standardization using fitted scalers
        features = (features - self.scalers["mean"]) / self.scalers["std"]
        return features


def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, k=16, emb_dim=8):
        super().__init__()
        self.k = k
        self.in_features = in_features

        # Define MLPs for EdgeConv layers
        mlp1 = nn.Sequential(
            nn.Linear(2 * in_features, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        mlp2 = nn.Sequential(
            nn.Linear(2 * 64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        mlp3 = nn.Sequential(
            nn.Linear(2 * 128, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )

        self.gnn1 = EdgeConv(mlp1, aggr='mean')
        self.gnn2 = EdgeConv(mlp2, aggr='mean')
        self.gnn3 = EdgeConv(mlp3, aggr='mean')

        # Post-GNN MLP to project concatenated features into the embedding space
        self.post_mlp = nn.Sequential(
            nn.Linear(in_features + 64 + 128 + 256, 256), nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, emb_dim),
        )

    def gnn_pass(self, pyg_batch):
        x0 = pyg_batch.x
        edge_index = pyg_batch.edge_index

        x1 = self.gnn1(x0, edge_index)
        x2 = self.gnn2(x1, edge_index)
        x3 = self.gnn3(x2, edge_index)

        # Use skip connections from all layers for the final embedding
        combined_features = torch.cat([x0, x1, x2, x3], dim=1) # [N_total_hits, F_combined]

        # Project to final embedding space
        embeddings = self.post_mlp(combined_features)

        # Normalize embeddings to lie on a unit hypersphere, which helps clustering losses
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def forward(self, batch):
        # The training loop provides a list of (X, y) tuples (_ragged collate).
        # We must build a torch_geometric.data.Batch object on the fly.
        data_list = []
        for X, y in batch:
            # Graph construction: k-NN on cartesian coords (features 4 and 5: x, y)
            edge_index = knn_graph(X[:, 4:6], self.k, loop=False, batch=None)
            data = Data(x=X, edge_index=edge_index, y=y)
            data_list.append(data)

        # Move the whole batch to the correct device
        device = batch[0][0].device
        pyg_batch = Batch.from_data_list(data_list).to(device)

        # Perform the GNN pass and return embeddings
        return self.gnn_pass(pyg_batch)

def make_model(in_features):
    # The harness might pass the first sample tuple (X,y) instead of an int.
    # We handle this defensively, but our preprocessor always produces 6 features.
    return HitClassifier(in_features=6)


# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25

def condensation_loss_and_acc(embeddings, batch_list, margin, alpha, beta):
    total_loss = 0.
    total_acc = 0.
    n_events = len(batch_list)
    device = embeddings.device

    # Create pointer array to slice the flattened embeddings tensor by event
    counts = [len(y) for _, y in batch_list]
    ptr = torch.cat([torch.tensor([0], device=device), torch.tensor(counts, device=device).cumsum(0)])

    for i in range(n_events):
        start, end = ptr[i], ptr[i+1]
        event_embeds = embeddings[start:end]
        event_y = batch_list[i][1]

        # Filter out noise hits (typically track_id <= 0)
        track_mask = event_y > 0
        pids = torch.unique(event_y[track_mask])

        if len(pids) < 2:
            # Not enough tracks to compute inter-cluster loss
            continue

        L_pull, L_push = 0., 0.
        cluster_centers = []
        pid_map = {pid.item(): i for i, pid in enumerate(pids)}

        # Pull force (intra-cluster attraction)
        for pid in pids:
            pid_mask = (event_y == pid)
            track_hits_embeds = event_embeds[pid_mask]
            center = track_hits_embeds.mean(dim=0)
            cluster_centers.append(center)

            # Attract hits to their cluster center
            pull = torch.norm(track_hits_embeds - center, p=2, dim=1).mean()
            L_pull += pull

        L_pull /= len(pids)

        # Push force (inter-cluster repulsion)
        cluster_centers = torch.stack(cluster_centers) # [n_clusters, emb_dim]
        # Calculate pairwise distances between cluster centers
        pdist = torch.pdist(cluster_centers, p=2)
        # Apply hinge loss to repel centers that are closer than the margin
        L_push = torch.mean(torch.relu(margin - pdist))

        total_loss += alpha * L_pull + beta * L_push

        # Proxy accuracy: For each hit, is its closest cluster center the correct one?
        hit_center_dists = torch.cdist(event_embeds[track_mask], cluster_centers)
        closest_center_idx = torch.argmin(hit_center_dists, dim=1)

        # Map true track IDs to cluster center indices for comparison
        true_center_idx = torch.tensor([pid_map[y.item()] for y in event_y[track_mask]], device=device)

        correct_preds = (closest_center_idx == true_center_idx).sum()
        event_acc = correct_preds / track_mask.sum() if track_mask.sum() > 0 else 0.
        total_acc += event_acc

    return total_loss / n_events, total_acc / n_events

def train_model(model, train_loader, val_loader, epochs):
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    # Hyperparameters for the condensation loss
    margin, alpha, beta = 1.0, 1.0, 1.0

    for epoch in range(epochs):
        model.train()
        running_loss, running_acc = 0., 0.
        for batch_list in train_loader:
            optimizer.zero_grad()

            # Move data for the current batch to the active device
            batch_on_device = [(X.to(device), y.to(device)) for X, y in batch_list]

            embeddings = model(batch_on_device)
            loss, acc = condensation_loss_and_acc(embeddings, batch_on_device, margin, alpha, beta)

            if torch.is_tensor(loss): # Ensure loss is a valid tensor
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_acc += acc.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = running_acc / len(train_loader)
        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)

        model.eval()
        running_val_loss, running_val_acc = 0., 0.
        with torch.no_grad():
            for batch_list in val_loader:
                batch_on_device = [(X.to(device), y.to(device)) for X, y in batch_list]
                embeddings = model(batch_on_device)
                loss, acc = condensation_loss_and_acc(embeddings, batch_on_device, margin, alpha, beta)

                if torch.is_tensor(loss):
                    running_val_loss += loss.item()
                    running_val_acc += acc.item()

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_acc = running_val_acc / len(val_loader)
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} - "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        scheduler.step(avg_val_loss)

        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return model, train_loss, val_loss, train_acc, val_acc
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


