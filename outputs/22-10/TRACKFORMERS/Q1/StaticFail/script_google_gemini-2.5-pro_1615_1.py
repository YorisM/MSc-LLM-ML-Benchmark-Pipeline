
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
from torch_geometric.nn import EdgeConv, knn_graph
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

    def __init__(self):
        # We will fit a StandardScaler for each of the geometric features.
        self.scalers = {}

    def _get_feature_names(self):
        # Defines the features we will scale.
        return ['r', 'z', 'x', 'y']

    # Uncomment to implement custom collate function.
    # @staticmethod
    # def _collate_fn(batch: list):
    #    <LLM: Apply optional custom collate logic here>

    def make_loader_cfg(self):
        # The default _ragged collate function is used, which returns a list of samples.
        # This is handled by the model's forward pass.
        # A batch size of 1 processes one event at a time.
        return {"batch_size": 1}

    def fit(self, data):
        # data: list of event dictionaries
        # We concatenate all hits from all events to get a global picture for scaling.
        all_features = []
        for evt in data:
            r, theta, z = evt["hit_r"], evt["hit_theta"], evt["hit_z"]
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            all_features.append(np.column_stack([r, z, x, y]))

        if not all_features:
            return self

        full_feature_matrix = np.concatenate(all_features, axis=0) # [N_total_hits, 4]

        # Fit a separate scaler for each feature for robustness.
        for i, name in enumerate(self._get_feature_names()):
            scaler = StandardScaler()
            self.scalers[name] = scaler.fit(full_feature_matrix[:, i:i+1])
        return self

    def transform(self, data: torch.Tensor):
        # data: [N_hits, 4] with columns (r, theta, z, layer_id) from a single event
        r, theta, z, layer_id = data.T.to(torch.float32)

        # Add Cartesian coordinates
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        raw_feats = {'r': r, 'z': z, 'x': x, 'y': y}
        scaled_feats = {}
        for name in self._get_feature_names():
            if name in self.scalers:
                feat_tensor = raw_feats[name].reshape(-1, 1)
                # Scaler expects a numpy array
                scaled_np = self.scalers[name].transform(feat_tensor.cpu().numpy())
                scaled_feats[name] = torch.from_numpy(scaled_np).to(data.device).view(-1)

        # Assemble the final feature tensor: [r_s, theta, z_s, layer_id, x_s, y_s]
        return torch.stack([
            scaled_feats.get('r', r),
            theta,
            scaled_feats.get('z', z),
            layer_id,
            scaled_feats.get('x', x),
            scaled_feats.get('y', y)
        ], dim=1).float() # [N_hits, 6]

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, embedding_dim=12, k_knn=16):
        super().__init__()
        self.k_knn = k_knn
        # Features for KNN graph: Scaled Cartesian coordinates (x_s, y_s, z_s)
        self.knn_feature_indices = [4, 5, 2]

        # A series of EdgeConv layers to learn relationships between hits
        mlp1 = nn.Sequential(nn.Linear(2 * in_features, 64), nn.BatchNorm1d(64), nn.ReLU(),
                             nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.conv1 = EdgeConv(mlp1, aggr='mean')

        mlp2 = nn.Sequential(nn a.Linear(2 * 64, 128), nn.BatchNorm1d(128), nn.ReLU(),
                             nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.conv2 = EdgeConv(mlp2, aggr='mean')

        mlp3 = nn.Sequential(nn.Linear(2 * 128, 256), nn.BatchNorm1d(256), nn.ReLU(),
                             nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.conv3 = EdgeConv(mlp3, aggr='mean')

        # Final MLP to project concatenated features to the embedding space
        self.post_mlp = nn.Sequential(
            nn.Linear(in_features + 64 + 128 + 256, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, batch: list[tuple[torch.Tensor, torch.Tensor]]):
        # This forward pass is designed to work with the default _ragged collate_fn.
        # It takes a list of (X, y) tuples and dynamically builds a PyG Batch.
        data_list = []
        device = batch[0][0].device # Get device from the first tensor in the batch
        for X, _ in batch:
            # Build k-NN graph on-the-fly using scaled cartesian coordinates
            knn_coords = X[:, self.knn_feature_indices]
            edge_index = knn_graph(knn_coords, k=self.k_knn, loop=False, batch=None)
            data_list.append(Data(x=X, edge_index=edge_index))

        # Collate list of Data objects into a single Batch object for efficient GNN processing
        pyg_batch = Batch.from_data_list(data_list).to(device)
        x, edge_index = pyg_batch.x, pyg_batch.edge_index

        # Apply GNN layers with skip connections
        x_in = x
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)

        # Concatenate features from all layers (DenseNet-style)
        x_all = torch.cat([x_in, x1, x2, x3], dim=-1) # [N_total_hits, in_features + 64 + 128 + 256]

        embeddings = self.post_mlp(x_all) # [N_total_hits, embedding_dim]

        return embeddings

def make_model(in_features):
    # in_features will be 6, as determined by the preprocessor.
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 100

def _object_condensation_loss(embeddings, truth_ids, batch_indices, num_graphs, device):
    q_min = 2.0
    repulsive_margin = 1.0
    attractive_weight = 1.0
    repulsive_weight = 1.0
    attractive_radius = 0.1 # Dead zone for attractive force

    attractive_term = torch.tensor(0.0, device=device)
    repulsive_term = torch.tensor(0.0, device=device)
    total_clusters = 0
    events_with_repulsion = 0

    for i in range(num_graphs):
        event_mask = (batch_indices == i)
        if not torch.any(event_mask): continue

        event_embeddings = embeddings[event_mask]
        event_truth_ids = truth_ids[event_mask]

        unique_track_ids = torch.unique(event_truth_ids[event_truth_ids != 0])
        if len(unique_track_ids) < 1: continue

        condensation_points = []
        for track_id in unique_track_ids:
            track_mask = (event_truth_ids == track_id)
            track_hits_embeddings = event_embeddings[track_mask]

            if track_hits_embeddings.shape[0] < q_min: continue

            condensation_point = torch.mean(track_hits_embeddings, dim=0)
            condensation_points.append(condensation_point)

            distances = torch.norm(track_hits_embeddings - condensation_point, p=2, dim=1)
            attractive_term += torch.mean(torch.clamp(distances - attractive_radius, min=0)**2)
            total_clusters += 1

        if len(condensation_points) > 1:
            events_with_repulsion += 1
            condensation_points = torch.stack(condensation_points)
            dist_matrix = torch.cdist(condensation_points, condensation_points, p=2)

            rep_loss = torch.clamp(repulsive_margin - dist_matrix, min=0)**2
            # Sum over off-diagonal elements and normalize
            repulsive_term += torch.sum(rep_loss) / (len(condensation_points) * (len(condensation_points) - 1))

    if total_clusters == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    attractive_loss = attractive_weight * attractive_term / total_clusters
    repulsive_loss = repulsive_weight * repulsive_term / max(1, events_with_repulsion)

    return attractive_loss + repulsive_loss

def _clustering_accuracy(embeddings, truth_ids, batch_indices, num_graphs):
    correct, total = 0, 0

    for i in range(num_graphs):
        event_mask = (batch_indices == i)
        if not torch.any(event_mask): continue

        event_embeddings = embeddings[event_mask]
        event_truth_ids = truth_ids[event_mask]

        unique_track_ids = torch.unique(event_truth_ids[event_truth_ids != 0])
        if len(unique_track_ids) < 1: continue

        id_map = {tid.item(): i for i, tid in enumerate(unique_track_ids)}
        true_centroids = torch.stack([event_embeddings[event_truth_ids == tid].mean(dim=0) for tid in unique_track_ids])

        dist_matrix = torch.cdist(event_embeddings, true_centroids)
        pred_indices = torch.argmin(dist_matrix, dim=1)

        true_indices = torch.tensor([id_map.get(tid.item(), -1) for tid in event_truth_ids], device=embeddings.device, dtype=torch.long)

        valid_mask = true_indices != -1
        correct += (pred_indices[valid_mask] == true_indices[valid_mask]).sum().item()
        total += valid_mask.sum().item()

    return correct / total if total > 0 else 0.0

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)

    best_val_loss = float('inf')
    best_model_state = None
    patience = 20
    epochs_no_improve = 0

    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss, running_acc, n_batches = 0.0, 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()

            embeddings = model(batch)
            truth_ids = torch.cat([y for _, y in batch]).to(device)
            batch_indices = torch.cat([torch.full_like(y, i) for i, (_, y) in enumerate(batch)]).to(device)

            loss = _object_condensation_loss(embeddings, truth_ids, batch_indices, len(batch), device)

            if torch.isnan(loss) or loss == 0.0: continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                running_acc += _clustering_accuracy(embeddings.detach(), truth_ids, batch_indices, len(batch))
            n_batches += 1

        avg_train_loss = running_loss / n_batches if n_batches > 0 else 0.0
        avg_train_acc = running_acc / n_batches if n_batches > 0 else 0.0
        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)

        model.eval()
        running_loss, running_acc, n_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                embeddings = model(batch)
                truth_ids = torch.cat([y for _, y in batch]).to(device)
                batch_indices = torch.cat([torch.full_like(y, i) for i, (_, y) in enumerate(batch)]).to(device)

                loss = _object_condensation_loss(embeddings, truth_ids, batch_indices, len(batch), device)
                if torch.isnan(loss): continue

                running_loss += loss.item()
                running_acc += _clustering_accuracy(embeddings, truth_ids, batch_indices, len(batch))
                n_batches += 1

        avg_val_loss = running_loss / n_batches if n_batches > 0 else 0.0
        avg_val_acc = running_acc / n_batches if n_batches > 0 else 0.0
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

        print(f"E{epoch+1:03d} | Tr L:{avg_train_loss:.4f} Tr A:{avg_train_acc:.3f} | Vl L:{avg_val_loss:.4f} Vl A:{avg_val_acc:.3f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

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


