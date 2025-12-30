
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
from torch_geometric.data import Data, Batch
from torch_geometric.nn import EdgeConv, MessagePassing
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from collections import Counter
import torch.nn.functional as F

# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit()
    #   - transform()

    def __init__(self, use_gnn=True):
        self.scalers = {}
        self.use_gnn = use_gnn
        self.graph_building_params = {'r_max': 1.0, 'theta_max': 0.1, 'z_extrap_max': 2.0}

        # Non-GNN (DBSCAN) feature engineering h-params
        self.dbscan_scalers = {}
        self.dbscan_features_use_log_r = True


    def _cartesian(self, hits):
        r, theta = hits[:, 0], hits[:, 1]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y], dim=1)

    def _dbscan_features(self, X):
        r, theta, z, layer_id = X.T
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # Conformal mapping-like features
        # Tracks are (almost) circles through the origin in x-y, lines in r-z
        # u = x/r^2, v = y/r^2 makes circles through origin into lines
        # z/r is constant for a track from origin
        r_ = r + 1e-6 # Avoid division by zero
        u = x / r_**2
        v = y / r_**2
        z_over_r = z / r_

        return torch.stack([u, v, z_over_r], dim=1)

    @staticmethod
    def _collate_fn(batch: list):
        # We handle preprocessor binding to collate_fn via the loader config
        preprocessor, batch = batch[0][0], [item[1] for item in batch]

        if not preprocessor.use_gnn:
            # For non-GNN methods, we can just batch tensors.
            # Here we expect ragged list of tuples
            return batch

        data_list = []
        for evt_idx, (X, y) in enumerate(batch):
            # Graph construction logic
            X_cartesian = preprocessor._cartesian(X)
            hit_positions = torch.cat([X_cartesian, X[:, 2:3]], dim=1) # x, y, z

            unique_layers, layer_counts = torch.unique(X[:, 3], return_counts=True)
            layer_indices = [torch.where(X[:, 3] == i)[0] for i in unique_layers]
            layer_map = {int(layer_id): indices for layer_id, indices in zip(unique_layers, layer_indices)}

            edge_src, edge_dst = [], []
            for i, l_idx in enumerate(unique_layers[:-1]):
                next_l_idx = unique_layers[i+1]
                if int(next_l_idx) != int(l_idx) + 1: continue # Only adjacent layers

                nodes_from = layer_map[int(l_idx)]
                nodes_to = layer_map[int(next_l_idx)]

                # Simple Cone Search
                # Extrapolate z from origin: z_ext = z_from * (r_to / r_from)
                r_from, r_to = X[nodes_from, 0].unsqueeze(1), X[nodes_to, 0].unsqueeze(0)
                z_from, z_to = X[nodes_from, 2].unsqueeze(1), X[nodes_to, 2].unsqueeze(0)
                theta_from, theta_to = X[nodes_from, 1].unsqueeze(1), X[nodes_to, 1].unsqueeze(0)

                with torch.no_grad():
                    z_extrap = z_from * (r_to / (r_from + 1e-9))
                    delta_z = torch.abs(z_to - z_extrap)
                    delta_theta = torch.abs(theta_to - theta_from)

                    mask = (delta_theta < preprocessor.graph_building_params['theta_max']) & \
                           (delta_z < preprocessor.graph_building_params['z_extrap_max'])

                    src_indices, dst_indices = torch.where(mask)
                    edge_src.append(nodes_from[src_indices])
                    edge_dst.append(nodes_to[dst_indices])

            if not edge_src: # Handle events with no valid edges
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_labels = torch.empty(0, dtype=torch.float)
            else:
                edge_src = torch.cat(edge_src)
                edge_dst = torch.cat(edge_dst)
                edge_index = torch.stack([edge_src, edge_dst])

                # Create edge labels
                y_src = y[edge_src]
                y_dst = y[edge_dst]
                edge_labels = (y_src == y_dst).float()

            data_list.append(Data(x=X, edge_index=edge_index, edge_attr=edge_labels, y=y))

        return Batch.from_data_list(data_list)

    def make_loader_cfg(self):
        if not self.use_gnn:
             return {"loader_class": "torch.utils.data.DataLoader", 
                     "collate_fn": "_ragged", "batch_size": 1}

        # Need to bind self to the collate_fn as it's static
        def bind_collate(batch):
            return MyPreprocessor._collate_fn([(self, item) for item in batch])

        return {
           "loader_class": "torch.utils.data.DataLoader",
           "collate_fn": bind_collate,
           "batch_size": 32,
        }

    def fit(self, data):
        if self.use_gnn:
            all_hits = np.vstack([np.column_stack((evt["hit_r"], evt["hit_theta"], evt["hit_z"])) for evt in data])
            self.scalers['standard'] = StandardScaler().fit(all_hits)
        else: # DBSCAN
            all_features = torch.cat([self._dbscan_features(torch.from_numpy(np.column_stack((evt["hit_r"], evt["hit_theta"], evt["hit_z"], evt["layer_id"])))) for evt in data])
            self.dbscan_scalers['features'] = StandardScaler().fit(all_features.numpy())
        return self

    def transform(self, data):
        if self.use_gnn:
            coords = data[:, :3] # r, theta, z
            scaled_coords = self.scalers['standard'].transform(coords.numpy())
            return torch.cat([torch.from_numpy(scaled_coords).float(), data[:, 3:]], dim=1) # [N_hits, 4]
        else: # DBSCAN
            features = self._dbscan_features(data)
            scaled_features = self.dbscan_scalers['features'].transform(features.numpy())
            return torch.from_numpy(scaled_features).float() # [N_hits, 3]

def make_preprocessor():
    # To switch to the GNN model, set use_gnn=True
    return MyPreprocessor(use_gnn=True)

# 2. ---------- MODEL ARCHITECTURE ----------
class EdgeClassifierGNN(nn.Module):
    def __init__(self, in_features, hidden_dim=128, n_graph_iters=3, edge_threshold=0.5):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.n_graph_iters = n_graph_iters
        self.edge_threshold = edge_threshold

        self.node_encoder = nn.Linear(self.in_features, hidden_dim)

        self.gnn_layers = nn.ModuleList()
        for _ in range(n_graph_iters):
             self.gnn_layers.append(
                EdgeConv(nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
            )

        self.edge_classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        # 1. Encode node features
        h = self.node_encoder(data.x) # [N_total_hits, hidden_dim]

        # 2. Run GNN layers
        for layer in self.gnn_layers:
            h = layer(h, data.edge_index) # [N_total_hits, hidden_dim]

        # 3. Classify edges
        edge_src, edge_dst = data.edge_index
        edge_features = torch.cat([h[edge_src], h[edge_dst]], dim=-1) # [N_edges, 2*hidden_dim]
        edge_logits = self.edge_classifier(edge_features).squeeze(-1) # [N_edges]

        if self.training:
            return edge_logits
        else:
            # Inference mode: build tracks from edges
            scores = torch.sigmoid(edge_logits)
            passing_edges = data.edge_index[:, scores > self.edge_threshold]

            # Use scipy to find connected components for the whole batch
            num_nodes = data.num_nodes
            if passing_edges.shape[1] == 0:
                labels = -1*np.ones(num_nodes, dtype=np.int64)
            else:
                sparse_adj = to_scipy_sparse_matrix(passing_edges, num_nodes=num_nodes)
                _, labels = connected_components(csgraph=sparse_adj, directed=False, return_labels=True)

            pred_track_ids = torch.from_numpy(labels).to(data.x.device)

            # Unpack batch into list of tensors
            event_ids = torch.unique(data.batch)
            output = []
            for event_id in event_ids:
                mask = data.batch == event_id
                output.append(pred_track_ids[mask])
            return output

class DBScanWrapper(nn.Module):
    def __init__(self, in_features): # in_features is the preprocessed feature dim
        super().__init__()
        self.eps = 0.9 
        self.min_samples = 2

    def forward(self, batch):
        all_preds = []
        # batch is a list of (X, y) tuples
        for X, y in batch:
            X_np = X.cpu().numpy()
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1).fit(X_np)
            preds = torch.from_numpy(db.labels_).long()
            all_preds.append(preds)
        return all_preds

def make_model(in_features):
    # If using the GNN
    if isinstance(in_features, torch_geometric.data.Batch):
        return EdgeClassifierGNN(in_features.num_node_features)
    # If using DBSCAN
    else: # Fallback to a simple non-parametric model
        return DBScanWrapper(in_features[1])


# 3. ---------- MODEL TRAINING ----------
EPOCHS = 35 

def _calculate_fit_accuracy(pred_tracks_batch, true_tracks_batch, device):
    total_matched_hits = 0
    total_true_hits = 0

    for event_idx, pred_ids in enumerate(pred_tracks_batch):
        true_ids = true_tracks_batch[event_idx].cpu()
        pred_ids = pred_ids.cpu()

        # Get true tracks
        true_tracks = {}
        for hit_idx, track_id in enumerate(true_ids):
            track_id = int(track_id.item())
            if track_id == -1: continue # noise
            if track_id not in true_tracks: true_tracks[track_id] = []
            true_tracks[track_id].append(hit_idx)

        total_true_hits += sum(len(hits) for hits in true_tracks.values())

        # Get reconstructed tracks
        reco_tracks = {}
        for hit_idx, track_id in enumerate(pred_ids):
            track_id = int(track_id.item())
            if track_id == -1: continue  # noise from DBSCAN/cc
            if track_id not in reco_tracks: reco_tracks[track_id] = []
            reco_tracks[track_id].append(hit_idx)

        # Filter reconstructed tracks
        surviving_reco_tracks = []
        for track_id, hits in reco_tracks.items():
            if len(hits) < 4: continue

            # Check purity
            hit_true_ids = [int(true_ids[h].item()) for h in hits]
            if not hit_true_ids: continue

            # Handle noise hits in purity calculation
            non_noise_ids = [tid for tid in hit_true_ids if tid != -1]
            if not non_noise_ids: continue

            majority = Counter(non_noise_ids).most_common(1)[0]
            purity = majority[1] / len(hits)

            if purity >= 0.5:
                # Store as set for efficient intersection
                surviving_reco_tracks.append((set(hits), majority[0]))

        # Match to true tracks
        for true_id, true_hits_list in true_tracks.items():
            true_hits_set = set(true_hits_list)
            n_match = 0
            for reco_hits_set, reco_majority_id in surviving_reco_tracks:
                # Optimization: only check tracks with matching majority ID
                if reco_majority_id == true_id:
                    intersect = len(true_hits_set.intersection(reco_hits_set))
                    if intersect > n_match:
                        n_match = intersect
            total_matched_hits += n_match

    if total_true_hits == 0: return 0.0
    return total_matched_hits / total_true_hits

def train_model(model, train_loader, val_loader, epochs):
    if isinstance(model, DBScanWrapper):
        return model, [], [], [], [] # Non-parametric

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0).to(device)) # Heuristic weight for class imbalance

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_val_acc = -1

    print(f"Starting training for {epochs} epochs on {device}...")
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        epoch_train_loss, epoch_train_correct, epoch_train_total = 0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            edge_logits = model(batch)            
            truth = batch.edge_attr

            loss = loss_fn(edge_logits, truth)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * batch.num_graphs
            preds = (torch.sigmoid(edge_logits) > 0.5).long()
            epoch_train_correct += (preds == truth).sum().item()
            epoch_train_total += len(truth)

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_train_acc = epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0
        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)

        # --- Validation ---
        model.eval()
        epoch_val_fit_acc_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # Get true tracks for unpacking
                true_tracks_batch = []
                event_ids = torch.unique(batch.batch)
                for event_id in event_ids:
                    mask = batch.batch == event_id
                    true_tracks_batch.append(batch.y[mask])

                # This forward call returns track IDs because model is in eval mode
                pred_tracks_batch = model(batch)

                fit_acc = _calculate_fit_accuracy(pred_tracks_batch, true_tracks_batch, device)
                epoch_val_fit_acc_sum += fit_acc * len(pred_tracks_batch)

        avg_val_acc = epoch_val_fit_acc_sum / len(val_loader.dataset)
        # Use 1 - FitAccuracy as the validation loss for early stopping
        avg_val_loss = 1.0 - avg_val_acc 
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

        scheduler.step(avg_val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val FitAccuracy: {avg_val_acc:.4f}")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f"  -> New best validation accuracy: {best_val_acc:.4f}. Saving model.")
            torch.save(model.state_dict(), "best_model.pt")

    # Load best model state for returning
    model.load_state_dict(torch.load("best_model.pt"))
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


