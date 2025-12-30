
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
from torch_geometric.nn import EdgeConv, knn_graph
from sklearn.cluster import DBSCAN
from sklearn.metrics import fowlkes_mallows_score
import itertools

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
        # We will store the mean and standard deviation for x, y, z coordinates.
        self.mean = None
        self.std = None

    def _raw_reshape(self, data: torch.Tensor) -> torch.Tensor:
        # Convert cylindrical to Cartesian coordinates for graph construction
        # and standardisation, as Euclidean distances are more natural in Cartesian space.
        # input `data` format: [N_hits, 4] (hit_r, hit_theta, hit_z, layer_id)
        r, theta, z, layer_id = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # New feature matrix: (x, y, z, layer_id). 
        # layer_id is kept as-is, not normalized, as it's a categorical identifier.
        return torch.stack([x, y, z, layer_id], dim=1) # [N_hits, 4]

    def make_loader_cfg(self):
        # Using batch_size=1 simplifies processing variable-sized events,
        # as each batch corresponds to a single event and its graph.
        return {"batch_size": 1}

    def fit(self, events: list):
        # Calculate mean and std for the first 3 features (x, y, z)
        # across the entire training dataset for normalization.

        all_features = []
        # Use a large subset of events to estimate statistics.
        for evt in events[:2000]: 
            X_raw, _ = _split_X_y(evt)
            # We only need the features that will be normalized (x,y,z).
            reshaped_X = self._raw_reshape(X_raw)
            all_features.append(reshaped_X[:, :3]) # Append only x, y, z

        if not all_features:
            self.mean = torch.zeros(3)
            self.std = torch.ones(3)
            return self

        full_feature_tensor = torch.cat(all_features, dim=0) # [Total_hits, 3]
        self.mean = torch.mean(full_feature_tensor, dim=0)
        self.std = torch.std(full_feature_tensor, dim=0)
        # Avoid division by zero in case a dimension has zero variance.
        self.std[self.std == 0] = 1.0

        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        # Apply the preprocessing: Cartesian conversion and feature normalization.
        reshaped_data = self._raw_reshape(data)

        if self.mean is None or self.std is None:
            raise RuntimeError("Preprocessor must be fit before transform is called.")

        # Normalize x, y, z features using the pre-computed stats.
        norm_features = (reshaped_data[:, :3] - self.mean) / self.std # [N_hits, 3]

        # Combine normalized features with the non-normalized layer_id.
        transformed_data = torch.cat([norm_features, reshaped_data[:, 3:4]], dim=1) # [N_hits, 4]

        return transformed_data

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, emb_dim=8, k_neighbors=8, dbscan_eps=0.25, dbscan_min_samples=2):
        super().__init__()
        self.in_features = in_features
        self.emb_dim = emb_dim
        self.k_neighbors = k_neighbors
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        # Define GNN layers using EdgeConv, which is effective for point cloud tasks.
        # It operates on local neighborhoods, dynamically defined in feature space.

        nn1 = nn.Sequential(nn.Linear(2 * in_features, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.gcn1 = EdgeConv(nn1, aggr='max')

        nn2 = nn.Sequential(nn.Linear(2 * 64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.gcn2 = EdgeConv(nn2, aggr='max')

        nn3 = nn.Sequential(nn.Linear(2 * 128, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.gcn3 = EdgeConv(nn3, aggr='max')

        # Final MLP to project concatenated GNN outputs to the embedding space.
        self.emb_mlp = nn.Sequential(
            nn.Linear(in_features + 64 + 128 + 256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, self.emb_dim)
        )

    def create_graph(self, x: torch.Tensor) -> torch.Tensor:
        # Build a k-NN graph for the current event's hits.
        # Since batch_size=1, we can pass batch=None.
        edge_index = knn_graph(x, k=self.k_neighbors, batch=None, loop=False)
        return edge_index

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        # Construct graph and pass through GNN layers.
        edge_index = self.create_graph(x)

        # GNN layers with skip connections concatenated at the end.
        x1 = self.gcn1(x, edge_index)           # [N_hits, 64]
        x2 = self.gcn2(x1, edge_index)          # [N_hits, 128]
        x3 = self.gcn3(x2, edge_index)          # [N_hits, 256]

        # Concatenate features from all layers (DenseNet-like) for a rich representation.
        out = torch.cat([x, x1, x2, x3], dim=1) # [N_hits, in + 64 + 128 + 256]

        # Project to embedding space.
        embs = self.emb_mlp(out)                # [N_hits, emb_dim]

        # Normalize embeddings to lie on a unit hypersphere. This stabilizes training
        # with a contrastive loss and makes clustering distances more uniform.
        embs = F.normalize(embs, p=2, dim=1)    # [N_hits, emb_dim]
        return embs

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:
        # batch_x shape: [N_hits_in_event, features] due to batch_size=1

        embs = self.get_embeddings(batch_x)

        # Use different behavior for training (metric learning) and inference (clustering).
        if self.training:
            # During training, return embeddings for calculation of the contrastive loss.
            return embs
        else:
            # During inference, perform clustering on embeddings to predict track IDs.
            embs_cpu = embs.detach().cpu().numpy()

            # DBSCAN is robust to noise and doesn't require pre-specifying the number of tracks.
            clusterer = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric='euclidean')
            labels = clusterer.fit_predict(embs_cpu)

            return torch.from_numpy(labels).to(batch_x.device)

def make_model(in_features: int) -> HitClassifier:
    # in_features will be 4 (x_norm, y_norm, z_norm, layer_id)
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50 

def contrastive_loss(embs: torch.Tensor, y_true: torch.Tensor, margin: float = 0.8, device='cpu'):
    # Implements contrastive loss by sampling positive (same track) and negative (different track) pairs.
    labels = y_true.cpu().numpy()
    unique_labels, counts = np.unique(labels, return_counts=True)
    true_track_ids = unique_labels[counts > 1]

    if len(true_track_ids) == 0:
        return torch.tensor(0.0, device=device)

    # Sample positive pairs
    pos_pairs = [list(itertools.combinations(np.where(labels == track_id)[0], 2)) for track_id in true_track_ids]
    pos_pairs = [pair for sublist in pos_pairs for pair in sublist]

    if not pos_pairs:
        return torch.tensor(0.0, device=device)

    # Subsample to a manageable number to keep training steps fast.
    if len(pos_pairs) > 2000:
        pos_pairs = [pos_pairs[i] for i in np.random.choice(len(pos_pairs), 2000, replace=False)]

    # Sample an equal number of negative pairs.
    neg_pairs = []
    n_hits = len(labels)
    n_samples = len(pos_pairs)

    attempts, max_attempts = 0, 5 * n_samples # Safety break
    while len(neg_pairs) < n_samples and attempts < max_attempts:
        i, j = np.random.randint(0, n_hits, 2)
        if labels[i] != labels[j]:
            neg_pairs.append((i, j))
        attempts += 1

    if not neg_pairs:
        return torch.tensor(0.0, device=device)

    pos_pairs_tensor = torch.LongTensor(pos_pairs).to(device)
    neg_pairs_tensor = torch.LongTensor(neg_pairs).to(device)

    # Calculate L2 distance for positive and negative pairs in the embedding space.
    pos_dist = F.pairwise_distance(embs[pos_pairs_tensor[:, 0]], embs[pos_pairs_tensor[:, 1]])
    neg_dist = F.pairwise_distance(embs[neg_pairs_tensor[:, 0]], embs[neg_pairs_tensor[:, 1]])

    # Contrastive Loss: pull positive pairs together, push negative pairs apart by a margin.
    loss = torch.mean(pos_dist) + torch.mean(F.relu(margin - neg_dist))
    return loss

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    best_val_acc = -1.0
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            X, y_true = view.batch_x, view.batch_y
            optimizer.zero_grad()
            embs = model(X) # Returns embeddings in training mode
            loss = contrastive_loss(embs, y_true, device=device)
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        train_acc.append(0)  # Omit train accuracy calculation for speed.

        # --- Validation Phase ---
        model.eval()
        epoch_val_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                X, y_true = view.batch_x, view.batch_y
                pred_ids = model(X) # Returns cluster IDs in eval mode

                # Use Fowlkes-Mallows score as a fast proxy for clustering quality.
                # It compares the similarity of two clusterings (predicted vs true).
                acc = fowlkes_mallows_score(y_true.cpu().numpy(), pred_ids.cpu().numpy())
                epoch_val_acc += acc

        avg_val_acc = epoch_val_acc / len(val_loader)
        val_acc.append(avg_val_acc)
        val_loss.append(0) # Not calculating validation loss for simplicity, focus on accuracy proxy.

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc (F-M): {avg_val_acc:.4f}")

        scheduler.step(avg_val_acc)

        # --- Early Stopping ---
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} as validation accuracy did not improve.")
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


