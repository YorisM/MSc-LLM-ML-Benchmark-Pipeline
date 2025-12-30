
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
from torch_geometric.nn import EdgeConv
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
import tempfile

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
        # We will store feature means and standard deviations for normalization.
        self.mean = None
        self.std = None
        self.fitted = False

    def _raw_reshape(self, data):           
        # Not used in this approach
        return data # Returns identity by default

    def make_loader_cfg(self):
        # Using the default loader configuration with ragged batches
        return None

    def fit(self, data):
        # data is a list of event dictionaries. We compute global statistics over all hits.
        all_r, all_theta, all_z = [], [], []
        for evt in data:
            all_r.append(evt["hit_r"])
            all_theta.append(evt["hit_theta"])
            all_z.append(evt["hit_z"])

        # Concatenate all hits from all training events
        r = np.concatenate(all_r)
        theta = np.concatenate(all_theta)
        z = np.concatenate(all_z)

        # Feature engineering: add Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # We will scale these 5 features
        features_to_scale = np.stack([r, theta, z, x, y], axis=1) # [N_total_hits, 5]

        self.mean = np.mean(features_to_scale, axis=0, dtype=np.float32)
        self.std = np.std(features_to_scale, axis=0, dtype=np.float32)
        self.std[self.std == 0] = 1.0  # Avoid division by zero for constant features

        self.fitted = True
        return self

    def transform(self, data):
        # data is a torch.Tensor of shape [N_hits, 4] for a single event.
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fit before transforming data.")

        # Deconstruct input
        r, theta, z, layer_id = data.T # [N_hits,], [N_hits,], ...

        # Re-create the engineered features
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        features_to_scale = torch.stack([r, theta, z, x, y], dim=1) # [N_hits, 5]

        # Load stats and move to the correct device
        mean_t = torch.from_numpy(self.mean).to(data.device)
        std_t = torch.from_numpy(self.std).to(data.device)

        # Apply standardization
        scaled_features = (features_to_scale - mean_t) / std_t # [N_hits, 5]

        # Append the unscaled layer_id as the final feature
        final_features = torch.cat([scaled_features, layer_id.unsqueeze(1)], dim=1) # [N_hits, 6]
        return final_features

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, embedding_dim=16, hidden_dim=64):
        super().__init__()
        # MLPs for the EdgeConv layers. Input to MLP is 2 * num_node_features
        mlp1 = nn.Sequential(nn.Linear(2 * in_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = EdgeConv(mlp1, aggr='mean')

        mlp2 = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = EdgeConv(mlp2, aggr='mean')

        mlp3 = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim))
        self.conv3 = EdgeConv(mlp3, aggr='mean')

        # Final projection head to get the desired embedding dimension
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.embedding_dim = embedding_dim

    @staticmethod
    def _build_edges(X, layer_feature_idx=-1):
        # X is [N_hits, N_features]
        # layer_id is the last feature
        layers = X[:, layer_feature_idx].long()
        unique_layers, inverse_indices = torch.unique(layers, sorted=True, return_inverse=True)

        hits_by_layer = [torch.where(inverse_indices == i)[0] for i in range(len(unique_layers))]

        edge_list = []
        for i in range(len(unique_layers) - 1):
            l1_hits = hits_by_layer[i]
            l2_hits = hits_by_layer[i+1]
            # Connect all hits in a layer to all hits in the next consecutive layer
            if len(l1_hits) > 0 and len(l2_hits) > 0:
                grid = torch.cartesian_prod(l1_hits, l2_hits)
                edge_list.append(grid)

        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long, device=X.device)

        edge_index = torch.cat(edge_list, dim=0).t()
        # Create bi-directional edges for message passing
        return torch.cat([edge_index, edge_index.flip(0)], dim=1)

    def forward(self, batch):
        # The default loader provides a list of (X, y) tuples
        data_list = []
        for _, (X, y) in enumerate(batch):
            edge_index = self._build_edges(X)
            data = Data(x=X, edge_index=edge_index, y=y)
            data_list.append(data)

        if not data_list:
            return torch.empty((0, self.embedding_dim), device=device)

        # Create a single batched graph object
        pyg_batch = Batch.from_data_list(data_list).to(device)

        x, edge_index = pyg_batch.x, pyg_batch.edge_index

        # Apply GNN layers
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        # Project to embedding space
        x = self.output_proj(x)

        # L2-normalize embeddings to live on a hypersphere. This often helps contrastive losses.
        x = F.normalize(x, p=2, dim=1)

        return x # Return [N_total_hits, embedding_dim] embeddings

def make_model(example_sample):
    # The preprocessor adds 2 features (x, y). The input has 4, so the model will see 6.
    # The harness builds the model before preprocessing, so we hard-code the feature count.
    in_features = 6 
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25 

def contrastive_loss(embeddings, track_ids, batch_indices, margin=1.0):
    total_loss = torch.tensor(0.0, device=embeddings.device)
    num_events = 0

    unique_batches = torch.unique(batch_indices)
    for i in unique_batches:
        event_mask = (batch_indices == i)
        embs = embeddings[event_mask] # [N_event_hits, D_emb]
        tids = track_ids[event_mask]  # [N_event_hits,]
        n_hits = embs.shape[0]

        if n_hits < 2:
            continue

        # Pairwise distance matrix for the event
        dists = torch.cdist(embs, embs, p=2)
        # Adjacency matrix based on true track IDs
        adj = tids.unsqueeze(1) == tids.unsqueeze(0)

        # Mask for positive pairs (same track, different hits)
        pos_mask = adj & ~torch.eye(n_hits, dtype=torch.bool, device=embs.device)
        # Mask for negative pairs (different tracks)
        neg_mask = ~adj

        # Positive loss: pull same-track hits together
        if pos_mask.sum() > 0:
            pos_dists = dists[pos_mask]
            pos_loss = torch.pow(pos_dists, 2).mean()
        else:
            pos_loss = torch.tensor(0.0, device=embs.device)

        # Negative loss: push different-track hits apart by a margin
        if neg_mask.sum() > 0:
            neg_dists = dists[neg_mask]
            neg_loss = torch.pow(torch.clamp(margin - neg_dists, min=0), 2).mean()
        else:
            neg_loss = torch.tensor(0.0, device=embs.device)

        total_loss += pos_loss + neg_loss
        num_events += 1

    return total_loss / num_events if num_events > 0 else total_loss


def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

    train_loss, val_loss = [], []
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    # Use a temporary file to save the best model to avoid cluttering the script directory
    temp_dir = tempfile.gettempdir()
    best_model_path = os.path.join(temp_dir, "best_model_state.pt")

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()

            embeddings = model(batch)
            if embeddings.shape[0] == 0: continue

            y_true = torch.cat([y for _, y in batch]).to(device)
            batch_indices = torch.cat([torch.full_like(y, i) for i, (_, y) in enumerate(batch)]).to(device)

            loss = contrastive_loss(embeddings, y_true, batch_indices)

            if loss > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0.0
        train_loss.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        num_batches_val = 0
        with torch.no_grad():
            for batch in val_loader:
                embeddings = model(batch)
                if embeddings.shape[0] == 0: continue

                y_true = torch.cat([y for _, y in batch]).to(device)
                batch_indices = torch.cat([torch.full_like(y, i) for i, (_, y) in enumerate(batch)]).to(device)

                loss = contrastive_loss(embeddings, y_true, batch_indices)
                epoch_val_loss += loss.item()
                num_batches_val += 1

        avg_val_loss = epoch_val_loss / num_batches_val if num_batches_val > 0 else 0.0
        val_loss.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load the best performing model state for the final return
    model.load_state_dict(torch.load(best_model_path))
    trained_model = model

    # No meaningful accuracy metric is calculated during this training process,
    # as it's based on an unsupervised loss. We return empty lists as per harness flexibility.
    train_acc, val_acc = [], []

    return trained_model, train_loss, val_loss, train_acc, val_acc

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


