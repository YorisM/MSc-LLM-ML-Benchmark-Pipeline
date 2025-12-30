
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
    X = np.column_stack((evt["hit_r"].astype(np.float32),
                        evt["hit_theta"].astype(np.float32),
                        evt["hit_z"].astype(np.float32),
                        evt["layer_id"].astype(np.float32)))
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
from torch.nn import functional as F
from torch_geometric.nn import EdgeConv
from torch_geometric.utils import knn_graph
from torch.optim.lr_scheduler import ReduceLROnPlateau
import collections

# 1.1 -------- OPTIONAL: CUSTOM DATASET / DATA-CLASS  --------
#   def make_dataset(events, pre, train: bool):
#       # This solution uses the default dataset.
#       return None

# 1.2 ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit()
    #   - transform()

    def __init__(self):
        self.coord_mean = None
        self.coord_std = None
        self.layer_id_mean = None
        self.layer_id_std = None


    def _raw_reshape(self, data):
        return data

    def make_loader_cfg(self):
        # We process one event at a time to build per-event graphs.
        return {"batch_size": 1}

    def fit(self, data):
        # data is a list of event dictionaries
        all_coords = []
        all_layer_ids = []
        for evt in data:
            X, _ = _split_X_y(evt)
            r, theta, z, layer_id = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            coords = torch.stack([x, y, z], dim=1)
            all_coords.append(coords)
            all_layer_ids.append(layer_id)

        all_coords_cat = torch.cat(all_coords, dim=0)
        self.coord_mean = all_coords_cat.mean(dim=0)
        self.coord_std = all_coords_cat.std(dim=0)
        self.coord_std[self.coord_std == 0] = 1.0 # Avoid division by zero

        all_layer_ids_cat = torch.cat(all_layer_ids, dim=0)
        self.layer_id_mean = all_layer_ids_cat.mean()
        self.layer_id_std = all_layer_ids_cat.std()
        if self.layer_id_std == 0:
            self.layer_id_std = 1.0

        return self

    def transform(self, data):
        # data is a tensor of hits for a single event
        r, theta, z, layer_id = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        coords = torch.stack([x, y, z], dim=1)

        # Normalize coordinates
        coords = (coords - self.coord_mean) / self.coord_std

        # Normalize layer_id
        layer_id_norm = (layer_id - self.layer_id_mean) / self.layer_id_std

        # Combine features: (x_norm, y_norm, z_norm, layer_id_norm)
        # return shape: (N_hits, 4)
        return torch.cat([coords, layer_id_norm.unsqueeze(1)], dim=1)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, emb_dim=8, k=16):
        super().__init__()
        self.k = k
        self.emb_dim = emb_dim

        # Define a helper for creating MLP blocks
        def create_mlp(in_channels, channels, add_last_activation=True):
            mlp_modules = []
            for i, out_channels in enumerate(channels):
                mlp_modules.append(nn.Linear(in_channels, out_channels))
                if i < len(channels) - 1 or add_last_activation:
                    mlp_modules.append(nn.BatchNorm1d(out_channels))
                    mlp_modules.append(nn.ReLU())
                in_channels = out_channels
            return nn.Sequential(*mlp_modules)

        # DGCNN-style feature extractor
        self.conv1 = EdgeConv(create_mlp(2 * in_features, [64, 64]))
        self.conv2 = EdgeConv(create_mlp(2 * 64, [64, 64]))
        self.conv3 = EdgeConv(create_mlp(2 * 64, [128]))

        # Final projection to embedding space
        self.projection = create_mlp(128, [64, self.emb_dim], add_last_activation=False)

    def forward(self, batch_x):
        # batch_x corresponds to one event due to batch_size=1
        # shape: (N_hits, in_features)

        # Spatial coordinates for graph building
        pos = batch_x[:, :3] # (x_norm, y_norm, z_norm)

        # Dynamic graph construction and convolution
        edge_index = knn_graph(pos, self.k, batch=None)
        h1 = self.conv1(batch_x, edge_index)

        edge_index = knn_graph(h1, self.k, batch=None)
        h2 = self.conv2(h1, edge_index)

        edge_index = knn_graph(h2, self.k, batch=None)
        h3 = self.conv3(h2, edge_index)

        # Project to final embedding space
        out = self.projection(h3) # shape: (N_hits, emb_dim)

        # L2-normalize embeddings for stable contrastive training
        out = F.normalize(out, p=2, dim=1)
        return out

def make_model(input_features):
    return HitClassifier(input_features, emb_dim=8, k=16)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30
def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,
                                  min_lr=1e-5)

    def contrastive_loss(embeddings, y_true, margin=1.0):
        valid_hits_mask = y_true > 0 # Ignore noise hits (track_id <= 0)
        y_true_valid = y_true[valid_hits_mask]
        embeddings_valid = embeddings[valid_hits_mask]

        n = len(y_true_valid)
        if n < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Get upper triangle indices to avoid duplicate pairs and self-pairs
        triu_indices = torch.triu_indices(n, n, 1, device=embeddings.device)

        # Pairwise adjacency and distances
        adj_pairs = (y_true_valid[triu_indices[0]] == y_true_valid[triu_indices[1]])
        dist_matrix = torch.cdist(embeddings_valid, embeddings_valid)
        dist_pairs = dist_matrix[triu_indices[0], triu_indices[1]]

        # Positive pairs loss (pull together)
        pos_dists = dist_pairs[adj_pairs]
        loss_pos = torch.mean(pos_dists**2) if pos_dists.numel() > 0 else 0.0

        # Negative pairs loss (push apart)
        neg_dists = dist_pairs[~adj_pairs]
        loss_neg = torch.mean(torch.clamp(margin - neg_dists, min=0)**2) if neg_dists.numel() > 0 else 0.0

        return loss_pos + loss_neg

    train_loss, val_loss = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            # batch_size=1 means batch is a list with one (X, y) tuple
            X, y_true = batch[0]
            X, y_true = X.to(device), y_true.to(device)

            optimizer.zero_grad()
            embeddings = model(X)
            loss = contrastive_loss(embeddings, y_true)

            if torch.is_grad_enabled():
                loss.backward()
                optimizer.step()

            epoch_train_loss += loss.item()

        train_loss.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X, y_true = batch[0]
                X, y_true = X.to(device), y_true.to(device)

                embeddings = model(X)
                loss = contrastive_loss(embeddings, y_true)
                epoch_val_loss += loss.item()

        current_val_loss = epoch_val_loss / len(val_loader)
        val_loss.append(current_val_loss)
        scheduler.step(current_val_loss)

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            # A mechanism to save the best model could be here, but for now we just track loss
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Return trained model and metrics. Accuracy is not tracked.
    return model, train_loss, val_loss, [], []

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

    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None

    train_loader, val_loader = make_loaders(raw_train, raw_val, pre,
                                            batch = cfg.get("batch_size", 128),
                                            collate_fn = _ragged,
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


