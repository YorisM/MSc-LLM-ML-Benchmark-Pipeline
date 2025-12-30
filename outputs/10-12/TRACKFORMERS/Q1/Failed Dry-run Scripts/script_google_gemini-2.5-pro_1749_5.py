
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
from torch.nn import functional as F
from torch_geometric.nn import EdgeConv, knn_graph
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import copy
from tqdm import tqdm

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
        self.scalers = {}

    def _raw_reshape(self, data):           
        return data # Returns identity by default

    def make_loader_cfg(self):
        return None

    def fit(self, events):
        features = {
            "x": [], "y": [], "z": [], "r": []
        }
        for evt in events:
            r, theta, z = evt["hit_r"], evt["hit_theta"], evt["hit_z"]
            features["x"].append(r * np.cos(theta))
            features["y"].append(r * np.sin(theta))
            features["z"].append(z)
            features["r"].append(r)

        for key, values in features.items():
            all_vals = np.concatenate(values)
            self.scalers[key] = {
                "mean": all_vals.mean(),
                "std": all_vals.std()
            }
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        # data is a (N, 4) tensor with (r, theta, z, layer_id)
        r, theta, z, layer_id = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # Ensure std is not zero
        x_std = self.scalers["x"]["std"] if self.scalers["x"]["std"] > 1e-8 else 1.
        y_std = self.scalers["y"]["std"] if self.scalers["y"]["std"] > 1e-8 else 1.
        z_std = self.scalers["z"]["std"] if self.scalers["z"]["std"] > 1e-8 else 1.
        r_std = self.scalers["r"]["std"] if self.scalers["r"]["std"] > 1e-8 else 1.

        x_scaled = (x - self.scalers["x"]["mean"]) / x_std
        y_scaled = (y - self.scalers["y"]["mean"]) / y_std
        z_scaled = (z - self.scalers["z"]["mean"]) / z_std
        r_scaled = (r - self.scalers["r"]["mean"]) / r_std

        return torch.stack([x_scaled, y_scaled, z_scaled, r_scaled, theta, layer_id], dim=1) # (N_hits, 6)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, embed_dim=8, k=10, dbscan_eps=0.25, dbscan_min_samples=2):
        super().__init__()
        self.k = k
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        # Node feature projection
        self.node_encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )

        # EdgeConv layers for message passing
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64)), aggr='mean')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64)), aggr='mean')

        # Output projection to embedding space
        self.embed_proj = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

    def forward(self, batch):
        embeddings_list = []

        model_device = next(self.parameters()).device
        for X, _ in batch:
            X = X.to(model_device)
            node_coords = X[:, :3] # Use scaled x, y, z for geometry

            edge_index = knn_graph(node_coords, self.k, loop=False)

            h = self.node_encoder(X)
            h = self.conv1(h, edge_index)
            h = self.conv2(h, edge_index)

            embeddings = self.embed_proj(h) # (N_hits, embed_dim)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings_list.append(embeddings)

        if self.training:
            return embeddings_list
        else:
            pred_ids_list = []
            for embeddings in embeddings_list:
                emb_np = embeddings.detach().cpu().numpy()
                dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric='euclidean', n_jobs=-1)
                pred_ids = dbscan.fit_predict(emb_np)
                pred_ids_list.append(torch.from_numpy(pred_ids).to(model_device))
            return pred_ids_list

def make_model(example_sample):
    if isinstance(example_sample, tuple):
        in_features = example_sample[0].shape[1]
    else:
        in_features = example_sample.shape[1]

    return HitClassifier(in_features=in_features, k=10, embed_dim=8)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30
def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2, verbose=False)
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.5)

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    best_val_acc = -1
    patience = 5
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        num_train_events_processed = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            optimizer.zero_grad()

            embeddings_list = model(batch)
            event_losses = []

            for i, (_, y_true) in enumerate(batch):
                embeddings = embeddings_list[i]
                y_true = y_true.to(device)

                unique_labels, counts = torch.unique(y_true, return_counts=True)
                if len(unique_labels) < 2: continue

                anchors, positives, negatives = [], [], []
                for label_idx, label in enumerate(unique_labels):
                    if counts[label_idx] < 2: continue

                    is_current_track = (y_true == label)
                    current_indices = torch.where(is_current_track)[0]
                    other_indices = torch.where(~is_current_track)[0]

                    for anchor_idx in current_indices:
                        pos_candidates = current_indices[current_indices != anchor_idx]
                        pos_idx = pos_candidates[torch.randperm(len(pos_candidates))[:1]]
                        neg_idx = other_indices[torch.randperm(len(other_indices))[:1]]

                        anchors.append(embeddings[anchor_idx].unsqueeze(0))
                        positives.append(embeddings[pos_idx])
                        negatives.append(embeddings[neg_idx])

                if not anchors: continue

                loss = triplet_loss_fn(torch.cat(anchors), torch.cat(positives), torch.cat(negatives))
                event_losses.append(loss)
                num_train_events_processed += 1

            if event_losses:
                batch_loss = torch.stack(event_losses).mean()
                batch_loss.backward()
                optimizer.step()
                total_train_loss += batch_loss.item()

        avg_train_loss = total_train_loss / len(train_loader) if train_loader else 0
        train_loss_hist.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_ari = 0
        num_val_events = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in pbar:
                pred_ids_list = model(batch)

                for i, (_, y_true) in enumerate(batch):
                    pred_ids = pred_ids_list[i]
                    valid_mask = pred_ids != -1
                    if torch.sum(valid_mask) > 1:
                         ari = adjusted_rand_score(y_true.cpu()[valid_mask.cpu()], pred_ids.cpu()[valid_mask.cpu()])
                         total_val_ari += ari

                    num_val_events += 1

        avg_val_ari = total_val_ari / num_val_events if num_val_events > 0 else 0
        val_acc_hist.append(avg_val_ari)
        val_loss_hist.append(0)  # No validation loss defined

        scheduler.step(avg_val_ari)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val ARI: {avg_val_ari:.4f}, LR: {current_lr:.6f}")

        if avg_val_ari > best_val_acc:
            best_val_acc = avg_val_ari
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    train_acc_hist = [0.0] * len(train_loss_hist)

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

# </end code template>

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


