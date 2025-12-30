
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import EdgeConv, knn_graph
from torch_geometric.data import Data, Batch

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
        self.mean = None
        self.std = None

    def _raw_reshape(self, data):           
        return data # Returns identity by default

    def make_loader_cfg(self):
        return None

    def fit(self, data):
        all_coords = []
        for evt in data:
            r, theta, z = evt["hit_r"], evt["hit_theta"], evt["hit_z"]
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            # Features to be scaled: r, z, x, y
            coords = np.stack([r, z, x, y], axis=1) # [N_hits_evt, 4]
            all_coords.append(coords)

        all_coords_np = np.concatenate(all_coords, axis=0) # [Total_hits, 4]

        self.mean = torch.from_numpy(all_coords_np.mean(axis=0)).float()
        self.std = torch.from_numpy(all_coords_np.std(axis=0)).float()
        # Add epsilon to std to avoid division by zero
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, data):
        # data is a tensor [N_hits, 4] with (r, theta, z, layer_id)
        r, theta, z, layer_id = data.T
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # Features to scale: r, z, x, y
        to_scale = torch.stack([r, z, x, y], dim=1) # [N_hits, 4]
        scaled = (to_scale - self.mean.to(data.device)) / self.std.to(data.device) # [N_hits, 4]
        r_s, z_s, x_s, y_s = scaled.T

        # Final features: (r_scaled, theta, z_scaled, layer_id, x_scaled, y_scaled)
        final_features = torch.stack([r_s, theta, z_s, layer_id, x_s, y_s], dim=1) # [N_hits, 6]
        return final_features

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim=64, emb_dim=8, k_neighbors=9):
        super().__init__()
        self.k = k_neighbors
        self.emb_dim = emb_dim

        # Input projection to lift features to the hidden dimension
        self.input_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Graph learning layers
        self.edge_conv1 = EdgeConv(nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        ), aggr='mean')

        self.edge_conv2 = EdgeConv(nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        ), aggr='mean')

        # Output projection to embedding space
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.emb_dim)
        )

    def forward(self, batch):
        # batch is a list of (X, y) tuples from the DataLoader
        data_list = []
        for X, y in batch:
            # Use scaled Cartesian coordinates for nearest neighbor search
            coords_for_knn = X[:, [4, 5, 2]] # x_s, y_s, z_s
            edge_index = knn_graph(coords_for_knn, self.k, loop=False, flow='source_to_target')
            data_list.append(Data(x=X, edge_index=edge_index, y=y))

        # Create a single batch object for PyG
        pyg_batch = Batch.from_data_list(data_list).to(device)

        # GNN forward pass
        h = self.input_net(pyg_batch.x) # [N_total_hits, hidden_dim]
        h = self.edge_conv1(h, pyg_batch.edge_index) # [N_total_hits, hidden_dim]
        h = self.edge_conv2(h, pyg_batch.edge_index) # [N_total_hits, hidden_dim]

        # Project to embedding space
        embeddings = self.output_net(h) # [N_total_hits, emb_dim]

        # Return embeddings and a batch index vector to separate hits by event
        return embeddings, pyg_batch.batch

def make_model(example_sample):
    if isinstance(example_sample, (list, tuple)):
        # example_sample is (X_tensor, y_tensor)
        in_features = example_sample[0].shape[1]
    else:
        # Fallback if example_sample is just the feature tensor
        in_features = example_sample.shape[1]
    return HitClassifier(in_features=in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25

def _calculate_loss_acc(embeddings, batch_idx, y_trues, device):
    """Helper function to compute metric learning loss and a proxy accuracy."""
    total_loss = torch.tensor(0.0, device=device)
    total_acc_hits = 0.0
    total_acc_attempts = 0.0
    margin = 1.0

    num_events = int(batch_idx.max().item() + 1)

    for i in range(num_events):
        event_mask = (batch_idx == i)
        event_embeddings = embeddings[event_mask]
        event_y = y_trues[i].to(device)
        n_hits = len(event_y)

        if n_hits < 2:
            continue

        dists = torch.cdist(event_embeddings, event_embeddings)

        is_same_track = event_y.unsqueeze(1) == event_y.unsqueeze(0)
        # Positive pairs: same track, not noise (track_id > 0), not self-pair
        is_pos = is_same_track & (event_y.unsqueeze(1) > 0) & ~torch.eye(n_hits, dtype=torch.bool, device=device)
        # Negative pairs: different tracks
        is_neg = ~is_same_track

        # Pull loss (L2 distance for positive pairs)
        pos_dists = dists[is_pos]
        pull_loss = torch.pow(pos_dists, 2).mean() if pos_dists.numel() > 0 else torch.tensor(0.0, device=device)

        # Push loss (hinge loss for negative pairs)
        neg_dists = dists[is_neg]
        push_loss = torch.pow(torch.relu(margin - neg_dists), 2).mean() if neg_dists.numel() > 0 else torch.tensor(0.0, device=device)

        total_loss += pull_loss + push_loss

        # Accuracy Calculation (proxy metric: triplet success rate)
        for hit_idx in range(n_hits):
            if event_y[hit_idx] <= 0: continue

            pos_mask = is_pos[hit_idx]
            neg_mask = is_neg[hit_idx] & (event_y > 0) # Compare to other tracks, not noise

            if pos_mask.any() and neg_mask.any():
                pos_idx = torch.where(pos_mask)[0][torch.randint(pos_mask.sum(), (1,))]
                neg_idx = torch.where(neg_mask)[0][torch.randint(neg_mask.sum(), (1,))]

                if dists[hit_idx, pos_idx] < dists[hit_idx, neg_idx]:
                    total_acc_hits += 1
                total_acc_attempts += 1

    batch_loss = total_loss / num_events if num_events > 0 else torch.tensor(0.0, device=device)
    batch_acc = total_acc_hits / total_acc_attempts if total_acc_attempts > 0 else 0.0
    return batch_loss, batch_acc

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss, running_acc, num_batches = 0.0, 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            y_trues = [evt[1] for evt in batch]
            embeddings, batch_idx = model(batch)
            loss, acc = _calculate_loss_acc(embeddings, batch_idx, y_trues, device)

            if torch.is_tensor(loss) and not torch.isnan(loss) and loss.requires_grad:
              loss.backward()
              optimizer.step()
              running_loss += loss.item()
              running_acc += acc
              num_batches += 1

        epoch_train_loss = running_loss / num_batches if num_batches > 0 else 0
        epoch_train_acc = running_acc / num_batches if num_batches > 0 else 0
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_loss, running_acc, num_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                y_trues = [evt[1] for evt in batch]
                embeddings, batch_idx = model(batch)
                loss, acc = _calculate_loss_acc(embeddings, batch_idx, y_trues, device)

                if torch.is_tensor(loss) and not torch.isnan(loss):
                  running_loss += loss.item()
                  running_acc += acc
                  num_batches += 1

        epoch_val_loss = running_loss / num_batches if num_batches > 0 else 0
        epoch_val_acc = running_acc / num_batches if num_batches > 0 else 0
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Early stopping logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Restore best model weights
    if 'best_model_state' in locals():
      model.load_state_dict(best_model_state)

    return model, train_loss, val_loss, train_acc, val_acc

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
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


