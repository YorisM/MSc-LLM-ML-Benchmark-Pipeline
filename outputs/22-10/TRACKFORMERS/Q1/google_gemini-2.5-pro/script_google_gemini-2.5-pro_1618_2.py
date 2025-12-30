
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
import torch.optim as optim
import copy

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
        self.mean_ = None
        self.std_ = None

    def fit(self, events):
        """
        Fits the preprocessor by calculating normalization statistics from the raw training data.
        It also performs feature engineering, converting cylindrical coordinates to Cartesian.
        """
        all_X_engineered = []
        for evt in events:
            # Replicate the logic from the harness's _split_X_y function
            X_np = np.column_stack((evt["hit_r"],
                                    evt["hit_theta"],
                                    evt["hit_z"],
                                    evt["layer_id"]))
            X = torch.from_numpy(X_np).float()

            # Feature engineering: add Cartesian coordinates
            r, theta, z, layer_id = X.T # Each is of shape (N_hits,)
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)

            # Use [x, y, z, r, layer_id] as features
            X_engineered = torch.stack([x, y, z, r, layer_id], dim=1) # Shape: (N_hits, 5)
            all_X_engineered.append(X_engineered)

        # Concatenate all events' hits to compute overall statistics
        all_X_tensor = torch.cat(all_X_engineered, dim=0)
        self.mean_ = all_X_tensor.mean(dim=0)
        self.std_ = all_X_tensor.std(dim=0)

        # Prevent division by zero if a feature has zero variance
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, data: torch.Tensor):
        """
        Applies the feature engineering and normalization to the input data tensor.
        """
        # data has shape (N_hits, 4) with [r, theta, z, layer_id]
        r, theta, z, layer_id = data.T
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        X_engineered = torch.stack([x, y, z, r, layer_id], dim=1) # Shape: (N_hits, 5)

        # Apply standardization using the pre-computed statistics
        X_normalized = (X_engineered - self.mean_) / self.std_
        return X_normalized

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, embedding_dim=8, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def _embed(self, x: torch.Tensor):
        """Processes a single tensor of hits and returns their embeddings."""
        embeddings = self.network(x) # Shape: (N, embedding_dim)
        # L2-normalize embeddings to place them on a hypersphere
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def forward(self, batch):
        """
        A flexible forward pass that handles both a single tensor of hits
        and a batch in the form of a list of (X, y) tuples.
        """
        if isinstance(batch, torch.Tensor):
            return self._embed(batch)
        elif isinstance(batch, list):
            # This case handles the harness sanity check `model(first_batch)`
            all_X = [item[0].to(device) for item in batch]
            return [self._embed(x) for x in all_X]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

def make_model(example_sample):
    # The harness provides the first sample from the first batch, which after
    # preprocessing is a tuple of tensors (X_transformed, y).
    in_features = example_sample[0].shape[1]
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 40  # Increased for better convergence

# Helper function to generate positive and negative pairs for contrastive loss
def get_pairs(y: torch.Tensor):
    n_hits = y.shape[0]
    if n_hits < 2:
        return None, None

    # Create a broadcasted equality matrix to find pairs from the same track
    y_matrix = y.unsqueeze(1)
    is_same_track = (y_matrix == y_matrix.T)

    # Use upper triangle indices to get unique pairs (i, j) where i < j
    triu_indices = torch.triu_indices(n_hits, n_hits, offset=1, device=y.device)

    is_same_track_vec = is_same_track[triu_indices[0], triu_indices[1]]

    # Positive pairs are hits from the same track
    pos_mask = is_same_track_vec
    pos_pairs = triu_indices[:, pos_mask].T # Shape: (N_pos, 2)

    # Negative pairs are hits from different tracks
    neg_mask = ~is_same_track_vec
    neg_pairs = triu_indices[:, neg_mask].T # Shape: (N_neg, 2)

    n_pos, n_neg = pos_pairs.shape[0], neg_pairs.shape[0]
    if n_pos == 0 or n_neg == 0:
        return None, None

    # Subsample negative pairs to maintain a reasonable balance and reduce computation
    if n_neg > n_pos * 2:
        rand_indices = torch.randperm(n_neg, device=y.device)[:n_pos * 2]
        neg_pairs = neg_pairs[rand_indices]

    return pos_pairs, neg_pairs


def train_model(model, train_loader, val_loader, epochs):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    margin = 1.0  # Margin for contrastive loss

    best_model_state = None
    best_val_loss = float('inf')
    patience = 8
    epochs_no_improve = 0

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0.0
        n_train_events = 0
        total_correct_train, total_pairs_train = 0, 0

        for batch in train_loader: # batch is a list of (X, y) tuples
            optimizer.zero_grad()

            # Efficiently process all hits in the batch with one model call
            list_of_X = [item[0] for item in batch]
            list_of_y = [item[1] for item in batch]
            hit_lengths = [len(x) for x in list_of_X]

            X_cat = torch.cat(list_of_X, dim=0).to(device)
            # Use _embed to process the concatenated tensor
            embeddings_cat = model._embed(X_cat) # Shape: (N_total_hits, D_embed)
            embeddings_list = torch.split(embeddings_cat, hit_lengths)

            batch_loss = 0
            for i in range(len(batch)):
                embeddings, y = embeddings_list[i], list_of_y[i].to(device)

                pairs = get_pairs(y)
                if pairs is None: continue
                pos_pairs, neg_pairs = pairs

                # Contrastive Loss Calculation
                pos_dists = torch.norm(embeddings[pos_pairs[:, 0]] - embeddings[pos_pairs[:, 1]], p=2, dim=1)
                neg_dists = torch.norm(embeddings[neg_pairs[:, 0]] - embeddings[neg_pairs[:, 1]], p=2, dim=1)

                loss_pos = torch.pow(pos_dists, 2).mean()
                loss_neg = torch.pow(torch.clamp(margin - neg_dists, min=0.0), 2).mean()
                loss = loss_pos + loss_neg
                batch_loss += loss

                with torch.no_grad():
                    total_correct_train += (pos_dists < margin).sum().item()
                    total_correct_train += (neg_dists > margin).sum().item()
                    total_pairs_train += len(pos_dists) + len(neg_dists)

            if batch_loss > 0:
                avg_batch_loss = batch_loss / len(batch)
                avg_batch_loss.backward()
                optimizer.step()
                epoch_train_loss += avg_batch_loss.item() * len(batch)
            n_train_events += len(batch)

        # --- Validation Phase ---
        model.eval()
        epoch_val_loss = 0.0
        n_val_events = 0
        total_correct_val, total_pairs_val = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                list_of_X = [item[0] for item in batch]
                list_of_y = [item[1] for item in batch]
                hit_lengths = [len(x) for x in list_of_X]

                X_cat = torch.cat(list_of_X, dim=0).to(device)
                embeddings_cat = model._embed(X_cat)
                embeddings_list = torch.split(embeddings_cat, hit_lengths)

                batch_loss = 0
                for i in range(len(batch)):
                    embeddings, y = embeddings_list[i], list_of_y[i].to(device)

                    pairs = get_pairs(y)
                    if pairs is None: continue
                    pos_pairs, neg_pairs = pairs

                    pos_dists = torch.norm(embeddings[pos_pairs[:, 0]] - embeddings[pos_pairs[:, 1]], p=2, dim=1)
                    neg_dists = torch.norm(embeddings[neg_pairs[:, 0]] - embeddings[neg_pairs[:, 1]], p=2, dim=1)

                    loss_pos = torch.pow(pos_dists, 2).mean()
                    loss_neg = torch.pow(torch.clamp(margin - neg_dists, min=0.0), 2).mean()
                    batch_loss += loss_pos + loss_neg

                    total_correct_val += (pos_dists < margin).sum().item()
                    total_correct_val += (neg_dists > margin).sum().item()
                    total_pairs_val += len(pos_dists) + len(neg_dists)

                if batch_loss > 0:
                    epoch_val_loss += (batch_loss / len(batch)).item() * len(batch)
                n_val_events += len(batch)

        # --- Epoch End ---
        avg_train_loss = epoch_train_loss / n_train_events if n_train_events > 0 else 0
        avg_val_loss = epoch_val_loss / n_val_events if n_val_events > 0 else 0

        current_train_acc = total_correct_train / total_pairs_train if total_pairs_train > 0 else 0
        current_val_acc = total_correct_val / total_pairs_val if total_pairs_val > 0 else 0

        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)
        train_acc.append(current_train_acc)
        val_acc.append(current_val_acc)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
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


