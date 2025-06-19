
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: Python 3.12, PyTorch 2.6.0, NumPy 2.2.3, SciKit-Learn 1.6.1
import os, sys, pickle, gzip, json, torch, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data"
TAG      = "10_50_linear"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"REDVID_{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def split_X_y(evt):
    lay = evt["layer_id"].astype(np.float32)
    lay_norm = lay / lay.max()
    X = np.column_stack([evt["hit_r"],
                         evt["hit_theta"],
                         evt["hit_z"],
                         lay_norm])
    t_id = evt["track_id"].astype(np.int32)
    return (torch.from_numpy(X),
            torch.from_numpy(t_id))

class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, track_id = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, track_id)

def _ragged(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    # batch[i] = (hits_i, track_id_i)      ← shapes: (N_i, F), (N_i,)
    return batch

def make_loaders(batch_size=128, workers=0):
    tr = EventDataset(_load_events("train"), pre=None, train=True)
    va = EventDataset(_load_events("val"),   pre=None, train=False)

    train_ld = DataLoader(tr, batch_size=batch_size, shuffle=True,
                          collate_fn=_ragged, num_workers=workers)
    val_ld   = DataLoader(va, batch_size=batch_size, collate_fn=_ragged)
    return train_ld, val_ld

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch or sklearn (sub-)modules you actually use.
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import copy

# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module.
    # May allocate NumPy arrays or Torch tensors internally, but:
    # transform() must be deterministic.
    # fit(events) receives the *raw* event dicts list, not a tensor batch.
    # Store only derived parameters needed for transform i.e. do not store the raw data
    # itself in the preprocessor object.
    def __init__(self):
        # We will create new features (x, y) and scale a total of 5 features.
        self.scaler = StandardScaler()
        self.fitted = False

    def _get_features_from_dict(self, event: dict) -> np.ndarray:
        """Extracts features for scaling from a raw event dict."""
        r, theta, z = event["hit_r"], event["hit_theta"], event["hit_z"]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([r, theta, z, x, y])

    def _get_features_from_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """Extracts features for scaling from the initial feature tensor."""
        # X has columns: r, theta, z, lay_norm
        r, theta, z = X[:, 0], X[:, 1], X[:, 2]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([r, theta, z, x, y], dim=1)

    def fit(self, events: list[dict]):
        """Fits the scaler on the raw training data."""
        all_features_to_scale = np.vstack([self._get_features_from_dict(evt) for evt in events])
        self.scaler.fit(all_features_to_scale)
        self.fitted = True
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Applies feature engineering and scaling."""
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        # Keep the normalized layer_id separate as it doesn't need scaling.
        lay_norm = X[:, 3].unsqueeze(1)

        # Get the 5 features to be scaled.
        features_to_scale = self._get_features_from_tensor(X) # (N_hits, 5)

        # Scikit-learn scaler requires a NumPy array.
        scaled_features_np = self.scaler.transform(features_to_scale.numpy())
        scaled_features_torch = torch.from_numpy(scaled_features_np.astype(np.float32))

        # Combine scaled features with unscaled lay_norm to get final 6 features.
        final_features = torch.cat([scaled_features_torch, lay_norm], dim=1) # (N_hits, 6)
        return final_features

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features: int, embed_dim: int = 8, dropout_rate: float = 0.2):
        super().__init__()
        # An MLP to learn a low-dimensional embedding for each hit.
        self.network = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, embed_dim),
        )

    def forward(self, batch: list[torch.Tensor]) -> list[torch.Tensor]:
        # batch: A list of tensors, where each tensor represents the hits of one event.
        # Process each event's hits independently.
        embeddings = []
        for hits_tensor in batch:
            # hits_tensor shape: (N_hits_i, in_features)
            emb = self.network(hits_tensor) # (N_hits_i, embed_dim)
            embeddings.append(emb)
        return embeddings

def make_model(in_features: int) -> HitClassifier:
    # The number of input features is determined by the preprocessor (6 in this case).
    return HitClassifier(in_features=in_features, embed_dim=8)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25

# --- HELPER FUNCTIONS FOR TRAINING ---
def contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float, device: torch.device) -> torch.Tensor:
    """Calculates contrastive loss with hard negative/positive mining for a single event."""
    n_hits = embeddings.shape[0]
    if n_hits < 2:
        return torch.tensor(0.0, device=device)

    # dist_matrix[i, j] = ||embeddings[i] - embeddings[j]||^2
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)**2

    # Create masks to identify pairs of hits from the same/different tracks.
    is_same_track = labels.unsqueeze(0) == labels.unsqueeze(1)
    is_different_track = ~is_same_track
    is_not_self = ~torch.eye(n_hits, dtype=torch.bool, device=device)
    positive_mask = is_same_track & is_not_self

    losses = []
    for i in range(n_hits):
        # Anchor hit i must have at least one positive and one negative partner.
        if torch.any(positive_mask[i]) and torch.any(is_different_track[i]):
            # Hard positive: the farthest hit belonging to the same track.
            pos_dist = torch.max(dist_matrix[i][positive_mask[i]])
            # Hard negative: the closest hit belonging to a different track.
            neg_dist = torch.min(dist_matrix[i][is_different_track[i]])

            loss = F.relu(pos_dist - neg_dist + margin)
            losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=device)

    return torch.mean(torch.stack(losses))

def calculate_clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates clustering accuracy using the Hungarian algorithm for label matching."""
    # Filter out noise points from DBSCAN (label -1).
    valid_indices = y_pred != -1
    if not np.any(valid_indices):
        return 0.0

    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    if y_true_valid.shape[0] == 0:
        return 0.0

    # Create a cost matrix where C[i,j] is the number of points in true cluster i and predicted cluster j.
    cm = confusion_matrix(y_true_valid, y_pred_valid)

    # Use the Hungarian algorithm to find the optimal one-to-one mapping between true and predicted labels.
    # We negate the matrix because linear_sum_assignment finds a minimum cost assignment.
    row_ind, col_ind = linear_sum_assignment(-cm)

    # The accuracy is the sum of matched elements divided by the total number of non-noise points.
    n_correct = cm[row_ind, col_ind].sum()
    return n_correct / len(y_true_valid)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
    # PARAMETERS
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)

    # HYPERPARAMETERS
    MARGIN = 1.0  # For contrastive loss
    DBSCAN_EPS = 0.5  # DBSCAN radius, heuristically margin/2
    DBSCAN_MIN_SAMPLES = 2  # A track needs at least 2 hits
    EARLY_STOPPING_PATIENCE = 5

    # TRAINING STATE
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # HISTORY
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    print(f"Training on {DEVICE} for up to {epochs} epochs...")
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            # Unpack batch and move to device
            hits_batch = [item[0].to(DEVICE) for item in batch] # list of (N_i, F) tensors
            labels_batch = [item[1].to(DEVICE) for item in batch] # list of (N_i,) tensors

            # Forward pass: get embeddings for all events in the batch
            embeddings_batch = model(hits_batch) # list of (N_i, D) tensors

            # Calculate loss per event and aggregate
            batch_loss = 0.0
            for embeddings, labels in zip(embeddings_batch, labels_batch):
                batch_loss += contrastive_loss(embeddings, labels, MARGIN, DEVICE)

            if len(batch) > 0:
                loss = batch_loss / len(batch)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        train_acc.append(0.0)  # Skip accuracy calculation on training set for speed

        # --- VALIDATION PHASE ---
        model.eval()
        epoch_val_loss, epoch_val_acc = 0.0, 0.0
        n_val_events = 0
        with torch.no_grad():
            for batch in val_loader:
                hits_batch = [item[0].to(DEVICE) for item in batch]
                labels_true_batch = [item[1] for item in batch]

                embeddings_batch = model(hits_batch)

                for i in range(len(batch)):
                    embeddings_i = embeddings_batch[i]
                    labels_true_i = labels_true_batch[i].to(DEVICE)

                    # Calculate validation loss for this event
                    epoch_val_loss += contrastive_loss(embeddings_i, labels_true_i, MARGIN, DEVICE).item()

                    # Calculate validation accuracy via clustering
                    embeddings_np = embeddings_i.cpu().numpy()
                    labels_true_np = labels_true_batch[i].cpu().numpy()
                    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
                    labels_pred_np = dbscan.fit_predict(embeddings_np)

                    epoch_val_acc += calculate_clustering_accuracy(labels_true_np, labels_pred_np)
                    n_val_events += 1

        avg_val_loss = epoch_val_loss / n_val_events if n_val_events > 0 else 0.0
        avg_val_acc = epoch_val_acc / n_val_events if n_val_events > 0 else 0.0
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {avg_val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} as validation loss did not improve for {EARLY_STOPPING_PATIENCE} epochs.")
            break

    if best_model_state:
        print("Restoring best model weights found during training.")
        model.load_state_dict(best_model_state)

    return model, train_loss, val_loss, train_acc, val_acc

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 
                         
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
    train_ds = EventDataset(raw_train, pre, train=True)
    val_ds   = EventDataset(raw_val , pre, train=False)
    train_ld = DataLoader(train_ds, batch_size=512,
                        shuffle=True, collate_fn=_ragged)
    val_ld   = DataLoader(val_ds,   batch_size=512,
                        collate_fn=_ragged)

    # 2. Build model
    in_features = train_ds[0][0].shape[-1]                   
    model = make_model(in_features)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_ld, val_ld, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single toy forward pass
    if dryrun:
        toy_event       = torch.zeros(10, in_features)
        toy_transformed = pre.transform(toy_event)
        toy_batch       = [toy_transformed]
        try:
            _ = trained_model(toy_batch)
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

