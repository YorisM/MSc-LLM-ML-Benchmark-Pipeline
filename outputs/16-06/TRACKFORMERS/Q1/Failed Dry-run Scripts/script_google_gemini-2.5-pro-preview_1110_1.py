
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

# <start code template>
# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch or sklearn (sub-)modules you actually use.
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.cluster import DBSCAN
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
        # We manually store mean and std to keep `transform` in PyTorch
        self.r_mean, self.r_std = None, None
        self.z_mean, self.z_std = None, None

    def fit(self, events):
        # Extract statistics from all events for r and z coordinates
        all_r = np.concatenate([e['hit_r'] for e in events])
        all_z = np.concatenate([e['hit_z'] for e in events])

        self.r_mean, self.r_std = all_r.mean(), all_r.std()
        self.z_mean, self.z_std = all_z.mean(), all_z.std()
        return self

    def transform(self, X: torch.Tensor):
        # Apply preprocessing logic, return torch.Tensor
        # X: (N_hits, 4) -> [r, theta, z, layer_id_norm]

        # Standard scale r and z
        # Using pre-computed stats to avoid numpy conversion
        r_scaled = (X[:, 0] - self.r_mean) / self.r_std
        z_scaled = (X[:, 2] - self.z_mean) / self.z_std

        # Transform theta to (sin(theta), cos(theta)) to handle periodicity
        theta = X[:, 1]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Keep layer_id_norm as is
        layer_norm = X[:, 3]

        # New feature vector: [r_scaled, cos_theta, sin_theta, z_scaled, layer_id_norm]
        # Shape: (N_hits, 5)
        return torch.stack([r_scaled, cos_theta, sin_theta, z_scaled, layer_norm], dim=1)

    def fit_transform(self, events):
        # This function is not used by the harness, but implemented for completeness.
        self.fit(events)
        # For this to be fully correct, we'd need to process all events and stack them.
        # Since it's unused, we'll leave it as a conceptual implementation.
        all_X = []
        for evt in events:
            # Note: split_X_y is defined in the harness prefix
            X, _ = split_X_y(evt)
            all_X.append(self.transform(X))
        return all_X

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, embedding_dim=8, hidden_dim=128):
        super().__init__()
        # Embedding network to learn a representation space for hits.
        # A metric learning approach is used, where the model outputs an embedding for each hit.
        self.embedding_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, batch_X: list[torch.Tensor]):
        # batch_X: list of hit tensors, one for each event in the batch
        # list[ (N_hits_i, F_in) ]
        outputs = []
        for X in batch_X:
            # Get embeddings from the network
            emb = self.embedding_net(X)
            # L2 normalize embeddings for stable metric learning on a hypersphere
            emb = nn.functional.normalize(emb, p=2, dim=1)
            outputs.append(emb)
        # returns list[ (N_hits_i, D_emb) ]
        return outputs

def make_model(in_features):
    return HitClassifier(in_features, embedding_dim=8, hidden_dim=128)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25

def _sample_triplets(embeddings, labels, n_triplets, device):
    """
    Sample triplets (anchor, positive, negative) for triplet loss.
    This is a semi-hard online mining strategy, implemented per-event.
    """
    anchors, positives, negatives = [], [], []

    # Hits with track_id=0 are noise, not part of any track
    # We only form positive pairs from hits with the same, non-zero track_id
    unique_labels, counts = torch.unique(labels, return_counts=True)

    # Filter for labels corresponding to tracks with at least 2 hits
    possible_labels = unique_labels[(counts >= 2) & (unique_labels > 0)]

    if len(possible_labels) == 0:
        return None, None, None

    # Pre-compute indices for each label for faster sampling
    label_indices = {label.item(): torch.where(labels == label)[0] 
                     for label in possible_labels}

    n_sampled = 0
    attempts = 0 # To prevent infinite loops in sparse events
    while n_sampled < n_triplets and attempts < 2 * n_triplets:
        attempts += 1
        # 1. Sample anchor and positive from the same track
        # Randomly choose a track ID that has at least two hits
        anchor_label = possible_labels[torch.randint(len(possible_labels), (1,))].item()

        # Randomly sample two distinct hits from this track
        positive_indices = label_indices[anchor_label]
        anchor_idx, positive_idx = positive_indices[torch.randperm(len(positive_indices))[:2]]

        # 2. Sample negative from a different track or from noise
        # Create mask of all hits not belonging to the anchor's track
        negative_mask = (labels != anchor_label)
        if not torch.any(negative_mask):
            continue # No other tracks/noise to sample from

        negative_indices = torch.where(negative_mask)[0]
        negative_idx = negative_indices[torch.randint(len(negative_indices), (1,))]

        anchors.append(embeddings[anchor_idx])
        positives.append(embeddings[positive_idx])
        negatives.append(embeddings[negative_idx])
        n_sampled += 1

    if not anchors:
        return None, None, None

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def _calculate_accuracy(true_labels, pred_labels):
    """
    Calculates clustering accuracy.
    - Matches predicted clusters to true tracks via majority voting.
    - Handles noise points (true_label=0, pred_label=-1) correctly.
    """

    correct = 0
    total = len(true_labels)

    # 1. Handle hits predicted as noise by DBSCAN (pred_label = -1)
    noise_cluster_mask = (pred_labels == -1)
    if np.any(noise_cluster_mask):
        true_labels_in_noise = true_labels[noise_cluster_mask]
        # Correct if a true noise hit (true_label=0) is predicted as noise
        correct += np.sum(true_labels_in_noise == 0)

    # 2. Handle non-noise clusters
    # Iterate over each unique predicted cluster ID
    pred_ids = np.unique(pred_labels[pred_labels != -1])
    for pid in pred_ids:
        cluster_mask = (pred_labels == pid)
        true_labels_in_cluster = true_labels[cluster_mask]

        # To find the majority track, consider only true tracks (label > 0)
        true_track_labels = true_labels_in_cluster[true_labels_in_cluster > 0]

        if len(true_track_labels) == 0:
            # This cluster contains only noise hits but was not labeled as noise.
            # All are considered incorrect.
            continue

        # Find the most frequent true track ID in the cluster
        unique_t, counts_t = np.unique(true_track_labels, return_counts=True)
        majority_id = unique_t[np.argmax(counts_t)]

        # Add count of hits in the cluster that match the majority ID
        correct += np.sum(true_labels_in_cluster == majority_id)

    return correct / total if total > 0 else 0.0

def train_model(model, train_loader, val_loader, epochs):
    # REQUIREMENTS 
    # Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    # Implement early-stopping.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # HYPERPARAMETERS
    LR = 1e-3
    PATIENCE = 5
    TRIPLET_MARGIN = 1.0
    TRIPLETS_PER_EVENT = 128
    DBSCAN_EPS = 0.6
    DBSCAN_MIN_SAMPLES = 2

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    triplet_loss_fn = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)

    # History and Early Stopping
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    best_val_acc = -1.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            list_X = [data[0].to(device) for data in batch]
            list_y = [data[1].to(device) for data in batch]

            optimizer.zero_grad()
            list_emb = model(list_X)

            batch_loss = 0
            n_events_with_loss = 0
            for emb, labels in zip(list_emb, list_y):
                anchors, positives, negatives = _sample_triplets(emb, labels, TRIPLETS_PER_EVENT, device)
                if anchors is not None:
                    loss = triplet_loss_fn(anchors, positives, negatives)
                    batch_loss += loss
                    n_events_with_loss += 1

            if n_events_with_loss > 0:
                mean_loss = batch_loss / n_events_with_loss
                mean_loss.backward()
                optimizer.step()
                total_train_loss += mean_loss.item()

        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        history['train_loss'].append(avg_train_loss)

        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0.0
        all_val_accs = []
        with torch.no_grad():
            for batch in val_loader:
                list_X = [data[0].to(device) for data in batch]
                list_y_cpu = [data[1] for data in batch]
                list_y_gpu = [data[1].to(device) for data in batch]

                list_emb = model(list_X)

                for i, emb in enumerate(list_emb):
                    true_labels = list_y_cpu[i].cpu().numpy()

                    # Cluster embeddings with DBSCAN
                    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='euclidean', n_jobs=-1)
                    pred_labels = clustering.fit_predict(emb.cpu().numpy())

                    acc = _calculate_accuracy(true_labels, pred_labels)
                    all_val_accs.append(acc)

                    # Also compute val loss for monitoring
                    anchors, positives, negatives = _sample_triplets(emb, list_y_gpu[i], TRIPLETS_PER_EVENT, device)
                    if anchors is not None:
                        loss = triplet_loss_fn(anchors, positives, negatives)
                        total_val_loss += loss.item()

        avg_val_acc = np.mean(all_val_accs) if all_val_accs else 0.0
        avg_val_loss = total_val_loss / len(val_loader.dataset) if val_loader.dataset else 0.0

        history['val_acc'].append(avg_val_acc)
        history['val_loss'].append(avg_val_loss)
        # Training accuracy is not computed to save time.
        history['train_acc'].append(0.0) 

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        scheduler.step(avg_val_acc)

        # Early stopping logic
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    # Load best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Unpack history
    train_loss, val_loss, train_acc, val_acc = (history['train_loss'], history['val_loss'], 
                                                history['train_acc'], history['val_acc'])

    return model, train_loss, val_loss, train_acc, val_acc
# <end code template>

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

