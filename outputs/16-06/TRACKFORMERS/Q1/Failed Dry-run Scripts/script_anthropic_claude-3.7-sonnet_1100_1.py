
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
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from collections import defaultdict

# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.r_mean = None
        self.r_std = None
        self.theta_mean = None
        self.theta_std = None
        self.z_mean = None
        self.z_std = None
        self.r_min = None
        self.r_max = None
        self.theta_min = None
        self.theta_max = None
        self.z_min = None
        self.z_max = None

    def fit(self, events):
        # Compute statistics across all events
        all_r = np.concatenate([event["hit_r"] for event in events])
        all_theta = np.concatenate([event["hit_theta"] for event in events])
        all_z = np.concatenate([event["hit_z"] for event in events])

        # Compute mean and std for normalization
        self.r_mean, self.r_std = np.mean(all_r), np.std(all_r)
        self.theta_mean, self.theta_std = np.mean(all_theta), np.std(all_theta)
        self.z_mean, self.z_std = np.mean(all_z), np.std(all_z)

        # Store min/max for robust scaling
        self.r_min, self.r_max = np.min(all_r), np.max(all_r)
        self.theta_min, self.theta_max = np.min(all_theta), np.max(all_theta)
        self.z_min, self.z_max = np.min(all_z), np.max(all_z)

        return self

    def transform(self, X):
        # X is a tensor with shape [N_hits, 4]
        # X contains [hit_r, hit_theta, hit_z, layer_id_norm]

        # Apply normalization
        X_normalized = X.clone()

        # Z-score normalization
        X_normalized[:, 0] = (X[:, 0] - self.r_mean) / self.r_std  # Normalize r
        X_normalized[:, 1] = (X[:, 1] - self.theta_mean) / self.theta_std  # Normalize theta
        X_normalized[:, 2] = (X[:, 2] - self.z_mean) / self.z_std  # Normalize z

        # Add additional features
        r_scaled = (X[:, 0] - self.r_min) / (self.r_max - self.r_min)
        theta_scaled = (X[:, 1] - self.theta_min) / (self.theta_max - self.theta_min)
        z_scaled = (X[:, 2] - self.z_min) / (self.z_max - self.z_min)

        # Create new features tensor with additional features
        new_features = torch.zeros((X.shape[0], 8), dtype=X.dtype, device=X.device)
        new_features[:, 0:4] = X_normalized  # Original normalized features
        new_features[:, 4] = r_scaled
        new_features[:, 5] = theta_scaled
        new_features[:, 6] = z_scaled
        new_features[:, 7] = X[:, 3]  # Original layer_id_norm

        return new_features

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim=256, embedding_dim=128, num_heads=8, num_layers=4):
        super().__init__()

        # Initial embedding
        self.embedding = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # Clustering parameters
        self.eps = 0.4
        self.min_samples = 2

    def forward(self, batch):
        # batch is a list of tuples (hits, track_ids)
        all_embeddings = []
        all_track_ids = []

        for item in batch:
            if isinstance(item, tuple):
                hits, track_ids = item
            else:
                hits = item
                track_ids = None

            # Move to device
            device = next(self.parameters()).device
            hits = hits.to(device)
            if track_ids is not None:
                track_ids = track_ids.to(device)

            # Initial embeddings
            embeddings = self.embedding(hits)  # [N_hits, hidden_dim]

            # Apply transformer
            transformer_out = self.transformer(embeddings)  # [N_hits, hidden_dim]

            # Final projection
            final_embeddings = self.output(transformer_out)  # [N_hits, embedding_dim]
            final_embeddings = F.normalize(final_embeddings, p=2, dim=1)

            all_embeddings.append(final_embeddings)
            if track_ids is not None:
                all_track_ids.append(track_ids)

        return all_embeddings, all_track_ids if all_track_ids else None

    def cluster_hits(self, embeddings):
        # Cluster the embeddings to predict track assignments
        embeddings_np = embeddings.detach().cpu().numpy()
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_ids = dbscan.fit_predict(embeddings_np)
        return cluster_ids

def make_model(in_features):
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # Contrastive loss parameters
    margin = 1.0

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    best_val_acc = 0
    best_model = None
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            embeddings_list, track_ids_list = model(batch)

            # Compute contrastive loss
            batch_loss = 0
            batch_acc = 0

            for embeddings, track_ids in zip(embeddings_list, track_ids_list):
                # Skip empty batches
                if embeddings.size(0) <= 1:
                    continue

                # Compute pairwise distances
                dist_matrix = torch.cdist(embeddings, embeddings)  # [N_hits, N_hits]

                # Create mask where 1 indicates same track, 0 different track
                track_mask = (track_ids.unsqueeze(0) == track_ids.unsqueeze(1)).float()  # [N_hits, N_hits]

                # Contrastive loss: pull same-track embeddings together, push different-track embeddings apart
                pos_loss = track_mask * dist_matrix
                neg_loss = (1 - track_mask) * torch.clamp(margin - dist_matrix, min=0)

                # Balance positive and negative samples
                pos_loss_mean = pos_loss.sum() / (track_mask.sum() + 1e-8)
                neg_loss_mean = neg_loss.sum() / ((1 - track_mask).sum() + 1e-8)
                loss = pos_loss_mean + neg_loss_mean

                batch_loss += loss

                # Compute accuracy by predicting pairwise assignments
                threshold = 0.5
                pred_same_track = (dist_matrix < threshold).float()
                accuracy = (pred_same_track == track_mask).float().mean()
                batch_acc += accuracy.item()

            # Skip if batch is empty
            if len(embeddings_list) == 0:
                continue

            batch_loss /= len(embeddings_list)
            batch_acc /= len(embeddings_list)

            # Backward pass
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc
            num_batches += 1

        if num_batches > 0:
            train_loss.append(epoch_loss / num_batches)
            train_acc.append(epoch_acc / num_batches)
        else:
            train_loss.append(0)
            train_acc.append(0)

        # Validation
        model.eval()
        val_epoch_loss = 0
        val_epoch_acc = 0
        val_num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Forward pass
                embeddings_list, track_ids_list = model(batch)

                # Compute contrastive loss and clustering accuracy
                batch_loss = 0
                batch_acc = 0

                for embeddings, track_ids in zip(embeddings_list, track_ids_list):
                    # Skip empty batches
                    if embeddings.size(0) <= 1:
                        continue

                    # Compute pairwise distances for contrastive loss
                    dist_matrix = torch.cdist(embeddings, embeddings)
                    track_mask = (track_ids.unsqueeze(0) == track_ids.unsqueeze(1)).float()

                    # Contrastive loss
                    pos_loss = track_mask * dist_matrix
                    neg_loss = (1 - track_mask) * torch.clamp(margin - dist_matrix, min=0)

                    pos_loss_mean = pos_loss.sum() / (track_mask.sum() + 1e-8)
                    neg_loss_mean = neg_loss.sum() / ((1 - track_mask).sum() + 1e-8)
                    loss = pos_loss_mean + neg_loss_mean

                    batch_loss += loss

                    # Compute clustering accuracy using DBSCAN
                    cluster_ids = model.cluster_hits(embeddings)

                    # Map cluster IDs to track IDs
                    track_ids_np = track_ids.cpu().numpy()

                    # Handle special case where all points are noise (-1)
                    if np.all(cluster_ids == -1) or len(np.unique(cluster_ids)) <= 1:
                        batch_acc += 0
                        continue

                    # Create a mapping from cluster ID to most common track ID
                    cluster_to_track = {}
                    for cluster_id in np.unique(cluster_ids):
                        if cluster_id == -1:  # Noise points
                            continue
                        mask = cluster_ids == cluster_id
                        track_counts = np.bincount(track_ids_np[mask])
                        most_common_track = np.argmax(track_counts)
                        cluster_to_track[cluster_id] = most_common_track

                    # Compute accuracy
                    correct = 0
                    total = 0

                    for i, (cluster_id, true_track) in enumerate(zip(cluster_ids, track_ids_np)):
                        if cluster_id in cluster_to_track:
                            total += 1
                            if cluster_to_track[cluster_id] == true_track:
                                correct += 1

                    if total > 0:
                        batch_acc += correct / total
                    else:
                        batch_acc += 0

                if len(embeddings_list) > 0:
                    batch_loss /= len(embeddings_list)
                    batch_acc /= len(embeddings_list)

                    val_epoch_loss += batch_loss.item()
                    val_epoch_acc += batch_acc
                    val_num_batches += 1

        if val_num_batches > 0:
            val_loss.append(val_epoch_loss / val_num_batches)
            val_acc.append(val_epoch_acc / val_num_batches)
        else:
            val_loss.append(0)
            val_acc.append(0)

        # Learning rate scheduling
        scheduler.step(val_acc[-1])

        # Early stopping
        if val_acc[-1] > best_val_acc:
            best_val_acc = val_acc[-1]
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

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

