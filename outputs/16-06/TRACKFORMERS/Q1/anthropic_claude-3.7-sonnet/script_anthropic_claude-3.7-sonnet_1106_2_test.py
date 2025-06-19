
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
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.r_mean = None
        self.r_std = None
        self.theta_mean = None
        self.theta_std = None
        self.z_mean = None
        self.z_std = None

    def fit(self, events):
        # Extract all hit coordinates
        r_all = np.concatenate([evt["hit_r"] for evt in events])
        theta_all = np.concatenate([evt["hit_theta"] for evt in events])
        z_all = np.concatenate([evt["hit_z"] for evt in events])

        # Compute statistics
        self.r_mean, self.r_std = np.mean(r_all), np.std(r_all)
        self.theta_mean, self.theta_std = np.mean(theta_all), np.std(theta_all)
        self.z_mean, self.z_std = np.mean(z_all), np.std(z_all)

        return self

    def transform(self, X):
        # X shape: (N_hits, 4) with columns [r, theta, z, layer_norm]
        X_norm = X.clone()

        # Normalize r, theta, z
        X_norm[:, 0] = (X[:, 0] - self.r_mean) / self.r_std
        X_norm[:, 1] = (X[:, 1] - self.theta_mean) / self.theta_std
        X_norm[:, 2] = (X[:, 2] - self.z_mean) / self.z_std

        # Layer is already normalized (as per the split_X_y function)
        return X_norm

    def fit_transform(self, events):
        self.fit(events)
        return self.transform(events)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim=256, num_layers=4, dropout=0.1):
        super().__init__()

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, batch):
        # batch is a list of tensors, each with shape (N_hits, in_features)
        embeddings = []

        for hits in batch:
            # hits shape: (N_hits, in_features)
            x = self.input_layer(hits)

            # Apply hidden layers with residual connections
            for layer in self.hidden_layers:
                residual = x
                x = layer(x)
                x = x + residual

            # Output layer
            emb = self.output_layer(x)

            # Normalize embeddings to unit length
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb)

        return embeddings

def make_model(in_features):
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 5
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    best_val_acc = 0
    best_model_state = None
    patience = 5
    patience_counter = 0
    margin = 0.5  # Margin for contrastive loss

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        num_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            hits_batch, track_ids_batch = zip(*batch)

            # Move to device
            hits_batch = [hits.to(device) for hits in hits_batch]
            track_ids_batch = [track_ids.to(device) for track_ids in track_ids_batch]

            # Forward pass
            embeddings = model(hits_batch)

            # Compute loss and accuracy
            batch_loss = 0
            batch_acc = 0

            for embeddings_event, track_ids_event in zip(embeddings, track_ids_batch):
                # Compute pairwise cosine similarity
                similarity = torch.matmul(embeddings_event, embeddings_event.t())

                # Create target similarity matrix
                track_ids_expanded1 = track_ids_event.unsqueeze(1)
                track_ids_expanded2 = track_ids_event.unsqueeze(0)
                target_similarity = (track_ids_expanded1 == track_ids_expanded2).float()

                # Margin-based contrastive loss
                pos_loss = (1 - similarity) * target_similarity
                neg_loss = torch.clamp(similarity - margin, min=0) * (1 - target_similarity)

                # Ignore diagonal elements (self-similarity)
                mask = 1 - torch.eye(track_ids_event.size(0), device=device)
                pos_loss = pos_loss * mask
                neg_loss = neg_loss * mask

                loss = (pos_loss.sum() + neg_loss.sum()) / (mask.sum() + 1e-8)
                batch_loss += loss

                # Compute accuracy using clustering
                embeddings_np = embeddings_event.detach().cpu().numpy()
                track_ids_np = track_ids_event.detach().cpu().numpy()

                # Estimate the number of clusters (tracks)
                n_tracks = len(torch.unique(track_ids_event))

                # Cluster the embeddings
                kmeans = KMeans(n_clusters=n_tracks, n_init=10).fit(embeddings_np)
                labels = kmeans.labels_

                # Compute adjusted Rand index as accuracy
                ari = adjusted_rand_score(track_ids_np, labels)
                batch_acc += ari

            batch_loss /= len(hits_batch)
            batch_acc /= len(hits_batch)

            # Backpropagation
            batch_loss.backward()
            optimizer.step()

            epoch_train_loss += batch_loss.item()
            epoch_train_acc += batch_acc
            num_train_batches += 1

        avg_train_loss = epoch_train_loss / num_train_batches
        avg_train_acc = epoch_train_acc / num_train_batches
        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)

        if (epoch + 1) % 1 == 0:   # print every epoch
            print(f"[TRAIN] epoch {epoch+1:02d}  "
                    f"loss={batch_loss.item():.4f}  "
                    f"ari={batch_acc:.3f}")

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                hits_batch, track_ids_batch = zip(*batch)

                # Move to device
                hits_batch = [hits.to(device) for hits in hits_batch]
                track_ids_batch = [track_ids.to(device) for track_ids in track_ids_batch]

                # Forward pass
                embeddings = model(hits_batch)

                # Compute loss and accuracy
                batch_loss = 0
                batch_acc = 0

                for embeddings_event, track_ids_event in zip(embeddings, track_ids_batch):
                    # Compute similarity
                    similarity = torch.matmul(embeddings_event, embeddings_event.t())

                    # Create target similarity matrix
                    track_ids_expanded1 = track_ids_event.unsqueeze(1)
                    track_ids_expanded2 = track_ids_event.unsqueeze(0)
                    target_similarity = (track_ids_expanded1 == track_ids_expanded2).float()

                    # Margin-based contrastive loss
                    pos_loss = (1 - similarity) * target_similarity
                    neg_loss = torch.clamp(similarity - margin, min=0) * (1 - target_similarity)

                    # Ignore diagonal elements
                    mask = 1 - torch.eye(track_ids_event.size(0), device=device)
                    pos_loss = pos_loss * mask
                    neg_loss = neg_loss * mask

                    loss = (pos_loss.sum() + neg_loss.sum()) / (mask.sum() + 1e-8)
                    batch_loss += loss

                    # Compute accuracy using clustering
                    embeddings_np = embeddings_event.detach().cpu().numpy()
                    track_ids_np = track_ids_event.detach().cpu().numpy()

                    # Estimate the number of clusters (tracks)
                    n_tracks = len(torch.unique(track_ids_event))

                    # Cluster the embeddings
                    kmeans = KMeans(n_clusters=n_tracks, n_init=10).fit(embeddings_np)
                    labels = kmeans.labels_

                    # Compute adjusted Rand index as accuracy
                    ari = adjusted_rand_score(track_ids_np, labels)
                    batch_acc += ari

                batch_loss /= len(hits_batch)
                batch_acc /= len(hits_batch)

                epoch_val_loss += batch_loss.item()
                epoch_val_acc += batch_acc
                num_val_batches += 1

        avg_val_loss = epoch_val_loss / num_val_batches
        avg_val_acc = epoch_val_acc / num_val_batches
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

        # Learning rate scheduling
        scheduler.step(avg_val_acc)

        # Early stopping
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    if best_model_state is not None:
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
    # 1. Load & preprocess  ── load just 5 % of each split to speed-test
    RAW_PCT = 0.05           # tweak here if you want more / less
    raw_train, raw_val = _load_events("train"), _load_events("val")

    n_tr = max(1, int(len(raw_train) * RAW_PCT))
    n_va = max(1, int(len(raw_val)   * RAW_PCT))
    raw_train, raw_val = raw_train[:n_tr], raw_val[:n_va]

    if dryrun:                                         # keep tiny dry-run slice
        raw_train, raw_val = raw_train[:32], raw_val[:8]

    print(f"[INFO] loaded {len(raw_train)} train events "
          f"and {len(raw_val)} val events (pct={RAW_PCT:.0%})")
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
    print(f"[INFO] model built – in_features={in_features}, "
        f"total params={sum(p.numel() for p in model.parameters()):,}")

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

