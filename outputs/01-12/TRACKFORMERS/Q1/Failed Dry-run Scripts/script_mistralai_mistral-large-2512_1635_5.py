
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import detect_and_assert_lane, assert_label_output_by_lane, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, build_trackformers_model, to_python

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

def split_X_y(evt):
    X = np.column_stack([
        evt["hit_r"].astype(np.float32),
        evt["hit_theta"].astype(np.float32),
        evt["hit_z"].astype(np.float32),
        evt["layer_id"].astype(np.float32)
    ])
    y = evt["track_id"].astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, labels = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, labels)

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from collections import defaultdict

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_mean = None
        self.layer_std = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": None,
            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Compute global statistics for normalization
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Only scale r, theta, z

        # Compute layer statistics
        layers = all_X[:, 3]
        self.layer_mean = np.mean(layers)
        self.layer_std = np.std(layers)

        return self

    def transform(self, X):
        # Normalize r, theta, z
        X_norm = X.clone()
        X_norm[:, :3] = torch.from_numpy(self.scaler.transform(X[:, :3].numpy()))

        # Normalize layer_id
        if self.layer_std > 0:
            X_norm[:, 3] = (X[:, 3] - self.layer_mean) / self.layer_std

        # Create edge indices using spatial proximity
        coords = X_norm[:, :3].numpy()
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=0.5)
        edge_index = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()

        # Create PyG Data object
        data = Data(x=X_norm, edge_index=edge_index)
        return data

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        num_features = example_batch_x.x.shape[1]

        # Graph layers
        self.conv1 = EdgeConv(num_features, 64)
        self.conv2 = EdgeConv(64, 128)
        self.conv3 = EdgeConv(128, 256)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Output layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

        # Track embedding
        self.track_embedding = nn.Linear(32, 32)

        # Final classification
        self.classifier = nn.Linear(32, 1)  # Will predict similarity scores

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Attention
        attn_weights = F.softmax(self.attention(x), dim=0)
        x = x * attn_weights

        # MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

    def predict_labels(self, data):
        with torch.no_grad():
            embeddings = self.forward(data)  # [N_hits, 32]

            # Get batch information
            batch = data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

            # Cluster embeddings using HDBSCAN (approximated with distance threshold)
            from sklearn.cluster import DBSCAN

            # Convert to numpy for clustering
            emb_np = embeddings.cpu().numpy()
            batch_np = batch.cpu().numpy()

            all_labels = []
            for b in torch.unique(batch):
                mask = (batch == b)
                if mask.sum() == 0:
                    continue

                # Cluster within this event
                X = emb_np[mask.cpu().numpy()]
                if len(X) < 2:
                    labels = np.zeros(len(X), dtype=int)
                else:
                    # Use DBSCAN with distance threshold
                    clustering = DBSCAN(eps=0.5, min_samples=3).fit(X)
                    labels = clustering.labels_

                # Convert to global labels
                max_label = len(all_labels)
                labels[labels >= 0] += max_label
                all_labels.extend(labels)

            # Convert to tensor
            labels = torch.tensor(all_labels, dtype=torch.long, device=embeddings.device)

            # Assign -1 to noise points
            labels[labels < 0] = -1

            return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            # Get embeddings
            embeddings = model(data)

            # Create positive and negative pairs for contrastive loss
            batch = data.batch
            pos_pairs = []
            neg_pairs = []

            for b in torch.unique(batch):
                mask = (batch == b)
                emb = embeddings[mask]
                y = data.y[mask]

                # Create positive pairs (same track)
                track_ids = torch.unique(y[y > 0])
                for track_id in track_ids:
                    track_mask = (y == track_id)
                    if track_mask.sum() >= 2:
                        indices = torch.where(track_mask)[0]
                        pairs = torch.combinations(indices, r=2)
                        pos_pairs.extend(pairs.tolist())

                # Create negative pairs (different tracks)
                if len(track_ids) >= 2:
                    for i in range(len(track_ids)):
                        for j in range(i+1, len(track_ids)):
                            track1_mask = (y == track_ids[i])
                            track2_mask = (y == track_ids[j])
                            if track1_mask.sum() > 0 and track2_mask.sum() > 0:
                                idx1 = torch.where(track1_mask)[0][0]
                                idx2 = torch.where(track2_mask)[0][0]
                                neg_pairs.append([idx1.item(), idx2.item()])

            # Convert to tensors
            if pos_pairs:
                pos_pairs = torch.tensor(pos_pairs, device=device)
                pos_emb1 = embeddings[pos_pairs[:, 0]]
                pos_emb2 = embeddings[pos_pairs[:, 1]]
                pos_sim = F.cosine_similarity(pos_emb1, pos_emb2)
            else:
                pos_sim = torch.tensor([], device=device)

            if neg_pairs:
                neg_pairs = torch.tensor(neg_pairs, device=device)
                neg_emb1 = embeddings[neg_pairs[:, 0]]
                neg_emb2 = embeddings[neg_pairs[:, 1]]
                neg_sim = F.cosine_similarity(neg_emb1, neg_emb2)
            else:
                neg_sim = torch.tensor([], device=device)

            # Contrastive loss
            margin = 0.5
            pos_loss = (1 - pos_sim).pow(2).mean() if pos_sim.numel() > 0 else 0
            neg_loss = (torch.clamp(margin - neg_sim, min=0).pow(2).mean()
                       if neg_sim.numel() > 0 else 0)
            loss = pos_loss + neg_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Training accuracy (approximate)
            with torch.no_grad():
                pred_labels = model.predict_labels(data)
                correct = (pred_labels == data.y).float().sum()
                total_correct += correct.item()
                total_samples += data.y.numel()

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples if total_samples > 0 else 0
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                embeddings = model(data)

                # Same loss calculation as training
                batch = data.batch
                pos_pairs = []
                neg_pairs = []

                for b in torch.unique(batch):
                    mask = (batch == b)
                    emb = embeddings[mask]
                    y = data.y[mask]

                    track_ids = torch.unique(y[y > 0])
                    for track_id in track_ids:
                        track_mask = (y == track_id)
                        if track_mask.sum() >= 2:
                            indices = torch.where(track_mask)[0]
                            pairs = torch.combinations(indices, r=2)
                            pos_pairs.extend(pairs.tolist())

                    if len(track_ids) >= 2:
                        for i in range(len(track_ids)):
                            for j in range(i+1, len(track_ids)):
                                track1_mask = (y == track_ids[i])
                                track2_mask = (y == track_ids[j])
                                if track1_mask.sum() > 0 and track2_mask.sum() > 0:
                                    idx1 = torch.where(track1_mask)[0][0]
                                    idx2 = torch.where(track2_mask)[0][0]
                                    neg_pairs.append([idx1.item(), idx2.item()])

                if pos_pairs:
                    pos_pairs = torch.tensor(pos_pairs, device=device)
                    pos_emb1 = embeddings[pos_pairs[:, 0]]
                    pos_emb2 = embeddings[pos_pairs[:, 1]]
                    pos_sim = F.cosine_similarity(pos_emb1, pos_emb2)
                else:
                    pos_sim = torch.tensor([], device=device)

                if neg_pairs:
                    neg_pairs = torch.tensor(neg_pairs, device=device)
                    neg_emb1 = embeddings[neg_pairs[:, 0]]
                    neg_emb2 = embeddings[neg_pairs[:, 1]]
                    neg_sim = F.cosine_similarity(neg_emb1, neg_emb2)
                else:
                    neg_sim = torch.tensor([], device=device)

                pos_loss = (1 - pos_sim).pow(2).mean() if pos_sim.numel() > 0 else 0
                neg_loss = (torch.clamp(margin - neg_sim, min=0).pow(2).mean()
                           if neg_sim.numel() > 0 else 0)
                loss = pos_loss + neg_loss
                val_loss += loss.item()

                # Validation accuracy
                pred_labels = model.predict_labels(data)
                correct = (pred_labels == data.y).float().sum()
                val_correct += correct.item()
                val_samples += data.y.numel()

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_samples if val_samples > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step(val_acc)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
        else:
            if epoch > 10 and val_acc < best_val_acc * 0.95:
                print("Early stopping triggered")
                break

    # Load best model
    model.load_state_dict(best_model)

    return model, train_losses, val_losses, train_accs, val_accs

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    Xs = [split_X_y(evt)[0] for evt in raw_train]
    pre = make_preprocessor().fit(Xs)

    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, raw_train, pre, train=True)
    val_ds       = build_dataset(spec, raw_val,   pre, train=False)
    train_loader = build_dataloader(spec, train_ds, is_eval=False)
    val_loader   = build_dataloader(spec, val_ds,   is_eval=True)

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane(spec, first_batch)

    # Build model
    model = build_trackformers_model(mode, first_batch, make_model, device)

    # Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        if not hasattr(trained_model, "predict_labels") or not callable(getattr(trained_model, "predict_labels")):
            raise TypeError("Contract error: trained model must implement predict_labels(batch_x).")

        trained_model.eval()
        try:
            with torch.no_grad():
                mode = None
                for i, batch in enumerate(val_loader):
                    if mode is None:
                        mode = detect_and_assert_lane(spec, batch)

                    if mode == "torch_ragged_xy":
                        Xs, _ys = batch
                        Xs = [x.to(device) for x in Xs]
                        out = trained_model.predict_labels(Xs)
                    elif mode == "pyg_batch":
                        G = batch.to(device)
                        out = trained_model.predict_labels(G)
                    else:
                        raise RuntimeError(f"Unknown lane mode: {mode}")

                    assert_label_output_by_lane(mode, batch, out, allow_noise_label=True)
                    if i >= 3:  # 4 batches
                        break
        except Exception as e:
            raise RuntimeError("Sanity-check predict_labels() failed") from e
        return

    if not dryrun:
        # Persist artefacts
        base = base_from_argv0()
        persist_artefacts(base, SCRIPT_DIR, trained_model, pre, spec)

        # Save plots
        plot_train_val(tr_loss, va_loss, f"{base} Loss", os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        plot_train_val(tr_acc, va_acc, f"{base} Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))
        
        # Write JSON Summary
        summary = {
            "epochs": n_epochs      if n_epochs else None,
            "train_loss": tr_loss   if tr_loss else None,
            "val_loss":   va_loss   if va_loss else None,
            "train_acc":  tr_acc    if tr_acc else None,
            "val_acc":    va_acc    if va_acc else None,
        }
        summary = to_python(summary)
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

