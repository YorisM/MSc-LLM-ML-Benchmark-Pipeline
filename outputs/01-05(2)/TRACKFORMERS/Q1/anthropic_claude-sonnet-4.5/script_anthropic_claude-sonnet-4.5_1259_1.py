
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import detect_and_assert_lane, assert_label_output_by_lane, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, build_trackformers_model

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

import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import hdbscan
import numpy as np

class TrackingDataset(Dataset):
    def __init__(self, events, pre, train=True, **kwargs):
        self.events = events
        self.pre = pre
        self.train = train
        self.k_neighbors = 8

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        X, y = split_X_y(self.events[idx])

        if self.pre is not None:
            X = self.pre.transform(X)

        edge_index = self._build_knn_edges(X)

        data = Data(x=X.float(), y=y.long(), edge_index=edge_index.long())
        return data

    def _build_knn_edges(self, X):
        N = X.shape[0]

        if N <= 1:
            return torch.zeros((2, 0), dtype=torch.long)

        k = min(self.k_neighbors, N - 1)

        coords = X[:, :3]
        dist = torch.cdist(coords, coords)

        _, indices = torch.topk(dist, k + 1, largest=False, dim=1)
        indices = indices[:, 1:]

        source = torch.arange(N).unsqueeze(1).expand(-1, k).reshape(-1)
        target = indices.reshape(-1)

        edge_index = torch.stack([source, target], dim=0)

        return edge_index

class MyPreprocessor:
    def __init__(self):
        self.r_mean = 0.0
        self.r_std = 1.0
        self.z_mean = 0.0
        self.z_std = 1.0
        self.layer_mean = 0.0
        self.layer_std = 1.0

    def make_loader_cfg(self):
        return {
            "dataset_builder": "llm_script:TrackingDataset",
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
        all_r = torch.cat([X[:, 0] for X in Xs])
        all_z = torch.cat([X[:, 2] for X in Xs])
        all_layer = torch.cat([X[:, 3] for X in Xs])

        self.r_mean = all_r.mean().item()
        self.r_std = all_r.std().item() + 1e-6
        self.z_mean = all_z.mean().item()
        self.z_std = all_z.std().item() + 1e-6
        self.layer_mean = all_layer.mean().item()
        self.layer_std = all_layer.std().item() + 1e-6

        return self

    def transform(self, X):
        r = X[:, 0]
        theta = X[:, 1]
        z = X[:, 2]
        layer = X[:, 3]

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        r_norm = (r - self.r_mean) / self.r_std
        z_norm = (z - self.z_mean) / self.z_std
        layer_norm = (layer - self.layer_mean) / self.layer_std

        features = torch.stack([x, y, z, r_norm, z_norm, layer_norm], dim=1)

        return features

def make_preprocessor():
    return MyPreprocessor()

class HitClassifier(nn.Module):
    def __init__(self, example_batch):
        super().__init__()

        in_dim = example_batch.x.shape[1]
        hidden_dim = 128
        emb_dim = 32

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, emb_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)

        return x

    def predict_labels(self, batch):
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(batch).cpu().numpy()
            batch_idx = batch.batch.cpu().numpy()

            labels_list = []

            for i in range(int(batch_idx.max()) + 1):
                mask = (batch_idx == i)
                event_emb = embeddings[mask]

                if event_emb.shape[0] < 4:
                    event_labels = np.full(event_emb.shape[0], -1, dtype=np.int32)
                else:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=4,
                        min_samples=2,
                        cluster_selection_epsilon=0.0
                    )
                    event_labels = clusterer.fit_predict(event_emb)

                labels_list.append(event_labels)

            all_labels = np.concatenate(labels_list)
            return torch.from_numpy(all_labels).long()

def make_model(example_batch):
    return HitClassifier(example_batch)

def train_model(model, train_loader, val_loader, epochs):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    def contrastive_loss(embeddings, labels, batch_idx):
        total_loss = 0.0
        margin = 1.0
        n_events = int(batch_idx.max()) + 1

        for i in range(n_events):
            mask = (batch_idx == i)
            emb = embeddings[mask]
            lbl = labels[mask]

            if emb.shape[0] <= 1:
                continue

            dist = torch.cdist(emb, emb)

            valid = (lbl > 0)
            same_track = (lbl.unsqueeze(1) == lbl.unsqueeze(0)) & valid.unsqueeze(1) & valid.unsqueeze(0)
            same_track = same_track & ~torch.eye(emb.shape[0], dtype=torch.bool, device=emb.device)

            diff_track = (lbl.unsqueeze(1) != lbl.unsqueeze(0)) & valid.unsqueeze(1) & valid.unsqueeze(0)

            if same_track.sum() > 0:
                pos_loss = (dist * same_track.float()).sum() / same_track.float().sum()
            else:
                pos_loss = 0.0

            if diff_track.sum() > 0:
                neg_loss = (F.relu(margin - dist) * diff_track.float()).sum() / diff_track.float().sum()
            else:
                neg_loss = 0.0

            total_loss += pos_loss + neg_loss

        return total_loss / n_events if n_events > 0 else torch.tensor(0.0, device=embeddings.device)

    best_val_loss = float('inf')
    patience = 0
    max_patience = 5

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            emb = model(batch)
            loss = contrastive_loss(emb, batch.y, batch.batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                emb = model(batch)
                loss = contrastive_loss(emb, batch.y, batch.batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        train_accs.append(0.0)
        val_accs.append(0.0)

        print(f"Epoch {epoch+1}/{epochs}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

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
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

