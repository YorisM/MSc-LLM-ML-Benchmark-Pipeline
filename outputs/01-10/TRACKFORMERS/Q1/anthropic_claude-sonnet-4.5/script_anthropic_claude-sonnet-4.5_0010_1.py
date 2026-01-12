
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

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import hdbscan
import numpy as np

class MyPreprocessor:
    def __init__(self):
        self.r_mean = 0.0
        self.r_std = 1.0
        self.z_mean = 0.0
        self.z_std = 1.0
        self.layer_max = 1.0

    def make_loader_cfg(self):
        return {
            "dataset_builder": "llm_script:GraphDataset",
            "dataset_kwargs": {"k": 10},
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

        self.r_mean = float(all_r.mean())
        self.r_std = float(all_r.std()) + 1e-6
        self.z_mean = float(all_z.mean())
        self.z_std = float(all_z.std()) + 1e-6
        self.layer_max = float(all_layer.max()) + 1e-6
        return self

    def transform(self, X):
        X = X.clone().float()
        X[:, 0] = (X[:, 0] - self.r_mean) / self.r_std
        X[:, 2] = (X[:, 2] - self.z_mean) / self.z_std
        X[:, 3] = X[:, 3] / self.layer_max
        return X

def make_preprocessor():
    return MyPreprocessor()

class GraphDataset(Dataset):
    def __init__(self, events, pre, train=True, k=10):
        self.events = events
        self.pre = pre
        self.train = train
        self.k = k

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        X, y = split_X_y(self.events[idx])

        if self.pre is not None:
            X = self.pre.transform(X)

        edge_index = self._build_knn_graph(X, self.k)
        return Data(x=X.float(), y=y.long(), edge_index=edge_index)

    def _build_knn_graph(self, X, k):
        n = X.shape[0]

        r = X[:, 0]
        theta = X[:, 1]
        z = X[:, 2]

        x_cart = r * torch.cos(theta)
        y_cart = r * torch.sin(theta)
        coords = torch.stack([x_cart, y_cart, z], dim=1)

        dist = torch.cdist(coords, coords)

        k_actual = min(k + 1, n)
        _, indices = torch.topk(dist, k=k_actual, largest=False, dim=1)

        edges = []
        for i in range(n):
            for j in indices[i]:
                j_int = int(j.item()) if hasattr(j, 'item') else int(j)
                if i != j_int:
                    edges.append([i, j_int])

        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).T

        return edge_index

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()

        in_dim = example_batch_x.x.shape[1]

        self.conv1 = GCNConv(in_dim, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv3(x, edge_index))

        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)

        return x

    def predict_labels(self, batch):
        embeddings = self.forward(batch)
        embeddings_np = embeddings.cpu().numpy()

        batch_idx = batch.batch.cpu().numpy()
        n_graphs = int(batch.batch.max().item()) + 1

        all_labels = []
        for i in range(n_graphs):
            mask = (batch_idx == i)
            emb = embeddings_np[mask]

            if len(emb) < 4:
                labels = np.full(len(emb), -1, dtype=np.int64)
            else:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=4,
                    min_samples=1,
                    cluster_selection_epsilon=0.0,
                    metric='euclidean'
                )
                labels = clusterer.fit_predict(emb)

            all_labels.append(labels)

        all_labels = np.concatenate(all_labels)
        return torch.tensor(all_labels, dtype=torch.long, device=batch.x.device)

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            embeddings = model(batch)
            labels = batch.y

            dist = torch.cdist(embeddings, embeddings, p=2)

            labels_eq = (labels.unsqueeze(0) == labels.unsqueeze(1))
            non_noise = (labels > 0)

            pos_mask = labels_eq & non_noise.unsqueeze(0) & non_noise.unsqueeze(1)
            pos_mask = pos_mask & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)

            neg_mask = (~labels_eq) & non_noise.unsqueeze(0) & non_noise.unsqueeze(1)

            loss = torch.tensor(0.0, device=embeddings.device)

            if pos_mask.sum() > 0:
                pos_loss = (dist[pos_mask] ** 2).mean()
                loss = loss + pos_loss

            if neg_mask.sum() > 0:
                margin = 1.0
                neg_loss = F.relu(margin - dist[neg_mask]).mean()
                loss = loss + neg_loss

            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

        if n_batches > 0:
            avg_train_loss = epoch_loss / n_batches
        else:
            avg_train_loss = 0.0
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                embeddings = model(batch)
                labels = batch.y

                dist = torch.cdist(embeddings, embeddings, p=2)

                labels_eq = (labels.unsqueeze(0) == labels.unsqueeze(1))
                non_noise = (labels > 0)

                pos_mask = labels_eq & non_noise.unsqueeze(0) & non_noise.unsqueeze(1)
                pos_mask = pos_mask & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)

                neg_mask = (~labels_eq) & non_noise.unsqueeze(0) & non_noise.unsqueeze(1)

                loss = 0.0

                if pos_mask.sum() > 0:
                    loss = loss + (dist[pos_mask] ** 2).mean().item()

                if neg_mask.sum() > 0:
                    loss = loss + F.relu(1.0 - dist[neg_mask]).mean().item()

                val_loss += loss
                n_val_batches += 1

        if n_val_batches > 0:
            avg_val_loss = val_loss / n_val_batches
        else:
            avg_val_loss = 0.0
        val_losses.append(avg_val_loss)

        scheduler.step()

        train_accs.append(0.0)
        val_accs.append(0.0)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

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

