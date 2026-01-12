
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

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": "ragged_xy",
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def _compute_features(self, X):
        # X: Tensor [N_hits, 4] with columns r, theta, z, layer_id
        if not torch.is_tensor(X):
            X_t = torch.as_tensor(X, dtype=torch.float32)
        else:
            X_t = X.float()
        r = X_t[:, 0]  # [N]
        theta = X_t[:, 1]  # [N]
        z = X_t[:, 2]  # [N]
        layer = X_t[:, 3]  # [N]
        x = r * torch.cos(theta)  # [N]
        y = r * torch.sin(theta)  # [N]
        feats = torch.stack([x, y, z, r, layer], dim=1)  # [N,5]
        return feats

    def fit(self, Xs):
        total = None
        total_sq = None
        count = 0
        for X in Xs:
            feats = self._compute_features(X)  # [N_i,5]
            if total is None:
                total = torch.zeros(feats.shape[1], dtype=torch.float64)
                total_sq = torch.zeros(feats.shape[1], dtype=torch.float64)
            total += feats.double().sum(dim=0)
            total_sq += (feats.double() * feats.double()).sum(dim=0)
            count += feats.shape[0]
        mean = total / float(count)
        var = total_sq / float(count) - mean * mean
        std = torch.sqrt(var + 1e-6)
        self.mean = mean.float()
        self.std = std.float()
        return self

    def transform(self, X):
        feats = self._compute_features(X).float()  # [N,5]
        if self.mean is not None and self.std is not None:
            feats = (feats - self.mean) / self.std
        return feats

def make_preprocessor():
    return MyPreprocessor()

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        if isinstance(example_batch_x, (list, tuple)):
            sample_x = example_batch_x[0]
            in_dim = sample_x.shape[1]
        else:
            in_dim = example_batch_x.shape[1]
        self.emb_dim = 32
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.emb_dim),
            nn.ReLU()
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.emb_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.k_neighbors = 10
        self.edge_threshold = 0.6
        self.min_cluster_size = 4

    def encode(self, X):
        # X: [N,F]
        return self.encoder(X)  # [N,emb_dim]

    def edge_logits(self, hi, hj):
        # hi,hj: [E,emb_dim]
        feats = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)  # [E, emb_dim*4]
        logits = self.edge_mlp(feats).squeeze(-1)  # [E]
        return logits

    def forward(self, batch_x):
        if isinstance(batch_x, (list, tuple)):
            return [self.encode(x) for x in batch_x]
        else:
            return self.encode(batch_x)

    def _sample_pairs(self, y_np, max_pairs=512):
        # y_np: numpy array [N]
        N = len(y_np)
        if N < 2:
            return None
        idx_all = np.arange(N)
        pos_pairs = []
        # generate all positive pairs
        tracks = [t for t in np.unique(y_np) if t > 0]
        for t in tracks:
            hits = np.where(y_np == t)[0]
            if len(hits) < 2:
                continue
            for i in range(len(hits)):
                for j in range(i + 1, len(hits)):
                    pos_pairs.append((hits[i], hits[j]))
        if len(pos_pairs) > max_pairs // 2:
            pos_pairs = list(np.random.choice(len(pos_pairs), size=max_pairs // 2, replace=False))
            pos_pairs = [pos_pairs[i] if isinstance(pos_pairs[0], tuple) else pos_pairs for i in range(len(pos_pairs))]
        neg_pairs = []
        rng = np.random.default_rng()
        while len(neg_pairs) < max_pairs - len(pos_pairs):
            i, j = rng.choice(idx_all, size=2, replace=False)
            if y_np[i] != y_np[j]:
                neg_pairs.append((i, j))
        pairs = []
        labels = []
        for p in pos_pairs:
            pairs.append(p)
            labels.append(1)
        for p in neg_pairs:
            pairs.append(p)
            labels.append(0)
        if len(pairs) == 0:
            return None
        pairs = np.array(pairs, dtype=np.int64)
        labels = np.array(labels, dtype=np.float32)
        return pairs[:, 0], pairs[:, 1], labels

    def compute_pair_loss(self, X, y, max_pairs=512):
        # X: [N,F], y: [N]
        y_np = y.cpu().numpy()
        sampled = self._sample_pairs(y_np, max_pairs=max_pairs)
        if sampled is None:
            return None, 0, 0
        idx1_np, idx2_np, lbl_np = sampled
        idx1 = torch.from_numpy(idx1_np).to(X.device)
        idx2 = torch.from_numpy(idx2_np).to(X.device)
        labels = torch.from_numpy(lbl_np).to(X.device)
        emb = self.encode(X)  # [N,emb_dim]
        hi = emb[idx1]  # [E,emb_dim]
        hj = emb[idx2]  # [E,emb_dim]
        logits = self.edge_logits(hi, hj)  # [E]
        pos_count = labels.sum().item()
        neg_count = labels.shape[0] - pos_count
        if pos_count > 0:
            pos_weight = torch.tensor(neg_count / (pos_count + 1e-6), device=X.device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = (preds == labels).sum().item()
        return loss, labels.shape[0], correct

    def predict_labels(self, batch_x):
        if isinstance(batch_x, (list, tuple)):
            outputs = []
            for X in batch_x:
                outputs.append(self._predict_single(X))
            return outputs
        else:
            return self._predict_single(batch_x)

    def _predict_single(self, X):
        # X: [N,F]
        with torch.no_grad():
            N = X.shape[0]
            if N == 0:
                return torch.zeros((0,), dtype=torch.long, device=X.device)
            emb = self.encode(X)  # [N,emb_dim]
            coords = X[:, :3]  # [N,3]
            # pairwise distances
            dist = torch.cdist(coords, coords, p=2)  # [N,N]
            k = min(self.k_neighbors, max(N - 1, 1))
            knn_idx = dist.topk(k=k + 1, largest=False).indices[:, 1:]  # [N,k]
            edge_set = set()
            for i in range(N):
                for j in knn_idx[i].tolist():
                    if i == j:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    edge_set.add((a, b))
            if len(edge_set) == 0:
                return torch.full((N,), -1, dtype=torch.long, device=X.device)
            edges = torch.tensor(list(edge_set), device=X.device, dtype=torch.long)  # [E,2]
            hi = emb[edges[:, 0]]  # [E,emb_dim]
            hj = emb[edges[:, 1]]  # [E,emb_dim]
            logits = self.edge_logits(hi, hj)  # [E]
            probs = torch.sigmoid(logits)
            mask = probs > self.edge_threshold
            sel_edges = edges[mask]
            parent = list(range(N))

            def find(u):
                while parent[u] != u:
                    parent[u] = parent[parent[u]]
                    u = parent[u]
                return u

            def union(u, v):
                ru, rv = find(u), find(v)
                if ru == rv:
                    return
                parent[rv] = ru

            for e in sel_edges.tolist():
                union(e[0], e[1])
            roots = [find(i) for i in range(N)]
            # count cluster sizes
            size_dict = {}
            for r in roots:
                size_dict[r] = size_dict.get(r, 0) + 1
            root_to_cluster = {}
            cluster_id = 0
            labels = torch.full((N,), -1, dtype=torch.long, device=X.device)
            for i, r in enumerate(roots):
                if size_dict[r] >= self.min_cluster_size:
                    if r not in root_to_cluster:
                        root_to_cluster[r] = cluster_id
                        cluster_id += 1
                    labels[i] = root_to_cluster[r]
                else:
                    labels[i] = -1
            return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

EPOCHS = 5
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_pairs = 0
        total_correct = 0
        for Xs, ys in train_loader:
            optimizer.zero_grad()
            batch_loss_sum = 0.0
            batch_pairs = 0
            batch_correct = 0
            for X, y in zip(Xs, ys):
                X = X.to(device)
                y = y.to(device)
                res = model.compute_pair_loss(X, y, max_pairs=512)
                if res is None:
                    continue
                loss_evt, n_pairs, correct = res
                batch_loss_sum += loss_evt * n_pairs
                batch_pairs += n_pairs
                batch_correct += correct
            if batch_pairs == 0:
                continue
            loss = batch_loss_sum / batch_pairs
            loss.backward()
            optimizer.step()
            total_loss += batch_loss_sum.item()
            total_pairs += batch_pairs
            total_correct += batch_correct
        if total_pairs > 0:
            epoch_loss = total_loss / total_pairs
            epoch_acc = total_correct / total_pairs
        else:
            epoch_loss = None
            epoch_acc = None
        train_loss_hist.append(epoch_loss)
        train_acc_hist.append(epoch_acc)
        # validation
        model.eval()
        val_loss_sum = 0.0
        val_pairs = 0
        val_correct = 0
        with torch.no_grad():
            for Xs, ys in val_loader:
                for X, y in zip(Xs, ys):
                    X = X.to(device)
                    y = y.to(device)
                    res = model.compute_pair_loss(X, y, max_pairs=512)
                    if res is None:
                        continue
                    loss_evt, n_pairs, correct = res
                    val_loss_sum += loss_evt.item() * n_pairs
                    val_pairs += n_pairs
                    val_correct += correct
        if val_pairs > 0:
            v_loss = val_loss_sum / val_pairs
            v_acc = val_correct / val_pairs
        else:
            v_loss = None
            v_acc = None
        val_loss_hist.append(v_loss)
        val_acc_hist.append(v_acc)
    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

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

