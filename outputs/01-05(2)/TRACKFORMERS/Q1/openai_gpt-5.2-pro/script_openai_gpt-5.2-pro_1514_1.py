
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

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import hdbscan
from sklearn.cluster import DBSCAN


class MyPreprocessor:
    def __init__(self):
        self.r_mean = 0.0
        self.r_std = 1.0
        self.z_mean = 0.0
        self.z_std = 1.0
        self.slope_mean = 0.0
        self.slope_std = 1.0
        self.layer_min = 0.0
        self.layer_max = 1.0

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": "ragged_xy",
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, 4] = (r, theta, z, layer_id)
        sum_r = 0.0
        sum_r2 = 0.0
        sum_z = 0.0
        sum_z2 = 0.0
        sum_s = 0.0
        sum_s2 = 0.0
        n = 0

        layer_min = float("inf")
        layer_max = float("-inf")

        eps = 1e-3
        for X in Xs:
            if not torch.is_tensor(X):
                X = torch.as_tensor(X)
            X = X.detach().cpu().float()
            r = X[:, 0]  # [N]
            z = X[:, 2]  # [N]
            layer = X[:, 3]  # [N]
            slope = z / (r.abs() + eps)  # [N]

            sum_r += float(r.sum().item())
            sum_r2 += float((r * r).sum().item())
            sum_z += float(z.sum().item())
            sum_z2 += float((z * z).sum().item())
            sum_s += float(slope.sum().item())
            sum_s2 += float((slope * slope).sum().item())
            n += int(r.numel())

            layer_min = min(layer_min, float(layer.min().item()))
            layer_max = max(layer_max, float(layer.max().item()))

        if n <= 0:
            return self

        r_mean = sum_r / n
        z_mean = sum_z / n
        s_mean = sum_s / n

        r_var = max(1e-8, sum_r2 / n - r_mean * r_mean)
        z_var = max(1e-8, sum_z2 / n - z_mean * z_mean)
        s_var = max(1e-8, sum_s2 / n - s_mean * s_mean)

        self.r_mean = float(r_mean)
        self.r_std = float(math.sqrt(r_var))
        self.z_mean = float(z_mean)
        self.z_std = float(math.sqrt(z_var))
        self.slope_mean = float(s_mean)
        self.slope_std = float(math.sqrt(s_var))
        self.layer_min = float(layer_min if math.isfinite(layer_min) else 0.0)
        self.layer_max = float(layer_max if math.isfinite(layer_max) else 1.0)
        if self.layer_max <= self.layer_min:
            self.layer_max = self.layer_min + 1.0

        return self

    def transform(self, X):
        # X: torch.FloatTensor [N_hits, 4] = (r, theta, z, layer_id)
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)
        X = X.float()

        r = X[:, 0]  # [N]
        theta = X[:, 1]  # [N]
        z = X[:, 2]  # [N]
        layer = X[:, 3]  # [N]

        eps = 1e-3
        slope = z / (r.abs() + eps)  # [N]
        sin_t = torch.sin(theta)  # [N]
        cos_t = torch.cos(theta)  # [N]

        r_n = (r - self.r_mean) / (self.r_std + 1e-8)  # [N]
        z_n = (z - self.z_mean) / (self.z_std + 1e-8)  # [N]
        slope_n = (slope - self.slope_mean) / (self.slope_std + 1e-8)  # [N]

        layer_n = (layer - self.layer_min) / (self.layer_max - self.layer_min + 1e-8)  # [N]
        layer_raw = layer  # [N] (float but integer-valued)

        # Output: [N, 7]
        out = torch.stack([r_n, z_n, slope_n, sin_t, cos_t, layer_n, layer_raw], dim=1)  # [N, 7]
        return out


def make_preprocessor():
    return MyPreprocessor()


class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()

        # Determine input dim from example
        if isinstance(example_batch_x, (tuple, list)) and len(example_batch_x) > 0:
            if isinstance(example_batch_x[0], (list, tuple)) and len(example_batch_x[0]) > 0 and torch.is_tensor(example_batch_x[0][0]):
                # Possibly (Xs, ys)
                ex = example_batch_x[0][0]
            elif torch.is_tensor(example_batch_x[0]):
                ex = example_batch_x[0]
            else:
                ex = None
        else:
            ex = None

        in_dim = int(ex.shape[1]) if (ex is not None and ex.ndim == 2) else 7
        if in_dim < 7:
            raise ValueError(f"Expected >=7 engineered features, got in_dim={in_dim}")

        self.cont_dim = 6  # use first 6 as continuous
        self.layer_col = 6  # last col is raw layer id float
        self.layer_vocab = 256
        self.layer_emb_dim = 8

        self.layer_emb = nn.Embedding(self.layer_vocab, self.layer_emb_dim)

        mlp_in = self.cont_dim + self.layer_emb_dim
        hid = 64
        out_dim = 12

        self.net = nn.Sequential(
            nn.Linear(mlp_in, hid),
            nn.GELU(),
            nn.LayerNorm(hid),
            nn.Dropout(0.10),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.LayerNorm(hid),
            nn.Dropout(0.10),
            nn.Linear(hid, out_dim),
        )

        self.out_dim = out_dim

        # Clustering hyperparams (tuned for linear-ish tracks)
        self.hdb_min_cluster = 4
        self.hdb_min_samples = 2
        self.db_eps = 0.35
        self.merge_thresh = 0.22
        self.reassign_k = 1

    def _embed_event(self, X):
        # X: [N, 7]
        cont = X[:, : self.cont_dim]  # [N, 6]
        layer_id = X[:, self.layer_col].round().clamp(0, self.layer_vocab - 1).long()  # [N]
        layer_vec = self.layer_emb(layer_id)  # [N, E]
        inp = torch.cat([cont, layer_vec], dim=1)  # [N, 6+E]
        emb = self.net(inp)  # [N, D]
        emb = F.normalize(emb, p=2, dim=1)  # [N, D]
        return emb

    def forward(self, batch_x):
        # batch_x: list of [N_i, 7]
        if torch.is_tensor(batch_x):
            # single event [N, 7]
            return self._embed_event(batch_x)

        if not isinstance(batch_x, list):
            raise TypeError(f"Expected list of tensors, got {type(batch_x)}")

        embs = []
        for X in batch_x:
            embs.append(self._embed_event(X))  # each [N_i, D]
        return embs

    @staticmethod
    def _union_find_merge(labels, centroids, merge_thresh):
        # labels: int64 [N], centroids: float64 [K, d]
        uniq = np.array([u for u in np.unique(labels) if u >= 0], dtype=np.int64)
        if uniq.size <= 1:
            return labels
        idx_map = {int(u): i for i, u in enumerate(uniq)}
        K = uniq.size
        parent = np.arange(K, dtype=np.int64)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Pairwise centroid distances
        for i in range(K):
            for j in range(i + 1, K):
                di = np.linalg.norm(centroids[i] - centroids[j])
                if di < merge_thresh:
                    union(i, j)

        # Remap labels by roots
        root_to_new = {}
        next_id = 0
        out = labels.copy()
        for u in uniq:
            r = find(idx_map[int(u)])
            if r not in root_to_new:
                root_to_new[r] = next_id
                next_id += 1

        for u in uniq:
            r = find(idx_map[int(u)])
            out[labels == u] = root_to_new[r]

        return out

    def _cluster_one(self, X_evt, emb_evt):
        # X_evt: torch [N, 7] preprocessed
        # emb_evt: torch [N, D] normalized
        N = int(X_evt.shape[0])
        if N == 0:
            return torch.empty((0,), dtype=torch.int64, device=X_evt.device)

        # Physics-informed features: slope_n, sin, cos
        phys = X_evt[:, 2:5]  # [N, 3]
        feat = torch.cat([phys * 1.0, emb_evt * 0.35], dim=1)  # [N, 3 + D]
        feat_np = feat.detach().float().cpu().numpy().astype(np.float32)

        labels = None
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.hdb_min_cluster,
                min_samples=self.hdb_min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                allow_single_cluster=False,
            )
            labels = clusterer.fit_predict(feat_np).astype(np.int64)  # [-1..K-1]
        except Exception:
            labels = None

        if labels is None or (np.unique(labels[labels >= 0]).size == 0):
            # Fallback: DBSCAN on phys-only
            phys_np = phys.detach().float().cpu().numpy().astype(np.float32)
            try:
                labels = DBSCAN(eps=self.db_eps, min_samples=self.hdb_min_cluster).fit_predict(phys_np).astype(np.int64)
            except Exception:
                labels = -np.ones((N,), dtype=np.int64)

        # Drop clusters smaller than 4 (safety)
        if np.any(labels >= 0):
            uniq, cnt = np.unique(labels[labels >= 0], return_counts=True)
            small = set(int(u) for u, c in zip(uniq, cnt) if int(c) < 4)
            if small:
                for u in small:
                    labels[labels == u] = -1

        # Compute centroids in feat space for reassignment / merging
        uniq = np.array([u for u in np.unique(labels) if u >= 0], dtype=np.int64)
        if uniq.size > 0:
            centroids = np.stack([feat_np[labels == u].mean(axis=0) for u in uniq], axis=0)  # [K, d]
            # Merge close centroids
            labels = self._union_find_merge(labels, centroids, self.merge_thresh)

        # Recompute centroids after merge for reassignment
        uniq = np.array([u for u in np.unique(labels) if u >= 0], dtype=np.int64)
        if uniq.size > 0:
            centroids = np.stack([feat_np[labels == u].mean(axis=0) for u in uniq], axis=0)  # [K, d]
            # Cluster radii from median intra distances
            radii = []
            for i, u in enumerate(uniq):
                pts = feat_np[labels == u]
                if pts.shape[0] <= 1:
                    radii.append(0.0)
                else:
                    d = np.linalg.norm(pts - centroids[i], axis=1)
                    med = float(np.median(d))
                    radii.append(2.0 * med + 0.06)
            radii = np.array(radii, dtype=np.float32)  # [K]

            # Reassign noise points if close enough
            noise_idx = np.where(labels < 0)[0]
            if noise_idx.size > 0:
                pts = feat_np[noise_idx]  # [M, d]
                # distances: [M, K]
                dists = np.linalg.norm(pts[:, None, :] - centroids[None, :, :], axis=2)
                nn = np.argmin(dists, axis=1)  # [M]
                best = dists[np.arange(dists.shape[0]), nn]
                # accept if within radius
                ok = best < radii[nn]
                if np.any(ok):
                    labels[noise_idx[ok]] = uniq[nn[ok]]

        # Final size filter and relabel consecutively
        uniq = np.array([u for u in np.unique(labels) if u >= 0], dtype=np.int64)
        if uniq.size > 0:
            uniq, cnt = np.unique(labels[labels >= 0], return_counts=True)
            small = set(int(u) for u, c in zip(uniq, cnt) if int(c) < 4)
            if small:
                for u in small:
                    labels[labels == u] = -1

        uniq = [int(u) for u in np.unique(labels) if u >= 0]
        remap = {u: i for i, u in enumerate(uniq)}
        for u in uniq:
            labels[labels == u] = remap[u]

        return torch.as_tensor(labels, dtype=torch.int64, device=X_evt.device)  # [N]

    def predict_labels(self, batch_x):
        if torch.is_tensor(batch_x):
            emb = self._embed_event(batch_x)  # [N, D]
            return self._cluster_one(batch_x, emb)  # [N]

        if not isinstance(batch_x, list):
            raise TypeError(f"Expected list of tensors, got {type(batch_x)}")

        self.eval()
        outs = []
        with torch.no_grad():
            for X in batch_x:
                emb = self._embed_event(X)  # [N, D]
                lab = self._cluster_one(X, emb)  # [N]
                outs.append(lab)
        return outs


def make_model(example_batch_x):
    return HitClassifier(example_batch_x)


def _discriminative_loss_and_proxy_acc(emb_list, y_list, delta_v=0.25, delta_d=0.70, w_var=1.0, w_dist=1.0, w_reg=0.001):
    # emb_list: list of [N_i, D]
    # y_list: list of [N_i]
    total_loss = 0.0
    total_acc_num = 0.0
    total_acc_den = 0.0
    n_events = 0

    for emb, y in zip(emb_list, y_list):
        # emb: [N, D], y: [N]
        y = y.long()
        mask = y > 0
        if mask.sum().item() < 4:
            continue

        emb_v = emb[mask]  # [Nv, D]
        y_v = y[mask]  # [Nv]

        tids = torch.unique(y_v)  # [K]
        K = int(tids.numel())
        if K <= 0:
            continue

        mus = []
        var_term = 0.0
        for t in tids:
            m = (y_v == t)
            e_t = emb_v[m]  # [n_t, D]
            if e_t.shape[0] == 0:
                continue
            mu = e_t.mean(dim=0)  # [D]
            mus.append(mu)
            d = torch.norm(e_t - mu[None, :], dim=1)  # [n_t]
            var_term = var_term + torch.mean(F.relu(d - delta_v) ** 2)

        if len(mus) == 0:
            continue
        mus = torch.stack(mus, dim=0)  # [K, D]
        var_term = var_term / mus.shape[0]

        dist_term = emb_v.new_tensor(0.0)
        if mus.shape[0] > 1:
            # pairwise distances [K, K]
            pd = torch.cdist(mus, mus, p=2)  # [K, K]
            iu = torch.triu_indices(pd.shape[0], pd.shape[1], offset=1, device=pd.device)
            d_ij = pd[iu[0], iu[1]]  # [K*(K-1)/2]
            dist_term = torch.mean(F.relu(2.0 * delta_d - d_ij) ** 2)

        reg_term = torch.mean(torch.norm(mus, dim=1))

        loss_evt = w_var * var_term + w_dist * dist_term + w_reg * reg_term
        total_loss = total_loss + loss_evt
        n_events += 1

        # Proxy centroid assignment accuracy (uses truth tids as centroids)
        with torch.no_grad():
            d = torch.cdist(emb_v, mus, p=2)  # [Nv, K]
            idx = torch.argmin(d, dim=1)  # [Nv]
            pred_tid = tids[idx]  # [Nv]
            total_acc_num += float((pred_tid == y_v).sum().item())
            total_acc_den += float(y_v.numel())

    if n_events == 0:
        return emb_list[0].new_tensor(0.0), 0.0

    total_loss = total_loss / n_events
    acc = float(total_acc_num / max(1.0, total_acc_den))
    return total_loss, acc


EPOCHS = 18


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=2e-4)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    best_state = None
    patience = 4
    bad = 0

    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    for ep in range(int(epochs)):
        model.train()
        tr_losses = []
        tr_accs = []

        for batch in train_loader:
            Xs, ys = batch
            Xs = [x.to(device, non_blocking=True) for x in Xs]
            ys = [y.to(device, non_blocking=True) for y in ys]

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                emb_list = model(Xs)  # list of [N_i, D]
                loss, acc = _discriminative_loss_and_proxy_acc(emb_list, ys)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            tr_losses.append(float(loss.detach().item()))
            tr_accs.append(float(acc))

        scheduler.step()

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("inf")
        tr_acc = float(np.mean(tr_accs)) if tr_accs else 0.0
        train_loss_hist.append(tr_loss)
        train_acc_hist.append(tr_acc)

        # Validation
        model.eval()
        va_losses = []
        va_accs = []
        with torch.no_grad():
            for batch in val_loader:
                Xs, ys = batch
                Xs = [x.to(device, non_blocking=True) for x in Xs]
                ys = [y.to(device, non_blocking=True) for y in ys]
                emb_list = model(Xs)
                loss, acc = _discriminative_loss_and_proxy_acc(emb_list, ys)
                va_losses.append(float(loss.detach().item()))
                va_accs.append(float(acc))

        va_loss = float(np.mean(va_losses)) if va_losses else float("inf")
        va_acc = float(np.mean(va_accs)) if va_accs else 0.0
        val_loss_hist.append(va_loss)
        val_acc_hist.append(va_acc)

        # Early stopping
        if va_loss + 1e-5 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    trained_model = model
    return trained_model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

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

