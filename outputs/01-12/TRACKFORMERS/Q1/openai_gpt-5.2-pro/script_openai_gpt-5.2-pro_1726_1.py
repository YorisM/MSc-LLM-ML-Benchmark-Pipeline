
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

import hdbscan
from sklearn.cluster import DBSCAN


class MyPreprocessor:
    def __init__(self):
        # global stats for standardization
        self.eps = 1e-6
        self.r_mean = 0.0
        self.r_std = 1.0
        self.z_mean = 0.0
        self.z_std = 1.0
        self.layer_mean = 0.0
        self.layer_std = 1.0
        self.zr_mean = 0.0
        self.zr_std = 1.0
        self.logr_mean = 0.0
        self.logr_std = 1.0
        self.absz_mean = 0.0
        self.absz_std = 1.0

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
        # Xs: list of per-event X, each [N_hits_i, 4]
        n = 0
        sum_r = 0.0
        sumsq_r = 0.0
        sum_z = 0.0
        sumsq_z = 0.0
        sum_layer = 0.0
        sumsq_layer = 0.0
        sum_zr = 0.0
        sumsq_zr = 0.0
        sum_logr = 0.0
        sumsq_logr = 0.0
        sum_absz = 0.0
        sumsq_absz = 0.0

        for X in Xs:
            if not torch.is_tensor(X):
                X = torch.as_tensor(X)
            X = X.detach().cpu().to(torch.float64)
            r = X[:, 0]  # [N]
            z = X[:, 2]  # [N]
            layer = X[:, 3]  # [N]

            zr = z / (r.abs() + 1e-3)  # [N]
            logr = torch.log1p(r.clamp_min(0.0))  # [N]
            absz = z.abs()  # [N]

            nn = int(r.numel())
            if nn == 0:
                continue

            n += nn
            sum_r += float(r.sum().item())
            sumsq_r += float((r * r).sum().item())
            sum_z += float(z.sum().item())
            sumsq_z += float((z * z).sum().item())
            sum_layer += float(layer.sum().item())
            sumsq_layer += float((layer * layer).sum().item())
            sum_zr += float(zr.sum().item())
            sumsq_zr += float((zr * zr).sum().item())
            sum_logr += float(logr.sum().item())
            sumsq_logr += float((logr * logr).sum().item())
            sum_absz += float(absz.sum().item())
            sumsq_absz += float((absz * absz).sum().item())

        if n <= 1:
            return self

        def _mean_std(s, ss):
            mean = s / n
            var = max(0.0, ss / n - mean * mean)
            std = math.sqrt(var) + 1e-6
            return mean, std

        self.r_mean, self.r_std = _mean_std(sum_r, sumsq_r)
        self.z_mean, self.z_std = _mean_std(sum_z, sumsq_z)
        self.layer_mean, self.layer_std = _mean_std(sum_layer, sumsq_layer)
        self.zr_mean, self.zr_std = _mean_std(sum_zr, sumsq_zr)
        self.logr_mean, self.logr_std = _mean_std(sum_logr, sumsq_logr)
        self.absz_mean, self.absz_std = _mean_std(sum_absz, sumsq_absz)
        return self

    def transform(self, X):
        # X: one event tensor [N_hits, 4]
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)
        X = X.to(torch.float32)

        r = X[:, 0]  # [N]
        theta = X[:, 1]  # [N]
        z = X[:, 2]  # [N]
        layer = X[:, 3]  # [N]

        # engineered
        sin_t = torch.sin(theta)  # [N]
        cos_t = torch.cos(theta)  # [N]
        zr = z / (r.abs() + 1e-3)  # [N]
        logr = torch.log1p(r.clamp_min(0.0))  # [N]
        absz = z.abs()  # [N]

        # standardize (keep sin/cos raw)
        r_s = (r - self.r_mean) / self.r_std  # [N]
        z_s = (z - self.z_mean) / self.z_std  # [N]
        layer_s = (layer - self.layer_mean) / self.layer_std  # [N]
        zr_s = (zr - self.zr_mean) / self.zr_std  # [N]
        logr_s = (logr - self.logr_mean) / self.logr_std  # [N]
        absz_s = (absz - self.absz_mean) / self.absz_std  # [N]

        # Output features: [N, 9]
        # 0 r_s, 1 z_s, 2 layer_s, 3 sin_t, 4 cos_t, 5 zr_s, 6 logr_s, 7 absz_s, 8 layer_raw
        out = torch.stack(
            [r_s, z_s, layer_s, sin_t, cos_t, zr_s, logr_s, absz_s, layer],
            dim=1,
        ).to(torch.float32)  # [N, 9]
        return out


def make_preprocessor():
    return MyPreprocessor()


def _supcon_loss_event(emb, y, temperature=0.12, max_hits=256):
    # emb: [N, D], y: [N]
    # ignore y<=0 (noise)
    device = emb.device
    y = y.to(device)
    valid = y > 0
    if valid.sum().item() < 2:
        return emb.new_tensor(0.0)

    idx = torch.nonzero(valid, as_tuple=False).squeeze(1)  # [Nv]
    if idx.numel() > max_hits:
        perm = torch.randperm(idx.numel(), device=device)[:max_hits]
        idx = idx[perm]

    e = emb[idx]  # [Nv, D]
    t = y[idx]  # [Nv]

    # Must have at least one positive for some anchors
    # Normalize
    e = F.normalize(e, p=2, dim=1)  # [Nv, D]
    # Similarities
    sim = (e @ e.t()) / temperature  # [Nv, Nv]
    # Mask out self
    Nv = sim.shape[0]
    sim = sim - torch.max(sim, dim=1, keepdim=True).values  # stable
    logits = sim  # [Nv, Nv]

    # Exclude self from denominators
    self_mask = torch.eye(Nv, device=device, dtype=torch.bool)
    # Positives: same label, not self
    pos_mask = (t.view(-1, 1) == t.view(1, -1)) & (~self_mask)  # [Nv, Nv]

    # Denominator over all except self
    exp_logits = torch.exp(logits) * (~self_mask).to(logits.dtype)  # [Nv, Nv]
    denom = exp_logits.sum(dim=1) + 1e-12  # [Nv]

    # Numerator over positives
    num = (exp_logits * pos_mask.to(logits.dtype)).sum(dim=1)  # [Nv]
    has_pos = num > 0
    if has_pos.sum().item() == 0:
        return emb.new_tensor(0.0)

    loss = -torch.log((num[has_pos] + 1e-12) / denom[has_pos])  # [Npos]
    return loss.mean()


def _nn1_accuracy_event(emb, y, max_hits=192):
    # 1-NN label agreement, ignoring noise (y<=0)
    device = emb.device
    y = y.to(device)
    valid = y > 0
    if valid.sum().item() < 2:
        return None
    idx = torch.nonzero(valid, as_tuple=False).squeeze(1)  # [Nv]
    if idx.numel() > max_hits:
        perm = torch.randperm(idx.numel(), device=device)[:max_hits]
        idx = idx[perm]
    e = F.normalize(emb[idx], p=2, dim=1)  # [Nv, D]
    t = y[idx]  # [Nv]
    # cosine distance via dot-product
    sim = e @ e.t()  # [Nv, Nv]
    Nv = sim.shape[0]
    sim.fill_diagonal_(-1e9)
    nn_idx = sim.argmax(dim=1)  # [Nv]
    acc = (t[nn_idx] == t).float().mean()
    return acc


class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        if isinstance(example_batch_x, (list, tuple)):
            # ragged lane: example_batch_x is list[Tensor] or (Xs, ys) handled by harness; we get Xs
            if isinstance(example_batch_x, (tuple, list)) and len(example_batch_x) > 0 and torch.is_tensor(example_batch_x[0]):
                F_in = int(example_batch_x[0].shape[1])
            else:
                F_in = 9
        elif torch.is_tensor(example_batch_x):
            F_in = int(example_batch_x.shape[1])
        else:
            F_in = 9

        self.F_in = F_in
        self.emb_dim = 4
        self.layer_raw_idx = F_in - 1  # last feature is raw layer_id float

        # Physics-informed base projection (sin, cos, z/r, r)
        self.base = nn.Linear(F_in, self.emb_dim, bias=False)
        nn.init.zeros_(self.base.weight)
        # indices in preprocessor output:
        # 0 r_s, 1 z_s, 2 layer_s, 3 sin_t, 4 cos_t, 5 zr_s, 6 logr_s, 7 absz_s, 8 layer_raw
        with torch.no_grad():
            if F_in >= 6:
                self.base.weight[0, 3] = 1.0  # sin(theta)
                self.base.weight[1, 4] = 1.0  # cos(theta)
                self.base.weight[2, 5] = 1.0  # z/r slope
                self.base.weight[3, 0] = 0.5  # weak r dependence to help density

        self.mlp = nn.Sequential(
            nn.Linear(F_in, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, self.emb_dim),
        )

        self.input_noise = 0.015  # only in training
        self.temperature = 0.12

        # clustering hyperparams
        self.hdb_min_cluster_size = 4
        self.hdb_min_samples = 2
        self.assign_thr = 0.23  # conservative attachment for noise points

    def forward(self, batch_x):
        # batch_x: list[Tensor [N_i, F]] OR Tensor [N, F]
        if isinstance(batch_x, list):
            lengths = [int(x.shape[0]) for x in batch_x]
            if sum(lengths) == 0:
                return [x.new_zeros((0, self.emb_dim)) for x in batch_x]
            xcat = torch.cat(batch_x, dim=0)  # [sumN, F]
            if self.training and self.input_noise > 0:
                xcat = xcat + self.input_noise * torch.randn_like(xcat)  # [sumN, F]
            emb = self.base(xcat) + self.mlp(xcat)  # [sumN, D]
            emb = F.normalize(emb, p=2, dim=1)  # [sumN, D]
            outs = list(torch.split(emb, lengths, dim=0))  # list[[N_i, D]]
            return outs
        else:
            x = batch_x
            if self.training and self.input_noise > 0:
                x = x + self.input_noise * torch.randn_like(x)
            emb = self.base(x) + self.mlp(x)  # [N, D]
            emb = F.normalize(emb, p=2, dim=1)  # [N, D]
            return emb

    def _cluster_event(self, emb, x_feats):
        # emb: torch.FloatTensor [N, D] (normalized), x_feats: [N, F]
        N = int(emb.shape[0])
        if N < self.hdb_min_cluster_size:
            return np.full(N, -1, dtype=np.int64)

        emb_np = emb.detach().cpu().numpy().astype(np.float32)  # [N, D]
        layer_raw = x_feats[:, self.layer_raw_idx].detach().cpu().numpy()
        layer_int = np.rint(layer_raw).astype(np.int64)  # [N]

        labels = None
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.hdb_min_cluster_size,
                min_samples=self.hdb_min_samples,
                metric="euclidean",
                cluster_selection_method="leaf",
                allow_single_cluster=False,
            )
            labels = clusterer.fit_predict(emb_np).astype(np.int64)  # [N]
        except Exception:
            labels = np.full(N, -1, dtype=np.int64)

        # If everything is noise, try a small-eps DBSCAN fallback
        if np.all(labels < 0):
            try:
                db = DBSCAN(eps=0.18, min_samples=self.hdb_min_cluster_size, metric="euclidean")
                labels = db.fit_predict(emb_np).astype(np.int64)
            except Exception:
                labels = np.full(N, -1, dtype=np.int64)

        # Postprocess: split clusters with duplicate layers (merging indicator)
        next_id = (labels[labels >= 0].max() + 1) if np.any(labels >= 0) else 0
        uniq_clusters = [c for c in np.unique(labels) if c >= 0]
        for c in uniq_clusters:
            idx = np.where(labels == c)[0]
            if idx.size < 8:
                continue
            li = layer_int[idx]
            _, counts = np.unique(li, return_counts=True)
            if counts.max(initial=0) <= 1:
                continue
            # Re-cluster inside this potentially-merged cluster
            sub_emb = emb_np[idx]
            try:
                sub = hdbscan.HDBSCAN(
                    min_cluster_size=self.hdb_min_cluster_size,
                    min_samples=1,
                    metric="euclidean",
                    cluster_selection_method="leaf",
                    allow_single_cluster=False,
                ).fit_predict(sub_emb).astype(np.int64)
            except Exception:
                continue
            sub_uniq = [sc for sc in np.unique(sub) if sc >= 0]
            if len(sub_uniq) <= 1:
                continue
            # Replace: assign new ids; drop sub-noise to -1 to avoid impurity
            labels[idx] = -1
            for sc in sub_uniq:
                sc_idx = idx[sub == sc]
                if sc_idx.size >= self.hdb_min_cluster_size:
                    labels[sc_idx] = next_id
                    next_id += 1

        # Remove clusters <4
        for c in [c for c in np.unique(labels) if c >= 0]:
            if np.sum(labels == c) < self.hdb_min_cluster_size:
                labels[labels == c] = -1

        # Attach some noise points to nearest cluster center, if consistent w/ layer uniqueness
        clusters = [c for c in np.unique(labels) if c >= 0]
        if len(clusters) > 0:
            centers = []
            used_layers = []
            for c in clusters:
                idx = np.where(labels == c)[0]
                centers.append(emb_np[idx].mean(axis=0))
                used_layers.append(set(layer_int[idx].tolist()))
            centers = np.stack(centers, axis=0).astype(np.float32)  # [K, D]

            noise_idx = np.where(labels < 0)[0]
            if noise_idx.size > 0:
                # distances [M, K]
                diff = emb_np[noise_idx, None, :] - centers[None, :, :]
                d2 = np.sum(diff * diff, axis=2)
                nn = np.argmin(d2, axis=1)
                dd = np.sqrt(np.min(d2, axis=1) + 1e-12)
                for m, j in enumerate(noise_idx):
                    if dd[m] > self.assign_thr:
                        continue
                    k = int(nn[m])
                    lyr = int(layer_int[j])
                    if lyr in used_layers[k]:
                        continue
                    labels[j] = int(clusters[k])
                    used_layers[k].add(lyr)

        # Final cleanup: remove any cluster <4 again after attachments
        for c in [c for c in np.unique(labels) if c >= 0]:
            if np.sum(labels == c) < self.hdb_min_cluster_size:
                labels[labels == c] = -1

        return labels.astype(np.int64)

    def predict_labels(self, batch_x):
        # ragged lane only: batch_x is list[Tensor [N_i, F]]
        if not isinstance(batch_x, list):
            # tensor case (not expected in this challenge lane)
            emb = self.forward(batch_x)
            N = int(emb.shape[0])
            return torch.full((N,), -1, dtype=torch.long, device=batch_x.device)

        self.eval()
        with torch.no_grad():
            emb_list = self.forward(batch_x)  # list[[N_i, D]]

        outs = []
        for x, e in zip(batch_x, emb_list):
            dev = x.device
            labels_np = self._cluster_event(e, x)  # [N] np.int64
            outs.append(torch.from_numpy(labels_np).to(device=dev, dtype=torch.long))  # [N]
        return outs


def make_model(example_batch_x):
    return HitClassifier(example_batch_x)


EPOCHS = 6


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    best_val = float("inf")
    best_state = None
    patience = 3
    bad = 0

    for epoch in range(int(epochs)):
        model.train()
        tr_losses = []
        tr_accs = []

        for Xs, ys in train_loader:
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            emb_list = model(Xs)  # list[[N_i, D]]
            loss = None
            for e, y in zip(emb_list, ys):
                l = _supcon_loss_event(e, y, temperature=model.temperature, max_hits=256)
                loss = l if loss is None else (loss + l)
            if loss is None:
                continue
            loss = loss / max(1, len(emb_list))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_losses.append(float(loss.detach().cpu().item()))

            # lightweight monitoring: 1-NN on a few events only
            if len(tr_accs) < 8:
                with torch.no_grad():
                    acc_vals = []
                    for e, y in zip(emb_list[:2], ys[:2]):
                        a = _nn1_accuracy_event(e, y, max_hits=160)
                        if a is not None:
                            acc_vals.append(float(a.detach().cpu().item()))
                    if len(acc_vals) > 0:
                        tr_accs.append(float(np.mean(acc_vals)))

        sched.step()

        train_loss = float(np.mean(tr_losses)) if len(tr_losses) else float("nan")
        train_acc = float(np.mean(tr_accs)) if len(tr_accs) else 0.0
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        model.eval()
        va_losses = []
        va_accs = []
        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]
                emb_list = model(Xs)

                loss = None
                for e, y in zip(emb_list, ys):
                    l = _supcon_loss_event(e, y, temperature=model.temperature, max_hits=256)
                    loss = l if loss is None else (loss + l)
                if loss is None:
                    continue
                loss = loss / max(1, len(emb_list))
                va_losses.append(float(loss.detach().cpu().item()))

                if len(va_accs) < 16:
                    # sample a couple per batch
                    for e, y in zip(emb_list[:2], ys[:2]):
                        a = _nn1_accuracy_event(e, y, max_hits=160)
                        if a is not None:
                            va_accs.append(float(a.detach().cpu().item()))

        val_loss = float(np.mean(va_losses)) if len(va_losses) else float("nan")
        val_acc = float(np.mean(va_accs)) if len(va_accs) else 0.0
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        # Early stopping
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

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

