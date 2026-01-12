
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
import copy
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class MyPreprocessor:
    def __init__(self):
        # Global normalization stats (python floats; picklable)
        self.r_mean = 0.0
        self.r_std = 1.0
        self.z_mean = 0.0
        self.z_std = 1.0
        self.slope_mean = 0.0
        self.slope_std = 1.0
        self.layer_mean = 0.0
        self.layer_std = 1.0

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
        # Xs: list[Tensor], each [N_i, 4] with cols [r, theta, z, layer_id]
        n = 0
        sum_r = 0.0
        sum_z = 0.0
        sum_slope = 0.0
        sum_layer = 0.0
        sumsq_r = 0.0
        sumsq_z = 0.0
        sumsq_slope = 0.0
        sumsq_layer = 0.0

        for X in Xs:
            Xd = X.double()
            r = Xd[:, 0]  # [N]
            z = Xd[:, 2]  # [N]
            layer = Xd[:, 3]  # [N]
            slope = z / (r + 1e-3)  # [N]

            n_i = int(r.numel())
            if n_i == 0:
                continue

            sum_r += float(r.sum().item())
            sum_z += float(z.sum().item())
            sum_slope += float(slope.sum().item())
            sum_layer += float(layer.sum().item())

            sumsq_r += float((r * r).sum().item())
            sumsq_z += float((z * z).sum().item())
            sumsq_slope += float((slope * slope).sum().item())
            sumsq_layer += float((layer * layer).sum().item())

            n += n_i

        if n <= 1:
            return self

        def mean_std(sum1, sumsq1):
            mu = sum1 / n
            var = max(1e-12, sumsq1 / n - mu * mu)
            return float(mu), float(math.sqrt(var))

        self.r_mean, self.r_std = mean_std(sum_r, sumsq_r)
        self.z_mean, self.z_std = mean_std(sum_z, sumsq_z)
        self.slope_mean, self.slope_std = mean_std(sum_slope, sumsq_slope)
        self.layer_mean, self.layer_std = mean_std(sum_layer, sumsq_layer)

        # Prevent pathological tiny std
        self.r_std = max(self.r_std, 1e-6)
        self.z_std = max(self.z_std, 1e-6)
        self.slope_std = max(self.slope_std, 1e-6)
        self.layer_std = max(self.layer_std, 1e-6)
        return self

    def transform(self, X):
        # X: torch.FloatTensor [N_hits, 4] raw -> torch.FloatTensor [N_hits, 7]
        # Columns: 0 sin(theta), 1 cos(theta), 2 slope_norm, 3 r_norm, 4 z_norm, 5 layer_norm, 6 layer_raw
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float32)

        r = X[:, 0]  # [N]
        theta = X[:, 1]  # [N]
        z = X[:, 2]  # [N]
        layer = X[:, 3]  # [N]

        sin_t = torch.sin(theta)  # [N]
        cos_t = torch.cos(theta)  # [N]
        slope = z / (r + 1e-3)  # [N]

        slope_norm = (slope - self.slope_mean) / self.slope_std  # [N]
        r_norm = (r - self.r_mean) / self.r_std  # [N]
        z_norm = (z - self.z_mean) / self.z_std  # [N]
        layer_norm = (layer - self.layer_mean) / self.layer_std  # [N]
        layer_raw = layer  # [N]

        out = torch.stack(
            [sin_t, cos_t, slope_norm, r_norm, z_norm, layer_norm, layer_raw], dim=1
        )  # [N, 7]
        return out.float()


def make_preprocessor():
    return MyPreprocessor()


def _cluster_event_dbscan(P_np, layer_raw_np):
    """
    P_np: np.ndarray [N, 3] (sin, cos, slope_norm-like)
    layer_raw_np: np.ndarray [N] raw layer id (float->int)
    returns labels np.int64 [N] with noise = -1
    """
    N = P_np.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.int64)
    if N < 4:
        return -np.ones((N,), dtype=np.int64)

    # Adaptive epsilon from kNN distances (robust to global scaling)
    k = min(6, max(2, N - 1))
    try:
        nnbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        nnbrs.fit(P_np)
        dists, _ = nnbrs.kneighbors(P_np, return_distance=True)  # [N, k]
        # Use distance to (k-1)-th neighbor (excluding self at 0)
        dk = dists[:, -1]
        med = float(np.median(dk))
        eps = 1.35 * med
    except Exception:
        eps = 0.12

    eps = float(np.clip(eps, 0.05, 0.35))

    # DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=2, metric="euclidean")
    labels = db.fit_predict(P_np).astype(np.int64)  # [N], noise=-1

    # Filter tiny clusters (<4 hits) as noise
    for lab in np.unique(labels):
        if lab < 0:
            continue
        idx = np.where(labels == lab)[0]
        if idx.size < 4:
            labels[idx] = -1

    # Enforce max 1 hit per layer within a cluster (keep closest to centroid)
    # This reduces impurity when two tracks overlap in parameter space.
    layer_int = layer_raw_np.astype(np.int64, copy=False)
    for lab in np.unique(labels):
        if lab < 0:
            continue
        idx = np.where(labels == lab)[0]
        if idx.size < 4:
            labels[idx] = -1
            continue

        centroid = P_np[idx].mean(axis=0, keepdims=True)  # [1,3]
        d = np.linalg.norm(P_np[idx] - centroid, axis=1)  # [n_c]
        layers = layer_int[idx]  # [n_c]

        # For each layer keep the min-distance hit
        keep_mask = np.zeros(idx.size, dtype=bool)
        for L in np.unique(layers):
            j = np.where(layers == L)[0]
            if j.size == 1:
                keep_mask[j[0]] = True
            else:
                best = j[np.argmin(d[j])]
                keep_mask[best] = True

        drop = idx[~keep_mask]
        if drop.size > 0:
            labels[drop] = -1

        # Re-check size after dropping duplicates
        idx2 = np.where(labels == lab)[0]
        if idx2.size < 4:
            labels[idx2] = -1

    # Optional: assign some noise hits to nearest cluster if very close and layer not used
    # (helps recall without strongly hurting purity)
    labs = [l for l in np.unique(labels) if l >= 0]
    if len(labs) > 0:
        centroids = {}
        used_layers = {}
        layer_owner = {}  # (lab, layer) -> (idx, dist)
        for lab in labs:
            idx = np.where(labels == lab)[0]
            if idx.size == 0:
                continue
            c = P_np[idx].mean(axis=0)  # [3]
            centroids[lab] = c
            used_layers[lab] = set(layer_int[idx].tolist())
            # store current best per (lab, layer)
            for ii in idx:
                L = int(layer_int[ii])
                dist = float(np.linalg.norm(P_np[ii] - c))
                key = (lab, L)
                if key not in layer_owner or dist < layer_owner[key][1]:
                    layer_owner[key] = (int(ii), dist)

        noise_idx = np.where(labels < 0)[0]
        assign_thr = 0.75 * eps
        for ii in noise_idx:
            p = P_np[ii]
            L = int(layer_int[ii])
            # find nearest centroid
            best_lab = None
            best_dist = None
            for lab, c in centroids.items():
                dist = float(np.linalg.norm(p - c))
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_lab = lab
            if best_lab is None or best_dist is None:
                continue
            if best_dist > assign_thr:
                continue

            # layer conflict resolution
            key = (best_lab, L)
            if key not in layer_owner:
                labels[ii] = best_lab
                used_layers[best_lab].add(L)
                layer_owner[key] = (int(ii), best_dist)
            else:
                # Replace existing hit at same layer if this one is closer
                existing_idx, existing_dist = layer_owner[key]
                if best_dist < existing_dist:
                    labels[existing_idx] = -1
                    labels[ii] = best_lab
                    layer_owner[key] = (int(ii), best_dist)

    # Reindex cluster labels to 0..K-1 for cleanliness (not required, but stable)
    labs = sorted([l for l in np.unique(labels) if l >= 0])
    if len(labs) > 0:
        remap = {old: new for new, old in enumerate(labs)}
        for old, new in remap.items():
            labels[labels == old] = new

    return labels.astype(np.int64)


def _fitaccuracy_counts(y_true_t, y_pred_t):
    """
    Returns (correct_hits, total_truth_hits) per FitAccuracy definition.
    y_true_t: torch.LongTensor [N], truth track_id (0=noise)
    y_pred_t: torch.LongTensor [N], pred cluster id (-1=noise)
    """
    y_true = y_true_t.detach().cpu().numpy().astype(np.int64, copy=False)
    y_pred = y_pred_t.detach().cpu().numpy().astype(np.int64, copy=False)

    truth_mask = y_true > 0
    total_truth = int(truth_mask.sum())
    if total_truth == 0:
        return 0, 0

    # truth track sizes
    truth_ids, truth_counts = np.unique(y_true[truth_mask], return_counts=True)
    truth_size = {int(t): int(c) for t, c in zip(truth_ids, truth_counts)}

    correct = 0
    pred_labels = np.unique(y_pred)
    for plab in pred_labels:
        if plab < 0:
            continue
        idx = np.where(y_pred == plab)[0]
        if idx.size < 4:
            continue

        # purity: include all hits in predicted cluster (including truth noise)
        y_t = y_true[idx]
        y_t_pos = y_t[y_t > 0]
        if y_t_pos.size == 0:
            continue

        tids, cnts = np.unique(y_t_pos, return_counts=True)
        j = int(np.argmax(cnts))
        match_tid = int(tids[j])
        match_cnt = int(cnts[j])

        purity = match_cnt / float(idx.size)
        coverage = match_cnt / float(truth_size.get(match_tid, 10**9))

        if purity >= 0.5 and coverage >= 0.5:
            correct += match_cnt

    return int(correct), int(total_truth)


class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # example_batch_x: list of [N_i, F]
        if isinstance(example_batch_x, (list, tuple)):
            F = int(example_batch_x[0].shape[1])
        else:
            F = int(example_batch_x.shape[1])

        self.F = F

        # Simple linear projection, initialized to copy first 3 engineered features:
        # 0 sin(theta), 1 cos(theta), 2 slope_norm
        self.proj = nn.Linear(F, 3, bias=True)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            if F >= 3:
                self.proj.weight[0, 0] = 1.0
                self.proj.weight[1, 1] = 1.0
                self.proj.weight[2, 2] = 1.0

    def forward(self, batch_x):
        # batch_x: list length B, each FloatTensor [N_i, F]
        if not isinstance(batch_x, (list, tuple)):
            raise TypeError("Expected batch_x to be a list of per-event tensors [N_i, F].")

        sizes = [int(x.shape[0]) for x in batch_x]
        if len(sizes) == 0:
            return []

        x_cat = torch.cat(batch_x, dim=0)  # [sumN, F]
        out_cat = self.proj(x_cat)  # [sumN, 3]
        outs = list(out_cat.split(sizes, dim=0))  # list of [N_i, 3]
        return outs

    def predict_labels(self, batch_x):
        # batch_x: list length B, each FloatTensor [N_i, F]
        self.eval()
        with torch.no_grad():
            outs = self.forward(batch_x)  # list of [N_i, 3] on device

        labels_out = []
        for x, p in zip(batch_x, outs):
            # x: [N, F], p: [N, 3]
            N = int(x.shape[0])
            if N == 0:
                labels_out.append(torch.zeros((0,), dtype=torch.int64))
                continue

            # Use raw layer id for duplicate-layer cleanup
            if x.shape[1] >= 7:
                layer_raw = x[:, 6]  # [N]
            else:
                layer_raw = torch.zeros((N,), device=x.device, dtype=x.dtype)

            P_np = p.detach().cpu().numpy().astype(np.float32, copy=False)  # [N, 3]
            layer_np = layer_raw.detach().cpu().numpy().astype(np.float32, copy=False)  # [N]
            lab_np = _cluster_event_dbscan(P_np, layer_np)  # [N] int64, noise=-1
            labels_out.append(torch.from_numpy(lab_np.astype(np.int64)))

        return labels_out


def make_model(example_batch_x):
    return HitClassifier(example_batch_x)


EPOCHS = 6


def _compute_batch_loss(Xs, ys, outs):
    # Xs: list of [N_i, F], ys: list of [N_i], outs: list of [N_i, 3]
    # Targets derived per truth track: mean(sin,cos,slope_norm) from input features cols [0,1,2]
    total_loss = None
    n_events = 0
    smooth_l1 = nn.SmoothL1Loss(reduction="mean")

    for x, y, out in zip(Xs, ys, outs):
        # x: [N, F], y: [N], out: [N, 3]
        if x.numel() == 0:
            continue
        mask = y > 0
        if int(mask.sum().item()) < 4:
            continue

        # Build target [N, 3]
        target = torch.zeros_like(out)  # [N, 3]
        tids = torch.unique(y[mask])
        for tid in tids.tolist():
            idx = (y == tid)
            if int(idx.sum().item()) == 0:
                continue
            mean_sin = x[idx, 0].mean()
            mean_cos = x[idx, 1].mean()
            norm = torch.sqrt(mean_sin * mean_sin + mean_cos * mean_cos + 1e-12)
            mean_sin = mean_sin / norm
            mean_cos = mean_cos / norm
            mean_slope = x[idx, 2].mean()
            target[idx, 0] = mean_sin
            target[idx, 1] = mean_cos
            target[idx, 2] = mean_slope

        data_loss = smooth_l1(out[mask], target[mask])
        circle_pen = ((out[mask, 0] ** 2 + out[mask, 1] ** 2 - 1.0) ** 2).mean()

        loss_evt = data_loss + 0.05 * circle_pen
        if total_loss is None:
            total_loss = loss_evt
        else:
            total_loss = total_loss + loss_evt
        n_events += 1

    if total_loss is None:
        # Return a differentiable zero on correct device
        dev = Xs[0].device if len(Xs) > 0 else torch.device("cpu")
        return torch.zeros((), device=dev, requires_grad=True)

    return total_loss / float(max(1, n_events))


def _eval_fitaccuracy(model, loader, max_batches=10):
    model.eval()
    total_correct = 0
    total_truth = 0
    with torch.no_grad():
        for b, batch in enumerate(loader):
            Xs, ys = batch
            Xs = [x.to(device) for x in Xs]
            preds = model.predict_labels(Xs)  # list of [N_i] on CPU
            for y_t, y_p in zip(ys, preds):
                c, t = _fitaccuracy_counts(y_t, y_p)
                total_correct += c
                total_truth += t
            if max_batches is not None and (b + 1) >= max_batches:
                break
    if total_truth == 0:
        return 0.0
    return float(total_correct) / float(total_truth)


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        # No trainable params: still provide metrics arrays
        train_loss_hist, val_loss_hist = [], []
        train_acc_hist, val_acc_hist = [], []
        for _ in range(int(epochs)):
            train_loss_hist.append(0.0)
            val_loss_hist.append(0.0)
            train_acc_hist.append(_eval_fitaccuracy(model, train_loader, max_batches=5))
            val_acc_hist.append(_eval_fitaccuracy(model, val_loader, max_batches=10))
        return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience = 3
    bad = 0

    for ep in range(int(epochs)):
        model.train()
        running = 0.0
        nb = 0

        for Xs, ys in train_loader:
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            opt.zero_grad(set_to_none=True)
            outs = model(Xs)  # list of [N_i, 3]
            loss = _compute_batch_loss(Xs, ys, outs)
            loss.backward()
            clip_grad_norm_(params, 1.0)
            opt.step()

            running += float(loss.item())
            nb += 1

        sched.step()

        tr_loss = running / float(max(1, nb))
        train_loss_hist.append(tr_loss)

        model.eval()
        with torch.no_grad():
            vrunning = 0.0
            vnb = 0
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]
                outs = model(Xs)
                vloss = _compute_batch_loss(Xs, ys, outs)
                vrunning += float(vloss.item())
                vnb += 1
            va_loss = vrunning / float(max(1, vnb))
            val_loss_hist.append(va_loss)

        # FitAccuracy proxy on limited batches for speed
        tr_acc = _eval_fitaccuracy(model, train_loader, max_batches=6)
        va_acc = _eval_fitaccuracy(model, val_loader, max_batches=12)
        train_acc_hist.append(tr_acc)
        val_acc_hist.append(va_acc)

        # Early stopping on val loss (stable)
        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

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

