
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
from sklearn.cluster import DBSCAN
import hdbscan

class MyPreprocessor:
    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    # TIPS
    #   - IMPORTANT Default data flow: events[idx] -> split_X_y(evt) -> X, y
    #   - When modifying data features or feature engineering: annotate tensor size as comments after each tensor operation to reduce dimension mismatches.

    def __init__(self):
        # store normalization parameters
        self.mean = None
        self.std = None

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "utils.llm_io:EventDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one: 
            "collate": "ragged_xy",  # or "identity" or "None"
            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, F_raw]
        # compute mean and std of engineered features
        total_sum = torch.zeros(7, dtype=torch.float64)
        total_sum_sq = torch.zeros(7, dtype=torch.float64)
        total_count = 0
        for X in Xs:
            if not torch.is_tensor(X):
                X_t = torch.from_numpy(np.asarray(X))
            else:
                X_t = X
            r = X_t[:, 0].double()
            theta = X_t[:, 1].double()
            z = X_t[:, 2].double()
            layer = X_t[:, 3].double()
            x_cart = r * torch.cos(theta)  # [N]
            y_cart = r * torch.sin(theta)  # [N]
            slope = z / (r + 1e-3)         # [N]
            feats = torch.stack([r, theta, z, layer, x_cart, y_cart, slope], dim=1)  # [N,7]
            total_sum += feats.sum(dim=0)
            total_sum_sq += (feats * feats).sum(dim=0)
            total_count += feats.shape[0]
        mean = total_sum / max(total_count, 1)
        var = total_sum_sq / max(total_count, 1) - mean * mean
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        self.mean = mean.float()
        self.std = std.float()
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, F_raw]
        if not torch.is_tensor(X):
            X_t = torch.from_numpy(np.asarray(X)).float()
        else:
            X_t = X.float()
        r = X_t[:, 0]          # [N]
        theta = X_t[:, 1]      # [N]
        z = X_t[:, 2]          # [N]
        layer = X_t[:, 3]      # [N]
        x_cart = r * torch.cos(theta)  # [N]
        y_cart = r * torch.sin(theta)  # [N]
        slope = z / (r + 1e-3)         # [N]
        feats = torch.stack([r, theta, z, layer, x_cart, y_cart, slope], dim=1)  # [N,7]
        if self.mean is not None and self.std is not None:
            feats = (feats - self.mean.to(feats.device)) / self.std.to(feats.device)  # [N,7]
        return feats.float()  # [N,7]

def make_preprocessor():
    return MyPreprocessor()

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # determine input dimension
        if isinstance(example_batch_x, list):
            sample_x = example_batch_x[0]
        elif torch.is_tensor(example_batch_x):
            sample_x = example_batch_x
        elif hasattr(example_batch_x, "x"):
            sample_x = example_batch_x.x
        else:
            sample_x = None
        in_dim = sample_x.shape[1] if sample_x is not None else 7
        hidden = 64
        self.embed_dim = 16
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.embed_dim)
        )

    def forward(self, batch_x):
        # Define your model's forward pass here
        if isinstance(batch_x, list):
            outs = []
            for x in batch_x:
                x_dev = x.to(next(self.parameters()).device)
                emb = self.net(x_dev)                 # [N_i, embed_dim]
                emb = F.normalize(emb, p=2, dim=1)    # [N_i, embed_dim]
                outs.append(emb)
            return outs
        elif torch.is_tensor(batch_x):
            x_dev = batch_x.to(next(self.parameters()).device)
            emb = self.net(x_dev)                     # [N, embed_dim]
            emb = F.normalize(emb, p=2, dim=1)        # [N, embed_dim]
            return emb
        else:
            if hasattr(batch_x, "x"):
                x_dev = batch_x.x.to(next(self.parameters()).device)
                emb = self.net(x_dev)                 # [N_total, embed_dim]
                emb = F.normalize(emb, p=2, dim=1)
                return emb
            else:
                raise TypeError("Unsupported batch_x type in forward")

    def _cluster_event(self, emb):
        # emb: torch tensor [N, embed_dim]
        if emb.shape[0] == 0:
            return torch.empty((0,), dtype=torch.int64)
        emb_np = emb.detach().cpu().numpy()
        # normalize for clustering
        from sklearn.preprocessing import normalize
        emb_np = normalize(emb_np)
        labels = None
        try:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=2, metric='euclidean', cluster_selection_method='leaf')
            cl = clusterer.fit_predict(emb_np)
        except Exception:
            # fallback to DBSCAN
            n = emb_np.shape[0]
            if n > 1:
                # approximate epsilon from nearest neighbor distances
                from sklearn.neighbors import NearestNeighbors
                nnbrs = NearestNeighbors(n_neighbors=min(10, n)).fit(emb_np)
                dists, _ = nnbrs.kneighbors(emb_np)
                nn_d = dists[:, 1]
                eps = float(np.median(nn_d) * 1.5 + 1e-3)
                if eps <= 0 or math.isnan(eps) or math.isinf(eps):
                    eps = 0.5
                db = DBSCAN(eps=eps, min_samples=3)
                cl = db.fit_predict(emb_np)
            else:
                cl = np.array([-1], dtype=np.int64)
        if cl.max() > -1:
            uniq = np.unique(cl[cl >= 0])
            mapping = {c: i for i, c in enumerate(uniq)}
            cl_mapped = np.array([mapping[c] if c >= 0 else -1 for c in cl], dtype=np.int64)
        else:
            cl_mapped = cl.astype(np.int64)
        # prune small clusters (<4 hits)
        cl_tensor = torch.from_numpy(cl_mapped)
        if cl_tensor.numel() > 0 and (cl_tensor.max().item() >= 0):
            for cid in cl_tensor.unique():
                if cid.item() == -1:
                    continue
                mask = cl_tensor == cid
                if mask.sum().item() < 4:
                    cl_tensor[mask] = -1
        return cl_tensor.long()

    def predict_labels(self, batch_x):
        # Define your model's prediction logic here
        self.eval()
        with torch.no_grad():
            if isinstance(batch_x, list):
                emb_list = self.forward(batch_x)  # list of [N_i, embed_dim]
                out_labels = []
                for emb in emb_list:
                    labels = self._cluster_event(emb)  # [N_i]
                    out_labels.append(labels)
                return out_labels
            elif torch.is_tensor(batch_x):
                emb = self.forward(batch_x)  # [N, embed_dim]
                labels = self._cluster_event(emb)  # [N]
                return labels
            else:
                if hasattr(batch_x, "x") and hasattr(batch_x, "batch"):
                    emb = self.forward(batch_x)  # [N_total, embed_dim]
                    batch_vec = batch_x.batch
                    num_graphs = int(batch_vec.max().item()) + 1
                    out_list = []
                    for i in range(num_graphs):
                        mask = batch_vec == i
                        emb_i = emb[mask]
                        labels_i = self._cluster_event(emb_i)
                        out_list.append(labels_i)
                    # flatten to tensor in same order as batch_x.x
                    return torch.cat(out_list, dim=0)
                else:
                    raise TypeError("Unsupported batch_x type in predict_labels")

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

EPOCHS = 5   # adjust if you wish
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    #   - Must return: trained_model, train_loss, val_loss, train_acc, val_acc
    #   - Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for xs, ys in train_loader:
            xs = [x.to(device) for x in xs]
            ys = [y.to(device) for y in ys]
            optimizer.zero_grad()
            emb_list = model(xs)  # list of [N_i, embed_dim]
            triplets = []
            for emb, y in zip(emb_list, ys):
                # exclude noise
                track_ids = torch.unique(y[y > 0])
                if track_ids.numel() == 0:
                    continue
                for tid in track_ids:
                    idxs = (y == tid).nonzero(as_tuple=False).flatten()
                    if idxs.numel() < 2:
                        continue
                    perm = torch.randperm(idxs.numel(), device=emb.device)
                    a_idx = idxs[perm[0]]
                    p_idx = idxs[perm[1 % idxs.numel()]]
                    neg_candidates = (y != tid).nonzero(as_tuple=False).flatten()
                    if neg_candidates.numel() == 0:
                        continue
                    n_idx = neg_candidates[torch.randint(0, neg_candidates.numel(), (1,), device=emb.device)]
                    triplets.append((emb[a_idx], emb[p_idx], emb[n_idx]))
            if len(triplets) == 0:
                continue
            anchor = torch.stack([t[0] for t in triplets], dim=0)    # [T, embed_dim]
            positive = torch.stack([t[1] for t in triplets], dim=0)  # [T, embed_dim]
            negative = torch.stack([t[2] for t in triplets], dim=0)  # [T, embed_dim]
            loss = criterion(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        train_loss_hist.append(avg_loss)
        # validation
        model.eval()
        with torch.no_grad():
            total_v_loss = 0.0
            num_v_batches = 0
            for xs, ys in val_loader:
                xs = [x.to(device) for x in xs]
                ys = [y.to(device) for y in ys]
                emb_list = model(xs)
                triplets = []
                for emb, y in zip(emb_list, ys):
                    track_ids = torch.unique(y[y > 0])
                    if track_ids.numel() == 0:
                        continue
                    for tid in track_ids:
                        idxs = (y == tid).nonzero(as_tuple=False).flatten()
                        if idxs.numel() < 2:
                            continue
                        a_idx = idxs[0]
                        p_idx = idxs[1]
                        neg_candidates = (y != tid).nonzero(as_tuple=False).flatten()
                        if neg_candidates.numel() == 0:
                            continue
                        n_idx = neg_candidates[0]
                        triplets.append((emb[a_idx], emb[p_idx], emb[n_idx]))
                if len(triplets) == 0:
                    continue
                anchor = torch.stack([t[0] for t in triplets], dim=0)    # [T, embed_dim]
                positive = torch.stack([t[1] for t in triplets], dim=0)  # [T, embed_dim]
                negative = torch.stack([t[2] for t in triplets], dim=0)  # [T, embed_dim]
                loss = criterion(anchor, positive, negative)
                total_v_loss += loss.item()
                num_v_batches += 1
            avg_v_loss = total_v_loss / num_v_batches if num_v_batches > 0 else 0.0
            val_loss_hist.append(avg_v_loss)
        # placeholder accuracy (not used for metric)
        train_acc_hist.append(0.0)
        val_acc_hist.append(0.0)
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

