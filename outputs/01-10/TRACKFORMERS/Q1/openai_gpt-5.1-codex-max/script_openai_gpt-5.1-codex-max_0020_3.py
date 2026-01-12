
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

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

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

    def fit(self, Xs):
        # Compute mean and std over all hits across events incrementally
        total_count = 0
        total_sum = None
        total_sumsq = None
        for X in Xs:
            if isinstance(X, torch.Tensor):
                x_np = X.cpu().numpy()
            else:
                x_np = np.asarray(X)
            if total_sum is None:
                total_sum = x_np.sum(axis=0, dtype=np.float64)
                total_sumsq = np.square(x_np, dtype=np.float64).sum(axis=0)
            else:
                total_sum += x_np.sum(axis=0, dtype=np.float64)
                total_sumsq += np.square(x_np, dtype=np.float64).sum(axis=0)
            total_count += x_np.shape[0]
        mean = total_sum / max(total_count, 1)
        var = total_sumsq / max(total_count, 1) - mean**2
        std = np.sqrt(np.clip(var, 1e-8, None))
        self.mean = torch.from_numpy(mean.astype(np.float32))
        self.std = torch.from_numpy(std.astype(np.float32))
        return self

    def transform(self, X):
        # X: torch tensor [N_hits, 4]
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X)
        if self.mean is not None and self.std is not None:
            mean = self.mean.to(X.device)
            std = self.std.to(X.device)
            X = (X - mean) / std
        return X.float()

def make_preprocessor():
    return MyPreprocessor()

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Determine input dimension
        in_dim = None
        if isinstance(example_batch_x, list) and len(example_batch_x) > 0:
            in_dim = example_batch_x[0].shape[1]
        elif isinstance(example_batch_x, tuple) and len(example_batch_x) > 0:
            if isinstance(example_batch_x[0], list) and len(example_batch_x[0]) > 0:
                in_dim = example_batch_x[0][0].shape[1]
            elif torch.is_tensor(example_batch_x[0]):
                in_dim = example_batch_x[0].shape[1]
        elif hasattr(example_batch_x, "x"):
            in_dim = example_batch_x.x.shape[1]
        else:
            raise ValueError("Cannot infer input dimension from example_batch_x")
        hidden = 64
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4)  # [sin_theta, cos_theta, slope, intercept]
        )
        self.dbscan_eps = 0.5
        self.dbscan_min_samples = 2

    def forward(self, batch_x):
        # batch_x: list of [N_i, F]
        if isinstance(batch_x, list):
            return [self.net(x) for x in batch_x]
        elif hasattr(batch_x, "x"):
            return self.net(batch_x.x)
        else:
            return self.net(batch_x)

    @torch.no_grad()
    def predict_labels(self, batch_x):
        self.eval()
        preds = self.forward(batch_x)
        labels_list = []
        if isinstance(batch_x, list):
            for p in preds:
                p_np = p.detach().cpu().numpy()
                # Features for clustering: [sin_theta, cos_theta, slope, intercept]
                feat = p_np
                # Standardize features per event
                mean = feat.mean(axis=0, keepdims=True)
                std = feat.std(axis=0, keepdims=True)
                std[std < 1e-6] = 1.0
                feat_std = (feat - mean) / std
                db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
                lbl = db.fit_predict(feat_std)
                # Post-process: invalidate small clusters (<4 hits)
                lbl_out = lbl.copy()
                for cid in np.unique(lbl):
                    if cid == -1:
                        continue
                    mask = lbl == cid
                    if mask.sum() < 4:
                        lbl_out[mask] = -1
                labels_list.append(torch.from_numpy(lbl_out.astype(np.int64)))
        else:
            # PyG batch not used in this implementation
            p = preds
            p_np = p.detach().cpu().numpy()
            mean = p_np.mean(axis=0, keepdims=True)
            std = p_np.std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            feat_std = (p_np - mean) / std
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
            lbl = db.fit_predict(feat_std)
            lbl_out = lbl.copy()
            for cid in np.unique(lbl):
                if cid == -1:
                    continue
                mask = lbl == cid
                if mask.sum() < 4:
                    lbl_out[mask] = -1
            labels_list = torch.from_numpy(lbl_out.astype(np.int64))
        return labels_list

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    def compute_targets(x, y):
        # x: [N,F], y: [N]
        target = torch.zeros((y.shape[0], 4), device=x.device, dtype=x.dtype)  # [N,4]
        track_ids = torch.unique(y)
        for tid in track_ids:
            if tid <= 0:
                continue
            mask_tid = y == tid
            if mask_tid.sum() == 0:
                continue
            r = x[mask_tid, 0]
            theta = x[mask_tid, 1]
            z = x[mask_tid, 2]
            sin_mean = torch.sin(theta).mean()
            cos_mean = torch.cos(theta).mean()
            theta_avg = torch.atan2(sin_mean, cos_mean)
            r_mean = r.mean()
            z_mean = z.mean()
            cov = ((r - r_mean) * (z - z_mean)).sum()
            var_r = ((r - r_mean) ** 2).sum()
            slope = cov / torch.clamp(var_r, min=1e-6)
            intercept = z_mean - slope * r_mean
            vec = torch.stack([torch.sin(theta_avg), torch.cos(theta_avg), slope, intercept])
            target[mask_tid, :] = vec
        return target  # [N,4]

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total_sqerr = 0.0
        total_hits = 0
        ari_list = []
        for Xs, ys in loader:
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]
            preds = model(Xs)
            # Loss
            for p, x, y in zip(preds, Xs, ys):
                mask = y > 0
                if mask.sum() == 0:
                    continue
                target = compute_targets(x, y)
                diff = p[mask] - target[mask]  # [n_hits_masked,4]
                total_sqerr += (diff * diff).sum().item()
                total_hits += mask.sum().item()
            # ARI
            pred_labels_list = model.predict_labels(Xs)
            for lbl_pred, y in zip(pred_labels_list, ys):
                y_cpu = y.cpu().numpy()
                lbl_pred_np = lbl_pred.cpu().numpy()
                mask_true = y_cpu > 0
                if mask_true.sum() < 2:
                    continue
                # Ensure labels for masked hits only
                true_labels = y_cpu[mask_true]
                pred_labels = lbl_pred_np[mask_true]
                if len(np.unique(pred_labels)) < 2 or len(np.unique(true_labels)) < 2:
                    ari = 0.0
                else:
                    ari = adjusted_rand_score(true_labels, pred_labels)
                ari_list.append(ari)
        avg_loss = total_sqerr / max(total_hits, 1)
        avg_ari = float(np.mean(ari_list)) if len(ari_list) > 0 else 0.0
        return avg_loss, avg_ari

    for epoch in range(epochs):
        model.train()
        epoch_sqerr = 0.0
        epoch_hits = 0
        for Xs, ys in train_loader:
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]
            optimizer.zero_grad()
            preds = model(Xs)
            batch_sqerr = 0.0
            batch_hits = 0
            for p, x, y in zip(preds, Xs, ys):
                mask = y > 0
                if mask.sum() == 0:
                    continue
                target = compute_targets(x, y)
                diff = p[mask] - target[mask]  # [n_mask,4]
                batch_sqerr += (diff * diff).sum()
                batch_hits += mask.sum()
            if batch_hits == 0:
                continue
            loss = batch_sqerr / batch_hits
            loss.backward()
            optimizer.step()
            epoch_sqerr += loss.item() * batch_hits.item()
            epoch_hits += batch_hits.item()
        epoch_loss = epoch_sqerr / max(epoch_hits, 1)
        train_loss_hist.append(epoch_loss)
        # Evaluate
        val_loss, val_ari = evaluate(val_loader)
        val_loss_hist.append(val_loss)
        # For simplicity, we do not compute train ARI to save time; set to 0
        train_acc_hist.append(0.0)
        val_acc_hist.append(val_ari)
        # Optional: early stopping could be implemented here

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
        summary = to_python(summary)
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

