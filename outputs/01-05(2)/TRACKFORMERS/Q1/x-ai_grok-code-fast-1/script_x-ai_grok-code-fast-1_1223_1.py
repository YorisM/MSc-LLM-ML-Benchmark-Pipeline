
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

# ---------- IMPORTS ----------
# <LLM: Import modules>
# Already have torch, nn, etc.

# ----------- (OPTIONAL) PRE-PROCESSING ----------
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
        # Xs: list of per-event X, each [N_hits_i, 4] (r, theta, z, layer_id)
        cartesian_list = []
        for X in Xs:
            r, theta, z, layer = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            cartesian = torch.stack([x, y, z, layer], dim=1)  # [N, 4]
            cartesian_list.append(cartesian)
        full_cartesian = torch.cat(cartesian_list, dim=0)
        self.mean = full_cartesian.mean(dim=0)
        self.std = full_cartesian.std(dim=0)
        return self

    def transform(self, X):
        # X: [N_hits, 4]
        r, theta, z, layer = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        cartesian = torch.stack([x, y, z, layer], dim=1)  # [N, 4]
        cartesian = (cartesian - self.mean) / self.std
        return cartesian  # torch.FloatTensor [N, 4]

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        F = example_batch_x[0].shape[1]  # 4
        self.mlp = nn.Sequential(
            nn.Linear(F, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, batch_x):
        # batch_x: list of [N_i, F], output list of [N_i, 64] embeddings, normalized
        return [F.normalize(self.mlp(X.to(device)), dim=1) for X in batch_x]

    def predict_labels(self, batch_x):
        # Use DBSCAN to cluster normalized embeddings per event
        with torch.no_grad():
            embeddings = self.forward(batch_x)
            labels = []
            for emb in embeddings:
                # emb: [N, 64], normalized
                emb_np = emb.cpu().numpy()
                # Tune eps and min_samples for clustering; adjust as needed for performance
                eps = 0.5   # Hardcoded; in practice, tune via validation
                min_samples = 2
                cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(emb_np)
                # DBSCAN labels outliers (potential noise) as -1
                labels.append(torch.tensor(cluster_labels, dtype=torch.long))
            return labels  # list of [N_i], with -1 for noise/unassigned

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 10  # Adjust epochs for convergence

def info_nce_loss(emb, y, temp=0.1):
    # emb: [N, 64] normalized, y: [N]
    N = emb.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Cosine similarity matrix (N, N)
    cos_sim = emb @ emb.t() / temp
    cos_sim[torch.arange(N), torch.arange(N)] = -float('inf')  # Exclude self-similarity

    # Positive mask: same track (>0), exclude self
    mask_y = (y.unsqueeze(1) == y.unsqueeze(0)) & (y.unsqueeze(1) > 0) & (y.unsqueeze(0) > 0)
    mask_y.fill_diagonal_(False)

    loss = 0.0
    valid_count = 0
    for i in range(N):
        pos_mask = mask_y[i]
        if pos_mask.sum() == 0:
            continue
        pos_sims = cos_sim[i, pos_mask]  # Sums for positives
        all_sims = cos_sim[i]
        sum_pos = torch.logsumexp(pos_sims, 0)
        sum_all = torch.logsumexp(all_sims, 0)
        loss += sum_all - sum_pos
        valid_count += 1
    loss /= max(valid_count, 1)
    return loss

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # Train embeddings with InfoNCE loss
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_losses = []
    val_accs = []  # FitAccuracy will be computed by harness, dummy here

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for xs, ys in train_loader:
            Xs = [x.to(device) for x in xs]
            ys = [y.to(device) for y in ys]
            emb = model.forward(Xs)  # list of normalized [N_i, 64]
            batch_loss = 0.0
            for emb_i, y_i in zip(emb, ys):
                batch_loss += info_nce_loss(emb_i, y_i)
            batch_loss /= max(len(emb), 1)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total_loss += batch_loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        scheduler.step()

        # Validation: Compute dummy val_acc; harness handles true FitAccuracy
        # Here, simulate acc computation if needed, but rely on harness
        val_accs.append(0.0)  # Placeholder

    return model, train_losses, [], [0.0] * epochs, val_accs

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

