
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

# ---------- IMPORTS ----------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from hdbscan import HDBSCAN

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.mean_r = 0.0
        self.std_r = 1.0
        self.mean_z = 0.0
        self.std_z = 1.0
        self.mean_layer = 0.0
        self.std_layer = 1.0
        self.mean_z_over_r = 0.0
        self.std_z_over_r = 1.0

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
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # concatenate to compute statistics
        all_r = torch.cat([X[:, 0] for X in Xs], dim=0)
        all_theta = torch.cat([X[:, 1] for X in Xs], dim=0)
        all_z = torch.cat([X[:, 2] for X in Xs], dim=0)
        all_layer = torch.cat([X[:, 3] for X in Xs], dim=0)
        all_z_over_r = all_z / (all_r + 1e-3)
        self.mean_r = all_r.mean().item()
        self.std_r = all_r.std(unbiased=False).item() + 1e-6
        self.mean_z = all_z.mean().item()
        self.std_z = all_z.std(unbiased=False).item() + 1e-6
        self.mean_layer = all_layer.mean().item()
        self.std_layer = all_layer.std(unbiased=False).item() + 1e-6
        self.mean_z_over_r = all_z_over_r.mean().item()
        self.std_z_over_r = all_z_over_r.std(unbiased=False).item() + 1e-6
        return self

    def transform(self, X):
        # X: [N_hits,4]
        r = X[:, 0]
        theta = X[:, 1]
        z = X[:, 2]
        layer = X[:, 3]
        r_norm = (r - self.mean_r) / self.std_r  # [N]
        z_norm = (z - self.mean_z) / self.std_z  # [N]
        layer_norm = (layer - self.mean_layer) / self.std_layer  # [N]
        z_over_r = z / (r + 1e-3)
        z_over_r_norm = (z_over_r - self.mean_z_over_r) / self.std_z_over_r  # [N]
        theta_sin = torch.sin(theta)  # [N]
        theta_cos = torch.cos(theta)  # [N]
        feats = torch.stack([r_norm, theta_sin, theta_cos, z_norm, layer_norm, z_over_r_norm], dim=1)  # [N,6]
        return feats.float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Determine input feature size from example
        if isinstance(example_batch_x, list):
            in_dim = example_batch_x[0].shape[1]
        else:
            in_dim = example_batch_x.shape[1]
        hidden = 64
        embed_dim = 16
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim)
        )
        self.clusterer_params = {
            "min_cluster_size": 4,
            "min_samples": 1,
            "cluster_selection_method": "leaf",
            "metric": "euclidean"
        }

    def forward(self, batch_x):
        # batch_x: list of [N_i, F]
        embeddings = []
        if isinstance(batch_x, list):
            for x in batch_x:
                emb = self.mlp(x)  # [N_i, embed_dim]
                emb = F.normalize(emb, dim=1)
                embeddings.append(emb)
        else:
            # handle PyG Batch if needed
            emb = self.mlp(batch_x.x)
            emb = F.normalize(emb, dim=1)
            embeddings = emb
        return embeddings

    def predict_labels(self, batch_x):
        self.eval()
        labels_out = []
        if isinstance(batch_x, list):
            with torch.no_grad():
                for x in batch_x:
                    emb = self.mlp(x)  # [N_i, embed_dim]
                    emb = F.normalize(emb, dim=1)
                    emb_np = emb.cpu().numpy()
                    if emb_np.shape[0] >= self.clusterer_params["min_cluster_size"]:
                        clusterer = HDBSCAN(**self.clusterer_params)
                        lbls = clusterer.fit_predict(emb_np)
                    else:
                        # Not enough points to form cluster; mark as noise
                        lbls = -1 * np.ones(emb_np.shape[0], dtype=np.int64)
                    labels_out.append(torch.from_numpy(lbls.astype(np.int64)))
        else:
            # PyG Batch
            with torch.no_grad():
                emb = self.mlp(batch_x.x)
                emb = F.normalize(emb, dim=1)
                emb_np = emb.cpu().numpy()
                clusterer = HDBSCAN(**self.clusterer_params)
                lbls = clusterer.fit_predict(emb_np)
                labels_out = torch.from_numpy(lbls.astype(np.int64))
        return labels_out

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 8
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    margin = 0.5
    neg_weight = 0.5

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_events = 0
        for Xs, ys in train_loader:
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]
            optimizer.zero_grad()
            embeddings_list = model(Xs)  # list of [N_i, D]
            loss_batch = 0.0
            valid_events = 0
            for emb, lbl in zip(embeddings_list, ys):
                mask = lbl > 0
                if mask.sum() == 0:
                    continue
                uniq_tracks = lbl[mask].unique()
                centroids = []
                track_ids = []
                for t in uniq_tracks:
                    track_ids.append(int(t.item()))
                    centroids.append(emb[lbl == t].mean(dim=0))
                centroids = torch.stack(centroids, dim=0)  # [T,D]
                # positive loss
                hit_centroids = []
                for t in lbl:
                    if t.item() > 0:
                        idx = (uniq_tracks == t).nonzero(as_tuple=False).item()
                        hit_centroids.append(centroids[idx])
                    else:
                        hit_centroids.append(torch.zeros_like(centroids[0]))
                hit_centroids = torch.stack(hit_centroids, dim=0)  # [N,D]
                pos_mask = mask.unsqueeze(1).float()
                diff = (emb - hit_centroids) * pos_mask  # [N,D]
                pos_loss = (diff.pow(2).sum(dim=1)).sum() / (pos_mask.sum() + 1e-6)
                # negative loss between centroids
                if centroids.shape[0] > 1:
                    dists = torch.pdist(centroids, p=2)
                    neg_loss = F.relu(margin - dists).mean()
                else:
                    neg_loss = torch.tensor(0.0, device=device)
                loss_event = pos_loss + neg_weight * neg_loss
                loss_batch += loss_event
                valid_events += 1
            if valid_events > 0:
                loss_batch = loss_batch / valid_events
                loss_batch.backward()
                optimizer.step()
                total_loss += loss_batch.item() * valid_events
                n_events += valid_events
        avg_loss = total_loss / max(n_events, 1)
        train_loss_hist.append(avg_loss)
        train_acc_hist.append(0.0)

        # Validation
        model.eval()
        val_total_loss = 0.0
        val_events = 0
        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]
                embeddings_list = model(Xs)
                loss_batch = 0.0
                valid_events = 0
                for emb, lbl in zip(embeddings_list, ys):
                    mask = lbl > 0
                    if mask.sum() == 0:
                        continue
                    uniq_tracks = lbl[mask].unique()
                    centroids = []
                    for t in uniq_tracks:
                        centroids.append(emb[lbl == t].mean(dim=0))
                    centroids = torch.stack(centroids, dim=0)
                    hit_centroids = []
                    for t in lbl:
                        if t.item() > 0:
                            idx = (uniq_tracks == t).nonzero(as_tuple=False).item()
                            hit_centroids.append(centroids[idx])
                        else:
                            hit_centroids.append(torch.zeros_like(centroids[0]))
                    hit_centroids = torch.stack(hit_centroids, dim=0)
                    pos_mask = mask.unsqueeze(1).float()
                    diff = (emb - hit_centroids) * pos_mask
                    pos_loss = (diff.pow(2).sum(dim=1)).sum() / (pos_mask.sum() + 1e-6)
                    if centroids.shape[0] > 1:
                        dists = torch.pdist(centroids, p=2)
                        neg_loss = F.relu(margin - dists).mean()
                    else:
                        neg_loss = torch.tensor(0.0, device=device)
                    loss_event = pos_loss + neg_weight * neg_loss
                    loss_batch += loss_event
                    valid_events += 1
                if valid_events > 0:
                    loss_batch = loss_batch / valid_events
                    val_total_loss += loss_batch.item() * valid_events
                    val_events += valid_events
        val_avg_loss = val_total_loss / max(val_events, 1)
        val_loss_hist.append(val_avg_loss)
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
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

