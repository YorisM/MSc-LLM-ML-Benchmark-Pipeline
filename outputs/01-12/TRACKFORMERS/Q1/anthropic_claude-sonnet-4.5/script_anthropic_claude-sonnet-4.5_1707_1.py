
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
import hdbscan
import numpy as np

class MyPreprocessor:
    def __init__(self):
        self.r_mean = 0.0
        self.r_std = 1.0
        self.z_mean = 0.0
        self.z_std = 1.0

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 16,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "collate": "ragged_xy",
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        all_r = []
        all_z = []
        for X in Xs:
            all_r.append(X[:, 0].numpy())
            all_z.append(X[:, 2].numpy())

        all_r = np.concatenate(all_r)
        all_z = np.concatenate(all_z)

        self.r_mean = float(np.mean(all_r))
        self.r_std = float(np.std(all_r)) + 1e-6
        self.z_mean = float(np.mean(all_z))
        self.z_std = float(np.std(all_z)) + 1e-6

        return self

    def transform(self, X):
        # X: [N_hits, 4] = [r, theta, z, layer_id]
        r = X[:, 0]  # [N_hits]
        theta = X[:, 1]  # [N_hits]
        z = X[:, 2]  # [N_hits]
        layer = X[:, 3]  # [N_hits]

        # Normalize r and z
        r_norm = (r - self.r_mean) / self.r_std  # [N_hits]
        z_norm = (z - self.z_mean) / self.z_std  # [N_hits]

        # Cartesian coordinates
        x = r * torch.cos(theta)  # [N_hits]
        y = r * torch.sin(theta)  # [N_hits]

        # Stack all features
        features = torch.stack([r_norm, theta, z_norm, layer, x, y], dim=1)  # [N_hits, 6]

        return features.float()

def make_preprocessor():
    return MyPreprocessor()

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        input_dim = example_batch_x[0].shape[1]  # 6
        hidden_dim = 256
        embed_dim = 32

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, batch_x):
        # batch_x: list of [N_i, F]
        embeddings = []
        for x in batch_x:
            emb = self.encoder(x)  # [N_i, embed_dim]
            emb = F.normalize(emb, p=2, dim=1)  # [N_i, embed_dim] L2 normalized
            embeddings.append(emb)
        return embeddings

    def predict_labels(self, batch_x):
        with torch.no_grad():
            embeddings = self.forward(batch_x)

        labels = []
        for emb in embeddings:
            emb_np = emb.cpu().numpy()
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=4,
                min_samples=2,
                cluster_selection_epsilon=0.0
            )
            pred = clusterer.fit_predict(emb_np)
            labels.append(torch.from_numpy(pred).long().to(emb.device))

        return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

EPOCHS = 20

def contrastive_loss(embeddings, labels, margin=1.0):
    # embeddings: [N, D]
    # labels: [N]

    # Filter out noise (label 0)
    mask = labels > 0
    if mask.sum() < 2:
        return torch.tensor(0.0, device=embeddings.device)

    embeddings = embeddings[mask]
    labels = labels[mask]

    N = len(labels)
    if N < 2:
        return torch.tensor(0.0, device=embeddings.device)

    # Pairwise squared distances
    dist = torch.cdist(embeddings, embeddings)  # [N, N]

    # Same track mask (excluding diagonal)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    same = same * (1 - torch.eye(N, device=same.device))

    # Different track mask
    diff = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()

    # Contrastive loss: minimize distance for same track, maximize for different
    same_loss = (dist * same).sum() / (same.sum() + 1e-6)
    diff_loss = (F.relu(margin - dist) * diff).sum() / (diff.sum() + 1e-6)

    return same_loss + diff_loss

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_count = 0

        for Xs, ys in train_loader:
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            optimizer.zero_grad()
            embeddings = model(Xs)

            # Compute contrastive loss for each event
            loss = 0.0
            for emb, y in zip(embeddings, ys):
                loss += contrastive_loss(emb, y)
            loss = loss / len(Xs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_count += 1

        train_loss /= train_count
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                embeddings = model(Xs)

                loss = 0.0
                for emb, y in zip(embeddings, ys):
                    loss += contrastive_loss(emb, y)
                loss = loss / len(Xs)

                val_loss += loss.item()
                val_count += 1

        val_loss /= val_count
        val_losses.append(val_loss)

        # Placeholder accuracy (not directly applicable for unsupervised clustering)
        train_accs.append(0.0)
        val_accs.append(0.0)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
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
        summary = to_python(summary)
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

