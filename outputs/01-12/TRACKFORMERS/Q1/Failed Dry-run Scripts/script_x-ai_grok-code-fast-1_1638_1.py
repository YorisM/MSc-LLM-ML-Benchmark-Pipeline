
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

# ---------- IMPORTS ----------
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
import hdbscan
import numpy as np

# ----------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # F = example_batch_x[0][0].shape[1]  # assuming 4
        self.d_model = 64
        self.emb_dim = 64
        self.num_heads = 8
        self.num_layers = 6
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.d_model, self.num_heads, batch_first=True, dim_feedforward=256),
            self.num_layers
        )
        self.feature_proj = nn.Linear(4, self.d_model)
        self.is_noise_head = nn.Linear(self.d_model, 2)  # 0: not noise, 1: noise
        self.track_embed = nn.Linear(self.d_model, self.emb_dim)
        self.dropout = nn.Dropout(0.1)
        # Example batch_x is list of X, but since __init__, to get F

    def forward(self, batch_x):
        # batch_x: list of [N_i, 4]
        padded_X = nn.utils.rnn.pad_sequence(batch_x, batch_first=True, padding_value=0)  # [batch_size, max_len, 4]
        lengths = torch.tensor([len(x) for x in batch_x], device=padded_X.device)
        max_len = padded_X.shape[1]
        # src_key_padding_mask: True for pad positions
        src_key_padding_mask = torch.zeros(len(batch_x), max_len, dtype=torch.bool, device=padded_X.device)
        for i, l in enumerate(lengths):
            src_key_padding_mask[i, l:] = True
        embeddings = self.feature_proj(padded_X)  # [batch_size, max_len, d_model]
        embeddings = self.dropout(embeddings)
        output = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)  # [batch_size, max_len, d_model]
        is_noise_logits = self.is_noise_head(output)  # [batch_size, max_len, 2]
        track_embeddings = self.track_embed(output)  # [batch_size, max_len, emb_dim]
        return {
            'is_noise_logits': is_noise_logits,
            'track_embeddings': track_embeddings,
            'lengths': lengths,
            'max_len': max_len
        }

    def predict_labels(self, batch_x):
        with torch.no_grad():
            out = self.forward(batch_x)
            is_noise_logits = out['is_noise_logits']
            track_emb = out['track_embeddings']  # [batch_size, max_len, emb_dim]
            lengths = out['lengths']
            max_len = out['max_len']
            all_labels = []
            for i, l in enumerate(lengths):
                length = int(l.item())
                logits = is_noise_logits[i, :length]  # [length, 2]
                probs = torch.softmax(logits, dim=1)
                is_noise = torch.argmax(probs, dim=1)  # [length] 0 or 1
                emb = track_emb[i, :length].cpu().numpy()  # [length, emb_dim]
                labels = torch.full(length, -1, dtype=torch.long)
                not_noise_mask = (is_noise == 0).cpu().numpy()
                if np.sum(not_noise_mask) == 0:
                    all_labels.append(labels)
                    continue
                emb_not_noise = emb[not_noise_mask]
                clusterer = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=1)
                cluster_ids = clusterer.fit_predict(emb_not_noise)
                # cluster_ids: array of -1 or 0 to num_clusters-1
                cluster_ids = cluster_ids.copy()
                cluster_ids += 1  # now -1 stays -1, 0+ becomes 1+
                cluster_ids[cluster_ids == 0] = -1  # -1+1=0, back to -1 for noise
                labels[not_noise_mask] = torch.tensor(cluster_ids, dtype=torch.long).to(device='cpu')
                all_labels.append(labels.to(device='cpu'))
        return all_labels


def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def make_loader_cfg(self):
        return {
            "dataset_builder": "utils.llm_io:EventDataset",   # default harness dataset
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 32,  # smaller for memory
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": "ragged_xy",  # or "identity" or "None"
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        all_X = torch.cat(Xs, 0).numpy()  # [sum N, 4]
        self.scaler.fit(all_X)
        return self

    def transform(self, X):
        # X: [N, 4]
        normalized = self.scaler.transform(X.numpy()).astype(np.float32)
        return torch.from_numpy(normalized)  # [N, 4]

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL TRAINING ----------
def train_model(model, train_loader, val_loader, epochs):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            Xs, ys = batch
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]
            out = model(Xs)
            loss = 0.0
            for i in range(len(Xs)):
                y = ys[i]
                length = out['lengths'][i]
                is_noise_logits = out['is_noise_logits'][i, :length]  # [length, 2]
                target = (y == 0).long()  # 0: not noise, 1: noise
                loss += F.cross_entropy(is_noise_logits, target, reduction='mean')
                track_emb = out['track_embeddings'][i, :length]  # [length, emb_dim]
                not_noise_mask = (y != 0)
                if not_noise_mask.sum() == 0:
                    continue
                emb = track_emb[not_noise_mask]
                y_tracks = y[not_noise_mask]
                unique_tracks, inv_indices = torch.unique(y_tracks, return_inverse=True)
                centers = []
                for j in range(len(unique_tracks)):
                    mask = (inv_indices == j)
                    if mask.sum() == 0:
                        centers.append(torch.zeros(model.emb_dim, device=device))
                        continue
                    center = emb[mask].mean(dim=0)
                    centers.append(center)
                centers = torch.stack(centers)  # [num_tracks, emb_dim]
                for k in range(len(emb)):
                    diff = emb[k] - centers[inv_indices[k]]
                    loss += (diff ** 2).sum()
            loss = loss / len(Xs)  # average over events in batch
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        train_losses.append(epoch_loss / num_batches if num_batches > 0 else 0)
        # scheduler.step()
        model.eval()
        epoch_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]
                out = model(Xs)
                loss = 0.0
                for i in range(len(Xs)):
                    y = ys[i]
                    length = out['lengths'][i]
                    is_noise_logits = out['is_noise_logits'][i, :length]
                    target = (y == 0).long()
                    loss += F.cross_entropy(is_noise_logits, target, reduction='mean')
                    track_emb = out['track_embeddings'][i, :length]
                    not_noise_mask = (y != 0)
                    if not_noise_mask.sum() == 0:
                        continue
                    emb = track_emb[not_noise_mask]
                    y_tracks = y[not_noise_mask]
                    unique_tracks, inv_indices = torch.unique(y_tracks, return_inverse=True)
                    centers = []
                    for j in range(len(unique_tracks)):
                        mask = (inv_indices == j)
                        if mask.sum() == 0:
                            centers.append(torch.zeros(model.emb_dim, device=device))
                            continue
                        center = emb[mask].mean(dim=0)
                        centers.append(center)
                    centers = torch.stack(centers)
                    for k in range(len(emb)):
                        diff = emb[k] - centers[inv_indices[k]]
                        loss += (diff ** 2).sum()
                loss = loss / len(Xs)
                epoch_loss += loss.item()
                num_batches += 1
        val_losses.append(epoch_loss / num_batches if num_batches > 0 else 0)
    return model, train_losses, val_losses, None, None

EPOCHS = 15  # adjusted

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

