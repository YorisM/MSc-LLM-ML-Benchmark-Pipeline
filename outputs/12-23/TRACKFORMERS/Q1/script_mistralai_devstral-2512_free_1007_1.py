
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

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

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from collections import defaultdict

# ----------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        self.events = events
        self.pre = pre
        self.train = train
        self._cache = {}

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        if idx not in self._cache:
            X, y = split_X_y(self.events[idx])
            X = self.pre.transform(X) if self.pre is not None else X
            self._cache[idx] = (X, y)
        return self._cache[idx]

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_encodings = None
        self.n_layers = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "collate": "ragged_xy",
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 16}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, 4]
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Scale r, theta, z
        self.n_layers = int(np.max([x[:, 3].max() for x in Xs]) + 1)
        # Create one-hot encoding for layer_id
        self.layer_encodings = np.eye(self.n_layers, dtype=np.float32)
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, 4]
        X = X.numpy() if isinstance(X, torch.Tensor) else X
        # Scale spatial coordinates
        scaled = self.scaler.transform(X[:, :3])
        # One-hot encode layer_id
        layer_idx = X[:, 3].astype(int)
        layer_onehot = self.layer_encodings[layer_idx]
        # Combine features: [r, theta, z, layer_onehot]
        out = np.hstack([scaled, layer_onehot])
        return torch.from_numpy(out).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Infer input features from example
        in_features = example_batch_x[0].shape[1]  # [N_hits, F]
        # Embedding for layer one-hot (last self.n_layers features)
        self.n_layers = in_features - 3
        # Spatial encoder
        self.spatial_enc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # Layer encoder
        self.layer_enc = nn.Sequential(
            nn.Linear(self.n_layers, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        # Combined encoder
        self.combined = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Attention for track assignment
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        # Track head
        self.track_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Noise classifier
        self.noise_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, batch_x):
        # batch_x: list of tensors, each [N_hits, F]
        device = batch_x[0].device
        all_preds = []
        for x in batch_x:
            # x: [N_hits, F]
            n_hits = x.shape[0]
            # Split features
            spatial = x[:, :3]
            layer_oh = x[:, 3:]
            # Encode
            s_feat = self.spatial_enc(spatial)  # [N, 64]
            l_feat = self.layer_enc(layer_oh)    # [N, 32]
            h = torch.cat([s_feat, l_feat], dim=1)  # [N, 96]
            h = self.combined(h)  # [N, 128]
            # Self-attention for track assignment
            h_attn, _ = self.attention(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
            h_attn = h_attn.squeeze(0)
            # Predict track scores
            track_scores = self.track_head(h_attn).squeeze(1)  # [N]
            # Predict noise scores
            noise_scores = self.noise_head(h).squeeze(1)  # [N]
            # Assign hits to tracks via clustering in embedding space
            with torch.no_grad():
                # Use DBSCAN on the combined embedding
                emb = h.cpu().numpy()
                clustering = DBSCAN(eps=0.5, min_samples=4, metric='cosine').fit(emb)
                labels = clustering.labels_
                # Map DBSCAN labels to track IDs
                # Noise points get label -1
                unique_labels = set(labels)
                track_id_map = {label: idx + 1 for idx, label in enumerate(unique_labels if -1 not in unique_labels else unique_labels - {-1})}
                pred_labels = torch.zeros(n_hits, dtype=torch.long, device=device) - 1
                for label in unique_labels:
                    if label == -1:
                        continue
                    mask = (labels == label)
                    pred_labels[mask] = track_id_map[label]
                # Override with noise predictions where noise_score > 0.5
                noise_mask = (noise_scores > 0.5).cpu()
                pred_labels[noise_mask] = -1
            all_preds.append(pred_labels)
        return all_preds

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=False)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_model = None
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        n_train = 0
        for batch in train_loader:
            Xs, ys = batch
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]
            optimizer.zero_grad()
            # Forward pass
            pred_labels = model(Xs)
            # Compute loss: treat as binary classification (noise vs track)
            # We'll use the noise head for supervision
            noise_targets = (ys[0] == 0).float().to(device)  # 1 if noise, 0 if track
            noise_scores = model.noise_head(model.combined(torch.cat([
                model.spatial_enc(Xs[0][:, :3]),
                model.layer_enc(Xs[0][:, 3:])
            ], dim=1))).squeeze(1)
            loss = criterion(noise_scores, noise_targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(ys[0])
            n_train += len(ys[0])
            # Compute accuracy
            pred_noise = (noise_scores > 0.5).float()
            acc = (pred_noise == noise_targets).float().mean().item()
            epoch_train_acc += acc * len(ys[0])
        epoch_train_loss /= n_train
        epoch_train_acc /= n_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]
                noise_targets = (ys[0] == 0).float().to(device)
                noise_scores = model.noise_head(model.combined(torch.cat([
                    model.spatial_enc(Xs[0][:, :3]),
                    model.layer_enc(Xs[0][:, 3:])
                ], dim=1))).squeeze(1)
                loss = criterion(noise_scores, noise_targets)
                epoch_val_loss += loss.item() * len(ys[0])
                n_val += len(ys[0])
                pred_noise = (noise_scores > 0.5).float()
                acc = (pred_noise == noise_targets).float().mean().item()
                epoch_val_acc += acc * len(ys[0])
        epoch_val_loss /= n_val
        epoch_val_acc /= n_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        scheduler.step(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model = model.state_dict()
            torch.save(best_model, 'best_model.pth')

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_losses, val_losses, train_accs, val_accs

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

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

    # Build model
    first_batch = next(iter(train_loader))
    view        = normalise_batch(first_batch, device=device)
    model       = make_model(view.batch_x).to(device)

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
        try:
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    view = normalise_batch(batch, device=device)
                    out  = model(view.batch_x)
                    assert_label_output(view.batch_x, out, allow_noise_label=True)
                    if i >= 4: # loop over 4 batches
                        break
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
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

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

