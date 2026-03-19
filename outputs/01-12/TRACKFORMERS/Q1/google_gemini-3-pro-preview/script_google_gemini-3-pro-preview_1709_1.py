
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

# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# class CustomDataset(Dataset):
#   REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
#    def __init__(self, events, pre, train: bool = True, **kwargs):
#        X, y = events
#        self.X = pre.transform(X) if pre is not None else X
#        self.y = y
#    def __len__(self):
#        return int(self.y.shape[0])
#    def __getitem__(self, idx):
#        return self.X[idx], self.y[idx]

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    # TIPS
    #   - IMPORTANT Default data flow: events[idx] -> split_X_y(evt) -> X, y
    #   - When modifying data features or feature engineering: annotate tensor size as comments after each tensor operation to reduce dimension mismatches.

    def __init__(self):
        # We will use a standard scaler to normalize features
        self.scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "utils.llm_io:EventDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 32, # Batch of 32 events
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
        # F_raw is 4: r, theta, z, layer_id

        # Collect a sample of data to fit the scaler
        # We transform raw coordinates to Cartesian and trig features
        data_list = []
        # Upper limit to avoid OOM during fit with list accumulation
        limit = 2000
        count = 0

        for X in Xs:
            if count >= limit: break
            if isinstance(X, torch.Tensor):
                X_np = X.numpy()
            else:
                X_np = X

            # X_np: [N, 4] -> r, theta, z, layer
            r = X_np[:, 0]
            theta = X_np[:, 1]
            z = X_np[:, 2]
            layer = X_np[:, 3]

            x = r * np.cos(theta)
            y = r * np.sin(theta)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Features: [x, y, z, r, cos, sin, layer]
            # Dimensions: [N, 7]
            feats = np.column_stack([x, y, z, r, cos_t, sin_t, layer])
            data_list.append(feats)
            count += 1

        if data_list:
            all_data = np.concatenate(data_list, axis=0)
            self.scaler.fit(all_data)

        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, 4]
        dev = X.device
        X_np = X.cpu().numpy()

        r = X_np[:, 0]
        theta = X_np[:, 1]
        z = X_np[:, 2]
        layer = X_np[:, 3]

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # [N, 7]
        feats = np.column_stack([x, y, z, r, cos_t, sin_t, layer])

        # Scale
        feats_scaled = self.scaler.transform(feats)

        return torch.from_numpy(feats_scaled).float().to(dev)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
# MODEL I/O BATCH CONTRACT (CHOOSE ONE LANE)
# You MUST choose exactly one of the two supported input lanes and keep it consistent:
#
# --- LANE A: Torch ragged tensors (default) ---
# ...

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Input dim is 7 from preprocessor
        input_dim = 7
        embedding_dim = 12

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, embedding_dim)
        )

    def forward(self, batch_x):
        # When called directly with a Tensor (concatenated batch)
        # Output: [N_total, embedding_dim]
        # Normalize to unit hypersphere for stable cosine/euclidean distance
        out = self.net(batch_x)
        return F.normalize(out, p=2, dim=1)

    def predict_labels(self, batch_x):
        # batch_x is List[Tensor] (Lane A)
        self.eval()
        results = []

        # Silence sklearn warnings
        # warnings.filterwarnings("ignore")

        with torch.no_grad():
            for x in batch_x: # x is [N_hits, 7]
                # Embed
                emb = self.forward(x) # [N_hits, emb_dim]
                emb_np = emb.cpu().numpy()

                # Cluster
                # eps=0.08 on unit sphere corresponds to small angular separation
                # min_samples=3 allows catching tracks with >=4 hits reasonably well
                db = DBSCAN(eps=0.08, min_samples=3, metric='euclidean', n_jobs=1)
                labels = db.fit_predict(emb_np)

                # Labels are -1 for noise, 0..K for clusters
                # This matches requirements directly.
                results.append(torch.from_numpy(labels).to(x.device).long())

        return results

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 15   # Increased slightly for convergence
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    #   - Must return: trained_model, train_loss, val_loss, train_acc, val_acc
    #   - Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Step scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    device = next(model.parameters()).device
    model.to(device)

    trained_model = model
    best_val_loss = float('inf')

    # Margin parameters for Hinge Loss
    margin_pos = 0.1 # Pull same track
    margin_neg = 1.0 # Push diff track

    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for Xs, ys in train_loader:
            # Lane A: Xs is list of Tensors, ys is list of Tensors
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            # Concatenate for efficient forward pass
            sizes = [x.shape[0] for x in Xs]
            X_flat = torch.cat(Xs, dim=0) # [SumN, F]

            # Forward
            emb_flat = model(X_flat) # [SumN, D]

            # Split for per-event loss calculation
            emb_splits = torch.split(emb_flat, sizes, dim=0)

            batch_loss_accum = 0.0
            valid_events = 0

            optimizer.zero_grad()

            for i, emb in enumerate(emb_splits):
                y = ys[i]

                # Mask noise (track_id == 0)
                mask_track = (y != 0)
                if mask_track.sum() < 2:
                    continue # Need at least 2 hits to form a pair

                # Subsample if event is huge to save memory/compute
                y_tracks = y[mask_track]
                emb_tracks = emb[mask_track]

                if len(y_tracks) > 800:
                    perm = torch.randperm(len(y_tracks))[:800]
                    y_tracks = y_tracks[perm]
                    emb_tracks = emb_tracks[perm]

                # Pairwise distance matrix
                # shapes: [N_sub, D]
                dists = torch.cdist(emb_tracks, emb_tracks, p=2)

                # Identity matrix for labels
                # pos_mask[i, j] = 1 if same track
                pos_mask = (y_tracks.unsqueeze(1) == y_tracks.unsqueeze(0))
                neg_mask = ~pos_mask

                # Loss computation
                # 1. Pull positives (exclude self-loop which is dist 0)
                # We can just optimize mean of all positives
                loss_pos = torch.mean(torch.clamp(dists[pos_mask] - margin_pos, min=0)**2)

                # 2. Push negatives
                # Hinge: only penalty if dist < margin_neg
                hard_negs = dists[neg_mask]
                loss_neg = 0.0
                if hard_negs.numel() > 0:
                    violating = torch.clamp(margin_neg - hard_negs, min=0)**2
                    loss_neg = torch.mean(violating)

                batch_loss_accum += (loss_pos + loss_neg)
                valid_events += 1

            # Average loss over batch
            if valid_events > 0:
                final_batch_loss = batch_loss_accum / valid_events
                final_batch_loss.backward()
                optimizer.step()
                epoch_loss += final_batch_loss.item()
                n_batches += 1

        avg_train_loss = epoch_loss / max(1, n_batches)
        train_loss_hist.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                X_flat = torch.cat(Xs, dim=0)
                emb_flat = model(X_flat)
                emb_splits = torch.split(emb_flat, [x.shape[0] for x in Xs], dim=0)

                b_loss = 0.0
                v_ev = 0
                for i, emb in enumerate(emb_splits):
                    y = ys[i]
                    mask_track = (y != 0)
                    if mask_track.sum() < 2: continue

                    y_tracks = y[mask_track]
                    emb_tracks = emb[mask_track]
                    if len(y_tracks) > 800:
                        y_tracks = y_tracks[:800]
                        emb_tracks = emb_tracks[:800]

                    dists = torch.cdist(emb_tracks, emb_tracks, p=2)
                    pos_mask = (y_tracks.unsqueeze(1) == y_tracks.unsqueeze(0))
                    neg_mask = ~pos_mask

                    l_pos = torch.mean(torch.clamp(dists[pos_mask] - margin_pos, min=0)**2)
                    hard_negs = dists[neg_mask]
                    l_neg = 0.0
                    if hard_negs.numel() > 0:
                        l_neg = torch.mean(torch.clamp(margin_neg - hard_negs, min=0)**2)

                    b_loss += (l_pos + l_neg)
                    v_ev += 1

                if v_ev > 0:
                    val_loss_sum += (b_loss / v_ev).item()
                    n_val_batches += 1

        avg_val_loss = val_loss_sum / max(1, n_val_batches)
        val_loss_hist.append(avg_val_loss)

        scheduler.step()

        # Track best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trained_model = model

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Dummy acc values as we don't compute full metrics inside training loop
    train_acc = 0.0
    val_acc = 0.0

    # Return last known losses
    return trained_model, train_loss_hist[-1], val_loss_hist[-1], train_acc, val_acc

# IMPORTANT: DO NOT execute the pipeline here - the harness will do that.
# <end code template>

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

