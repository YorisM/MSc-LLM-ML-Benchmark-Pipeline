
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.cluster import DBSCAN
from torch.nn.utils.rnn import pad_sequence

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
        # Stats for normalization
        self.mean = None # [3]
        self.scale = None # [3]

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "utils.llm_io:EventDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 32,
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
        # F_raw is 4: r, theta, z, layer

        # We calculate mean and std of cartesian coordinates over a subset of events
        subset = Xs[:min(len(Xs), 1000)]
        data_list = []
        for x in subset:
            r = x[:, 0]
            theta = x[:, 1]
            z = x[:, 2]

            x_c = r * torch.cos(theta)
            y_c = r * torch.sin(theta)

            # [N, 3]
            coords = torch.stack([x_c, y_c, z], dim=1)
            data_list.append(coords)

        if data_list:
            all_data = torch.cat(data_list, dim=0)
            self.mean = all_data.mean(dim=0).cpu()
            self.scale = (all_data.std(dim=0) + 1e-6).cpu()
        else:
            self.mean = torch.zeros(3)
            self.scale = torch.ones(3)

        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, F_raw]
        r = X[:, 0]
        theta = X[:, 1]
        z = X[:, 2]

        # Convert to Cartesian
        x_c = r * torch.cos(theta)
        y_c = r * torch.sin(theta)

        coords = torch.stack([x_c, y_c, z], dim=1) # [N_hits, 3]

        # Normalize
        if self.mean is not None:
            coords = (coords - self.mean.to(coords.device)) / self.scale.to(coords.device)

        return coords # MUST return torch.FloatTensor [N_hits, F_out] for the default EventDataset path.

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
# MODEL I/O BATCH CONTRACT (CHOOSE ONE LANE)
# You MUST choose exactly one of the two supported input lanes and keep it consistent:
#
# --- LANE A: Torch ragged tensors (default) ---
# Loader:
#   - loader_class: "torch.utils.data:DataLoader"
#   - collate: "ragged_xy"
# Batch from DataLoader:
#   (Xs, ys) where
#     Xs: list[FloatTensor], length B, each Xs[i] shape [N_i, F]
#     ys: list[LongTensor],  length B, each ys[i] shape [N_i]
# Model output:
#   out = model.predict_labels(Xs)
#   out must be list[IntegerTensor], length B, each out[i] shape [N_i]
# Noise label:
#   Use -1 for noise/unassigned labels in the OUTPUT.
#
# --- LANE B: PyTorch Geometric (PyG) graphs ---
# ...

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Lane A: example_batch_x is list of tensors [N, 3]
        input_dim = 3
        hidden_dim = 64
        self.embed_dim = 16

        # Input projection
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer Encoder
        # batch_first=True => [Batch, Seq, Feature]
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, 
                                                   dim_feedforward=256, dropout=0.0, 
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Embedding Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embed_dim)
        )

        # Clustering parameters
        self.eps = 0.4
        self.min_samples = 3

    def forward(self, batch_x):
        # batch_x: list of [N_i, 3]
        lengths = [x.shape[0] for x in batch_x]

        # Pad sequence: [B, MaxN, 3]
        padded = pad_sequence(batch_x, batch_first=True)
        device = padded.device

        # Create padding mask (True where interactions are valid? No, PyTorch src_key_padding_mask is True for PADDED)
        # Create boolean mask: True for padded positions
        max_len = padded.size(1)
        # [B, MaxN]
        mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) >= torch.tensor(lengths, device=device).unsqueeze(1)

        # Network
        x = self.in_proj(padded) # [B, MaxN, H]
        x = self.encoder(x, src_key_padding_mask=mask) # [B, MaxN, H]
        emb = self.head(x) # [B, MaxN, E]

        # Unpack to list
        out_list = []
        for i, l in enumerate(lengths):
            out_list.append(emb[i, :l])

        return out_list

    def predict_labels(self, batch_x):
        self.eval()
        with torch.no_grad():
            embeddings_list = self.forward(batch_x)

        labels_list = []
        for emb in embeddings_list:
            # emb: [N, E]
            X_np = emb.detach().cpu().numpy()

            # DBSCAN clustering
            # eps needs to match the margin of the metric loss
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='euclidean', n_jobs=1)
            y_pred = db.fit_predict(X_np)

            labels_list.append(torch.tensor(y_pred, dtype=torch.long, device=emb.device))

        return labels_list # must return integer labels per hit

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # Requirements: return trained_model, train_loss, val_loss, train_acc, val_acc

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    device = next(model.parameters()).device

    # Hinge margin for contrastive loss
    MARGIN = 1.0

    for epoch in range(epochs):
        model.train()
        sum_loss = 0.0
        n_batches = 0

        for Xs, ys in train_loader:
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            optimizer.zero_grad()
            embeddings_list = model(Xs)

            batch_loss = torch.tensor(0.0, device=device)
            n_events = 0

            for emb, y in zip(embeddings_list, ys):
                if y.shape[0] < 2: continue
                n_events += 1

                # Pairwise euclidean distances
                dists = torch.cdist(emb, emb) # [N, N]

                # Masks
                # Noise is label 0.
                is_noise = (y == 0)
                y_col = y.unsqueeze(0)
                y_row = y.unsqueeze(1)

                # Positive pairs: same track, neither is noise
                pos_mask = (y_col == y_row) & ~(is_noise.unsqueeze(0) | is_noise.unsqueeze(1))

                # Negative pairs: different track, or noise-vs-valid
                # We ignore noise-vs-noise in loss
                neg_mask = (y_col != y_row) & ~(is_noise.unsqueeze(0) & is_noise.unsqueeze(1))

                # Loss terms
                loss_ev = torch.tensor(0.0, device=device)

                # Pull positives together
                if pos_mask.any():
                    loss_ev += (dists[pos_mask] ** 2).mean()

                # Push negatives apart (Hinge)
                if neg_mask.any():
                    d_neg = dists[neg_mask]
                    # encourage d > MARGIN
                    loss_ev += (F.relu(MARGIN - d_neg) ** 2).mean()

                batch_loss += loss_ev

            if n_events > 0:
                batch_loss = batch_loss / n_events
                batch_loss.backward()
                optimizer.step()
                sum_loss += batch_loss.item()
                n_batches += 1

        train_loss = sum_loss / max(n_batches, 1)

        # Valid Loop
        model.eval()
        val_sum_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]
                embeddings_list = model(Xs)

                batch_loss = 0.0
                n_events = 0
                for emb, y in zip(embeddings_list, ys):
                    if y.shape[0] < 2: continue
                    n_events += 1
                    dists = torch.cdist(emb, emb)
                    is_noise = (y == 0)
                    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & ~(is_noise.unsqueeze(0)|is_noise.unsqueeze(1))
                    neg_mask = (y.unsqueeze(0) != y.unsqueeze(1)) & ~(is_noise.unsqueeze(0)&is_noise.unsqueeze(1))

                    if pos_mask.any():
                        batch_loss += (dists[pos_mask]**2).mean().item()
                    if neg_mask.any():
                        batch_loss += (F.relu(MARGIN - dists[neg_mask])**2).mean().item()

                if n_events > 0:
                    val_sum_loss += batch_loss / n_events
                    val_batches += 1

        val_loss = val_sum_loss / max(val_batches, 1)

        # Logging handled by harness, but we can print checks
        # print(f"Epoch {epoch}: Train {train_loss:.4f}, Val {val_loss:.4f}")

    # Return dummy accuracy 0.0 as computing FitAccuracy inside loop is expensive
    # The harness suffix will compute the final metric.
    return model, train_loss, val_loss, 0.0, 0.0

# IMPORTS: DO NOT execute the pipeline here - the harness will do that.
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

