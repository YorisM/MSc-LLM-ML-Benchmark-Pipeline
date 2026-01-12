
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
from torch.nn.utils.rnn import pad_sequence
import sklearn.cluster as skc
from sklearn.preprocessing import StandardScaler
import warnings

# Use HDBSCAN if available, otherwise fallback
try:
    import hdbscan
except ImportError:
    hdbscan = None

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
        self.scaler = StandardScaler()
        self.fitted = False

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
        # Collect sample data to fit scaler on derived features
        # F_raw = 4: (r, theta, z, layer_id)

        data_buffer = []
        # Limit points to avoid excessive memory usage during fitting
        MAX_POINTS = 100000
        current_points = 0

        for x in Xs:
            if current_points >= MAX_POINTS:
                break

            if isinstance(x, torch.Tensor):
                x = x.numpy()

            # Derive features: convert cylindrical to cartesian + raw
            r = x[:, 0]
            theta = x[:, 1]
            z = x[:, 2]
            lid = x[:, 3]

            x_c = r * np.cos(theta)
            y_c = r * np.sin(theta)

            # Feature vector: [x, y, z, r, theta, layer_id] (Dim=6)
            feats = np.column_stack([x_c, y_c, z, r, theta, lid])

            data_buffer.append(feats)
            current_points += feats.shape[0]

        if data_buffer:
            all_feats = np.concatenate(data_buffer, axis=0)
            self.scaler.fit(all_feats)
            self.fitted = True

        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, F_raw]
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X

        r = X_np[:, 0]
        theta = X_np[:, 1]
        z = X_np[:, 2]
        lid = X_np[:, 3]

        x_c = r * np.cos(theta)
        y_c = r * np.sin(theta)

        feats = np.column_stack([x_c, y_c, z, r, theta, lid]) # [N_hits, 6]

        if self.fitted:
            feats = self.scaler.transform(feats)

        return torch.from_numpy(feats).float() # MUST return torch.FloatTensor [N_hits, F_out] for the default EventDataset path.

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
# Loader:
#   - loader_class: "torch_geometric.loader:DataLoader"
#   - collate: None
# Dataset samples MUST be torch_geometric.data.Data with at least:
#   data.x : FloatTensor [N_i, F]
#   data.y : LongTensor  [N_i]
# Batch from DataLoader:
#   G : torch_geometric.data.Batch (has G.x, G.y, and G.batch)
# Model forward:
#   out = model.predict_labels(G)
#   out must be an IntegerTensor of shape [G.x.shape[0]] (one label per node/hit)
#
# --- BOTH LANES ---
# Noise label: Use -1 for noise/unassigned labels in the OUTPUT.
# Any other batch shapes are NOT supported.

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Lane A: example_batch_x is a list of tensors
        input_dim = example_batch_x[0].shape[1] # Should be 6

        self.d_model = 64
        self.embed_dim = 16 # Dimension for embedding space
        self.nhead = 4
        self.num_layers = 3

        # Mapping input to hidden
        self.input_proj = nn.Linear(input_dim, self.d_model)

        # Transformer Encounter for context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=128,
            dropout=0.05,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Embedding head
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.embed_dim)
        )

    def forward(self, batch_x):
        # batch_x: list of [N_i, F]
        device = batch_x[0].device

        # Pad sequence for batch processing
        lengths = [x.shape[0] for x in batch_x]
        max_len = max(lengths)

        # Pad with 0
        padded_x = pad_sequence(batch_x, batch_first=True, padding_value=0.0) # [B, MaxL, F]

        # Create mask for Transformer (True = ignore)
        mask = torch.zeros((len(batch_x), max_len), dtype=torch.bool, device=device)
        for i, l in enumerate(lengths):
            if l < max_len:
                mask[i, l:] = True

        # Project and Encode
        x = self.input_proj(padded_x) # [B, MaxL, d_model]
        x = self.transformer(x, src_key_padding_mask=mask)

        # Extract and compute embeddings
        outputs = []
        for i, l in enumerate(lengths):
            # Slice valid part
            valid_x = x[i, :l, :]

            # Project to embedding space
            emb = self.head(valid_x)

            # Normalize to hypersphere (Metric Learning standard)
            emb = F.normalize(emb, p=2, dim=1)

            outputs.append(emb)

        return outputs # List of [N_i, embed_dim]

    def predict_labels(self, batch_x):
        self.eval()
        cluster_labels = []

        with torch.no_grad():
            embeddings_list = self.forward(batch_x)

            for emb in embeddings_list:
                # Convert to numpy for HDBSCAN
                X_np = emb.cpu().numpy().astype(np.float64)
                device = emb.device

                # Clustering
                if hdbscan is not None:
                    # min_cluster_size=4 matches FitAccuracy requirement
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=2, cluster_selection_epsilon=0.05)
                    labels = clusterer.fit_predict(X_np)
                else:
                    # Fallback
                    clusterer = skc.DBSCAN(eps=0.15, min_samples=4)
                    labels = clusterer.fit(X_np).labels_

                # HDBSCAN uses -1 for noise. FitAccuracy treats track_id > 0 as valid.
                # Just return raw labels (where -1 is mapped to noise).
                cluster_labels.append(torch.tensor(labels, dtype=torch.int32, device=device))

        return cluster_labels # must return integer labels per hit

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 15   # Increased epochs for convergence
def discriminative_loss(embedding, labels, delta_v=0.1, delta_d=0.6):
    # Discriminative Loss for instance segmentation (L2 on hypersphere)
    # embedding: [N, D] normalized
    # labels: [N] (0 is noise)

    # Only consider valid tracks
    valid_mask = labels > 0
    unique_tracks = torch.unique(labels[valid_mask])

    if len(unique_tracks) == 0:
        return torch.tensor(0.0, device=embedding.device, requires_grad=True)

    means = []
    l_var = torch.tensor(0.0, device=embedding.device)

    # 1. Variance term (Pull to centers)
    for tid in unique_tracks:
        mask = (labels == tid)
        track_pts = embedding[mask]
        mu = track_pts.mean(dim=0)
        means.append(mu)

        dist = torch.norm(track_pts - mu, dim=1)
        hinge = F.relu(dist - delta_v)
        l_var += torch.mean(hinge ** 2)

    l_var /= len(unique_tracks)

    if len(means) < 2:
        return l_var

    # 2. Distance term (Push centers apart)
    means = torch.stack(means)
    n_c = means.shape[0]

    # Pairwise distances
    diff = means.unsqueeze(1) - means.unsqueeze(0) # [Nc, Nc, D]
    dists = torch.norm(diff, dim=2)

    hinge_d = F.relu(2 * delta_d - dists)

    # Mask diagonal
    eye = torch.eye(n_c, device=embedding.device)
    hinge_d = hinge_d * (1 - eye)

    l_dist = torch.sum(hinge_d ** 2) / (n_c * (n_c - 1))

    return l_var + l_dist

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 12], gamma=0.25)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for Xs, ys in train_loader:
            # Transfer to device
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            optimizer.zero_grad()

            embeddings_list = model(Xs)

            loss = torch.tensor(0.0, device=device)
            valid_batch_count = 0

            for i in range(len(embeddings_list)):
                l = discriminative_loss(embeddings_list[i], ys[i])
                loss += l
                valid_batch_count += 1

            if valid_batch_count > 0:
                loss = loss / valid_batch_count
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses) if batch_losses else 0.0

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                embeddings_list = model(Xs)
                for i in range(len(embeddings_list)):
                    l = discriminative_loss(embeddings_list[i], ys[i])
                    val_losses.append(l.item())

        val_loss = np.mean(val_losses) if val_losses else 0.0
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Return required metrics (Accuracies are computed by harness, returning placeholders)
    return model, train_loss, val_loss, 0.0, 0.0

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

