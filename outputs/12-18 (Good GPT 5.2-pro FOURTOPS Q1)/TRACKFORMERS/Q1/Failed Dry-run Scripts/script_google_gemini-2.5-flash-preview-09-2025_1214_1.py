
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy 
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy, write_loaderspec
from utils.suffix_utils import base_from_argv0, write_json, plot_train_val, persist_artefacts

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

import torch.nn.functional as F
import hdbscan
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Margin for Contrastive Loss
CONTRASTIVE_MARGIN = 0.5 

def contrastive_loss_event(E_i, Y_i, margin):
    # E_i: [N_i, D_embed] embeddings for one event
    # Y_i: [N_i] true track IDs (0 is noise/unassigned)

    N_i = E_i.size(0)
    if N_i < 2: return torch.tensor(0.0).to(E_i.device)

    # Calculate pairwise squared distances
    D_sq = torch.cdist(E_i, E_i, p=2)**2 # [N_i, N_i]

    # Mask noise track IDs (0) temporarily to -1 so they don't form positive pairs
    Y_i_no_noise = Y_i.clone()
    Y_i_no_noise[Y_i_no_noise == 0] = -1 

    # Y_sim = 1 if i and j belong to the same true track (and that track is not noise/unassigned)
    Y_sim = (Y_i_no_noise.unsqueeze(1) == Y_i_no_noise.unsqueeze(0)) & (Y_i_no_noise.unsqueeze(1) > 0)
    Y_sim = Y_sim.float() 

    # We only care about unique pairs (upper triangle excluding diagonal)
    indices = torch.triu_indices(N_i, N_i, offset=1, device=E_i.device)

    D_sq_pairs = D_sq[indices[0], indices[1]] # [N_pairs]
    Y_sim_pairs = Y_sim[indices[0], indices[1]] # [N_pairs]

    # 1. Positive loss: Minimize distance
    L_pos = Y_sim_pairs * D_sq_pairs 

    # 2. Negative loss: Maximize distance up to margin
    D_pairs = torch.sqrt(D_sq_pairs) # [N_pairs]
    hinge = torch.clamp(margin - D_pairs, min=0.0)**2
    L_neg = (1.0 - Y_sim_pairs) * hinge

    L_total = (L_pos.sum() + L_neg.sum()) 

    N_pairs = D_sq_pairs.size(0)
    if N_pairs > 0:
        return L_total / N_pairs
    else:
        return torch.tensor(0.0).to(E_i.device)

def calculate_metrics_event(E_i, Y_i, margin):
    N_i = E_i.size(0)
    if N_i < 2: return 0, 0, 0, 0 

    D = torch.cdist(E_i, E_i, p=2) 

    Y_i_no_noise = Y_i.clone()
    Y_i_no_noise[Y_i_no_noise == 0] = -1 
    Y_sim = (Y_i_no_noise.unsqueeze(1) == Y_i_no_noise.unsqueeze(0)) & (Y_i_no_noise.unsqueeze(1) > 0)

    indices = torch.triu_indices(N_i, N_i, offset=1, device=E_i.device)

    D_pairs = D[indices[0], indices[1]] 
    Y_sim_pairs = Y_sim[indices[0], indices[1]] 

    is_pos = Y_sim_pairs.bool()
    is_neg = ~is_pos

    N_pos_total = is_pos.sum().item()
    N_pos_ok = (D_pairs[is_pos] < margin).sum().item()

    N_neg_total = is_neg.sum().item()
    N_neg_ok = (D_pairs[is_neg] >= margin).sum().item() 

    return N_pos_ok, N_pos_total, N_neg_ok, N_neg_total

# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import hdbscan
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------- (OPTIONAL) CUSTOM DATASET  --------
# def make_dataset(events, pre, train: bool, **kwargs):
#   REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:make_dataset"
#   k = kwargs.get("k", 16)
#   <LLM: Insert custom dataset logic here>
#   return CustomDataset(events, pre, train=train, k=k)

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit()
    #   - transform()

    R_IDX, THETA_IDX, Z_IDX, LAYER_IDX = 0, 1, 2, 3

    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        self.means = None # [4,]
        self.stds = None  # [4,]

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one:
            "collate": "ragged_xy",  # or "identity" or None

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, 4]

        # <LLM: Extract statistics for transform>
        X_flat = torch.cat(Xs, dim=0).numpy() # [N_total, 4]

        # Calculate stats for R, Z, Layer_ID (Indices 0, 2, 3)
        data_to_normalize = X_flat[:, [self.R_IDX, self.Z_IDX, self.LAYER_IDX]]

        self.means = np.zeros(4, dtype=np.float32)
        self.stds = np.ones(4, dtype=np.float32)

        self.means[[self.R_IDX, self.Z_IDX, self.LAYER_IDX]] = data_to_normalize.mean(axis=0)
        self.stds[[self.R_IDX, self.Z_IDX, self.LAYER_IDX]] = data_to_normalize.std(axis=0)

        self.stds[self.stds < 1e-6] = 1.0

        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, 4]

        # <LLM: Apply pre-processing logic>
        X = X.clone() 

        # 1. Normalization (R, Z, Layer_ID)
        X[:, [self.R_IDX, self.Z_IDX, self.LAYER_IDX]] = (
            X[:, [self.R_IDX, self.Z_IDX, self.LAYER_IDX]] - self.means[[self.R_IDX, self.Z_IDX, self.LAYER_IDX]]
        ) / self.stds[[self.R_IDX, self.Z_IDX, self.LAYER_IDX]]
        # X: [N_hits, 4]

        # 2. Angular encoding for Theta
        theta = X[:, self.THETA_IDX:self.THETA_IDX+1] # [N_hits, 1]
        X_sin_theta = torch.sin(theta)               # [N_hits, 1]
        X_cos_theta = torch.cos(theta)               # [N_hits, 1]

        # Output features: R_norm, Z_norm, Layer_norm, Sin(T), Cos(T)
        X_out = torch.cat([
            X[:, [self.R_IDX, self.Z_IDX, self.LAYER_IDX]], # R_norm, Z_norm, Layer_norm [N_hits, 3]
            X_sin_theta,                                    # Sin(T) [N_hits, 1]
            X_cos_theta,                                    # Cos(T) [N_hits, 1]
        ], dim=1) # [N_hits, 5]

        return X_out # MUST return torch.FloatTensor [N_hits, 5]

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # IMPORTANT: Default harness input:
        #   batch_x is ragged list[Tensor], one per event, each shaped [N_hits, F].

        # <LLM: Define and initialize any stateful components here>
        if isinstance(example_batch_x, list):
            F_in = example_batch_x[0].size(1)
        else:
             # Should infer from the example tensor, if available
            F_in = example_batch_x.size(1)

        D_MODEL = 128
        N_LAYERS = 3
        N_HEADS = 8
        D_FF = 512
        D_EMBED = 8 # Target embedding dimension for clustering / metric space

        # 1. Input embedding layer
        self.input_projection = nn.Linear(F_in, D_MODEL) # [N_hits, F_in] -> [N_hits, D_MODEL]

        # 2. Transformer Encoder Layer Definition 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, 
            nhead=N_HEADS, 
            dim_feedforward=D_FF, 
            dropout=0.1, 
            batch_first=True,
            device=device
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=N_LAYERS
        )

        # 3. Output head (Metric Embedding space)
        self.output_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_EMBED) 
        )

    def _pad_batch(self, batch_x):
        lengths = [x.size(0) for x in batch_x]
        N_max = max(lengths)
        F = batch_x[0].size(1)
        B = len(batch_x)
        device = batch_x[0].device 

        # Pad input features
        padded_x = torch.zeros((B, N_max, F), dtype=batch_x[0].dtype, device=device) # [B, N_max, F_in]

        # Create mask (True means masked/ignored)
        mask = torch.ones((B, N_max), dtype=torch.bool, device=device)

        for i, x in enumerate(batch_x):
            padded_x[i, :lengths[i], :] = x
            mask[i, :lengths[i]] = False

        return padded_x, mask, lengths # padded_x: [B, N_max, F], mask: [B, N_max]

    def forward(self, batch_x):
        # <LLM: Define your model's forward pass here>

        # 1. Pad batch for Transformer, get mask and lengths
        padded_x, src_key_padding_mask, lengths = self._pad_batch(batch_x) 
        # padded_x: [B, N_max, F_in], mask: [B, N_max], lengths: list[B]

        # 2. Input embedding / projection
        x = self.input_projection(padded_x) # [B, N_max, D_MODEL]

        # 3. Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask) # [B, N_max, D_MODEL]

        # 4. Output head: Metric Embedding space
        E_padded = self.output_head(x) # [B, N_max, D_EMBED]

        # 5. Unpack: Get the embeddings only for actual hits
        E_list = []
        for i in range(len(lengths)):
            if lengths[i] > 0:
                E_list.append(E_padded[i, :lengths[i], :])

        if not E_list:
            # Handle empty batch scenario if any event was empty
            return torch.empty(0, dtype=torch.int64, device=padded_x.device)

        E_flat = torch.cat(E_list, dim=0) # [N_total, D_EMBED]

        if self.training:
            # Return embeddings for metric learning loss calculation
            return E_flat 
        else:
            # Inference: Perform HDBSCAN clustering
            device = E_flat.device
            cluster_labels_list = []

            # Iterate through events embeddings E_i
            for i, N_i in enumerate(lengths):
                if N_i == 0:
                     continue

                E_i = E_list[i].detach().cpu().numpy() # [N_i, D_EMBED]

                # Check for minimum constraint for clustering (4 hits for tracking efficiency)
                if N_i < 4: 
                    labels_i = np.full(N_i, -1, dtype=np.int64) # Assign to noise
                else:
                    try:
                        # Use min_cluster_size=4 to align loosely with FitAccuracy requirements
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=4, 
                            min_samples=1, 
                            metric='euclidean',
                            allow_single_cluster=True,
                            core_dist_n_jobs=4 # Parallel processing for distance calculation
                        ).fit(E_i)

                        labels_i = clusterer.labels_.astype(np.int64) # Labels are -1 (noise), 0, 1, 2, ...

                        # Re-index labels: True track IDs must be > 0. Noise is -1.
                        unique_labels = np.unique(labels_i[labels_i != -1])

                        if len(unique_labels) > 0:
                            # Map 0, 1, 2... to 1, 2, 3...
                            mapping = {old: new + 1 for new, old in enumerate(unique_labels)}

                            new_labels_i = np.full_like(labels_i, -1)
                            for old, new in mapping.items():
                                new_labels_i[labels_i == old] = new
                            labels_i = new_labels_i

                    except Exception as e:
                        # Fallback
                        # print(f"Warning: HDBSCAN failed. Error: {e}")
                        labels_i = np.full(N_i, -1, dtype=np.int64)

                cluster_labels_list.append(torch.from_numpy(labels_i).to(device))

            if not cluster_labels_list:
                return torch.empty(0, dtype=torch.int64, device=device)

            Y_pred = torch.cat(cluster_labels_list, dim=0) # [N_total]

            return Y_pred

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 35   # <LLM: adjust if you wish>   
def train_model(model, train_loader, val_loader, epochs):
    # <LLM: Write code to define training loop>
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6
    )

    clip_value = 1.0

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    MAX_PATIENCE = 10

    margin = CONTRASTIVE_MARGIN

    def run_epoch(loader, model, optimizer=None, is_train=True):
        if is_train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0

        # Metric tracking
        N_pos_ok_total = 0
        N_pos_pairs_total = 0
        N_neg_ok_total = 0
        N_neg_pairs_total = 0
        N_events = 0

        with torch.set_grad_enabled(is_train):
            for batch_x, batch_y in loader:

                # 1. Forward Pass: Get embeddings or predicted labels
                E_flat = model(batch_x) 

                # Check for empty output (e.g., if batch was empty or all hits were filtered)
                if E_flat.size(0) == 0:
                    continue

                # 2. Extract corresponding labels Y_flat
                Y_flat = torch.cat(batch_y).to(E_flat.device) 

                current_loss = torch.tensor(0.0, device=E_flat.device)

                # We need the event lengths (N_i) to reconstruct the splits
                lengths = [x.size(0) for x in batch_x if x.size(0) > 0] # Only consider events that actually contributed hits

                if not lengths:
                    continue

                E_list = torch.split(E_flat, lengths)
                Y_list = torch.split(Y_flat, lengths)

                N_events += len(E_list)

                batch_metric_pos_ok, batch_metric_pos_total, batch_metric_neg_ok, batch_metric_neg_total = 0, 0, 0, 0

                for E_i, Y_i in zip(E_list, Y_list):
                    if E_i.size(0) > 0:
                        l_i = contrastive_loss_event(E_i, Y_i, margin=margin)
                        current_loss += l_i

                        pos_ok, N_pos, neg_ok, N_neg = calculate_metrics_event(E_i, Y_i, margin=margin)
                        batch_metric_pos_ok += pos_ok
                        batch_metric_pos_total += N_pos
                        batch_metric_neg_ok += neg_ok
                        batch_metric_neg_total += N_neg

                # Average loss over the events in the batch
                N_events_in_batch = len(E_list)
                if N_events_in_batch > 0:
                    current_loss /= N_events_in_batch

                total_loss += current_loss.item() * N_events_in_batch

                N_pos_ok_total += batch_metric_pos_ok
                N_pos_pairs_total += batch_metric_pos_total
                N_neg_ok_total += batch_metric_neg_ok
                N_neg_pairs_total += batch_metric_neg_total

                if is_train:
                    optimizer.zero_grad()
                    current_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    optimizer.step()

        # Calculate average proxy metrics
        if N_events == 0:
            return 0.0, 0.0

        avg_loss = total_loss / N_events

        if (N_pos_pairs_total + N_neg_pairs_total) > 0:
            Purity_metric = N_pos_ok_total / (N_pos_pairs_total + 1e-8)
            Recall_metric = N_neg_ok_total / (N_neg_pairs_total + 1e-8)
            avg_acc = 0.5 * Purity_metric + 0.5 * Recall_metric
        else:
            avg_acc = 1.0

        return avg_loss, avg_acc

    # --- Training Loop Execution ---
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(train_loader, model, optimizer, is_train=True)
        val_loss, val_acc = run_epoch(val_loader, model, is_train=False)

        scheduler.step(val_loss)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Train Acc (Proxy) {train_acc:.4f}, Val Acc (Proxy) {val_acc:.4f}")

        # Early stopping check based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= MAX_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model state before returning
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
# <end code template>

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
        write_json(
            {"train_loss": tr_loss, "val_loss": va_loss, "train_acc": tr_acc, "val_acc": va_acc},
            out_path=os.path.join(SCRIPT_DIR, f"{base}_train_summary.json"),
        )

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

