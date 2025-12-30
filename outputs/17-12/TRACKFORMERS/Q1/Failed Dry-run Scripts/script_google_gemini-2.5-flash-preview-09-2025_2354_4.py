
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy 
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader, split_X_y, EventDataset
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

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple
import hdbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
import math

# -------- (OPTIONAL) CUSTOM DATASET  --------
# Not overriding make_dataset, using default EventDataset.

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        self.mean = None
        self.std = None

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 16, # Reduced batch size for variable-length GNN processing
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": "ragged_xy",  # Required collate for ragged batches

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, data: List[torch.Tensor]):
        # <LLM: Extract statistics or fit transform>
        # data is list[Tensor] where Tensor is [N_i, 4]
        all_hits = torch.cat(data, dim=0) # [Total_N_hits, 4]

        # Calculate standardization parameters
        self.mean = torch.mean(all_hits, dim=0) # [4]
        self.std = torch.std(all_hits, dim=0)   # [4]

        # Stabilize stddev
        self.std[self.std < 1e-6] = 1.0 

        return self

    def transform(self, data: List[torch.Tensor]):
        # <LLM: Apply preprocessing logic, return torch.Tensor>
        if self.mean is None or self.std is None:
            raise RuntimeError("Preprocessor must be fitted first.")

        transformed_data = []
        # Ensure normalization tensors are on CPU/non-device if data list is constructed on CPU
        mean_cpu = self.mean.cpu() # [4]
        std_cpu = self.std.cpu()   # [4]

        for x in data:
            # x: [N_i, 4]
            x_norm = (x - mean_cpu) / std_cpu # [N_i, 4]
            transformed_data.append(x_norm)

        return transformed_data # must return an indexable, picklable object

def make_preprocessor():
    return MyPreprocessor()

# Function Definition for Loss
def calculate_margin_loss(embeddings: torch.Tensor, ys_list: List[torch.Tensor], margin: float = 1.0):
    """Calculates Metric Learning Margin Loss using randomly sampled pairs within events."""

    y = torch.cat(ys_list, dim=0).to(embeddings.device) # [Total_N]

    N_total = y.shape[0]

    if N_total == 0:
        return torch.tensor(0.0, device=embeddings.device)

    N_samples_target = 200000 

    start_idx = 0
    all_losses = []

    # Iterate through events to ensure pairs are only sampled within an event boundary
    for event_y in ys_list:
        N_i = event_y.shape[0]
        if N_i < 2: 
            start_idx += N_i
            continue

        # Dynamic sampling based on event size: min 100 samples, max N_i^2/4, total max N_samples_target
        samples_i = max(500, min(N_i * N_i // 4, N_samples_target // len(ys_list) + 1))

        # Sample indices within this event
        i_idx = torch.randint(N_i, (samples_i,), device=embeddings.device)
        j_idx = torch.randint(N_i, (samples_i,), device=embeddings.device)

        # Global indices
        i_global = start_idx + i_idx
        j_global = start_idx + j_idx

        # Ensure we don't sample the exact same hit pair twice (i!=j or use mask later)

        Ei = embeddings[i_global] # [samples_i, D_out]
        Ej = embeddings[j_global]
        Yi = y[i_global]
        Yj = y[j_global]

        # Target classification
        is_positive = (Yi == Yj) & (Yi > 0) # Positive if same track ID AND not noise [samples_i]
        y_pair = is_positive.float()

        # Distance squared
        d2 = torch.sum((Ei - Ej)**2, dim=1) # [samples_i]

        # Distance for negative comparison
        d = torch.sqrt(d2 + 1e-6) 

        # Loss: Minimize d2 for positives
        loss_pos = y_pair * d2

        # Loss: Maximize separation for negatives up to margin M
        loss_neg = (1 - y_pair) * torch.clamp(margin - d, min=0.0)**2 

        all_losses.append(loss_pos + loss_neg)
        start_idx += N_i

    if not all_losses:
        return torch.tensor(0.0, device=embeddings.device)

    return torch.cat(all_losses).mean()

# Utility for graph building
def _build_graph(x: torch.Tensor, k: int):
    # x: [N_hits, D_features] 
    N_hits = x.shape[0]
    if N_hits <= k:
        k = max(1, N_hits - 1)
        if k == 0:
            return torch.empty((2, 0), dtype=torch.long, device=x.device)

    # Use CPU for Scikit-learn Nearest Neighbors
    X_np = x.detach().cpu().numpy()

    # Search for k nearest neighbors (k+1 including self)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(X_np)
    distances, indices = nn.kneighbors(X_np) # indices: [N_hits, k+1]

    source_nodes = torch.arange(N_hits, device=x.device).repeat_interleave(k) # [N_hits * k]
    # indices[:, 1:] takes the k nearest neighbors, excluding the point itself.
    target_nodes = torch.tensor(indices[:, 1:], dtype=torch.long, device=x.device).flatten() # [N_hits * k]

    edge_index = torch.stack([source_nodes, target_nodes], dim=0) # [2, N_hits * k]

    # Make graph explicitly bidirectional and unique
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).unique(dim=1) # [2, E_total]

    return edge_index

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x: List[torch.Tensor]):
        super().__init__()
        # <LLM: Define and initialize any stateful components here>

        if isinstance(example_batch_x, list) and len(example_batch_x) > 0:
            F_in = example_batch_x[0].shape[-1] # F_in = 4
        else:
            F_in = 4 

        H = 64 # Hidden dimension
        D_out = 8 # Embedding dimension
        self.k = 8 # K for k-NN graph construction
        self.min_cluster_size = 4 
        self.min_samples = 1 

        # Initial transformation layer
        self.in_transform = nn.Sequential(
            nn.Linear(F_in, H), # [N_hits, 4] -> [N_hits, H]
            nn.ReLU(),
            nn.LayerNorm(H)
        ) 

        # GNN layers
        self.conv1 = GCNConv(H, H) 
        self.bn1 = nn.InstanceNorm1d(H)

        self.conv2 = GCNConv(H, H) 
        self.bn2 = nn.InstanceNorm1d(H)

        # Output embedding layer
        self.out_transform = nn.Sequential(
            nn.Linear(H, D_out), # [N_hits, H] -> [N_hits, D_out]
            nn.LayerNorm(D_out) 
        )

    def forward(self, batch_x: List[torch.Tensor]) -> torch.Tensor:

        x = torch.cat(batch_x, dim=0).to(device) # x: [Total_N, 4]
        N_total = x.shape[0]

        if N_total == 0:
            return torch.tensor([], dtype=torch.long, device=device)

        # Initial transformation
        h = self.in_transform(x) # h: [Total_N, H]

        start_idx = 0
        embeddings_list = []
        cluster_id_list = []

        is_evaluating = not self.training

        for idx in range(len(batch_x)):
            # Note: event_x features are the normalized coordinates
            event_x = batch_x[idx]
            N_i = event_x.shape[0]
            if N_i == 0: continue

            # Slice transformed features h_i
            h_i = h[start_idx:start_idx + N_i] # [N_i, H]

            # --- Graph Construction (coordinates are already normalized in x/event_x) ---
            edge_index_i = _build_graph(x[start_idx:start_idx+N_i], k=self.k).to(device) # [2, E_i]

            # --- GNN Propagation ---
            # Using feature propagation only on hit features, without edge weights
            h_i = F.relu(self.bn1(self.conv1(h_i, edge_index_i))) # [N_i, H]
            h_i = F.relu(self.bn2(self.conv2(h_i, edge_index_i))) # [N_i, H]

            # --- Embedding Output ---
            e_i = self.out_transform(h_i) # e_i: [N_i, D_out]
            embeddings_list.append(e_i)

            # --- Clustering (Inference Mode Only) ---
            if is_evaluating:
                E_np = e_i.cpu().detach().numpy()

                # Clustering requires at least min_cluster_size members
                if N_i < self.min_cluster_size:
                    pred_labels = torch.zeros(N_i, dtype=torch.long, device=device)
                else:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=self.min_cluster_size, 
                        metric='euclidean',
                        min_samples=self.min_samples,
                        # cluster_selection_epsilon=0.0, # default is fine
                        core_dist_n_jobs=1 
                    )

                    cluster_labels = clusterer.fit_predict(E_np) 
                    pred_labels = torch.tensor(cluster_labels, dtype=torch.long, device=device)

                    # Shift labels: -1 -> 0 (Noise); >=0 -> >=1 (Tracks)
                    shifted_labels = pred_labels + 1
                    shifted_labels[pred_labels == -1] = 0 # Mark HDBSCAN noise as Label 0

                cluster_id_list.append(shifted_labels)

            start_idx += N_i

        embeddings = torch.cat(embeddings_list, dim=0) # [Total_N, D_out]

        if is_evaluating:
            return torch.cat(cluster_id_list, dim=0) # [Total_N] integer labels
        else:
            # During training, return embeddings for metric learning loss calculation
            return embeddings # [Total_N, D_out]

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 30   
def train_model(model, train_loader, val_loader, epochs):

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    M = 0.5 # Margin for contrastive loss

    train_loss_history = []
    val_loss_history = []

    # We use placeholder/proxy metrics for classification accuracy tracking
    train_acc_history = [] 
    val_acc_history = [] 

    best_val_score = -float('inf')
    best_model_state = None
    patience = 8
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0

        # Training loop
        for batch in train_loader:
            Xs, Ys = batch 
            Xs_device = [x.to(device) for x in Xs]
            Ys_device = [y.to(device) for y in Ys]

            optimizer.zero_grad()

            # Forward pass returns embeddings [Total_N, D_out]
            embeddings = model(Xs_device) 

            loss = calculate_margin_loss(embeddings, Ys_device, margin=M)

            if loss.item() > 0:
                loss.backward()
                # Gradient clipping to stabilize training
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step()

            total_train_loss += loss.item() * len(Xs)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(1.0) # Placeholder

        scheduler.step()

        # Validation loop
        model.eval()
        total_val_loss = 0
        all_y_true = []
        all_y_pred = [] 

        with torch.no_grad():
            for batch in val_loader:
                Xs, Ys = batch 

                Xs_device = [x.to(device) for x in Xs]
                Ys_device = [y.to(device) for y in Ys]

                # 1. Calculate embeddings for proxy loss calculation (model is in eval=False here)
                # Need to manually get embeddings back for loss calculation despite eval mode.

                # Temporary switch to training mode just to get embeddings, but keep no_grad()
                model.train()
                embeddings = model(Xs_device) 
                model.eval()

                loss = calculate_margin_loss(embeddings, Ys_device, margin=M)
                total_val_loss += loss.item() * len(Xs)

                # 2. Get clustered labels (model is in eval mode)
                y_pred_total = model(Xs_device) 

                # Extract truth concatenated labels
                y_true_total = torch.cat(Ys_device, dim=0) # [Total_N]

                # Filter out noise for ACC calculation proxy
                non_noise_mask = (y_true_total > 0)
                y_true_non_noise = y_true_total[non_noise_mask]
                y_pred_non_noise = y_pred_total[non_noise_mask]

                if len(y_true_non_noise) > 0:
                    all_y_true.append(y_true_non_noise.cpu().numpy())
                    all_y_pred.append(y_pred_non_noise.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_loss_history.append(avg_val_loss)

        # Calculate surrogate accuracy (Adjusted Rand Index)
        current_val_score = -1.0
        if all_y_true:
            Y_true_np = np.concatenate(all_y_true)
            Y_pred_np = np.concatenate(all_y_pred)

            # Filter out predicted noise (label 0) for ARI calculation, as it penalizes mismatch unfairly.
            # ARI is robust to noise labels (-1), but here we map noise to 0. 
            # Skip noise (0) for ARI calculation if possible, or accept performance hit.

            # Since HDBSCAN noise is usually mapped to 0, let's include it for consistency with label space, 
            # but rely on ARI robustness.

            acr_score = adjusted_rand_score(Y_true_np, Y_pred_np)
            val_acc_history.append(acr_score)
            current_val_score = acr_score

        else:
            val_acc_history.append(0.0)

        print(f"Epoch {epoch}/{epochs}: Train Loss={avg_train_loss:.5f}, Val Loss={avg_val_loss:.5f}, Val ARI={current_val_score:.4f}")

        # Early Stopping based on ARI
        if current_val_score > best_val_score:
            best_val_score = current_val_score
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 10: # Minimum epochs for stability
                print(f"Early stopping at epoch {epoch}. Best ARI: {best_val_score:.4f}")
                break

    # Load best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

