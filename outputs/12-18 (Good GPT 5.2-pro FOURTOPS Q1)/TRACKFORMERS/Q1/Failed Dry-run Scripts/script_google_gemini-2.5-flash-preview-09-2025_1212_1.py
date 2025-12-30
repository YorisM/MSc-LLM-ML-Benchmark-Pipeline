
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

# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, knn_graph
import hdbscan
import numpy as np
import sys 

# -------- (OPTIONAL) CUSTOM DATASET  --------
def make_dataset(events, pre, train: bool, **kwargs):
  # k determines the k used in kNN graph construction
  K_NEIGHBORS = kwargs.get("k", 5) 

  data_list = []

  for evt in events:
      X_raw, labels = split_X_y(evt) 

      # Preprocessing (Normalization)
      X = pre.transform(X_raw) # [N_hits, 4] Normalized features (r, theta, z, layer_id)

      # Graph Construction using 3D coordinates (r, theta, z) -- indices 0, 1, 2
      coords = X[:, 0:3] # [N_hits, 3]
      N_hits = X.size(0)

      # Build kNN graph based on spatial coordinates
      if N_hits > K_NEIGHBORS:
           # knn_graph computes edges based on features X, k
           # flow='source_to_target' is default, directed edges
           edge_index = knn_graph(coords, k=K_NEIGHBORS, loop=False, flow='source_to_target') # [2, E]
      else:
           edge_index = torch.empty((2, 0), dtype=torch.long)

      # Create PyG Data object
      data = Data(x=X.float(), edge_index=edge_index.long(), y=labels.long()) 
      data_list.append(data)

  return data_list

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit()
    #   - transform()

    def __init__(self):
        self.mean = None
        self.std = None
        self.k = 5 # k parameter for kNN graph construction

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "llm_script:make_dataset",
            "dataset_kwargs": {"k": self.k},

            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 16, # Optimized for PyG batching and memory
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": "identity",  # PyG DataLoader handles batching

            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, 4]
        X_all = torch.cat(Xs, dim=0) # [N_total, 4]

        self.mean = X_all.mean(dim=0).cpu().numpy()
        self.std = X_all.std(dim=0).cpu().numpy()

        # Prevent division by zero
        self.std[self.std < 1e-6] = 1.0

        self.mean = torch.from_numpy(self.mean).float() # [4]
        self.std = torch.from_numpy(self.std).float()   # [4]

        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, 4]

        X = (X - self.mean.to(X.device)) / self.std.to(X.device) # [N_hits, 4]
        return X # MUST return torch.FloatTensor [N_hits, F_out] for the default EventDataset path.

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        F_in = example_batch_x.x.shape[1] # Expected 4
        D_hidden = 64
        D_emb = 8
        self.D_emb = D_emb

        # Initial projection MLP
        self.mlp_in = nn.Sequential(
            nn.Linear(F_in, D_hidden), # [N, 4] -> [N, 64]
            nn.LeakyReLU(),
            nn.LayerNorm(D_hidden)
        )

        # GNN layers (3 layers of GCNConv)
        self.conv1 = GCNConv(D_hidden, D_hidden)
        self.conv2 = GCNConv(D_hidden, D_hidden)
        self.conv3 = GCNConv(D_hidden, D_hidden)

        # Output embedding layer (using concatenation of features for enriched representation)
        self.output_mlp = nn.Sequential(
            nn.Linear(D_hidden * 3, D_hidden), # [N, 192] -> [N, 64]
            nn.LeakyReLU(),
            nn.Linear(D_hidden, D_emb)         # [N, 64] -> [N, 8]
        )

    def forward(self, batch_x):
        # Input batch_x is a torch_geometric.data.Batch object
        x = batch_x.x.to(device) # Features [N_total, F]
        edge_index = batch_x.edge_index.to(device) # [2, E]
        batch_idx = batch_x.batch.to(device) # [N_total] index mapping nodes to graph batches

        # 1. Input MLP
        h = self.mlp_in(x) # [N_total, 64]

        # 2. GNN layers
        h1 = F.leaky_relu(self.conv1(h, edge_index)) # [N_total, 64]
        h2 = F.leaky_relu(self.conv2(h1, edge_index)) # [N_total, 64]
        h3 = F.leaky_relu(self.conv3(h2, edge_index)) # [N_total, 64]

        # Concatenate outputs (simple skip connection mechanism)
        h_concat = torch.cat([h1, h2, h3], dim=1) # [N_total, 192]

        # 3. Output embeddings
        embeddings = self.output_mlp(h_concat) # [N_total, D_emb=8]

        if self.training:
            # During training, return embeddings and batch index for explicit loss calculation
            return embeddings, batch_idx

        else:
            # Inference: Apply HDBSCAN clustering event by event

            predictions = []

            # Iterate over events using the batch index helper from PyG
            unique_batches = torch_geometric.utils.scatter.scatter_max(batch_idx)[0] + 1

            # Since PyG Batch nodes are ordered by graph, we can use cumulative sum of nodes (ptr)
            # or iterate over the indices defined by the cluster map `batch_idx`

            start_index = 0

            for i in range(unique_batches.max().item() + 1):
                mask = (batch_idx == i)
                if not mask.any():
                    continue

                E_event = embeddings[mask] # [N_i, D_emb]
                N_hits = E_event.size(0)

                # Minimum track length for FitAccuracy is 4
                if N_hits < 4: 
                    labels = torch.full((N_hits,), -1, dtype=torch.long, device=device)
                else:
                    # HDBSCAN clustering on CPU
                    E_np = E_event.detach().cpu().numpy()

                    min_cluster_size = 4

                    try:
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=min_cluster_size, 
                            min_samples=1, 
                            metric='euclidean', 
                            allow_single_cluster=True,
                            core_dist_n_jobs=4
                        )
                        clusterer.fit(E_np)
                        raw_labels = clusterer.labels_ # -1 for noise, 0, 1, 2... for clusters

                        # Map labels to 1-based indexing for cluster IDs (>0) and keep noise -1
                        unique_ids = np.unique(raw_labels[raw_labels != -1])
                        mapping = {orig_id: new_id + 1 for new_id, orig_id in enumerate(unique_ids)}

                        mapped_labels = np.array([mapping.get(label, -1) for label in raw_labels], dtype=np.int64)

                        labels = torch.from_numpy(mapped_labels).to(device)

                    except Exception as e:
                        # Fallback to noise assignment if clustering fails
                        # print(f"HDBSCAN failed: {e}", file=sys.stderr)
                        labels = torch.full((N_hits,), -1, dtype=torch.long, device=device)

                predictions.append(labels)
                start_index += N_hits

            # Return concatenated labels for all hits in the batch [N_total]
            return torch.cat(predictions, dim=0) 

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 40    
LR = 1e-3

def contrastive_loss(embeddings, labels, batch, margin=1.0):
    # Calculation of Contrastive Loss for Metric Learning on PyG Batches
    N = embeddings.size(0)

    # Calculate squared Euclidean distances
    distance_matrix = torch.cdist(embeddings, embeddings, p=2).pow(2) # [N, N]

    # Create masks for comparison
    labels_expanded = labels.unsqueeze(1).expand(N, N)
    batch_expanded = batch.unsqueeze(1).expand(N, N)

    M_Y = (labels_expanded == labels_expanded.T) # Same Track ID
    M_B = (batch_expanded == batch_expanded.T) # Same Event
    M_diag = torch.eye(N, dtype=torch.bool, device=embeddings.device).logical_not() 

    # --- Positive Pairs (P): Same track (ID > 0) and same event
    M_P = M_Y & M_B & M_diag
    M_P_valid = M_P & (labels_expanded > 0)

    distances_P = distance_matrix[M_P_valid]
    L_P = distances_P.mean() if M_P_valid.any() else torch.tensor(0.0, device=embeddings.device)

    # --- Negative Pairs (N): Different track ID OR Noise vs Track, and same event
    M_N = (~M_Y) & M_B & M_diag

    distances_N = distance_matrix[M_N]

    # Maximize margin (M) by penalizing close negative pairs
    L_N = torch.relu(margin - distances_N).pow(2).mean() if M_N.any() else torch.tensor(0.0, device=embeddings.device)

    L_total = L_P + L_N

    return L_total

def train_model(model, train_loader, val_loader, epochs):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], [] # Proxy metrics

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10 

    def evaluate_loss(model, loader):
        model.train() # Keep model in training mode for consistent embedding output during evaluation
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)

                # Returns embeddings and batch index
                embeddings, batch_idx = model(batch) 

                # Compute Loss
                loss = contrastive_loss(embeddings, batch.y.to(device), batch_idx, margin=1.0)
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        # Proxy Accuracy based on normalized loss
        proxy_acc = max(0.0, 1.0 - (avg_loss / 1.0))

        return avg_loss, proxy_acc

    # Main training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            # Forward pass (returns embeddings and batch index)
            embeddings, batch_idx = model(batch)

            loss = contrastive_loss(embeddings, batch.y.to(device), batch_idx, margin=1.0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(max(0.0, 1.0 - (avg_train_loss / 1.0)))

        # Validation step relies on loss calculation (HDBSCAN is too slow to run every epoch)
        val_loss, val_acc_proxy = evaluate_loss(model, val_loader)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc_proxy)

        print(f"Epoch {epoch}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Proxy Acc: {val_acc_proxy:.4f}, LR: {optimizer.param_groups[0]['lr']:.6e}")

        # Learning Rate scheduler step
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break

    # Load best model state before returning
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)

    # Ensure model is back in evaluation mode for inference later (although harness handles setting model.eval())
    model.eval()
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history
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

