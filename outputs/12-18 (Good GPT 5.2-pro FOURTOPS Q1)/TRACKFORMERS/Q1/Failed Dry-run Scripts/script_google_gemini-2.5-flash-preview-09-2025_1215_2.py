
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
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import hdbscan

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

    def __init__(self):
        self.scaler_r = StandardScaler()
        self.scaler_z = StandardScaler()
        self.fitted = False

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4, 
            "pin_memory": True,

            "collate": "ragged_xy",

            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, F_raw=4]

        R_list = []
        Z_list = []

        for X in Xs:
            X_np = X.numpy()
            R_list.append(X_np[:, 0]) # hit_r
            Z_list.append(X_np[:, 2]) # hit_z

        R_all = np.concatenate(R_list).reshape(-1, 1) # [N_total, 1]
        Z_all = np.concatenate(Z_list).reshape(-1, 1) # [N_total, 1]

        self.scaler_r.fit(R_all)
        self.scaler_z.fit(Z_all)
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, F_raw=4]
        N_hits = X.shape[0]
        X_np = X.numpy() 

        # 0: R, 1: Theta, 2: Z, 3: Layer_ID

        # Scale R and Z
        R_scaled = self.scaler_r.transform(X_np[:, 0].reshape(-1, 1)).flatten() # [N_hits]
        Z_scaled = self.scaler_z.transform(X_np[:, 2].reshape(-1, 1)).flatten() # [N_hits]

        # Handle Theta (cyclical)
        Theta = X_np[:, 1] # [N_hits]
        SinT = np.sin(Theta) # [N_hits]
        CosT = np.cos(Theta) # [N_hits]

        # Layer ID normalization 
        Layer_ID = X_np[:, 3] / 10.0 # [N_hits]

        # Feature vector: [R_scaled, SinT, CosT, Z_scaled, Layer_ID]
        X_transformed = np.column_stack([
            R_scaled, SinT, CosT, Z_scaled, Layer_ID
        ]).astype(np.float32) # [N_hits, 5]

        return torch.from_numpy(X_transformed) # MUST return torch.FloatTensor [N_hits, 5]

def make_preprocessor():
    return MyPreprocessor()

# Constants for Discriminative Loss
DISTANCE_MARGIN = 0.5 
VARIANCE_BANDWIDTH = 0.1 

def discriminative_loss_event(E: torch.Tensor, Y: torch.Tensor, delta_v=VARIANCE_BANDWIDTH, delta_d=DISTANCE_MARGIN):
    # E: [N_hits, D_emb] embeddings
    # Y: [N_hits] true track IDs (0 is noise, > 0 are tracks)

    Y = Y.to(E.device)

    # 1. Separate noise (Track ID 0)
    non_noise_mask = Y > 0
    E_track = E[non_noise_mask]      # [N_track_hits, D_emb]
    Y_track = Y[non_noise_mask]      # [N_track_hits]

    if E_track.shape[0] == 0:
        return torch.tensor(0.0, device=E.device)

    unique_tracks = torch.unique(Y_track)
    N_tracks = len(unique_tracks)

    L_var = torch.tensor(0.0, device=E.device)
    C_list = [] # List of centroids

    # 2. Variance Loss (Attractive Term)
    # L_var = (1/N_tracks) * sum_k { (1/N_k) * sum_i in T_k { max(0, ||E_i - C_k|| - delta_v)^2 } }

    for k in unique_tracks:
        track_mask = (Y_track == k)
        E_k = E_track[track_mask]

        # Calculate centroid C_k
        C_k = torch.mean(E_k, dim=0) # [D_emb]
        C_list.append(C_k)

        # Calculate intra-cluster distance
        dist_sq = torch.sum((E_k - C_k)**2, dim=1) # [N_k]
        dist = torch.sqrt(dist_sq)

        # Variance term: max(0, dist - delta_v)**2
        var_loss_k = torch.mean(torch.pow(torch.clamp(dist - delta_v, min=0.0), 2))
        L_var += var_loss_k

    L_var /= N_tracks

    # 3. Distance Loss (Repulsive Term)
    # L_dist = (1/N_pairs) * sum_{k!=l} { max(0, delta_d - ||C_k - C_l||)^2 }
    L_dist = torch.tensor(0.0, device=E.device)
    if N_tracks > 1:
        C = torch.stack(C_list) # [N_tracks, D_emb]

        # Calculate pairwise L2 distances between centroids
        D_pairs = torch.cdist(C, C, p=2) 

        # Extract upper triangle (excluding diagonal)
        triu_indices = torch.triu_indices(N_tracks, N_tracks, offset=1)
        D_pairs_unique = D_pairs[triu_indices[0], triu_indices[1]] # [N_pairs]

        N_pairs = D_pairs_unique.shape[0]

        # Distance term: max(0, delta_d - D_kl)**2
        dist_loss_pairs = torch.pow(torch.clamp(delta_d - D_pairs_unique, min=0.0), 2)
        L_dist = torch.sum(dist_loss_pairs) / N_pairs

    # Total loss: L = L_var + W_dist * L_dist
    W_dist = 1.5 
    L_total = L_var + W_dist * L_dist

    return L_total

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Define and initialize any stateful components here
        if isinstance(example_batch_x, list):
            F_in = example_batch_x[0].shape[-1]
        else:
            F_in = example_batch_x.shape[-1]

        D_emb = 8 # Embedding dimension [N_hits, 8]
        H = 64 # Hidden dimension

        self.F_in = F_in
        self.D_emb = D_emb

        self.mlp = nn.Sequential(
            nn.Linear(F_in, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Linear(H, D_emb)
        )

        # HDBSCAN parameters
        self.hdbscan_params = {
            'min_cluster_size': 4,
            'cluster_selection_epsilon': 0.1, 
            'min_samples': 1,
            'allow_single_cluster': True,
            'metric': 'euclidean'
        }

    def process_embeddings(self, X):
        # X: [N_hits, F_in]
        E = self.mlp(X) # E: [N_hits, D_emb]
        return E

    def cluster_embeddings(self, E):
        # E: [N_hits, D_emb]. Assumed to be on CPU if using HDBSCAN.

        E_np = E.cpu().numpy()
        N_hits = E_np.shape[0]

        # Handle small events
        if N_hits < self.hdbscan_params['min_cluster_size']:
             return np.full(N_hits, -1, dtype=np.int64) 

        clusterer = hdbscan.HDBSCAN(**self.hdbscan_params)
        labels = clusterer.fit_predict(E_np) # labels: [N_hits] (-1 for noise, 0, 1, 2... for clusters)

        # Relabel clusters to ensure IDs start at 1, keeping noise at -1.
        noise_mask = (labels == -1)

        unique_pos_labels = np.unique(labels[~noise_mask])

        predicted_labels = np.full(N_hits, -1, dtype=np.int64)

        if len(unique_pos_labels) > 0:
            label_map = {old: new + 1 for new, old in enumerate(unique_pos_labels)}

            # Apply mapping only to non-noise cluster indices
            for old_id, new_id in label_map.items():
                predicted_labels[labels == old_id] = new_id

        return predicted_labels # [N_hits], numpy int64 array

    def forward(self, batch_x):
        # batch_x is ragged list[Tensor], one per event, each shaped [N_hits, F].

        # Define your model's forward pass here
        if self.training:
            # Training mode: return list of embeddings [N_i, D_emb]
            return [self.process_embeddings(X.to(device)) for X in batch_x]

        else: # Evaluation mode: Apply clustering and return labels
            predicted_labels_list = []
            with torch.no_grad():
                for X in batch_x:
                    X = X.to(device)
                    E = self.process_embeddings(X) # [N_i, D_emb]

                    labels_np = self.cluster_embeddings(E) # [N_i]

                    # Convert back to torch tensor, dtype int64
                    predicted_labels_list.append(torch.from_numpy(labels_np).long())

                return predicted_labels_list

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50   
def train_model(model, train_loader, val_loader, epochs):

    device = next(model.parameters()).device

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 10

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        n_train_events = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # E_list: list of [N_i, D_emb]
            E_list = model(batch_x) 

            batch_loss = 0.0

            # Calculate discriminative loss event by event
            for E, Y in zip(E_list, batch_y):
                Y = Y.to(device)

                loss_i = discriminative_loss_event(E, Y)
                batch_loss += loss_i

            if len(batch_x) > 0:
                batch_loss /= len(batch_x) 

            batch_loss.backward()
            optimizer.step()

            total_train_loss += batch_loss.item() * len(batch_x)
            n_train_events += len(batch_x)

        avg_train_loss = total_train_loss / n_train_events
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        n_val_events = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                E_list = model(batch_x)

                batch_loss = 0.0
                for E, Y in zip(E_list, batch_y):
                    Y = Y.to(device)
                    loss_i = discriminative_loss_event(E, Y)
                    batch_loss += loss_i

                if len(batch_x) > 0:
                    batch_loss /= len(batch_x)

                total_val_loss += batch_loss.item() * len(batch_x)
                n_val_events += len(batch_x)

        avg_val_loss = total_val_loss / n_val_events
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}, LR={optimizer.param_groups[0]['lr']:.2e}")

        # Early Stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered at Epoch {epoch+1}.")
                break

    n_trained_epochs = len(train_losses)
    train_acc = [0.0] * n_trained_epochs # Dummy accuracy placeholder
    val_acc = [0.0] * n_trained_epochs # Dummy accuracy placeholder

    return model, train_losses, val_losses, train_acc, val_acc

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

