
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

import math
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import numpy as np
import hdbscan

# -------- (OPTIONAL) CUSTOM DATASET  --------
# <LLM: Insert custom dataset logic here>

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit()
    #   - transform()

    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        self.scaler = None
        self.feature_names = ["hit_r", "hit_theta", "hit_z", "layer_id"]


    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 16, # Adjusted batch size for variable length/Transformer processing
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": "ragged_xy",  # Use ragged collate

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, data: list[torch.Tensor]):
        # <LLM: Extract statistics or fit transform>
        # data is a list of [N_i, 4] tensors (X)

        # Concatenate all data for standardization
        all_data_list = []
        for x in data:
            if x.numel() > 0:
                all_data_list.append(x)

        if not all_data_list: # Handle empty dataset case
             return self

        all_data = torch.cat(all_data_list, dim=0).numpy() # (N_total, 4)

        # Features to scale: r, z, layer_id (indices 0, 2, 3)
        relevant_features = all_data[:, [0, 2, 3]] 

        scaler = StandardScaler()
        scaler.fit(relevant_features)
        self.scaler = scaler

        return self

    def transform(self, data: list[torch.Tensor]) -> list[torch.Tensor]:
        # <LLM: Apply preprocessing logic, return torch.Tensor>
        transformed_events = []

        if self.scaler is None:
             return data

        for X in data: # X is [N_hits, 4]
            if X.shape[0] == 0:
                transformed_events.append(X.clone())
                continue

            X_np = X.numpy()

            # 1. Scale non-angular features (r, z, layer_id)
            X_scaled_part = self.scaler.transform(X_np[:, [0, 2, 3]]) # [N_hits, 3]

            # 2. Handle theta (index 1) using sin/cos transformation
            theta = X_np[:, 1] # [N_hits]
            sin_theta = np.sin(theta)[:, None] # [N_hits, 1]
            cos_theta = np.cos(theta)[:, None] # [N_hits, 1]

            # New features: (r_norm, z_norm, layer_id_norm, cos_theta, sin_theta) -> 5 features
            X_transformed = np.concatenate([
                X_scaled_part[:, 0:1], # r_norm [N_hits, 1]
                X_scaled_part[:, 1:2], # z_norm [N_hits, 1]
                X_scaled_part[:, 2:3], # layer_id_norm [N_hits, 1] 
                cos_theta,             # cos_theta [N_hits, 1]
                sin_theta,             # sin_theta [N_hits, 1]
            ], axis=1) # [N_hits, 5]

            # Append features as torch tensor
            transformed_events.append(torch.tensor(X_transformed, dtype=torch.float32))

        return transformed_events # must return an indexable, picklable object

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()

        # <LLM: Define and initialize any stateful components here>
        if isinstance(example_batch_x, list) and len(example_batch_x) > 0:
            F = example_batch_x[0].size(1) # F=5 (preprocessed features)
        elif isinstance(example_batch_x, torch.Tensor):
             F = example_batch_x.size(-1) 
        else:
             F = 5 

        D_MODEL = 64
        N_LAYERS = 3
        N_HEADS = 4
        D_CLUSTER = 4 # Target embedding dimension for clustering

        self.D_CLUSTER = D_CLUSTER

        # Initial Embedding Layer
        self.embedding = nn.Sequential(
            nn.Linear(F, D_MODEL), # [N_hits, F] -> [N_hits, D_MODEL=64]
            nn.GELU(),
            nn.LayerNorm(D_MODEL)
        )

        # Transformer Encoder Block (Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, 
            nhead=N_HEADS, 
            dim_feedforward=2*D_MODEL, 
            dropout=0.1, 
            batch_first=True,
            device=device
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        # Output Head: Mapping to final clustering space
        self.output_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.GELU(),
            nn.Linear(D_MODEL // 2, D_CLUSTER), # [N_hits, D_MODEL] -> [N_hits, D_CLUSTER=4]
        )

        # HDBSCAN parameters (4 hits minimum required per FitAccuracy definition)
        self.hdbscan_params = {
            'min_cluster_size': 4,
            'min_samples': 1,
            'metric': 'euclidean',
            'cluster_selection_method': 'eom',
        }

    def _process_batch(self, batch_x: list[torch.Tensor]):
         # Helper routine to compute embeddings from ragged batch

        event_lengths = [x.size(0) for x in batch_x]

        if not event_lengths or max(event_lengths) == 0:
            # Return empty lists if batch is entirely empty events
            return [torch.tensor([], dtype=torch.float32, device=device)] * len(batch_x), event_lengths, None

        max_len = max(event_lengths)

        # 1. Pad sequences and create mask
        # Pad the list of tensors [N_i, F] -> [B, Max_N, F]
        padded_x = pad_sequence(batch_x, batch_first=True).to(device) # [B, Max_N, F]

        # src_key_padding_mask: [B, Max_N], Padded indices are True.
        mask = torch.arange(max_len, device=device).unsqueeze(0) < torch.tensor(event_lengths, device=device).unsqueeze(1)
        src_key_padding_mask = ~mask

        # 2. Embedding
        h = self.embedding(padded_x) # [B, Max_N, D_MODEL]

        # 3. Transformer Encoder
        h_enc = self.transformer_encoder(
            src=h, 
            src_key_padding_mask=src_key_padding_mask
        ) # [B, Max_N, D_MODEL]

        # 4. Output Head (Embeddings for Clustering)
        embeddings_padded = self.output_head(h_enc) # [B, Max_N, D_CLUSTER]

        # 5. Unpack/Unpad results
        embeddings_list = []
        for i, N in enumerate(event_lengths):
            if N > 0:
                embeddings_list.append(embeddings_padded[i, :N, :]) # [N_i, D_CLUSTER]
            else:
                embeddings_list.append(None)

        return embeddings_list, event_lengths, src_key_padding_mask


    def forward(self, batch_x):
        # <LLM: Define your model's forward pass here>

        embeddings_list, event_lengths, _ = self._process_batch(batch_x)

        if not event_lengths:
             return [torch.tensor([], dtype=torch.int64, device=device)] * len(batch_x)

        predicted_labels_list = []

        for N_h, E in zip(event_lengths, embeddings_list):

            if N_h == 0 or E is None:
                predicted_labels_list.append(torch.tensor([], dtype=torch.int64, device=device))
                continue

            # Move data to CPU and convert to NumPy for HDBSCAN
            E_np = E.detach().cpu().numpy()

            if N_h < self.hdbscan_params['min_cluster_size']:
                labels = np.full(N_h, -1, dtype=np.int64)
            else:
                try:
                    # Run HDBSCAN (CPU operation)
                    clusterer = hdbscan.HDBSCAN(**self.hdbscan_params)
                    clusterer.fit(E_np)
                    labels = clusterer.labels_.astype(np.int64)
                except Exception:
                    labels = np.full(N_h, -1, dtype=np.int64)

            # Map HDBSCAN labels: 
            # -1 (Noise) -> -1 (Predicted Noise)
            # >= 0 (Cluster IDs) -> >= 1 (Predicted Track IDs)

            labels[labels >= 0] += 1

            predicted_labels_list.append(torch.tensor(labels, dtype=torch.int64, device=device))

        return predicted_labels_list 

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 30   

# Custom loss function for Metric Learning (Contrastive Loss)
def contrastive_loss(embeddings, true_labels, margin=1.0):
    # Filter out noise hits (track_id == 0) for loss calculation
    non_noise_mask = true_labels > 0 # [N]

    if non_noise_mask.sum() < 2:
        return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

    E = embeddings[non_noise_mask] # [N_non_noise, D]
    Y = true_labels[non_noise_mask] # [N_non_noise]
    N = E.size(0)

    # Calculate squared Euclidean distance matrix
    E_sq = torch.sum(E**2, dim=1, keepdim=True) # [N, 1]
    D_sq = E_sq + E_sq.T - 2 * (E @ E.T) # [N, N]
    D_sq = torch.clamp(D_sq, min=1e-8) 

    Y_matrix = (Y.unsqueeze(0) == Y.unsqueeze(1)) # Positive pairs mask [N, N]
    identity_mask = torch.eye(N, dtype=torch.bool, device=D_sq.device)
    Y_matrix[identity_mask] = False 

    positive_mask = Y_matrix 
    negative_mask = ~Y_matrix 

    # Attractive Loss (Pull positive pairs together)
    L_attractive = D_sq[positive_mask].mean() if positive_mask.any() else torch.tensor(0.0, device=D_sq.device)

    # Repulsive Loss (Push negative pairs apart)
    margin_sq = margin * margin
    D_neg = D_sq[negative_mask]

    L_repulsive = torch.relu(margin_sq - D_neg).mean() if negative_mask.any() else torch.tensor(0.0, device=D_sq.device)

    return L_attractive + L_repulsive

def compute_accuracy_proxy(embeddings, true_labels):
    # Proxy metric measuring separation quality

    non_noise_mask = true_labels > 0
    if non_noise_mask.sum() < 2:
         return 1.0

    E = embeddings[non_noise_mask]
    Y = true_labels[non_noise_mask]
    N = E.size(0)

    E_sq = torch.sum(E**2, dim=1, keepdim=True)
    D = torch.sqrt(torch.clamp(E_sq + E_sq.T - 2 * (E @ E.T), min=1e-8)) # Distance matrix [N, N]

    Y_matrix = (Y.unsqueeze(0) == Y.unsqueeze(1))
    identity_mask = torch.eye(N, dtype=torch.bool, device=D.device)
    Y_matrix[identity_mask] = False 

    positive_mask = Y_matrix 
    negative_mask = ~Y_matrix 

    pos_distances = D[positive_mask]
    neg_distances = D[negative_mask]

    T_P = 0.5   
    T_N = 1.0   

    P_total = pos_distances.numel()
    N_total = neg_distances.numel()

    metric_sum = 0
    count = 0 

    if P_total > 0:
        P_correct = (pos_distances < T_P).float().sum()
        metric_sum += P_correct / P_total
        count += 1

    if N_total > 0:
        N_correct = (neg_distances > T_N).float().sum()
        metric_sum += N_correct / N_total
        count += 1

    return metric_sum / count if count > 0 else 0.0

def train_model(model, train_loader, val_loader, epochs):

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3
    )

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5

    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        train_event_count = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            embeddings_list, _, _ = model._process_batch(batch_x)

            # Check if any event in the batch had hits
            if not embeddings_list or all(e is None for e in embeddings_list):
                 continue

            current_loss = 0.0
            current_acc = 0.0
            event_count = 0

            for E, Y_true in zip(embeddings_list, batch_y):
                if E is None: continue

                Y_true = Y_true.to(device) # [N_i]

                loss = contrastive_loss(E, Y_true, margin=1.0)
                current_loss += loss

                acc = compute_accuracy_proxy(E.detach(), Y_true)
                current_acc += acc
                event_count += 1

            if event_count > 0:
                current_loss /= event_count
                current_acc /= event_count

                current_loss.backward()
                optimizer.step()

                total_train_loss += current_loss.item()
                total_train_acc += current_acc
                train_event_count += event_count

        if train_event_count > 0:
            avg_train_loss = total_train_loss / train_event_count
            avg_train_acc = total_train_acc / train_event_count
            train_loss_list.append(avg_train_loss)
            train_acc_list.append(avg_train_acc)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        val_event_count = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:

                embeddings_list, _, _ = model._process_batch(batch_x)

                if not embeddings_list or all(e is None for e in embeddings_list):
                    continue

                current_loss = 0.0
                current_acc = 0.0
                event_count = 0

                for E, Y_true in zip(embeddings_list, batch_y):
                    if E is None: continue

                    Y_true = Y_true.to(device)

                    loss = contrastive_loss(E, Y_true, margin=1.0)
                    current_loss += loss

                    acc = compute_accuracy_proxy(E, Y_true)
                    current_acc += acc
                    event_count += 1

                if event_count > 0:
                    total_val_loss += (current_loss.item() / event_count)
                    total_val_acc += (current_acc / event_count)
                    val_event_count += event_count

        if val_event_count > 0:
            avg_val_loss = total_val_loss / val_event_count
            avg_val_acc = total_val_acc / val_event_count
            val_loss_list.append(avg_val_loss)
            val_acc_list.append(avg_val_acc)

            scheduler.step(avg_val_loss)

            # Early stopping check based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_weights = model.state_dict()
            else:
                patience_counter += 1

        if patience_counter >= max_patience:
            break

    # Load best weights before returning
    if 'best_model_weights' in locals():
        model.load_state_dict(best_model_weights)

    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

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

