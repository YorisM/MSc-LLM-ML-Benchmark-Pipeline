
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import detect_and_assert_lane, assert_label_output_by_lane, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, build_trackformers_model

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_stats = None

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
            "eval_overrides": {"shuffle": False, "batch_size": 32}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, F_raw]
        all_features = []
        for X in Xs:
            if isinstance(X, torch.Tensor):
                X = X.numpy()
            # Convert cylindrical to cartesian for better spatial understanding
            r, theta, z = X[:, 0], X[:, 1], X[:, 2]
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            # Use cartesian coordinates + layer_id
            features = np.column_stack([x, y, z, X[:, 3]])
            all_features.append(features)

        all_features = np.vstack(all_features)
        self.scaler.fit(all_features)

        # Layer statistics
        layer_vals = np.concatenate([X[:, 3] for X in Xs])
        self.layer_stats = {
            'min': float(layer_vals.min()),
            'max': float(layer_vals.max()),
            'num_unique': int(len(np.unique(layer_vals)))
        }
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, F_raw]
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X

        r, theta, z, layer = X_np[:, 0], X_np[:, 1], X_np[:, 2], X_np[:, 3]

        # Convert to cartesian coordinates
        x = r * np.cos(theta)  # [N_hits]
        y = r * np.sin(theta)  # [N_hits]

        # Create enhanced features: cartesian + cylindrical + layer info
        features = np.column_stack([
            x, y, z,           # Cartesian coordinates
            r, theta,          # Cylindrical coordinates
            layer              # Layer ID
        ])

        # Scale features
        features = self.scaler.transform(features)

        return torch.FloatTensor(features)  # [N_hits, 6]

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm2(x)
        return x

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Determine input dimension from example batch
        if isinstance(example_batch_x, list):
            input_dim = example_batch_x[0].shape[1]
        else:
            input_dim = example_batch_x.x.shape[1]

        hidden_dim = 256
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Transformer encoder for hit interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Residual blocks for deeper processing
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=0.1) for _ in range(4)
        ])

        # Output projection for clustering
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )

        # Final clustering head
        self.cluster_head = nn.Linear(hidden_dim // 4, 128)  # Embedding dimension

    def forward(self, batch_x):
        # Handle both ragged tensor and PyG formats
        if isinstance(batch_x, list):
            # Ragged tensor format
            embeddings = []
            for x in batch_x:
                x = x.to(device)
                # Project input
                h = self.input_proj(x)  # [N_i, hidden_dim]

                # Add positional encoding based on spatial coordinates
                pos_enc = self._get_positional_encoding(x)
                h = h + pos_enc

                # Transformer encoding with attention mask
                attn_mask = self._get_causal_mask(x.shape[0]).to(x.device)
                h = self.transformer(h, mask=attn_mask)  # [N_i, hidden_dim]

                # Residual blocks
                for block in self.res_blocks:
                    h = block(h)

                # Project to embedding space
                h = self.output_proj(h)  # [N_i, hidden_dim//4]
                h = self.cluster_head(h)  # [N_i, 128]
                embeddings.append(h)
            return embeddings
        else:
            # PyG format
            x = batch_x.x.to(device)
            batch = batch_x.batch.to(device)

            # Process each graph separately
            unique_batches = torch.unique(batch)
            all_embeddings = []

            for b in unique_batches:
                mask = (batch == b)
                x_batch = x[mask]

                # Project input
                h = self.input_proj(x_batch)  # [N_i, hidden_dim]

                # Add positional encoding
                pos_enc = self._get_positional_encoding(x_batch)
                h = h + pos_enc

                # Transformer encoding
                attn_mask = self._get_causal_mask(x_batch.shape[0]).to(x.device)
                h = self.transformer(h, mask=attn_mask)

                # Residual blocks
                for block in self.res_blocks:
                    h = block(h)

                # Project to embedding space
                h = self.output_proj(h)
                h = self.cluster_head(h)
                all_embeddings.append(h)

            # Reconstruct full tensor
            embeddings = torch.zeros(x.shape[0], 128, device=device)
            for b, emb in zip(unique_batches, all_embeddings):
                mask = (batch == b)
                embeddings[mask] = emb
            return embeddings

    def _get_positional_encoding(self, x):
        # Create sinusoidal positional encoding based on spatial coordinates
        n_points = x.shape[0]
        dim = 256  # hidden_dim

        # Use first 3 features (x, y, z) for position
        if x.shape[1] >= 3:
            pos = x[:, :3]
        else:
            pos = x

        # Create sinusoidal encodings
        position = torch.arange(0, n_points, dtype=torch.float32, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) * 
                            -(np.log(10000.0) / dim))

        pe = torch.zeros(n_points, dim, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def _get_causal_mask(self, seq_len):
        # Create upper triangular mask for causal attention
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

    def predict_labels(self, batch_x):
        # Get embeddings
        if isinstance(batch_x, list):
            # Ragged tensor format
            embeddings_list = self.forward(batch_x)
            labels_list = []

            for i, embeddings in enumerate(embeddings_list):
                if embeddings.shape[0] < 4:
                    # Too few hits for meaningful clustering
                    labels_list.append(torch.full((embeddings.shape[0],), -1, 
                                                dtype=torch.int64, device=device))
                    continue

                # Use DBSCAN-like clustering on embeddings
                labels = self._cluster_embeddings(embeddings)
                labels_list.append(labels)

            return labels_list
        else:
            # PyG format
            embeddings = self.forward(batch_x)
            batch = batch_x.batch.to(device)
            unique_batches = torch.unique(batch)

            all_labels = torch.full((embeddings.shape[0],), -1, 
                                  dtype=torch.int64, device=device)

            for b in unique_batches:
                mask = (batch == b)
                emb_batch = embeddings[mask]

                if emb_batch.shape[0] < 4:
                    all_labels[mask] = -1
                    continue

                labels = self._cluster_embeddings(emb_batch)
                all_labels[mask] = labels

            return all_labels

    def _cluster_embeddings(self, embeddings):
        # Move to CPU for clustering
        emb_cpu = embeddings.detach().cpu().numpy()
        n_points = emb_cpu.shape[0]

        if n_points < 4:
            return torch.full((n_points,), -1, dtype=torch.int64, device=device)

        # Adaptive clustering based on local density
        # Compute pairwise distances
        distances = cdist(emb_cpu, emb_cpu, metric='euclidean')

        # Find adaptive epsilon based on k-nearest neighbor distances
        k = min(10, n_points - 1)
        k_distances = np.sort(distances, axis=1)[:, k]
        eps = np.percentile(k_distances, 50)  # Median of k-distances

        # Simple connected components with adaptive threshold
        visited = np.zeros(n_points, dtype=bool)
        labels = np.full(n_points, -1, dtype=int)
        label_counter = 0

        for i in range(n_points):
            if not visited[i]:
                # Find neighbors within eps
                neighbors = np.where(distances[i] < eps)[0]

                if len(neighbors) >= 4:  # Minimum cluster size
                    # Expand cluster
                    stack = list(neighbors)
                    cluster_points = []

                    while stack:
                        point = stack.pop()
                        if not visited[point]:
                            visited[point] = True
                            cluster_points.append(point)

                            # Find neighbors of this point
                            new_neighbors = np.where(distances[point] < eps)[0]
                            for nb in new_neighbors:
                                if not visited[nb]:
                                    stack.append(nb)

                    if len(cluster_points) >= 4:
                        labels[cluster_points] = label_counter
                        label_counter += 1
                else:
                    visited[i] = True  # Mark as noise

        return torch.from_numpy(labels).to(device)

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )

    # Loss function - contrastive loss for clustering
    def contrastive_loss(embeddings, labels, margin=1.0):
        """Compute contrastive loss for clustering."""
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.t())  # [B, B]

        # Create mask for positive pairs (same label)
        label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        positive_mask = label_mask - torch.eye(batch_size, device=device)

        # Positive loss
        pos_loss = (1 - similarity) * positive_mask
        pos_loss = torch.sum(pos_loss) / (torch.sum(positive_mask) + 1e-8)

        # Negative loss
        negative_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
        neg_loss = torch.clamp(similarity - margin, min=0) * negative_mask
        neg_loss = torch.sum(neg_loss) / (torch.sum(negative_mask) + 1e-8)

        return pos_loss + neg_loss

    best_val_acc = 0
    patience = 10
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        batch_count = 0

        for xs, ys in train_loader:
            Xs = [x.to(device) for x in xs]
            ys_list = [y.to(device) for y in ys]

            optimizer.zero_grad()

            # Get embeddings for each event in batch
            embeddings_list = model(Xs)

            # Compute loss for each event
            batch_loss = 0
            batch_acc = 0
            event_count = 0

            for i, embeddings in enumerate(embeddings_list):
                if embeddings.shape[0] < 2:
                    continue

                labels = ys_list[i]
                # Remove noise hits (label 0) for training
                mask = labels != 0
                if mask.sum() < 2:
                    continue

                emb_filtered = embeddings[mask]
                labels_filtered = labels[mask]

                # Remap labels to 0..N for contrastive loss
                unique_labels = torch.unique(labels_filtered)
                label_map = {old: new for new, old in enumerate(unique_labels)}
                labels_mapped = torch.tensor([label_map[l.item()] for l in labels_filtered], 
                                           device=device)

                loss = contrastive_loss(emb_filtered, labels_mapped)
                batch_loss += loss

                # Compute accuracy (clustering purity)
                with torch.no_grad():
                    pred_labels = model._cluster_embeddings(emb_filtered)
                    if pred_labels.max() >= 0:
                        # Compute purity
                        pred_unique = torch.unique(pred_labels[pred_labels >= 0])
                        purity_sum = 0
                        for p_label in pred_unique:
                            mask_p = (pred_labels == p_label)
                            true_labels = labels_filtered[mask_p]
                            if len(true_labels) > 0:
                                majority = true_labels.mode().values[0]
                                purity = (true_labels == majority).float().mean()
                                purity_sum += purity * mask_p.sum()
                        accuracy = purity_sum / mask.sum() if mask.sum() > 0 else 0
                        batch_acc += accuracy.item()

                event_count += 1

            if event_count > 0:
                avg_loss = batch_loss / event_count
                avg_acc = batch_acc / event_count

                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_train_loss += avg_loss.item()
                epoch_train_acc += avg_acc
                batch_count += 1

        if batch_count > 0:
            avg_train_loss = epoch_train_loss / batch_count
            avg_train_acc = epoch_train_acc / batch_count
        else:
            avg_train_loss = 0
            avg_train_acc = 0

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        val_batch_count = 0

        with torch.no_grad():
            for xs, ys in val_loader:
                Xs = [x.to(device) for x in xs]
                ys_list = [y.to(device) for y in ys]

                # Get embeddings and compute loss
                embeddings_list = model(Xs)

                batch_loss = 0
                batch_acc = 0
                event_count = 0

                for i, embeddings in enumerate(embeddings_list):
                    if embeddings.shape[0] < 2:
                        continue

                    labels = ys_list[i]
                    mask = labels != 0
                    if mask.sum() < 2:
                        continue

                    emb_filtered = embeddings[mask]
                    labels_filtered = labels[mask]

                    unique_labels = torch.unique(labels_filtered)
                    label_map = {old: new for new, old in enumerate(unique_labels)}
                    labels_mapped = torch.tensor([label_map[l.item()] for l in labels_filtered], 
                                               device=device)

                    loss = contrastive_loss(emb_filtered, labels_mapped)
                    batch_loss += loss.item()

                    # Compute clustering accuracy
                    pred_labels = model._cluster_embeddings(emb_filtered)
                    if pred_labels.max() >= 0:
                        pred_unique = torch.unique(pred_labels[pred_labels >= 0])
                        purity_sum = 0
                        for p_label in pred_unique:
                            mask_p = (pred_labels == p_label)
                            true_labels = labels_filtered[mask_p]
                            if len(true_labels) > 0:
                                majority = true_labels.mode().values[0]
                                purity = (true_labels == majority).float().mean()
                                purity_sum += purity * mask_p.sum()
                        accuracy = purity_sum / mask.sum() if mask.sum() > 0 else 0
                        batch_acc += accuracy.item()

                    event_count += 1

                if event_count > 0:
                    epoch_val_loss += batch_loss / event_count
                    epoch_val_acc += batch_acc / event_count
                    val_batch_count += 1

        if val_batch_count > 0:
            avg_val_loss = epoch_val_loss / val_batch_count
            avg_val_acc = epoch_val_acc / val_batch_count
        else:
            avg_val_loss = 0
            avg_val_acc = 0

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)

        # Update scheduler
        scheduler.step()

        # Early stopping
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

    # Load best model
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

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
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

