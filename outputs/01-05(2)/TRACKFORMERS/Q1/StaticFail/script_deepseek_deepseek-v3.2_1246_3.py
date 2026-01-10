
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import hdbscan
from typing import List

# ---------- IMPORTS ----------
import warnings
warnings.filterwarnings('ignore')

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.r_scale = None
        self.z_scale = None
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
            "eval_overrides": {"shuffle": False, "batch_size": 16}
        }

    def fit(self, Xs):
        # Concatenate all events for statistics
        all_features = []
        for X in Xs:
            X_np = X.numpy() if isinstance(X, torch.Tensor) else X
            all_features.append(X_np)

        all_features = np.vstack(all_features)  # [N_total, 4]

        # Feature engineering: Add cylindrical coordinates and normalized features
        r = all_features[:, 0]
        theta = all_features[:, 1]
        z = all_features[:, 2]
        layer = all_features[:, 3]

        # Compute statistics for normalization
        self.r_scale = np.max(r) - np.min(r)
        self.z_scale = np.max(z) - np.min(z)

        # Layer statistics
        self.layer_stats = {
            'min': np.min(layer),
            'max': np.max(layer),
            'mean': np.mean(layer),
            'std': np.std(layer)
        }

        # Fit scaler on engineered features
        engineered = self._engineer_features(all_features)
        self.scaler.fit(engineered)

        return self

    def _engineer_features(self, X):
        # X: [N, 4] original features
        r = X[:, 0:1]
        theta = X[:, 1:2]
        z = X[:, 2:3]
        layer = X[:, 3:4]

        # Cylindrical to Cartesian
        x = r * np.cos(theta)  # [N, 1]
        y = r * np.sin(theta)  # [N, 1]

        # Normalize
        r_norm = (r - np.min(r)) / self.r_scale if self.r_scale > 0 else r
        z_norm = (z - np.min(z)) / self.z_scale if self.z_scale > 0 else z
        layer_norm = (layer - self.layer_stats['mean']) / self.layer_stats['std'] if self.layer_stats['std'] > 0 else layer

        # Additional features
        r2 = r ** 2  # [N, 1]
        z2 = z ** 2  # [N, 1]
        phi = np.arctan2(y, x)  # [N, 1]

        # Combine all features: total 11 features
        features = np.hstack([
            x, y, z,                     # Cartesian coordinates [3]
            r_norm, theta, z_norm,       # Normalized cylindrical [3]
            layer_norm,                  # Normalized layer [1]
            r2, z2, phi,                 # Derived features [3]
            np.sqrt(r2 + z2)            # Distance from origin [1]
        ])  # [N, 11]

        return features

    def transform(self, X):
        # X: [N, 4] original features
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X

        # Engineer features
        features = self._engineer_features(X_np)  # [N, 11]

        # Scale features
        features_scaled = self.scaler.transform(features)  # [N, 11]

        return torch.FloatTensor(features_scaled)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()

        # Analyze input shape from example batch
        input_dim = example_batch_x[0].shape[1]  # Feature dimension from first event

        # Enhanced architecture with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Attention pooling for context
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=0)
        )

        # Final projection for clustering
        self.projection = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),  # Embedding dimension for clustering
        )

        # Skip connection for stability
        self.skip = nn.Linear(input_dim, 32) if input_dim != 32 else nn.Identity()

    def forward(self, batch_x):
        # Process each event in batch
        batch_embeddings = []
        for x in batch_x:
            # x: [N_i, F]
            if x.shape[0] == 0:
                batch_embeddings.append(torch.zeros((0, 32), device=x.device))
                continue

            # Encode features
            encoded = self.encoder(x)  # [N_i, 128]

            # Add attention-weighted context
            attn_weights = self.attention(encoded)  # [N_i, 1]
            context = torch.sum(encoded * attn_weights, dim=0, keepdim=True)  # [1, 128]
            context = context.repeat(encoded.shape[0], 1)  # [N_i, 128]

            # Combine with encoded features
            combined = encoded + 0.1 * context  # [N_i, 128]

            # Project to embedding space
            proj = self.projection(combined)  # [N_i, 32]

            # Add skip connection
            skip = self.skip(x)  # [N_i, 32]
            embedding = proj + 0.1 * skip  # [N_i, 32]

            # Normalize embeddings for stable clustering
            embedding = F.normalize(embedding, p=2, dim=1)  # [N_i, 32]

            batch_embeddings.append(embedding)

        return batch_embeddings

    def predict_labels(self, batch_x):
        embeddings_list = self.forward(batch_x)
        predictions = []

        for embeddings in embeddings_list:
            if embeddings.shape[0] == 0:
                predictions.append(torch.tensor([], dtype=torch.int, device=embeddings.device))
                continue

            # Convert to numpy for HDBSCAN
            emb_np = embeddings.detach().cpu().numpy()

            # Adaptive HDBSCAN parameters based on event size
            n_points = emb_np.shape[0]
            min_cluster_size = max(4, int(0.05 * n_points))
            min_samples = max(1, int(0.01 * n_points))

            # Cluster using HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )

            labels = clusterer.fit_predict(emb_np)

            # Convert to torch tensor
            pred_tensor = torch.from_numpy(labels).to(embeddings.device)

            # HDBSCAN uses -1 for noise, which matches our requirement
            predictions.append(pred_tensor)

        return predictions

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Contrastive loss for learning embeddings
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0

        for xs, ys in train_loader:
            Xs = [x.to(device) for x in xs]
            ys = [y.to(device) for y in ys]

            optimizer.zero_grad()

            # Get embeddings
            embeddings_list = model(Xs)

            # Compute triplet loss
            batch_loss = 0
            batch_samples = 0

            for embeddings, y in zip(embeddings_list, ys):
                if embeddings.shape[0] < 3:
                    continue

                # Create triplets: anchor, positive, negative
                n_points = embeddings.shape[0]

                # Find pairs from same track (positive) and different tracks (negative)
                same_track_mask = (y.unsqueeze(1) == y.unsqueeze(0)) & (torch.eye(n_points, device=device) == 0)
                diff_track_mask = (y.unsqueeze(1) != y.unsqueeze(0))

                # Sample triplets
                for _ in range(min(100, n_points)):
                    # Random anchor
                    anchor_idx = torch.randint(0, n_points, (1,)).item()
                    anchor = embeddings[anchor_idx]

                    # Find positive sample
                    positive_candidates = torch.where(same_track_mask[anchor_idx])[0]
                    if len(positive_candidates) > 0:
                        pos_idx = positive_candidates[torch.randint(0, len(positive_candidates), (1,))]
                        positive = embeddings[pos_idx]

                        # Find negative sample
                        negative_candidates = torch.where(diff_track_mask[anchor_idx])[0]
                        if len(negative_candidates) > 0:
                            neg_idx = negative_candidates[torch.randint(0, len(negative_candidates), (1,))]
                            negative = embeddings[neg_idx]

                            # Compute triplet loss
                            triplet_loss = criterion(anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0))
                            batch_loss += triplet_loss
                            batch_samples += 1

            if batch_samples > 0:
                loss = batch_loss / batch_samples
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_train_loss += loss.item() * batch_samples

            # Track accuracy (approximate)
            with torch.no_grad():
                pred_labels = model.predict_labels(Xs)
                for pred, y_true in zip(pred_labels, ys):
                    if len(pred) > 0:
                        # Simple accuracy: same track if same predicted cluster (ignoring permutation)
                        # This is just for monitoring
                        pred_clean = pred[pred != -1]
                        y_clean = y_true[pred != -1]
                        if len(pred_clean) > 0:
                            # Use adjusted rand index as approximate accuracy
                            from sklearn import metrics
                            ari = metrics.adjusted_rand_score(y_clean.cpu().numpy(), pred_clean.cpu().numpy())
                            epoch_train_correct += max(0, ari) * len(pred_clean)
                            epoch_train_total += len(pred_clean)

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_correct = 0
        epoch_val_total = 0

        with torch.no_grad():
            for xs, ys in val_loader:
                Xs = [x.to(device) for x in xs]
                ys = [y.to(device) for y in ys]

                # Get embeddings and compute loss
                embeddings_list = model(Xs)

                batch_loss = 0
                batch_samples = 0

                for embeddings, y in zip(embeddings_list, ys):
                    if embeddings.shape[0] < 3:
                        continue

                    n_points = embeddings.shape[0]

                    # Similar triplet sampling for validation loss
                    for _ in range(min(50, n_points)):
                        anchor_idx = torch.randint(0, n_points, (1,)).item()
                        anchor = embeddings[anchor_idx]

                        # Find positive and negative samples
                        same_track = torch.where(y == y[anchor_idx])[0]
                        diff_track = torch.where(y != y[anchor_idx])[0]

                        if len(same_track) > 1 and len(diff_track) > 0:
                            pos_idx = same_track[torch.randint(0, len(same_track), (1,))]
                            while pos_idx.item() == anchor_idx:
                                pos_idx = same_track[torch.randint(0, len(same_track), (1,))]

                            neg_idx = diff_track[torch.randint(0, len(diff_track), (1,))]

                            positive = embeddings[pos_idx]
                            negative = embeddings[neg_idx]

                            triplet_loss = criterion(anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0))
                            batch_loss += triplet_loss.item()
                            batch_samples += 1

                if batch_samples > 0:
                    epoch_val_loss += batch_loss

                # Compute validation accuracy
                pred_labels = model.predict_labels(Xs)
                for pred, y_true in zip(pred_labels, ys):
                    if len(pred) > 0:
                        pred_clean = pred[pred != -1]
                        y_clean = y_true[pred != -1]
                        if len(pred_clean) > 0:
                            from sklearn import metrics
                            ari = metrics.adjusted_rand_score(y_clean.cpu().numpy(), pred_clean.cpu().numpy())
                            epoch_val_correct += max(0, ari) * len(pred_clean)
                            epoch_val_total += len(pred_clean)

        # Calculate epoch metrics
        avg_train_loss = epoch_train_loss / max(1, len(train_loader))
        avg_val_loss = epoch_val_loss / max(1, len(val_loader))

        train_acc = epoch_train_correct / max(1, epoch_train_total)
        val_acc = epoch_val_correct / max(1, epoch_val_total)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        scheduler.step()

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)
    model.to(device)

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

