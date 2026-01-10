
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

# ---------- IMPORTS ----------
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict

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

            "eval_overrides": {"shuffle": False, "num_workers": 0}
        }

    def fit(self, Xs):
        # Compute global statistics for normalization
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Only scale r, theta, z

        # Compute layer statistics
        layer_ids = np.unique(all_X[:, 3])
        self.layer_stats = {int(lid): {
            'count': np.sum(all_X[:, 3] == lid),
            'mean_r': np.mean(all_X[all_X[:, 3] == lid, 0]),
            'std_r': np.std(all_X[all_X[:, 3] == lid, 0])
        } for lid in layer_ids}

        return self

    def transform(self, X):
        # X shape: [N_hits, 4]
        X = X.clone().numpy() if torch.is_tensor(X) else X.copy()

        # Normalize r, theta, z
        X[:, :3] = self.scaler.transform(X[:, :3])

        # Add layer-specific features
        layer_id = X[:, 3].astype(int)
        layer_features = np.zeros((X.shape[0], 3))
        for i, lid in enumerate(layer_id):
            if lid in self.layer_stats:
                stats = self.layer_stats[lid]
                layer_features[i, 0] = stats['count']
                layer_features[i, 1] = stats['mean_r']
                layer_features[i, 2] = stats['std_r']

        # Combine features
        X = np.hstack([X, layer_features])  # [N_hits, 7]

        return torch.FloatTensor(X)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()

        # Determine input feature dimension
        if isinstance(example_batch_x, list):
            input_dim = example_batch_x[0].shape[1]
        else:
            input_dim = example_batch_x.shape[1]

        # Embedding layers
        self.hit_embed = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Graph neural network components
        self.edge_mlp = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Output layers
        self.output = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Track classification head
        self.classifier = nn.Linear(16, 1)  # Will use sigmoid for edge classification

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, Xs):
        # Xs: list of tensors [N_i, F]
        device = next(self.parameters()).device

        # Process each event in batch
        all_embeddings = []
        all_batch_indices = []

        for i, X in enumerate(Xs):
            X = X.to(device)
            embeddings = self.hit_embed(X)  # [N_i, 128]
            all_embeddings.append(embeddings)
            all_batch_indices.append(torch.full((X.shape[0],), i, device=device))

        # Concatenate all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)  # [N_total, 128]
        batch_indices = torch.cat(all_batch_indices, dim=0)  # [N_total]

        # Build complete graph for each event
        edge_scores = []
        edge_indices = []

        for i, X in enumerate(Xs):
            # Get embeddings for this event
            mask = (batch_indices == i)
            event_emb = embeddings[mask]  # [N_i, 128]

            # Build k-NN graph (k=10)
            k = min(10, event_emb.shape[0] - 1)
            if k <= 0:
                continue

            # Compute pairwise distances
            dist = torch.cdist(event_emb, event_emb)  # [N_i, N_i]
            _, topk_indices = dist.topk(k, largest=False)  # [N_i, k]

            # Create edges
            src = torch.arange(event_emb.shape[0], device=device).repeat_interleave(k)
            dst = topk_indices.flatten()

            # Remove self-loops
            valid_edges = src != dst
            src = src[valid_edges]
            dst = dst[valid_edges]

            # Get edge features
            edge_feat = torch.cat([
                event_emb[src],
                event_emb[dst]
            ], dim=1)  # [N_edges, 256]

            # Compute edge scores
            scores = self.edge_mlp(edge_feat).squeeze(1)  # [N_edges]
            scores = torch.sigmoid(scores)

            # Store edges and scores
            edge_indices.append(torch.stack([src, dst], dim=0))
            edge_scores.append(scores)

        if not edge_indices:
            # Handle case with no edges (single hit events)
            return embeddings

        # Stack all edges and scores
        edge_index = torch.cat(edge_indices, dim=1)  # [2, N_total_edges]
        edge_score = torch.cat(edge_scores, dim=0)  # [N_total_edges]

        # Apply edge filtering
        edge_mask = edge_score > 0.5
        edge_index = edge_index[:, edge_mask]

        # Message passing (simple GCN-style)
        row, col = edge_index
        deg = torch.bincount(row, minlength=embeddings.shape[0]).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Aggregate neighbor messages
        msg = embeddings[col] * norm.unsqueeze(1)
        agg = torch.zeros_like(embeddings)
        agg.scatter_add_(0, row.unsqueeze(1).expand(-1, embeddings.shape[1]), msg)

        # Update embeddings
        embeddings = embeddings + agg

        # Final output
        out = self.output(embeddings)  # [N_total, 16]

        return out

    def predict_labels(self, Xs):
        device = next(self.parameters()).device
        Xs = [x.to(device) for x in Xs]

        # Get embeddings
        with torch.no_grad():
            embeddings = self.forward(Xs)  # [N_total, 16]

        # Split by event
        labels_list = []
        start_idx = 0

        for X in Xs:
            n_hits = X.shape[0]
            event_emb = embeddings[start_idx:start_idx + n_hits]  # [N_i, 16]
            start_idx += n_hits

            # Skip if too few hits
            if n_hits < 2:
                labels_list.append(torch.zeros(n_hits, dtype=torch.long, device=device) - 1)
                continue

            # Build similarity matrix
            sim = torch.mm(event_emb, event_emb.t())  # [N_i, N_i]
            sim = torch.sigmoid(sim)

            # Threshold to create adjacency matrix
            adj = (sim > 0.5).float().cpu().numpy()

            # Find connected components
            n_components, component_labels = connected_components(
                csr_matrix(adj), directed=False, return_labels=True
            )

            # Convert to torch tensor
            component_labels = torch.from_numpy(component_labels).to(device)

            # Assign noise (-1) to small components
            component_sizes = torch.bincount(component_labels)
            noise_mask = component_sizes[component_labels] < 4
            component_labels[noise_mask] = -1

            labels_list.append(component_labels)

        return labels_list

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for Xs, ys in train_loader:
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            optimizer.zero_grad()

            # Forward pass
            embeddings = model(Xs)  # [N_total, 16]

            # Compute loss (we'll use a simple contrastive loss)
            loss = 0.0
            start_idx = 0

            for i, (X, y) in enumerate(zip(Xs, ys)):
                n_hits = X.shape[0]
                event_emb = embeddings[start_idx:start_idx + n_hits]  # [N_i, 16]
                event_y = y  # [N_i]

                # Skip noise hits (track_id = 0)
                non_noise_mask = (event_y != 0)
                if non_noise_mask.sum() < 2:
                    start_idx += n_hits
                    continue

                event_emb = event_emb[non_noise_mask]
                event_y = event_y[non_noise_mask]

                # Create positive and negative pairs
                same_track = (event_y.unsqueeze(0) == event_y.unsqueeze(1)).float()
                diff_track = 1 - same_track

                # Compute pairwise distances
                dist = torch.cdist(event_emb, event_emb)  # [N, N]

                # Contrastive loss
                pos_loss = same_track * dist.pow(2)
                neg_loss = diff_track * torch.clamp(1.0 - dist, min=0).pow(2)
                loss += (pos_loss + neg_loss).sum() / (non_noise_mask.sum() ** 2)

                start_idx += n_hits

            if loss > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                embeddings = model(Xs)
                start_idx = 0

                for i, (X, y) in enumerate(zip(Xs, ys)):
                    n_hits = X.shape[0]
                    event_emb = embeddings[start_idx:start_idx + n_hits]
                    event_y = y

                    non_noise_mask = (event_y != 0)
                    if non_noise_mask.sum() < 2:
                        start_idx += n_hits
                        continue

                    event_emb = event_emb[non_noise_mask]
                    event_y = event_y[non_noise_mask]

                    dist = torch.cdist(event_emb, event_emb)
                    same_track = (event_y.unsqueeze(0) == event_y.unsqueeze(1)).float()
                    diff_track = 1 - same_track

                    pos_loss = same_track * dist.pow(2)
                    neg_loss = diff_track * torch.clamp(1.0 - dist, min=0).pow(2)
                    val_loss += (pos_loss + neg_loss).sum() / (non_noise_mask.sum() ** 2)

                    start_idx += n_hits

        # Update learning rate
        scheduler.step()

        # Calculate average losses
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

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
        summary = to_python(summary)
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

