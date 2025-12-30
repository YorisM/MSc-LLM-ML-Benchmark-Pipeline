
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

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
import torch.nn.functional as F
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add, scatter_max
from sklearn.preprocessing import StandardScaler
import hdbscan
from scipy.spatial import cKDTree
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_mean = None
        self.layer_std = None
        self.r_mean = None
        self.r_std = None
        self.z_mean = None
        self.z_std = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,

            "collate": None,

            "extra_loader_kwargs": {
                "follow_batch": ['x']
            },

            "eval_overrides": {"shuffle": False, "num_workers": 2}
        }

    def fit(self, Xs):
        # Compute global statistics for normalization
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Only scale r, theta, z

        # Compute layer-wise statistics
        layer_ids = all_X[:, 3].astype(int)
        unique_layers = np.unique(layer_ids)
        self.layer_mean = {}
        self.layer_std = {}
        for layer in unique_layers:
            mask = layer_ids == layer
            if mask.sum() > 1:
                self.layer_mean[layer] = np.mean(all_X[mask, :3], axis=0)
                self.layer_std[layer] = np.std(all_X[mask, :3], axis=0) + 1e-8

        # Compute global r and z statistics for spatial features
        self.r_mean = np.mean(all_X[:, 0])
        self.r_std = np.std(all_X[:, 0]) + 1e-8
        self.z_mean = np.mean(all_X[:, 2])
        self.z_std = np.std(all_X[:, 2]) + 1e-8

        return self

    def transform(self, X):
        # X: [N_hits, 4] - r, theta, z, layer_id

        # Normalize r, theta, z
        X_norm = X.clone()
        X_norm[:, :3] = torch.from_numpy(self.scaler.transform(X[:, :3].numpy())).float()

        # Add layer-wise normalized features
        layer_ids = X[:, 3].long()
        for layer in torch.unique(layer_ids):
            mask = layer_ids == layer
            if layer.item() in self.layer_mean:
                layer_mean = torch.tensor(self.layer_mean[layer.item()], dtype=torch.float32)
                layer_std = torch.tensor(self.layer_std[layer.item()], dtype=torch.float32)
                X_norm[mask, :3] = (X_norm[mask, :3] - layer_mean) / layer_std

        # Add spatial features
        r = X[:, 0]
        z = X[:, 2]
        r_normalized = (r - self.r_mean) / self.r_std
        z_normalized = (z - self.z_mean) / self.z_std

        # Add cylindrical coordinates features
        x = r * torch.cos(X[:, 1])
        y = r * torch.sin(X[:, 1])
        phi = X[:, 1]

        # Add distance to origin
        dist_origin = torch.sqrt(x**2 + y**2 + z**2)

        # Add layer embedding
        layer_embedding = F.one_hot(layer_ids, num_classes=20).float()  # Assuming max 20 layers

        # Combine all features
        features = torch.cat([
            X_norm,
            x.unsqueeze(1),
            y.unsqueeze(1),
            phi.unsqueeze(1),
            r_normalized.unsqueeze(1),
            z_normalized.unsqueeze(1),
            dist_origin.unsqueeze(1),
            layer_embedding
        ], dim=1)  # [N_hits, 4 + 3 + 1 + 1 + 1 + 1 + 20] = [N_hits, 31]

        return features

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()

        # Determine input feature dimension from example batch
        if isinstance(example_batch_x, list):
            input_dim = example_batch_x[0].shape[1]
        else:
            input_dim = example_batch_x.shape[1]

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Edge feature computation
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, 64),  # delta_r, delta_phi, delta_z, layer_diff
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Graph neural network layers
        self.gcn_layers = nn.ModuleList()
        for i in range(8):
            conv = GENConv(128, 128, aggr='softmax', t=1.0, learn_t=True, num_layers=2)
            act = nn.ReLU()
            norm = nn.BatchNorm1d(128)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1, ckpt_grad=i % 2)
            self.gcn_layers.append(layer)

        # Track embedding
        self.track_embed = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Will use softmax in loss
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Noise prediction head
        self.noise_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def build_graph(self, x, batch_idx):
        # x: [N_hits, F]
        # batch_idx: [N_hits]

        # Create complete graph within each event
        num_nodes = x.size(0)
        device = x.device

        # Get event boundaries
        event_ptr = torch.cat([torch.tensor([0]), torch.bincount(batch_idx).cumsum(0)]).to(device)

        # Build edges within each event
        edge_indices = []
        edge_attrs = []

        for i in range(len(event_ptr) - 1):
            start, end = event_ptr[i], event_ptr[i+1]
            if end - start < 2:
                continue

            # Get nodes in this event
            event_nodes = torch.arange(start, end, device=device)
            event_x = x[start:end]

            # Create complete graph
            src, dst = torch.combinations(event_nodes, r=2).t()
            edge_indices.append(torch.stack([src, dst], dim=0))
            edge_indices.append(torch.stack([dst, src], dim=0))

            # Compute edge features
            src_x = event_x[src - start]
            dst_x = event_x[dst - start]

            delta_r = (src_x[:, 0] - dst_x[:, 0]).unsqueeze(1)
            delta_phi = (src_x[:, 1] - dst_x[:, 1]).unsqueeze(1)
            delta_z = (src_x[:, 2] - dst_x[:, 2]).unsqueeze(1)
            layer_diff = (src_x[:, 3] - dst_x[:, 3]).unsqueeze(1)

            edge_attr = torch.cat([delta_r, delta_phi, delta_z, layer_diff], dim=1)
            edge_attrs.append(edge_attr)
            edge_attrs.append(edge_attr)

        if len(edge_indices) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0, 4), device=device)

        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)

        return edge_index, edge_attr

    def forward(self, batch_x):
        # Handle different input formats
        if isinstance(batch_x, list):
            # Convert list of tensors to PyG Batch
            data_list = []
            for i, x in enumerate(batch_x):
                edge_index, edge_attr = self.build_graph(x, torch.full((x.size(0),), i, device=x.device))
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data_list.append(data)

            batch = Batch.from_data_list(data_list)
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
            batch_idx = batch.batch
        else:
            # Assume it's already a PyG Batch
            x = batch_x.x
            edge_index = batch_x.edge_index
            edge_attr = batch_x.edge_attr
            batch_idx = batch_x.batch

        # Encode node features
        h = self.node_encoder(x)  # [N_hits, 128]

        # Encode edge features
        edge_attr = self.edge_encoder(edge_attr)  # [N_edges, 64]

        # Graph neural network
        h_in = h
        for layer in self.gcn_layers:
            h = layer(h, edge_index, edge_attr)
            h = h + h_in  # Residual connection
            h_in = h

        # Attention mechanism
        attn_weights = self.attention(h)  # [N_hits, 1]
        attn_weights = torch.softmax(attn_weights, dim=0)
        h_attn = h * attn_weights

        # Track embedding (learned clustering)
        track_emb = self.track_embed(h_attn)  # [N_hits, 32]

        # Combine features
        combined = torch.cat([h_attn, track_emb], dim=1)  # [N_hits, 160]

        # Predict track IDs (using softmax for clustering)
        logits = self.classifier(combined)  # [N_hits, 1]

        # Reshape for clustering
        logits = logits.squeeze(1)  # [N_hits]

        # Predict noise probability
        noise_prob = self.noise_head(h_attn).squeeze(1)  # [N_hits]

        # Combine predictions
        # We'll use the logits for clustering and noise_prob to identify noise
        # Convert to cluster IDs using differentiable clustering
        cluster_ids = self.differentiable_cluster(logits, batch_idx)

        # Apply noise mask
        noise_mask = noise_prob > 0.5
        cluster_ids[noise_mask] = -1  # Mark as noise

        return cluster_ids

    def differentiable_cluster(self, logits, batch_idx):
        # Convert logits to cluster IDs using softmax and argmax
        # This is differentiable and allows gradient flow

        # Get unique batch indices
        unique_batches = torch.unique(batch_idx)

        cluster_ids = torch.zeros_like(logits, dtype=torch.long)

        for batch in unique_batches:
            mask = batch_idx == batch
            batch_logits = logits[mask]

            # Softmax to get probabilities
            probs = torch.softmax(batch_logits, dim=0)

            # Use straight-through estimator for argmax
            max_idx = torch.argmax(probs, dim=0)
            cluster_ids[mask] = max_idx

        return cluster_ids

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Custom loss function for track classification
    def track_loss(pred, target, batch_idx):
        # pred: [N_hits] cluster IDs (-1 for noise)
        # target: [N_hits] true track IDs (0 for noise)

        # Create mask for non-noise hits
        non_noise_mask = target > 0
        if non_noise_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Get unique batch indices
        unique_batches = torch.unique(batch_idx)

        loss = 0.0
        count = 0

        for batch in unique_batches:
            batch_mask = batch_idx == batch
            batch_non_noise = non_noise_mask & batch_mask

            if batch_non_noise.sum() == 0:
                continue

            # Get predictions and targets for this batch
            batch_pred = pred[batch_mask]
            batch_target = target[batch_mask]
            batch_non_noise_pred = batch_pred[batch_non_noise]
            batch_non_noise_target = batch_target[batch_non_noise]

            # Create one-hot targets
            num_classes = batch_non_noise_pred.max().item() + 1
            target_one_hot = F.one_hot(batch_non_noise_target, num_classes=num_classes).float()

            # Create predicted probabilities (softmax)
            pred_probs = torch.softmax(batch_non_noise_pred.float(), dim=0)

            # Compute cross-entropy loss
            ce_loss = -torch.sum(target_one_hot * torch.log(pred_probs + 1e-8)) / batch_non_noise.sum()

            # Add noise prediction loss
            noise_mask = batch_target == 0
            if noise_mask.sum() > 0:
                noise_pred = model.noise_head(model.node_encoder(batch_x[batch_mask][noise_mask]))
                noise_loss = F.binary_cross_entropy(noise_pred.squeeze(), torch.ones_like(noise_pred.squeeze()))
                ce_loss = ce_loss + 0.1 * noise_loss

            loss += ce_loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)

        return loss / count

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            x, y = view.batch_x, view.batch_y

            optimizer.zero_grad()

            # Handle different input formats
            if isinstance(x, list):
                batch_x = x
                batch_y = y
                batch_idx = torch.cat([torch.full((xi.size(0),), i, device=device) for i, xi in enumerate(x)])
            else:
                batch_x = x
                batch_y = y
                batch_idx = x.batch if hasattr(x, 'batch') else torch.zeros(x.size(0), device=device)

            pred = model(batch_x)
            loss = track_loss(pred, batch_y, batch_idx)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            # Simple accuracy calculation (not perfect for clustering)
            non_noise_mask = batch_y > 0
            if non_noise_mask.sum() > 0:
                pred_non_noise = pred[non_noise_mask]
                y_non_noise = batch_y[non_noise_mask]

                # Convert predictions to track IDs (simple approach)
                unique_pred = torch.unique(pred_non_noise)
                pred_to_track = torch.zeros_like(pred_non_noise)
                for i, p in enumerate(unique_pred):
                    pred_to_track[pred_non_noise == p] = i

                unique_y = torch.unique(y_non_noise)
                y_to_track = torch.zeros_like(y_non_noise)
                for i, y_val in enumerate(unique_y):
                    y_to_track[y_non_noise == y_val] = i

                train_correct += (pred_to_track == y_to_track).sum().item()
                train_total += non_noise_mask.sum().item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                x, y = view.batch_x, view.batch_y

                if isinstance(x, list):
                    batch_x = x
                    batch_y = y
                    batch_idx = torch.cat([torch.full((xi.size(0),), i, device=device) for i, xi in enumerate(x)])
                else:
                    batch_x = x
                    batch_y = y
                    batch_idx = x.batch if hasattr(x, 'batch') else torch.zeros(x.size(0), device=device)

                pred = model(batch_x)
                loss = track_loss(pred, batch_y, batch_idx)

                val_loss += loss.item()

                non_noise_mask = batch_y > 0
                if non_noise_mask.sum() > 0:
                    pred_non_noise = pred[non_noise_mask]
                    y_non_noise = batch_y[non_noise_mask]

                    unique_pred = torch.unique(pred_non_noise)
                    pred_to_track = torch.zeros_like(pred_non_noise)
                    for i, p in enumerate(unique_pred):
                        pred_to_track[pred_non_noise == p] = i

                    unique_y = torch.unique(y_non_noise)
                    y_to_track = torch.zeros_like(y_non_noise)
                    for i, y_val in enumerate(unique_y):
                        y_to_track[y_non_noise == y_val] = i

                    val_correct += (pred_to_track == y_to_track).sum().item()
                    val_total += non_noise_mask.sum().item()

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

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

