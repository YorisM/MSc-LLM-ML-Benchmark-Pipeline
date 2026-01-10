
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
                        
DATASET = {
    "X_train": "./challenges/FOURTOPS/data/train/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/train/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/train/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/train/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv(DATASET["X_train"], dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv(DATASET["Y_train"], dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv(DATASET["X_val"], dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv(DATASET['Y_val'], dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class FourTopsDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        X2 = pre.transform(X) if pre is not None else X
        if not torch.is_tensor(X2):
            X2 = torch.as_tensor(X2)
        self.X = X2.float()
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        self.y = y.long()
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

# -------------------------- START OF LLM BLOCK ------------------------------
# <start code template>
# ---------- IMPORTS ----------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from sklearn.preprocessing import StandardScaler

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.node_scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.max_objects = 18
        self.obj_dim = 5

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True if torch.cuda.is_available() else False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        # Extract global and object features
        n_samples = X.shape[0]
        global_feats = X[:, :2]  # [n_samples, 2]

        # Reshape objects: [n_samples, max_objects, obj_dim]
        obj_feats = X[:, 2:].reshape(n_samples, self.max_objects, self.obj_dim)

        # Mask for valid objects (obj_id != 0)
        valid_mask = obj_feats[:, :, 0] != 0  # [n_samples, max_objects]

        # Collect all valid object features for scaling
        valid_objs = []
        for i in range(n_samples):
            mask = valid_mask[i]
            if mask.any():
                # Use E, pT, eta, phi (skip obj_id)
                valid_objs.append(obj_feats[i, mask, 1:])  # [n_valid, 4]

        if valid_objs:
            all_valid = np.vstack(valid_objs)  # [total_valid, 4]
            self.node_scaler.fit(all_valid)

        # Fit global features
        self.global_scaler.fit(global_feats)

        # Fit edge features (pairwise mass and deltaR)
        edge_features = []
        for i in range(n_samples):
            mask = valid_mask[i]
            if mask.sum() > 1:
                obj_data = obj_feats[i, mask]  # [n_valid, 5]
                n_valid = obj_data.shape[0]

                # Extract kinematics
                E = obj_data[:, 1]  # [n_valid]
                pT = obj_data[:, 2]  # [n_valid]
                eta = obj_data[:, 3]  # [n_valid]
                phi = obj_data[:, 4]  # [n_valid]

                # Compute pairwise features
                for j in range(n_valid):
                    for k in range(j+1, n_valid):
                        # Invariant mass
                        p1 = [E[j], pT[j], eta[j], phi[j]]
                        p2 = [E[k], pT[k], eta[k], phi[k]]
                        mass = self._compute_invariant_mass(p1, p2)

                        # DeltaR
                        delta_eta = eta[j] - eta[k]
                        delta_phi = self._delta_phi(phi[j], phi[k])
                        deltaR = np.sqrt(delta_eta**2 + delta_phi**2)

                        edge_features.append([mass, deltaR])

        if edge_features:
            edge_features = np.array(edge_features)  # [n_pairs, 2]
            self.edge_scaler.fit(edge_features)

        return self

    def _delta_phi(self, phi1, phi2):
        """Compute cyclic difference in phi."""
        diff = phi1 - phi2
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def _compute_invariant_mass(self, p1, p2):
        """Compute invariant mass from 4-vectors."""
        E1, pT1, eta1, phi1 = p1
        E2, pT2, eta2, phi2 = p2

        # Convert to Cartesian 4-momentum
        pz1 = pT1 * np.sinh(eta1)
        px1 = pT1 * np.cos(phi1)
        py1 = pT1 * np.sin(phi1)

        pz2 = pT2 * np.sinh(eta2)
        px2 = pT2 * np.cos(phi2)
        py2 = pT2 * np.sin(phi2)

        # Sum 4-vectors
        E_sum = E1 + E2
        px_sum = px1 + px2
        py_sum = py1 + py2
        pz_sum = pz1 + pz2

        # Invariant mass
        m2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
        return np.sqrt(max(m2, 0))

    def transform(self, X):
        # X: [batch_size, 92]
        batch_size = X.shape[0]

        # Extract features
        global_feats = X[:, :2]  # [batch_size, 2]
        obj_feats = X[:, 2:].reshape(batch_size, self.max_objects, self.obj_dim)  # [batch_size, 18, 5]

        # Create mask for valid objects
        valid_mask = (obj_feats[:, :, 0] != 0).float()  # [batch_size, 18]

        # Scale global features
        if hasattr(self.global_scaler, 'scale_'):
            global_feats_np = global_feats.numpy() if torch.is_tensor(global_feats) else global_feats
            global_feats_scaled = self.global_scaler.transform(global_feats_np)
            global_feats = torch.tensor(global_feats_scaled, dtype=torch.float32)

        # Process each event
        processed_features = []
        for i in range(batch_size):
            mask = valid_mask[i]  # [18]
            n_valid = int(mask.sum().item())

            if n_valid == 0:
                # Fallback: use zeros
                obj_data = torch.zeros((1, 4), dtype=torch.float32)
                valid_idx = torch.zeros(1, dtype=torch.float32)
                edge_features = torch.zeros((1, 2), dtype=torch.float32)
                edge_indices = torch.zeros((2, 1), dtype=torch.long)
            else:
                # Get valid objects
                valid_indices = torch.where(mask > 0)[0]  # [n_valid]
                obj_data = obj_feats[i, valid_indices, 1:]  # [n_valid, 4] (skip obj_id)

                # Scale object features
                if hasattr(self.node_scaler, 'scale_'):
                    obj_data_np = obj_data.numpy() if torch.is_tensor(obj_data) else obj_data
                    obj_data_scaled = self.node_scaler.transform(obj_data_np)
                    obj_data = torch.tensor(obj_data_scaled, dtype=torch.float32)

                # Add global features to each object
                global_rep = global_feats[i].unsqueeze(0).repeat(n_valid, 1)  # [n_valid, 2]
                obj_data = torch.cat([obj_data, global_rep], dim=1)  # [n_valid, 6]

                # Create complete graph edges
                if n_valid > 1:
                    edges = []
                    edge_feats = []
                    for j in range(n_valid):
                        for k in range(j+1, n_valid):
                            edges.append([j, k])
                            edges.append([k, j])  # undirected

                            # Compute edge features
                            E1, pT1, eta1, phi1 = obj_feats[i, valid_indices[j], 1:5]
                            E2, pT2, eta2, phi2 = obj_feats[i, valid_indices[k], 1:5]

                            # Invariant mass
                            m = self._compute_invariant_mass(
                                [E1.item(), pT1.item(), eta1.item(), phi1.item()],
                                [E2.item(), pT2.item(), eta2.item(), phi2.item()]
                            )

                            # DeltaR
                            delta_eta = eta1.item() - eta2.item()
                            delta_phi = self._delta_phi(phi1.item(), phi2.item())
                            deltaR = math.sqrt(delta_eta**2 + delta_phi**2)

                            edge_feats.append([m, deltaR])
                            edge_feats.append([m, deltaR])  # symmetric

                    edge_indices = torch.tensor(edges, dtype=torch.long).t()  # [2, 2*edges]
                    edge_features = torch.tensor(edge_feats, dtype=torch.float32)  # [2*edges, 2]

                    # Scale edge features
                    if hasattr(self.edge_scaler, 'scale_') and len(edge_feats) > 0:
                        edge_features_np = edge_features.numpy()
                        edge_features_scaled = self.edge_scaler.transform(edge_features_np)
                        edge_features = torch.tensor(edge_features_scaled, dtype=torch.float32)
                else:
                    # Single node, no edges
                    edge_indices = torch.zeros((2, 1), dtype=torch.long)
                    edge_features = torch.zeros((1, 2), dtype=torch.float32)

            # Store as dictionary for easy access
            processed_features.append({
                'x': obj_data,  # [n_valid, 6]
                'edge_index': edge_indices,  # [2, n_edges]
                'edge_attr': edge_features,  # [n_edges, 2]
                'batch': torch.full((obj_data.shape[0],), i, dtype=torch.long),  # [n_valid]
                'valid_mask': mask  # [18]
            })

        return processed_features

def make_preprocessor():
    return MyPreprocessor()

#  -------- CUSTOM DATASET  --------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X
        self.y = torch.as_tensor(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------- MODEL ARCHITECTURE ----------
class ParticleAttentionLayer(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, key_padding_mask=None):
        # x: [batch_size, seq_len, dim]
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        node_dim = sample_object['x'].shape[1]  # 6
        edge_dim = sample_object['edge_attr'].shape[1]  # 2

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.GELU()
        )

        # Graph attention layers
        self.attention_layers = nn.ModuleList([
            ParticleAttentionLayer(256, heads=8, dropout=0.1) for _ in range(4)
        ])

        # Global pooling and classifier
        self.pool = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, batch_data):
        # batch_data is a list of dictionaries from the dataloader
        if isinstance(batch_data, list):
            # Handle list input from dataloader
            return self._process_batch(batch_data)
        else:
            # Direct forward pass
            return self._process_batch([batch_data])

    def _process_batch(self, batch_list):
        batch_size = len(batch_list)

        # Pad node features to same length
        max_nodes = max(d['x'].shape[0] for d in batch_list)
        node_features = []
        masks = []

        for data in batch_list:
            n_nodes = data['x'].shape[0]
            pad_len = max_nodes - n_nodes

            # Pad node features
            padded = F.pad(data['x'], (0, 0, 0, pad_len), value=0)  # [max_nodes, 6]
            node_features.append(padded)

            # Create mask (1 for valid, 0 for padded)
            mask = torch.cat([
                torch.ones(n_nodes, dtype=torch.bool, device=data['x'].device),
                torch.zeros(pad_len, dtype=torch.bool, device=data['x'].device)
            ])
            masks.append(mask)

        # Stack everything
        node_features = torch.stack(node_features)  # [batch_size, max_nodes, 6]
        masks = torch.stack(masks)  # [batch_size, max_nodes]

        # Encode nodes
        x = self.node_encoder(node_features)  # [batch_size, max_nodes, 256]

        # Apply attention layers
        for layer in self.attention_layers:
            # Reshape for attention: [batch_size * max_nodes, 256]
            batch_size, max_nodes, dim = x.shape
            x_flat = x.reshape(-1, dim)  # [batch_size * max_nodes, 256]
            mask_flat = ~masks.reshape(-1)  # [batch_size * max_nodes]

            # Reshape back for attention
            x_reshaped = x_flat.reshape(batch_size, max_nodes, dim)
            x = layer(x_reshaped, key_padding_mask=mask_flat)

        # Mask out padded nodes
        x = x * masks.unsqueeze(-1).float()

        # Global pooling (mean over valid nodes)
        valid_counts = masks.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
        pooled = x.sum(dim=1) / valid_counts.clamp(min=1)  # [batch_size, 256]

        # Final classification
        pooled = self.pool(pooled)  # [batch_size, 64]
        logits = self.classifier(pooled).squeeze(-1)  # [batch_size]

        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Early stopping
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    best_state = None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            data, targets = batch
            targets = targets.float().to(device)

            optimizer.zero_grad()

            # Handle batch processing
            if isinstance(data, list):
                # Move each element to device
                data_on_device = []
                for d in data:
                    device_data = {}
                    for k, v in d.items():
                        if torch.is_tensor(v):
                            device_data[k] = v.to(device)
                        else:
                            device_data[k] = v
                    data_on_device.append(device_data)
                data = data_on_device

            logits = model(data)
            loss = criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                data, targets = batch
                targets = targets.float().to(device)

                if isinstance(data, list):
                    data_on_device = []
                    for d in data:
                        device_data = {}
                        for k, v in d.items():
                            if torch.is_tensor(v):
                                device_data[k] = v.to(device)
                            else:
                                device_data[k] = v
                        data_on_device.append(device_data)
                    data = data_on_device

                logits = model(data)
                loss = criterion(logits, targets)

                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Update scheduler
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            if best_state is not None:
                model.load_state_dict(best_state)
            break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model, train_losses, val_losses, train_accs, val_accs

# <end code template>
# ---------------------------  END OF LLM-CODE BLOCK  ---------------------------

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:20]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre     = make_preprocessor().fit(X_train, Y_train)
    
    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec, require_torch_collate=False)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, (X_train, Y_train), pre, train=True)
    val_ds       = build_dataset(spec, (X_val,   Y_val),   pre, train=False)
    train_loader = build_dataloader(spec, train_ds, is_eval=False)
    val_loader   = build_dataloader(spec, val_ds,   is_eval=True)

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane_fourtops(spec, first_batch)
    view = make_view_by_lane_fourtops(mode, first_batch, device)

    # Build model
    model = make_model(view.batch_x).to(device)

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
                mode = detect_and_assert_lane_fourtops(spec, first_batch)
                view = make_view_by_lane_fourtops(mode, first_batch, device)
                out  = trained_model(view.batch_x)
                scores, kind = assert_binary_output(view, out)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e

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

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

