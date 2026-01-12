
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, to_python
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops, dryrun_finite_check_fourtops

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.spatial.distance import pdist, squareform
import math

# ---------- IMPORTS ----------
# Additional imports
from torch.optim import AdamW, lr_scheduler
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import warnings
warnings.filterwarnings('ignore')

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler_global = StandardScaler()
        self.scaler_kinematic = RobustScaler()  # More robust for particle physics
        self.object_type_stats = {}
        self.valid_object_types = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:GraphDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True if torch.cuda.is_available() else False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 1024}
        }

    def fit(self, X, y=None):
        # Extract global features (first 2)
        global_features = X[:, :2]
        self.scaler_global.fit(global_features)

        # Extract all kinematic features (E, pT, eta, phi for each object)
        kinematic_features = []
        object_types = []

        for i in range(18):
            start_idx = 2 + i * 5 + 1  # Skip object type, start at E
            if start_idx + 3 < X.shape[1]:
                kinematic = X[:, start_idx:start_idx + 3]  # E, pT, eta
                # Add phi with special handling for circular nature
                phi = X[:, start_idx + 3:start_idx + 4]
                kinematic_features.append(kinematic)
                object_types.append(X[:, 2 + i * 5:2 + i * 5 + 1])  # Object type

        if kinematic_features:
            kinematic_features = np.concatenate(kinematic_features, axis=0)
            self.scaler_kinematic.fit(kinematic_features)

        # Analyze object types
        object_types = np.concatenate(object_types, axis=0) if object_types else np.array([])
        unique_types = np.unique(object_types[object_types != 0])  # Exclude padding
        self.valid_object_types = unique_types

        return self

    def _compute_pairwise_features(self, event_features):
        """Compute pairwise features for graph construction"""
        num_objects = 18
        pairwise_features = []
        edge_indices = []

        # Get object features (excluding padding)
        obj_mask = event_features[:, 0] != 0  # Object type != 0

        for i in range(num_objects):
            if not obj_mask[i]:
                continue
            for j in range(i + 1, num_objects):
                if not obj_mask[j]:
                    continue

                # Get features for objects i and j
                features_i = event_features[i]
                features_j = event_features[j]

                # Extract kinematic features
                eta_i, phi_i = features_i[3], features_i[4]
                eta_j, phi_j = features_j[3], features_j[4]
                E_i, pT_i = features_i[1], features_i[2]
                E_j, pT_j = features_j[1], features_j[2]

                # Compute deltaR
                delta_eta = eta_i - eta_j
                delta_phi = (phi_i - phi_j + math.pi) % (2 * math.pi) - math.pi
                delta_r = math.sqrt(delta_eta**2 + delta_phi**2)

                # Compute approximate invariant mass (simplified)
                # For massless particles, m^2 ≈ 2*pT_i*pT_j*(cosh(delta_eta)-cos(delta_phi))
                cosh_term = math.cosh(delta_eta)
                cos_term = math.cos(delta_phi)
                approx_mass = math.sqrt(2 * pT_i * pT_j * (cosh_term - cos_term)) if pT_i > 0 and pT_j > 0 else 0

                # Additional features
                sum_pt = pT_i + pT_j
                pt_ratio = pT_i / pT_j if pT_j > 0 else 1.0

                pairwise_features.append([
                    delta_r,
                    approx_mass / 1000.0,  # Scale to GeV
                    sum_pt / 1000.0,
                    pt_ratio,
                    delta_eta,
                    delta_phi
                ])
                edge_indices.append([i, j])

        return np.array(pairwise_features), np.array(edge_indices).T if edge_indices else np.array([]).reshape(2, 0)

    def transform(self, X):
        processed_data = []

        for event_idx in range(X.shape[0]):
            event = X[event_idx]

            # Extract global features and normalize
            global_features = event[:2].reshape(1, -1)
            global_features_norm = self.scaler_global.transform(global_features)[0]

            # Process each object
            obj_features_list = []
            valid_objects = []

            for obj_idx in range(18):
                start_idx = 2 + obj_idx * 5
                obj_type = event[start_idx]

                if obj_type == 0:  # Padding
                    continue

                # Get kinematic features
                kinematic = event[start_idx + 1:start_idx + 5]  # E, pT, eta, phi
                kinematic_norm = self.scaler_kinematic.transform(kinematic.reshape(1, -1))[0]

                # Create object feature vector
                obj_features = np.concatenate([
                    [obj_type],  # Keep raw type for embedding
                    kinematic_norm,
                    global_features_norm  # Add global context to each node
                ])

                obj_features_list.append(obj_features)
                valid_objects.append(obj_idx)

            if not obj_features_list:
                # Empty event (shouldn't happen)
                obj_features_list.append(np.zeros(7))
                valid_objects.append(0)

            # Create node features tensor
            node_features = np.array(obj_features_list, dtype=np.float32)

            # Compute pairwise features for graph edges
            edge_features, edge_index = self._compute_pairwise_features(
                np.array([event[2 + i*5:2 + i*5 + 5] for i in range(18)])
            )

            # Create PyG Data object
            data = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long) if edge_index.size > 0 else torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.tensor(edge_features, dtype=torch.float32) if edge_features.size > 0 else torch.empty((0, 6), dtype=torch.float32),
                y=torch.tensor([0], dtype=torch.long)  # Placeholder, will be set by dataset
            )
            processed_data.append(data)

        return processed_data

def make_preprocessor():
    return MyPreprocessor()

# Custom Dataset for Graph Data
class GraphDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.graphs = pre.transform(X)  # List of PyG Data objects
        self.labels = torch.as_tensor(y).long()

        # Set labels in each graph
        for i, graph in enumerate(self.graphs):
            graph.y = torch.tensor([self.labels[i]], dtype=torch.long)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

# ---------- MODEL ARCHITECTURE ----------
class ParticleAttention(nn.Module):
    """Multi-head attention for particle features"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        return self.norm(x + self.dropout(attn_output))

class ParticleTransformerBlock(nn.Module):
    """Transformer block for particle sequence"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = ParticleAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.attention(x, mask)
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Sample object is a PyG Data object
        node_dim = sample_object.x.shape[1]  # 7 features per node
        edge_dim = sample_object.edge_attr.shape[1] if hasattr(sample_object, 'edge_attr') and sample_object.edge_attr.numel() > 0 else 0

        # Object type embedding
        self.obj_embedding = nn.Embedding(50, 8)  # Assume max 50 object types

        # Enhanced node feature processing
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim - 1 + 8, 128),  # -1 for obj_type, +8 for embedding
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Edge feature processing
        if edge_dim > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 256)
            )

        # GCN layers for graph processing
        self.gcn1 = GCNConv(256, 512)
        self.gcn2 = GCNConv(512, 512)
        self.gcn3 = GCNConv(512, 256)

        # Transformer layers for sequence processing
        self.transformer1 = ParticleTransformerBlock(256, 8, 512)
        self.transformer2 = ParticleTransformerBlock(256, 8, 512)

        # Global feature processing
        self.global_processor = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256 + 128, 512),  # Graph features + Sequence features + Global
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Attention pooling for sequence
        self.seq_attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, batch):
        # batch is a PyG Batch object
        x, edge_index, edge_attr, batch_vec = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        # Separate object type from other features
        obj_types = x[:, 0].long()
        other_features = x[:, 1:]

        # Embed object types
        obj_embeddings = self.obj_embedding(obj_types)

        # Combine features
        node_features = torch.cat([other_features, obj_embeddings], dim=1)

        # Encode node features
        node_encoded = self.node_encoder(node_features)  # [num_nodes, 256]

        # Process with GCN if edges exist
        if edge_index.shape[1] > 0:
            # Encode edge features if available
            if edge_attr is not None and edge_attr.shape[0] > 0:
                edge_encoded = self.edge_encoder(edge_attr)
                # Use edge features by concatenating to node features (simplified approach)
                pass

            # GCN layers
            x1 = F.relu(self.gcn1(node_encoded, edge_index))
            x1 = self.dropout(x1)
            x2 = F.relu(self.gcn2(x1, edge_index))
            x2 = self.dropout(x2)
            x3 = F.relu(self.gcn3(x2, edge_index))
        else:
            x3 = node_encoded

        # Graph-level pooling
        graph_features = global_mean_pool(x3, batch_vec)  # [batch_size, 256]

        # Transformer processing on node sequence
        # Get batch indices and create sequence
        batch_size = batch_vec.max().item() + 1
        seq_features = []

        for i in range(batch_size):
            mask = batch_vec == i
            nodes = x3[mask]  # [num_nodes_i, 256]
            if nodes.shape[0] > 0:
                # Add positional encoding
                pos_enc = torch.arange(nodes.shape[0], device=nodes.device).float().unsqueeze(1)
                pos_enc = torch.cat([torch.sin(pos_enc / 10000**(2*torch.arange(256, device=nodes.device)/256)).unsqueeze(0),
                                   torch.cos(pos_enc / 10000**(2*torch.arange(256, device=nodes.device)/256)).unsqueeze(0)], dim=0)
                pos_enc = pos_enc.mean(0)  # Simplified
                nodes = nodes + pos_enc[:nodes.shape[0]]

                # Transformer layers
                nodes = nodes.unsqueeze(0)  # [1, num_nodes_i, 256]
                nodes = self.transformer1(nodes)
                nodes = self.transformer2(nodes)

                # Attention pooling
                attn_weights = self.seq_attention(nodes)
                pooled = (nodes * attn_weights).sum(dim=1)
                seq_features.append(pooled)
            else:
                seq_features.append(torch.zeros(1, 256, device=graph_features.device))

        seq_features = torch.cat(seq_features, dim=0)  # [batch_size, 256]

        # Extract and process global features from the batch
        global_feats = []
        for i in range(batch_size):
            mask = batch_vec == i
            if mask.any():
                # Get global features from first node (they're concatenated to each node)
                global_feats.append(x[mask][0, -2:].unsqueeze(0))

        global_feats = torch.cat(global_feats, dim=0) if global_feats else torch.zeros(batch_size, 2, device=graph_features.device)
        global_encoded = self.global_processor(global_feats)

        # Combine all features
        combined = torch.cat([graph_features, seq_features, global_encoded], dim=1)

        # Final classification
        logits = self.classifier(combined)

        return logits.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int = EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )

    # Loss function with label smoothing
    criterion = nn.BCEWithLogitsLoss()

    # For early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch)
            targets = batch.y.float()

            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == targets).sum().item()
            train_total += targets.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                outputs = model(batch)
                targets = batch.y.float()

                loss = criterion(outputs, targets)
                val_loss += loss.item()

                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == targets).sum().item()
                val_total += targets.size(0)

        # Update learning rate
        scheduler.step()

        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break

    # Restore best model
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    X_fit, Y_fit = X_train, Y_train
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:200]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre = make_preprocessor().fit(X_fit, Y_fit)
    
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
    n_epochs = 10 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        try:
            dryrun_finite_check_fourtops(trained_model, spec, val_loader, device, batches=10)
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
        summary = to_python(summary)
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

