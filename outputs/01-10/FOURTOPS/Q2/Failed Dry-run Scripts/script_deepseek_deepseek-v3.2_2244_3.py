
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
from torch.utils.data import DataLoader, Dataset
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, TransformerConv, GATv2Conv
import scipy
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings("ignore")

class MyPreprocessor:
    def __init__(self):
        self.scaler_global = RobustScaler()
        self.scaler_kinematic = RobustScaler()
        self.obj_type_stats = {}
        self.node_feat_dim = 16
        self.edge_feat_dim = 10
        self.num_obj_types = 10
        self.max_objs = 18

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {
                "shuffle": False,
                "batch_size": 512,
                "num_workers": 4,
                "pin_memory": True
            }
        }

    def fit(self, X, y=None):
        # Separate global and object features
        global_feats = X[:, :2].numpy()
        obj_features = X[:, 2:].reshape(-1, 5).numpy()

        # Fit scalers
        self.scaler_global.fit(global_feats)

        # Only fit on non-zero objects (obj_type != 0)
        mask = obj_features[:, 0] != 0
        kinematic_feats = obj_features[mask, 1:]
        if len(kinematic_feats) > 0:
            self.scaler_kinematic.fit(kinematic_feats)

        # Collect object type statistics
        obj_types = obj_features[mask, 0].astype(int)
        for obj_type in np.unique(obj_types):
            self.obj_type_stats[obj_type] = {
                'count': np.sum(obj_types == obj_type),
                'mean_pt': np.mean(obj_features[mask & (obj_features[:, 0] == obj_type), 2])
            }

        return self

    def compute_pairwise_features(self, obj_tensors):
        """Compute invariant mass and deltaR for all object pairs"""
        batch_size = obj_tensors.shape[0]
        num_objs = obj_tensors.shape[1]

        # Extract 4-vectors: E, pT, eta, phi
        E = obj_tensors[:, :, 1]  # [B, N]
        pT = obj_tensors[:, :, 2]  # [B, N]
        eta = obj_tensors[:, :, 3]  # [B, N]
        phi = obj_tensors[:, :, 4]  # [B, N]

        # Compute 3-momentum components
        px = pT * torch.cos(phi)  # [B, N]
        py = pT * torch.sin(phi)  # [B, N]
        pz = pT * torch.sinh(eta)  # [B, N]

        # Prepare for pairwise computation
        Ei = E.unsqueeze(2)  # [B, N, 1]
        Ej = E.unsqueeze(1)  # [B, 1, N]

        pxi = px.unsqueeze(2)  # [B, N, 1]
        pxj = px.unsqueeze(1)  # [B, 1, N]
        pyi = py.unsqueeze(2)  # [B, N, 1]
        pyj = py.unsqueeze(1)  # [B, 1, N]
        pzi = pz.unsqueeze(2)  # [B, N, 1]
        pzj = pz.unsqueeze(1)  # [B, 1, N]

        # Invariant mass: m_ij = sqrt((E_i + E_j)^2 - |p_i + p_j|^2)
        E_sum = Ei + Ej  # [B, N, N]
        px_sum = pxi + pxj  # [B, N, N]
        py_sum = pyi + pyj  # [B, N, N]
        pz_sum = pzi + pzj  # [B, N, N]

        m2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)  # [B, N, N]
        m = torch.sqrt(torch.clamp(m2, min=1e-6))  # [B, N, N]

        # DeltaR: sqrt((Δη)^2 + (Δφ)^2)
        eta_i = eta.unsqueeze(2)  # [B, N, 1]
        eta_j = eta.unsqueeze(1)  # [B, 1, N]
        phi_i = phi.unsqueeze(2)  # [B, N, 1]
        phi_j = phi.unsqueeze(1)  # [B, 1, N]

        deta = eta_i - eta_j  # [B, N, N]
        dphi = torch.remainder(phi_i - phi_j + np.pi, 2 * np.pi) - np.pi  # [B, N, N]
        deltaR = torch.sqrt(deta**2 + dphi**2)  # [B, N, N]

        return m, deltaR

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X

        batch_size = X_np.shape[0]
        processed_data = []

        for i in range(batch_size):
            # Global features
            global_feats = X_np[i, :2].reshape(1, -1)
            global_feats = self.scaler_global.transform(global_feats).flatten()

            # Object features
            obj_features = X_np[i, 2:].reshape(-1, 5)  # [18, 5]

            # Create mask for real objects (non-zero padded)
            obj_mask = obj_features[:, 0] != 0
            num_real_objs = np.sum(obj_mask)

            if num_real_objs == 0:
                # Empty event - create dummy node
                obj_features_clean = np.zeros((1, 5))
                obj_mask = np.array([True])
                num_real_objs = 1
            else:
                obj_features_clean = obj_features[obj_mask]

            # Scale kinematic features
            obj_types = obj_features_clean[:, 0].copy()
            kinematic_feats = obj_features_clean[:, 1:].copy()
            if len(kinematic_feats) > 0:
                kinematic_feats = self.scaler_kinematic.transform(kinematic_feats)

            # Reconstruct with scaled features
            obj_features_scaled = np.zeros((num_real_objs, 5))
            obj_features_scaled[:, 0] = obj_types
            obj_features_scaled[:, 1:] = kinematic_feats

            # Convert to tensors
            global_tensor = torch.FloatTensor(global_feats)  # [2]
            obj_tensor = torch.FloatTensor(obj_features_scaled)  # [num_real_objs, 5]
            obj_type_tensor = torch.LongTensor(obj_types.astype(int))  # [num_real_objs]

            # Compute pairwise features
            obj_tensor_batch = obj_tensor.unsqueeze(0)  # [1, num_real_objs, 5]
            m, deltaR = self.compute_pairwise_features(obj_tensor_batch)
            m = m.squeeze(0)  # [num_real_objs, num_real_objs]
            deltaR = deltaR.squeeze(0)  # [num_real_objs, num_real_objs]

            # Create edge index (fully connected)
            edge_index = []
            edge_attr = []
            for j in range(num_real_objs):
                for k in range(num_real_objs):
                    if j != k:
                        edge_index.append([j, k])
                        # Edge features: invariant mass, deltaR, and kinematic differences
                        edge_features = torch.cat([
                            m[j, k].unsqueeze(0),
                            deltaR[j, k].unsqueeze(0),
                            obj_tensor[j, 1:] - obj_tensor[k, 1:]
                        ])
                        edge_attr.append(edge_features)

            edge_index = torch.LongTensor(edge_index).t() if edge_index else torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.stack(edge_attr) if edge_attr else torch.zeros((0, self.edge_feat_dim))

            # Create PyG Data object
            data = Data(
                x=obj_tensor,  # Node features: [num_real_objs, 5]
                edge_index=edge_index,  # [2, num_edges]
                edge_attr=edge_attr,  # [num_edges, edge_feat_dim]
                u=global_tensor,  # Global features: [2]
                num_nodes=num_real_objs
            )

            processed_data.append(data)

        return processed_data

def make_preprocessor():
    return MyPreprocessor()

class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.pre = pre
        self.data_list = pre.transform(X)
        self.labels = torch.LongTensor(y)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data.y = self.labels[idx].unsqueeze(0)  # Graph-level label
        return data

class ParticleEmbedding(nn.Module):
    def __init__(self, obj_type_dim=16, kinematic_dim=32):
        super().__init__()
        self.obj_type_embedding = nn.Embedding(20, obj_type_dim)  # 20 types max
        self.kinematic_proj = nn.Sequential(
            nn.Linear(4, kinematic_dim),
            nn.LayerNorm(kinematic_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.combine = nn.Linear(obj_type_dim + kinematic_dim, obj_type_dim + kinematic_dim)

    def forward(self, x):
        # x: [batch_size, 5] or [num_nodes, 5]
        obj_types = x[:, 0].long()  # [batch_size]
        kinematic = x[:, 1:]  # [batch_size, 4]

        type_emb = self.obj_type_embedding(obj_types)  # [batch_size, obj_type_dim]
        kinematic_emb = self.kinematic_proj(kinematic)  # [batch_size, kinematic_dim]

        combined = torch.cat([type_emb, kinematic_emb], dim=-1)  # [batch_size, obj_type_dim + kinematic_dim]
        combined = self.combine(combined)  # [batch_size, obj_type_dim + kinematic_dim]
        return combined

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.node_dim = 48  # ParticleEmbedding output
        self.edge_dim = sample_object.edge_attr.shape[1] if hasattr(sample_object, 'edge_attr') and sample_object.edge_attr.shape[0] > 0 else 10
        self.global_dim = 2

        # Particle embedding
        self.particle_embedding = ParticleEmbedding(obj_type_dim=16, kinematic_dim=32)

        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # Transformer layers
        self.transformer1 = TransformerConv(
            self.node_dim, 128,
            edge_dim=64,
            heads=4,
            dropout=0.1,
            concat=True
        )
        self.transformer2 = TransformerConv(
            128 * 4, 256,
            edge_dim=64,
            heads=4,
            dropout=0.1,
            concat=False
        )
        self.transformer3 = TransformerConv(
            256, 256,
            edge_dim=64,
            heads=4,
            dropout=0.1,
            concat=False
        )

        # Global attention pooling
        self.global_attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + self.global_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(128 * 4)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, batch):
        # Handle both single graph and batch
        if isinstance(batch, Batch):
            x, edge_index, edge_attr, batch_idx, u = batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u
            num_graphs = batch.num_graphs
        else:
            x, edge_index, edge_attr, u = batch.x, batch.edge_index, batch.edge_attr, batch.u
            batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            num_graphs = 1

        # Encode nodes
        x_embed = self.particle_embedding(x)  # [num_nodes, node_dim]

        # Encode edges
        edge_embed = self.edge_encoder(edge_attr)  # [num_edges, 64]

        # Apply transformer layers
        x1 = self.transformer1(x_embed, edge_index, edge_embed)  # [num_nodes, 128*4]
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.1, training=self.training)

        x2 = self.transformer2(x1, edge_index, edge_embed)  # [num_nodes, 256]
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.1, training=self.training)

        x3 = self.transformer3(x2, edge_index, edge_embed)  # [num_nodes, 256]
        x3 = self.bn3(x3)
        x3 = F.relu(x3)

        # Attention pooling
        attention_scores = self.global_attention(x3)  # [num_nodes, 1]
        attention_weights = torch.softmax(attention_scores, dim=0)
        graph_embed = torch.sum(x3 * attention_weights, dim=0, keepdim=True)  # [1, 256]

        # Repeat for each graph in batch
        if num_graphs > 1:
            graph_embed = global_mean_pool(x3, batch_idx)  # [num_graphs, 256]

        # Add global features
        if num_graphs > 1:
            global_feats = u  # [num_graphs, 2]
        else:
            global_feats = u.unsqueeze(0)  # [1, 2]

        combined = torch.cat([graph_embed, global_feats], dim=-1)  # [num_graphs, 258]

        # Final classification
        logits = self.classifier(combined)  # [num_graphs, 1]

        return logits.squeeze(-1)  # [num_graphs]

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 80

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Early stopping
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None

    # Metrics tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch)
            targets = batch.y.float()

            # Handle single graph vs batch
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            loss = criterion(logits, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            total_train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train if total_train > 0 else 0

        # Validation
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                logits = model(batch)
                targets = batch.y.float()

                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)

                loss = criterion(logits, targets)
                total_val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val if total_val > 0 else 0

        # Update scheduler
        scheduler.step()

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Load best model
    if best_model_state is not None:
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

