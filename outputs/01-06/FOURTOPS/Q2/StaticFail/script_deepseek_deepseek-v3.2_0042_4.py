
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math
from collections import defaultdict

# ---------- IMPORTS ----------
# <LLM: Import modules>
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader as PyGDataLoader
import warnings
warnings.filterwarnings('ignore')


#  -------- CUSTOM DATASET (PyG version) --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        # Transform X into list of PyG Data objects
        self.graphs = pre.transform(X)
        # Assign labels to each graph
        for i, g in enumerate(self.graphs):
            g.y = torch.tensor([y[i]], dtype=torch.long)
        self.train = train

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler_global = StandardScaler()
        self.scaler_node = StandardScaler()
        self.scaler_edge = StandardScaler()
        self.obj_categories = defaultdict(int)
        self.obj_to_idx = {}
        self.max_objects = 18
        self.obj_features = 5
        self.global_features = 2

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def _extract_features(self, event_tensor):
        # event_tensor: [92] flat
        # Extract global features
        global_feat = event_tensor[:2].numpy()  # [2]

        # Extract object features and reshape
        obj_flat = event_tensor[2:].reshape(self.max_objects, self.obj_features)  # [18, 5]

        # Create mask for real objects (obj_id != 0)
        obj_ids = obj_flat[:, 0]
        mask = obj_ids != 0

        # Filter real objects
        real_obj_feat = obj_flat[mask]  # [N_real, 5]

        return global_feat, real_obj_feat, mask

    def _compute_pairwise_features(self, real_obj_feat):
        # real_obj_feat: [N_real, 5] -> [obj_id, E, pT, eta, phi]
        N = real_obj_feat.shape[0]
        if N <= 1:
            return np.zeros((0, 2), dtype=np.float32)  # No edges

        # Extract kinematics
        E = real_obj_feat[:, 1]  # [N]
        pT = real_obj_feat[:, 2]  # [N]
        eta = real_obj_feat[:, 3]  # [N]
        phi = real_obj_feat[:, 4]  # [N]

        # Compute px, py, pz
        px = pT * np.cos(phi)  # [N]
        py = pT * np.sin(phi)  # [N]
        pz = pT * np.sinh(eta)  # [N]

        # Initialize pairwise features
        pairwise_features = []

        # Compute for all pairs (i < j)
        for i in range(N):
            for j in range(i+1, N):
                # Invariant mass m_ij = sqrt((E_i + E_j)^2 - |p_i + p_j|^2)
                E_sum = E[i] + E[j]
                px_sum = px[i] + px[j]
                py_sum = py[i] + py[j]
                pz_sum = pz[i] + pz[j]

                m2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
                m = np.sqrt(np.maximum(m2, 0.0))

                # Angular distance ΔR_ij = sqrt((η_i - η_j)^2 + (φ_i - φ_j)^2)
                delta_eta = eta[i] - eta[j]
                delta_phi = phi[i] - phi[j]
                # Normalize delta_phi to [-π, π]
                while delta_phi > math.pi:
                    delta_phi -= 2 * math.pi
                while delta_phi < -math.pi:
                    delta_phi += 2 * math.pi
                deltaR = np.sqrt(delta_eta**2 + delta_phi**2)

                pairwise_features.append([m, deltaR])

        return np.array(pairwise_features, dtype=np.float32)  # [N*(N-1)/2, 2]

    def _build_graph(self, global_feat, real_obj_feat, mask):
        # Build PyG Data object
        # Node features: [obj_id (encoded), E, pT, eta, phi] -> normalize
        node_feat = real_obj_feat.copy()
        # Encode obj_id (first column) as categorical
        obj_ids = node_feat[:, 0].astype(int)
        node_feat[:, 0] = np.array([self.obj_to_idx.get(oid, 0) for oid in obj_ids], dtype=np.float32)

        # Build edge_index (fully connected for real objects)
        N = real_obj_feat.shape[0]
        if N <= 1:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            # Create all pairs (i < j) and both directions for undirected graph
            edges = []
            for i in range(N):
                for j in range(N):
                    if i != j:
                        edges.append([i, j])
            edge_index = np.array(edges, dtype=np.int64).T  # [2, E]

        # Compute pairwise features for edges
        if N > 1:
            edge_attr = self._compute_pairwise_features(real_obj_feat)  # [E/2, 2]
            # Repeat for both directions (i->j and j->i)
            edge_attr = np.vstack([edge_attr, edge_attr])  # [E, 2]
        else:
            edge_attr = np.zeros((0, 2), dtype=np.float32)

        # Store global features as graph attribute
        global_feat_tensor = torch.tensor(global_feat, dtype=torch.float32).unsqueeze(0)  # [1, 2]

        # Create PyG Data object
        data = Data(
            x=torch.tensor(node_feat, dtype=torch.float32),  # [N_real, 5]
            edge_index=torch.tensor(edge_index, dtype=torch.long),  # [2, E]
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),  # [E, 2]
            global_feat=global_feat_tensor,
            y=None  # Will be set later
        )

        return data

    def fit(self, X, y=None):
        # X: [N_events, 92]
        all_global = []
        all_node = []
        all_edge = []

        # First pass: collect object categories
        for i in range(X.shape[0]):
            event = X[i]
            global_feat, real_obj_feat, mask = self._extract_features(event)
            all_global.append(global_feat)
            all_node.append(real_obj_feat)

            # Collect object IDs for categorical encoding
            obj_ids = real_obj_feat[:, 0].astype(int)
            for oid in obj_ids:
                self.obj_categories[oid] += 1

        # Create mapping for object IDs (0 reserved for padding/unknown)
        sorted_objs = sorted(self.obj_categories.keys())
        for idx, oid in enumerate(sorted_objs, 1):
            self.obj_to_idx[oid] = idx

        # Second pass: collect edge features
        for i in range(X.shape[0]):
            event = X[i]
            global_feat, real_obj_feat, mask = self._extract_features(event)
            edge_feat = self._compute_pairwise_features(real_obj_feat)
            if edge_feat.shape[0] > 0:
                all_edge.append(edge_feat)

        # Fit scalers
        all_global = np.vstack(all_global)  # [N_events, 2]
        self.scaler_global.fit(all_global)

        if len(all_node) > 0:
            all_node = np.vstack(all_node)  # [N_total_objects, 5]
            self.scaler_node.fit(all_node)

        if len(all_edge) > 0:
            all_edge = np.vstack(all_edge)  # [N_total_edges, 2]
            self.scaler_edge.fit(all_edge)

        return self

    def transform(self, X):
        # X: [N_events, 92]
        graphs = []
        for i in range(X.shape[0]):
            event = X[i]
            global_feat, real_obj_feat, mask = self._extract_features(event)

            # Normalize features
            global_feat_norm = self.scaler_global.transform(global_feat.reshape(1, -1)).flatten()

            if real_obj_feat.shape[0] > 0:
                node_feat_norm = self.scaler_node.transform(real_obj_feat)
            else:
                node_feat_norm = real_obj_feat

            # Build graph
            graph = self._build_graph(global_feat_norm, node_feat_norm, mask)
            graphs.append(graph)

        return graphs


def make_preprocessor():
    return MyPreprocessor()


# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is a Data object from PyG
        node_feat_dim = sample_object.x.size(1)  # 5
        edge_feat_dim = sample_object.edge_attr.size(1) if sample_object.edge_attr.size(0) > 0 else 2  # 2
        global_feat_dim = 2

        # Node processing layers
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Edge processing layers
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # GAT layers
        self.gat1 = GATConv(128, 256, heads=4, dropout=0.2, edge_dim=64)
        self.gat2 = GATConv(256 * 4, 256, heads=4, dropout=0.2, edge_dim=64)
        self.gat3 = GATConv(256 * 4, 128, heads=4, dropout=0.2, edge_dim=64, concat=False)

        # Global feature processing
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Readout layers
        self.readout = nn.Sequential(
            nn.Linear(128 + 64, 256),  # Node features + global features
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x is a PyG Batch object
        # Encode nodes
        x = self.node_encoder(batch_x.x)  # [total_nodes, 128]

        # Encode edges
        if batch_x.edge_attr.size(0) > 0:
            edge_attr = self.edge_encoder(batch_x.edge_attr)  # [total_edges, 64]
        else:
            edge_attr = torch.zeros((batch_x.edge_index.size(1), 64), 
                                   device=batch_x.x.device)

        # Apply GAT layers
        x = F.relu(self.gat1(x, batch_x.edge_index, edge_attr))  # [total_nodes, 256*4]
        x = F.relu(self.gat2(x, batch_x.edge_index, edge_attr))  # [total_nodes, 256*4]
        x = F.relu(self.gat3(x, batch_x.edge_index, edge_attr))  # [total_nodes, 128]

        # Global pooling
        graph_x = global_mean_pool(x, batch_x.batch)  # [batch_size, 128]

        # Process global features (E_T_miss, phi_Et_miss)
        global_feat = batch_x.global_feat  # [batch_size, 2]
        global_encoded = self.global_encoder(global_feat)  # [batch_size, 64]

        # Concatenate pooled node features with global features
        combined = torch.cat([graph_x, global_encoded], dim=1)  # [batch_size, 192]

        # Final classification
        out = self.readout(combined)  # [batch_size, 1]
        return out.squeeze(-1)  # [batch_size]


def make_model(example_object):
    return BinaryClassifier(example_object)


# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )

    # Early stopping
    best_val_loss = float('inf')
    best_val_acc = 0
    patience = 10
    patience_counter = 0

    # Track metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch)
            targets = batch.y.float()

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track metrics
            total_train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train if total_train > 0 else 0

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # Forward pass
                outputs = model(batch)
                targets = batch.y.float()

                # Compute loss
                loss = criterion(outputs, targets)

                # Track metrics
                total_val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val if total_val > 0 else 0

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Update scheduler
        scheduler.step(val_acc)

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            model.load_state_dict(best_model_state)
            break

    # Return trained model and metrics
    return model, train_losses, val_losses, train_accs, val_accs

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

