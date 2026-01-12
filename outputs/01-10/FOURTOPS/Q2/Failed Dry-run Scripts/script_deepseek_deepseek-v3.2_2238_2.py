
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

# -------------------------- START OF LLM BLOCK ------------------------------
# ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional
import math

#  -------- CUSTOM DATASET FOR GRAPH CONSTRUCTION --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        # Transform data through preprocessor
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # self.X[idx] is already a PyG Data object from the preprocessor
        return self.X[idx], self.y[idx]

# ----------- PRE-PROCESSOR WITH GRAPH CONSTRUCTION ----------
class MyPreprocessor:
    def __init__(self):
        # State for normalization
        self.global_mean = None
        self.global_std = None
        self.object_mean = None
        self.object_std = None
        self.obj_type_min = None
        self.obj_type_max = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",   # Use our custom dataset
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",  # PyG DataLoader for graphs
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        # X shape: [N_samples, 92]
        # Extract global and object features for statistics
        n_samples = X.shape[0]

        # Global features: ET_miss, phi_Et_miss
        global_features = X[:, :2].reshape(-1, 2)  # [N_samples*1, 2]

        # Object features: reshape to [N_samples*18, 5]
        obj_features = X[:, 2:].reshape(-1, 5)    # [N_samples*18, 5]

        # Mask for real objects (obj_type != 0)
        obj_type = obj_features[:, 0]
        mask = obj_type != 0
        real_obj_features = obj_features[mask]

        # Compute statistics
        self.global_mean = global_features.mean(axis=0)
        self.global_std = global_features.std(axis=0) + 1e-8

        if len(real_obj_features) > 0:
            # For object kinematic features (E, pT, eta, phi)
            obj_kinematics = real_obj_features[:, 1:]  # [N_real_objects, 4]
            self.object_mean = obj_kinematics.mean(axis=0)
            self.object_std = obj_kinematics.std(axis=0) + 1e-8
        else:
            self.object_mean = np.zeros(4, dtype=np.float32)
            self.object_std = np.ones(4, dtype=np.float32)

        # Object type statistics for embedding
        self.obj_type_min = int(obj_type.min())
        self.obj_type_max = int(obj_type.max())

        return self

    def _compute_pairwise_features(self, obj_vectors):
        # obj_vectors: [N_objects, 6] where columns: [eta, phi, px, py, pz, E]
        n_objects = obj_vectors.shape[0]
        if n_objects < 2:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.int64)

        # Compute deltaR and invariant mass for all pairs
        indices_i, indices_j = np.triu_indices(n_objects, k=1)

        # Extract features
        eta_i = obj_vectors[indices_i, 0]
        eta_j = obj_vectors[indices_j, 0]
        phi_i = obj_vectors[indices_i, 1]
        phi_j = obj_vectors[indices_j, 1]
        px_i = obj_vectors[indices_i, 2]
        px_j = obj_vectors[indices_j, 2]
        py_i = obj_vectors[indices_i, 3]
        py_j = obj_vectors[indices_j, 3]
        pz_i = obj_vectors[indices_i, 4]
        pz_j = obj_vectors[indices_j, 4]
        E_i = obj_vectors[indices_i, 5]
        E_j = obj_vectors[indices_j, 5]

        # DeltaR
        delta_eta = eta_i - eta_j
        delta_phi = phi_i - phi_j
        delta_phi = np.mod(delta_phi + np.pi, 2*np.pi) - np.pi
        delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

        # Invariant mass: m^2 = (E_i + E_j)^2 - (p_i + p_j)^2
        sum_E = E_i + E_j
        sum_px = px_i + px_j
        sum_py = py_i + py_j
        sum_pz = pz_i + pz_j
        sum_p2 = sum_px**2 + sum_py**2 + sum_pz**2

        m2 = sum_E**2 - sum_p2
        # Handle negative m^2 due to numerical errors
        m2 = np.maximum(m2, 0.0)
        inv_mass = np.sqrt(m2)

        # Stack features
        pair_features = np.stack([delta_R, inv_mass], axis=1)

        # Edge indices
        edge_indices = np.stack([indices_i, indices_j], axis=0)

        return pair_features, edge_indices

    def transform(self, X):
        from torch_geometric.data import Data

        # Normalize global features
        global_norm = (X[:, :2] - self.global_mean) / self.global_std

        # Process each event into a graph
        data_list = []
        for i in range(X.shape[0]):
            # Extract objects for this event
            event = X[i]
            obj_features = event[2:].reshape(-1, 5)  # [18, 5]

            # Find real objects (obj_type != 0)
            mask = obj_features[:, 0] != 0
            real_objects = obj_features[mask]
            n_real = real_objects.shape[0]

            if n_real == 0:
                # Create minimal graph with just global node
                x = torch.tensor(global_norm[i:i+1], dtype=torch.float32)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 2), dtype=torch.float32)
            else:
                # Prepare object features
                obj_type = real_objects[:, 0].astype(int)
                obj_kinematics = real_objects[:, 1:]  # [E, pT, eta, phi]

                # Normalize kinematics
                obj_kinematics_norm = (obj_kinematics - self.object_mean) / self.object_std

                # Convert to Cartesian coordinates for invariant mass calculation
                pT = obj_kinematics[:, 1]
                eta = obj_kinematics[:, 2]
                phi = obj_kinematics[:, 3]
                E = obj_kinematics[:, 0]

                px = pT * np.cos(phi)
                py = pT * np.sin(phi)
                pz = pT * np.sinh(eta)

                obj_vectors = np.stack([eta, phi, px, py, pz, E], axis=1)

                # Compute pairwise features
                pair_features, edge_idx = self._compute_pairwise_features(obj_vectors)

                # Create node features: [obj_type_embedding, normalized kinematics]
                # We'll embed obj_type in the model, here just pass raw types
                node_features = np.concatenate([
                    obj_type.reshape(-1, 1),          # [n_real, 1]
                    obj_kinematics_norm               # [n_real, 4]
                ], axis=1)

                # Add global node as node 0
                global_node_feat = global_norm[i].reshape(1, -1)  # [1, 2]
                # Pad global node features to match object node dimension
                global_node_feat_padded = np.zeros((1, 5), dtype=np.float32)
                global_node_feat_padded[:, :2] = global_node_feat
                global_node_feat_padded[:, 2] = -1  # Special marker for global node

                # Combine all nodes: global node + object nodes
                x_nodes = np.vstack([global_node_feat_padded, node_features])

                # Add edges from global node to all objects
                n_nodes = n_real + 1
                global_to_obj_edges = np.array([
                    [0] * n_real,                     # source: global node
                    list(range(1, n_real + 1))        # targets: object nodes
                ])

                # Combine all edges: object-object + global-object
                if edge_idx.size > 0:
                    # Offset object-object edges by 1 (global node is 0)
                    edge_idx_offset = edge_idx + 1
                    all_edge_idx = np.concatenate([edge_idx_offset, global_to_obj_edges], axis=1)
                else:
                    all_edge_idx = global_to_obj_edges

                # Edge features: for global-object edges, use zeros
                if pair_features.shape[0] > 0:
                    global_edge_features = np.zeros((n_real, 2), dtype=np.float32)
                    all_edge_features = np.vstack([pair_features, global_edge_features])
                else:
                    all_edge_features = np.zeros((n_real, 2), dtype=np.float32)

                # Convert to tensors
                x = torch.tensor(x_nodes, dtype=torch.float32)
                edge_index = torch.tensor(all_edge_idx, dtype=torch.long)
                edge_attr = torch.tensor(all_edge_features, dtype=torch.float32)

            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- GRAPH NEURAL NETWORK MODEL ----------
class ParticleGNN(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is a Data object from the dataset
        node_dim = sample_object.x.size(1)  # Should be 5
        edge_dim = sample_object.edge_attr.size(1) if sample_object.edge_attr is not None else 0

        # Object type embedding (global node has type -1)
        self.obj_type_embedding = nn.Embedding(num_embeddings=100, embedding_dim=16, padding_idx=99)

        # Actual input dimension: 16 (type embedding) + 4 (kinematics)
        input_dim = 16 + 4

        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Graph Transformer layers
        self.conv1 = nn.TransformerConv(input_dim, 128, edge_dim=32)
        self.conv2 = nn.TransformerConv(128, 128, edge_dim=32)
        self.conv3 = nn.TransformerConv(128, 128, edge_dim=32)

        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, batch_x):
        from torch_geometric.data import Batch

        # batch_x is a Batch object from PyG DataLoader
        if isinstance(batch_x, Batch):
            data = batch_x
        else:
            data = batch_x

        # Extract node features: first column is object type
        obj_types = data.x[:, 2].long()  # Object type is in column 2
        # Handle global node (type -1)
        obj_types = torch.where(obj_types == -1, torch.tensor(99, device=obj_types.device), obj_types)
        obj_types = torch.clamp(obj_types, 0, 99)

        kinematics = data.x[:, [0, 1, 3, 4]]  # Global features + normalized kinematics

        # Embed object types
        type_embedding = self.obj_type_embedding(obj_types)

        # Combine features
        x = torch.cat([type_embedding, kinematics], dim=1)

        # Process edge features
        edge_attr_encoded = self.edge_encoder(data.edge_attr)

        # Graph convolution layers
        x = F.relu(self.conv1(x, data.edge_index, edge_attr_encoded))
        x = F.relu(self.conv2(x, data.edge_index, edge_attr_encoded))
        x = F.relu(self.conv3(x, data.edge_index, edge_attr_encoded))

        # Global pooling (graph-level readout)
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Sum pooling over nodes in each graph
        unique_batches = torch.unique(batch)
        graph_embeddings = []
        for b in unique_batches:
            mask = batch == b
            graph_embed = x[mask].mean(dim=0, keepdim=True)
            graph_embeddings.append(graph_embed)

        graph_embeddings = torch.cat(graph_embeddings, dim=0)

        # Classifier
        out = self.classifier(graph_embeddings)
        return out.squeeze(-1)

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.gnn = ParticleGNN(sample_object)

    def forward(self, batch_x):
        return self.gnn(batch_x)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))

    best_val_auc = 0.0
    best_model_state = None
    patience = 15
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.float().to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            # Compute accuracy
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == targets.bool()).sum().item()
            total += targets.size(0)

            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_preds = []
        val_labels = []

        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device)
                targets = targets.float().to(device)

                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == targets.bool()).sum().item()
                val_total += targets.size(0)

                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Compute AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except:
            val_auc = 0.5

        # Early stopping based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Val AUC: {val_auc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# --------------------------- END OF LLM-CODE BLOCK ---------------------------

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

