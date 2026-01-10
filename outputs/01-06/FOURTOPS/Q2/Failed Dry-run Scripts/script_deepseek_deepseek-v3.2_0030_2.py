
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, GCNConv
import math
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

#  -------- CUSTOM DATASET FOR GRAPH CONVERSION --------
class GraphDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.graphs = pre.transform(X)  # Returns list of PyG Data objects
        self.y = y  # Keep original labels for validation
        self.train = train

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        # Ensure label is scalar for graph classification
        if hasattr(data, 'y'):
            data.y = torch.tensor([data.y], dtype=torch.long)
        else:
            data.y = torch.tensor([self.y[idx]], dtype=torch.long)
        return data

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.node_feature_stats = None
        self.edge_feature_stats = None
        self.obj_id_mapping = None
        self.node_norm_mean = None
        self.node_norm_std = None
        self.edge_norm_mean = None
        self.edge_norm_std = None

    def _extract_physics_features(self, X_batch):
        """Convert flat tensor to graph representation with physics features."""
        batch_size = X_batch.shape[0]
        graphs = []

        for i in range(batch_size):
            event = X_batch[i]

            # Extract global features [2]
            met = event[0].item()
            met_phi = event[1].item()

            # Extract object features [18 objects × 5 features]
            obj_features = event[2:].reshape(-1, 5)  # [18, 5]

            # Identify non-zero objects (obj_id != 0)
            obj_ids = obj_features[:, 0]  # First column is obj_id
            non_zero_mask = obj_ids != 0
            valid_objs = obj_features[non_zero_mask]  # [N_valid, 5]

            if len(valid_objs) == 0:
                # Create dummy graph with global features only
                x = torch.tensor([[met, met_phi, 0, 0, 0]], dtype=torch.float32)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 4), dtype=torch.float32)
            else:
                N = len(valid_objs)

                # Node features: [E, pT, eta, phi, obj_id, met, met_phi]
                # Convert obj_id to one-hot or keep as categorical
                obj_ids_norm = valid_objs[:, 0:1] / 10.0  # Normalize obj_id
                node_feats = torch.cat([
                    valid_objs[:, 1:5],  # [E, pT, eta, phi]
                    obj_ids_norm,  # Normalized obj_id
                    torch.ones(N, 1) * met,  # MET repeated
                    torch.ones(N, 1) * met_phi  # MET phi repeated
                ], dim=1)  # [N, 7]

                # Create fully connected graph
                edge_index = []
                edge_features = []

                for j in range(N):
                    for k in range(j+1, N):
                        edge_index.append([j, k])
                        edge_index.append([k, j])  # Undirected

                        # Compute physics-based edge features
                        p1 = valid_objs[j]
                        p2 = valid_objs[k]

                        # Delta R
                        deta = p1[3] - p2[3]  # eta difference
                        dphi = p1[4] - p2[4]  # phi difference
                        # Wrap phi difference to [-pi, pi]
                        while dphi > math.pi:
                            dphi -= 2*math.pi
                        while dphi < -math.pi:
                            dphi += 2*math.pi
                        delta_r = torch.sqrt(deta**2 + dphi**2)

                        # Approximate invariant mass (simplified)
                        E1, E2 = p1[1], p2[1]
                        pT1, pT2 = p1[2], p2[2]
                        eta1, eta2 = p1[3], p2[3]
                        phi1, phi2 = p1[4], p2[4]

                        # Calculate 3-momentum components
                        px1 = pT1 * torch.cos(phi1)
                        py1 = pT1 * torch.sin(phi1)
                        pz1 = pT1 * torch.sinh(eta1)

                        px2 = pT2 * torch.cos(phi2)
                        py2 = pT2 * torch.sin(phi2)
                        pz2 = pT2 * torch.sinh(eta2)

                        # Invariant mass squared
                        m2 = (E1 + E2)**2 - ((px1 + px2)**2 + (py1 + py2)**2 + (pz1 + pz2)**2)
                        m2 = torch.clamp(m2, min=1e-6)
                        inv_mass = torch.sqrt(m2) / 1000.0  # Convert to GeV

                        # Additional physics features
                        sum_pt = pT1 + pT2
                        pt_ratio = torch.min(pT1, pT2) / torch.max(pT1, pT2)

                        edge_feat = torch.tensor([
                            delta_r.item(),
                            inv_mass.item(),
                            sum_pt.item() / 1000.0,  # Convert to GeV
                            pt_ratio.item()
                        ], dtype=torch.float32)
                        edge_features.append(edge_feat)
                        edge_features.append(edge_feat)  # Both directions same

                edge_index = torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.stack(edge_features) if edge_features else torch.zeros((0, 4), dtype=torch.float32)
                x = node_feats.float()

            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([0], dtype=torch.long)  # Placeholder
            )
            graphs.append(data)

        return graphs

    def _normalize_features(self, graphs, fit=False):
        """Normalize node and edge features."""
        if fit:
            # Collect all node features
            all_node_feats = []
            all_edge_feats = []
            for g in graphs:
                all_node_feats.append(g.x)
                if g.edge_attr.shape[0] > 0:
                    all_edge_feats.append(g.edge_attr)

            node_feats = torch.cat(all_node_feats, dim=0)
            edge_feats = torch.cat(all_edge_feats, dim=0) if all_edge_feats else torch.zeros((0, 4))

            # Compute statistics
            self.node_norm_mean = node_feats.mean(dim=0)
            self.node_norm_std = node_feats.std(dim=0) + 1e-8

            if edge_feats.shape[0] > 0:
                self.edge_norm_mean = edge_feats.mean(dim=0)
                self.edge_norm_std = edge_feats.std(dim=0) + 1e-8
            else:
                self.edge_norm_mean = torch.zeros(4)
                self.edge_norm_std = torch.ones(4)

        # Apply normalization
        normalized_graphs = []
        for g in graphs:
            # Normalize node features
            g.x = (g.x - self.node_norm_mean) / self.node_norm_std

            # Normalize edge features if present
            if g.edge_attr.shape[0] > 0:
                g.edge_attr = (g.edge_attr - self.edge_norm_mean) / self.edge_norm_std

            normalized_graphs.append(g)

        return normalized_graphs

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:GraphDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {"follow_batch": [], "exclude_keys": []},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        # Convert batch to graphs
        graphs = self._extract_physics_features(X)
        # Compute normalization statistics
        self._normalize_features(graphs, fit=True)
        return self

    def transform(self, X):
        graphs = self._extract_physics_features(X)
        graphs = self._normalize_features(graphs, fit=False)
        return graphs

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Sample object is a PyG Data object from the dataset
        node_feat_dim = sample_object.x.shape[1]  # Typically 7
        edge_feat_dim = sample_object.edge_attr.shape[1] if sample_object.edge_attr.shape[0] > 0 else 4

        # Graph Attention Networks with edge features
        self.gat1 = GATConv(node_feat_dim, 128, edge_dim=edge_feat_dim)
        self.gat2 = GATConv(128, 256, edge_dim=edge_feat_dim)
        self.gat3 = GATConv(256, 512, edge_dim=edge_feat_dim)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

        # Global pooling then MLP
        self.mlp = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(512, 256)),
            ('bn1', nn.BatchNorm1d(256)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.3)),
            ('lin2', nn.Linear(256, 128)),
            ('bn2', nn.BatchNorm1d(128)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.2)),
            ('lin3', nn.Linear(128, 1))
        ]))

    def forward(self, batch):
        # batch is a PyG Batch object
        x, edge_index, edge_attr, batch_vector = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        # Apply GAT layers with residual connections
        x1 = F.elu(self.gat1(x, edge_index, edge_attr))
        x1 = self.bn1(x1)

        x2 = F.elu(self.gat2(x1, edge_index, edge_attr))
        x2 = self.bn2(x2)

        x3 = self.gat3(x2, edge_index, edge_attr)

        # Global mean pooling
        graph_emb = global_mean_pool(x3, batch_vector)

        # Final classification
        out = self.mlp(graph_emb)
        return out.squeeze(-1)  # [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 80

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 10

    # AUC calculation helper
    from sklearn.metrics import roc_auc_score

    def compute_auc(loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                preds = torch.sigmoid(model(batch))
                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        # Convert to binary labels (assuming y is 0/1)
        all_labels = (all_labels > 0.5).astype(int)
        return roc_auc_score(all_labels, all_preds)

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch)
            labels = batch.y.float().squeeze()

            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_train_loss += loss.item() * batch.num_graphs
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += batch.num_graphs

        train_loss = epoch_train_loss / len(train_loader.dataset)
        train_acc = correct / total if total > 0 else 0

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                labels = batch.y.float().squeeze()

                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * batch.num_graphs

                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += batch.num_graphs

        val_loss = epoch_val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        # Compute AUC
        val_auc = compute_auc(val_loader)

        # Update scheduler based on AUC
        scheduler.step(val_auc)

        # Early stopping based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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

