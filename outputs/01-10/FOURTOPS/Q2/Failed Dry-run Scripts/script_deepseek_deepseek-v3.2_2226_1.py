
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import stats
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, GATv2Conv, global_max_pool
from torch_geometric.utils import dense_to_sparse, add_self_loops
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler_means = None
        self.scaler_stds = None
        self.obj_id_min = None
        self.obj_id_max = None
        self.obj_id_embedding_dim = 8

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True if torch.cuda.is_available() else False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Compute statistics for normalization (only on non-zero padded entries)
        # Shape: [N, 92] -> we need to handle objects separately
        N = X_np.shape[0]

        # Extract all non-zero object entries (obj_id != 0)
        obj_features = []
        for i in range(N):
            for obj_idx in range(18):
                start_idx = 2 + obj_idx * 5
                obj_id = X_np[i, start_idx]
                if obj_id != 0:  # Real object
                    features = X_np[i, start_idx:start_idx+5]
                    obj_features.append(features)

        obj_features = np.array(obj_features)  # [M, 5]

        # Normalize continuous features (E, pT, eta, phi)
        # Skip obj_id (index 0) as it's categorical
        cont_features = obj_features[:, 1:]  # [M, 4]

        self.scaler_means = cont_features.mean(axis=0)  # [4]
        self.scaler_stds = cont_features.std(axis=0)    # [4]
        self.scaler_stds[self.scaler_stds < 1e-8] = 1.0

        # For obj_id embedding
        all_obj_ids = obj_features[:, 0]
        self.obj_id_min = int(all_obj_ids.min())
        self.obj_id_max = int(all_obj_ids.max())

        return self

    def transform(self, X):
        if torch.is_tensor(X):
            X_np = X.numpy()
        else:
            X_np = X

        N = X_np.shape[0]
        data_list = []

        for i in range(N):
            # Extract global features
            et_miss = X_np[i, 0]
            phi_et_miss = X_np[i, 1]

            # Extract objects
            objects = []
            valid_mask = []

            for obj_idx in range(18):
                start_idx = 2 + obj_idx * 5
                obj_id = X_np[i, start_idx]

                if obj_id != 0:  # Real object
                    E = X_np[i, start_idx + 1]
                    pT = X_np[i, start_idx + 2]
                    eta = X_np[i, start_idx + 3]
                    phi = X_np[i, start_idx + 4]

                    # Normalize continuous features
                    cont_features = np.array([E, pT, eta, phi])
                    cont_features = (cont_features - self.scaler_means) / self.scaler_stds

                    # Create node features: [obj_id (normalized), normalized E, pT, eta, phi, px, py, pz]
                    # Compute px, py, pz from pT, eta, phi
                    px = pT * np.cos(phi)
                    py = pT * np.sin(phi)
                    pz = pT * np.sinh(eta)

                    # Normalize px, py, pz (optional but helps training)
                    node_feat = np.array([
                        (obj_id - self.obj_id_min) / (self.obj_id_max - self.obj_id_min + 1e-8),
                        cont_features[0],  # E
                        cont_features[1],  # pT
                        cont_features[2],  # eta
                        cont_features[3],  # phi
                        px / 1000.0,  # Convert to GeV scale
                        py / 1000.0,
                        pz / 1000.0
                    ], dtype=np.float32)

                    objects.append(node_feat)
                    valid_mask.append(True)
                else:
                    valid_mask.append(False)

            if len(objects) == 0:
                # Create dummy object if no real objects (shouldn't happen)
                objects.append(np.zeros(8, dtype=np.float32))
                valid_mask = [True]

            # Build graph
            num_nodes = len(objects)
            node_features = np.array(objects)  # [num_nodes, 8]

            # Create complete graph (all pairs) for now
            # We'll let the model handle edge features
            edge_index = []
            for src in range(num_nodes):
                for dst in range(num_nodes):
                    if src != dst:
                        edge_index.append([src, dst])

            if len(edge_index) == 0:
                # Self-loop if only one node
                edge_index.append([0, 0])

            edge_index = np.array(edge_index).T  # [2, num_edges]

            # Create edge features: deltaR and invariant mass
            edge_attr = []
            for src, dst in edge_index.T:
                if src == dst:
                    # Self-loop: zero features
                    edge_attr.append([0.0, 0.0])
                else:
                    eta1 = node_features[src, 3] * self.scaler_stds[2] + self.scaler_means[2]
                    phi1 = node_features[src, 4] * self.scaler_stds[3] + self.scaler_means[3]
                    eta2 = node_features[dst, 3] * self.scaler_stds[2] + self.scaler_means[2]
                    phi2 = node_features[dst, 4] * self.scaler_stds[3] + self.scaler_means[3]

                    # deltaR
                    delta_eta = eta1 - eta2
                    delta_phi = phi1 - phi2
                    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
                    delta_r = np.sqrt(delta_eta**2 + delta_phi**2)

                    # Approximate invariant mass (simplified)
                    E1 = node_features[src, 1] * self.scaler_stds[0] + self.scaler_means[0]
                    E2 = node_features[dst, 1] * self.scaler_stds[0] + self.scaler_means[0]
                    px1 = node_features[src, 5] * 1000.0
                    py1 = node_features[src, 6] * 1000.0
                    pz1 = node_features[src, 7] * 1000.0
                    px2 = node_features[dst, 5] * 1000.0
                    py2 = node_features[dst, 6] * 1000.0
                    pz2 = node_features[dst, 7] * 1000.0

                    # M^2 = (E1+E2)^2 - (p1+p2)^2
                    E_sum = E1 + E2
                    px_sum = px1 + px2
                    py_sum = py1 + py2
                    pz_sum = pz1 + pz2
                    m2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
                    m = np.sqrt(max(0, m2)) / 1000.0  # Convert to GeV

                    edge_attr.append([delta_r, m])

            edge_attr = np.array(edge_attr, dtype=np.float32)

            # Add global features as node features
            # Normalize global features
            et_miss_norm = (et_miss - self.scaler_means[1]) / self.scaler_stds[1] if self.scaler_stds[1] > 0 else 0
            phi_et_miss_norm = phi_et_miss / np.pi  # Normalize to [-1, 1]

            # Add global features to all nodes
            global_feats = np.array([et_miss_norm, phi_et_miss_norm], dtype=np.float32)
            global_feats_expanded = np.tile(global_feats, (num_nodes, 1))  # [num_nodes, 2]

            node_features = np.concatenate([node_features, global_feats_expanded], axis=1)  # [num_nodes, 10]

            # Create PyG Data object
            data = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32) if len(edge_attr) > 0 else None
            )

            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

#  -------- CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.data_list = pre.transform(X)
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        self.y = y.long()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data.y = self.y[idx].unsqueeze(0)  # Graph-level label
        return data

# ---------- MODEL ARCHITECTURE ----------
class ParticleGNN(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim=256, output_dim=1, num_layers=4, heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.edge_dim = edge_dim

        # Node feature projection
        self.node_proj = nn.Linear(input_dim, hidden_dim)

        # Edge feature projection
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # GNN layers with residual connections
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            # GATv2Conv for attention-based message passing
            conv = GATv2Conv(
                hidden_dim, hidden_dim // heads, heads=heads,
                edge_dim=hidden_dim, dropout=0.1
            )
            self.gnn_layers.append(conv)

        # Batch norms
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        # FFNs after each GNN layer
        self.ffns = nn.ModuleList()
        for i in range(num_layers):
            ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.ffns.append(ffn)

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Global feature processor
        self.global_processor = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Project features
        x = F.elu(self.node_proj(x))  # [N, hidden_dim]

        if edge_attr is not None:
            edge_attr = F.elu(self.edge_proj(edge_attr))  # [E, hidden_dim]

        # GNN layers with residuals
        for conv, bn, ffn in zip(self.gnn_layers, self.batch_norms, self.ffns):
            # Message passing
            if edge_attr is not None:
                x_new = conv(x, edge_index, edge_attr)
            else:
                x_new = conv(x, edge_index)

            # Residual connection
            x_new = x + x_new
            x_new = bn(x_new)
            x_new = F.elu(x_new)

            # FFN
            x_new = ffn(x_new)
            x_new = x + x_new  # Another residual
            x = F.elu(x_new)

        # Global pooling
        x_mean = global_mean_pool(x, batch)  # [B, hidden_dim]
        x_max = global_max_pool(x, batch)    # [B, hidden_dim]
        x_pooled = torch.cat([x_mean, x_max], dim=1)  # [B, hidden_dim * 2]

        # Process global features (extract from node features)
        # Global features are the last 2 features of each node
        global_feats = data.x[:, -2:]  # [N, 2]
        # Aggregate per graph
        global_mean = global_mean_pool(global_feats, batch)  # [B, 2]
        global_processed = self.global_processor(global_mean)  # [B, hidden_dim//2]

        # Combine graph features with processed global features
        combined = torch.cat([x_pooled, global_processed], dim=1)  # [B, hidden_dim*2 + hidden_dim//2]

        # Final classification
        out = self.readout(combined)  # [B, 1]
        return out.squeeze(-1)  # [B]

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is a Data object from the dataset
        input_dim = sample_object.x.shape[1]  # Should be 10
        edge_dim = sample_object.edge_attr.shape[1] if sample_object.edge_attr is not None else 0

        # Create GNN
        self.gnn = ParticleGNN(
            input_dim=input_dim,
            edge_dim=edge_dim if edge_dim > 0 else 2,  # Default to 2 for deltaR, mass
            hidden_dim=256,
            output_dim=1,
            num_layers=4,
            heads=8
        )

    def forward(self, batch_x):
        # batch_x is a Batch object from PyG
        return self.gnn(batch_x)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 100

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-5
    )

    # Loss function with label smoothing
    criterion = nn.BCEWithLogitsLoss()

    # For AUC computation
    from sklearn.metrics import roc_auc_score

    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_aucs = []

    # Early stopping
    best_val_auc = 0.0
    patience = 20
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch)
            targets = batch.y.float().squeeze()

            # Compute loss
            loss = criterion(logits, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            train_loss += loss.item() * batch.num_graphs
            predictions = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (predictions == targets).sum().item()
            train_total += batch.num_graphs

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                logits = model(batch)
                targets = batch.y.float().squeeze()

                loss = criterion(logits, targets)
                val_loss += loss.item() * batch.num_graphs

                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).float()
                val_correct += (predictions == targets).sum().item()
                val_total += batch.num_graphs

                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Compute AUC
        val_auc = roc_auc_score(all_targets, all_probs)

        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)

        # Update scheduler
        scheduler.step()

        # Early stopping based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# ---------------------------  END OF LLM-CODE BLOCK  ---------------------------

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

