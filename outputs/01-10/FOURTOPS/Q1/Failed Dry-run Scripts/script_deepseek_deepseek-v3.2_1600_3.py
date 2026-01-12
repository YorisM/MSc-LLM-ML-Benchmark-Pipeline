
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# -------------------------- START OF LLM BLOCK ------------------------------
# ---------- IMPORTS ----------
# Additional imports
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

#  -------- CUSTOM DATASET (PyG graphs) --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X
        self.y = y.long() if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.long)
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        event = self.X[idx]  # shape [92]
        label = self.y[idx]  # scalar

        # Extract global features
        et_miss = event[0:1]       # ETmiss magnitude [1]
        phi_et_miss = event[1:2]   # phi_ETmiss [1]
        global_feats = torch.cat([et_miss, phi_et_miss])  # [2]

        # Extract objects: reshape to [18, 5]
        objects = event[2:].reshape(18, 5)  # [18, 5]

        # Create mask for valid objects (obj_id != 0 or E > 0)
        valid_mask = (objects[:, 0] != 0) & (objects[:, 1] > 1e-6)  # [18]
        valid_objects = objects[valid_mask]  # [num_valid, 5]

        if len(valid_objects) == 0:
            # Fallback: use all objects if none valid (shouldn't happen)
            valid_objects = objects
            valid_mask = torch.ones(18, dtype=torch.bool)

        # Build node features: concatenate kinematics with global features
        # [num_valid, 5] -> [num_valid, 7] (add global features to each node)
        global_expanded = global_feats.unsqueeze(0).expand(valid_objects.size(0), 2)
        node_features = torch.cat([valid_objects, global_expanded], dim=1)  # [num_valid, 7]

        # Build fully connected edges between valid nodes
        num_nodes = node_features.size(0)
        if num_nodes > 1:
            adj_matrix = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
            edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()  # [2, num_edges]
        else:
            # Self-loop for single node
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        # Normalize node features
        node_features[:, 1:3] = torch.log1p(node_features[:, 1:3])  # log transform E, pT
        node_features[:, 3:5] = torch.tanh(node_features[:, 3:5])  # bound eta, phi

        # Create PyG Data object
        data = Data(x=node_features, edge_index=edge_index, y=label)
        data.num_nodes = num_nodes

        return data

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.means = None
        self.stds = None
        self.eps = 1e-8

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
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        X_np = X.numpy() if torch.is_tensor(X) else X
        self.means = np.nanmean(X_np, axis=0)
        self.stds = np.nanstd(X_np, axis=0) + self.eps
        return self

    def transform(self, X):
        if torch.is_tensor(X):
            X_norm = (X - torch.from_numpy(self.means).to(X.device)) / torch.from_numpy(self.stds).to(X.device)
        else:
            X_norm = (X - self.means) / self.stds
        return X_norm

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
# Lane B: PyTorch Geometric graphs
class ParticleGNN(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.pooling = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node encoding
        x = self.node_encoder(x)  # [num_nodes, hidden_dim]

        # GCN layers with residual connections
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2 + x1, edge_index))

        # Global pooling (concat mean and max)
        mean_pool = global_mean_pool(x3, batch)  # [batch_size, hidden_dim]
        max_pool = global_max_pool(x3, batch)    # [batch_size, hidden_dim]
        x_pool = torch.cat([mean_pool, max_pool], dim=1)  # [batch_size, hidden_dim*2]

        # Pooling and classification
        x_pool = self.pooling(x_pool)            # [batch_size, hidden_dim//2]
        logits = self.classifier(x_pool)         # [batch_size, 1]
        return logits.squeeze(-1)                # [batch_size]

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        hidden_dim = 256
        dropout = 0.3
        self.gnn = ParticleGNN(hidden_dim, dropout)

    def forward(self, batch_x):
        # batch_x is a PyG Batch object
        return self.gnn(batch_x)  # [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    patience = 10
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            outputs = model(data)
            labels = data.y.float()
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * data.num_graphs
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels.bool()).sum().item()
            total += data.num_graphs

            all_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total if total > 0 else 0
        train_auc = roc_auc_score(all_labels, all_preds) if all_labels else 0.5

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                outputs = model(data)
                labels = data.y.float()
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * data.num_graphs
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels.bool()).sum().item()
                val_total += data.num_graphs

                val_all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())

        val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_auc = roc_auc_score(val_all_labels, val_all_preds) if val_all_labels else 0.5

        # Update learning rate
        scheduler.step(val_auc)

        # Early stopping check
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
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

    # Load best model weights
    model.load_state_dict(best_model_wts)

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

