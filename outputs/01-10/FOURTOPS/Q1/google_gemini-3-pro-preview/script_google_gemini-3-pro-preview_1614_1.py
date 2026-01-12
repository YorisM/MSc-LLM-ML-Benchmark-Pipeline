
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
# <start code template>
# ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool
from sklearn.metrics import roc_auc_score

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        # pre.transform returns a Tensor (N, 18, 9)
        self.X = pre.transform(X) if pre is not None else X
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        self.y = y.long()

        # Precompute edge indices for fully connected graphs of size 1 to 18
        # This saves significant overhead during __getitem__
        self.edge_indices = {}
        for n in range(1, 19):
            row = torch.arange(n, dtype=torch.long)
            # Fully connected with self-loops: all-to-all
            # cartesian_prod returns (N*N, 2), so we transpose to (2, NumEdges)
            edges = torch.cartesian_prod(row, row).t()
            self.edge_indices[n] = edges

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        # self.X[idx] shape: (18, 9)
        # Features map:
        # 0: is_real (mask), 1: id, 2: logE, 3: logpT, 4: eta, 
        # 5: sin_phi, 6: cos_phi, 7: logMET, 8: cos(dphi_met)
        feats = self.X[idx]

        # Determine real nodes (non-padded)
        mask_idx = feats[:, 0] > 0.5
        x = feats[mask_idx] # (num_real_nodes, 9)

        num_nodes = x.size(0)

        if num_nodes == 0:
            # Fallback for completely empty event (should be rare)
            x = torch.zeros((1, 9), dtype=feats.dtype)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            # Use precomputed edge index
            edge_index = self.edge_indices[num_nodes]

        y = self.y[idx].unsqueeze(0) # (1,) Long

        # Return PyG Data object
        return Data(x=x, edge_index=edge_index, y=y)

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.stats = {}

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            # Use PyG DataLoader for batched graphs
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 256, 
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "collate": None, 
            "eval_overrides": {"shuffle": False, "batch_size": 256}
        }

    def fit(self, X, y=None):
        if torch.is_tensor(X):
            X = X.numpy()

        # Layout: MET(0), METphi(1), and 18 blocks of 5 [id, E, pT, eta, phi] starting at 2
        objs = X[:, 2:].reshape(-1, 18, 5)
        pT = objs[:, :, 2]

        # Mask valid objects (pT > epsilon)
        mask = pT > 1e-4
        valid_objs = objs[mask] # Flattened valid objects

        # Compute stats for normalization
        logE = np.log1p(valid_objs[:, 1])
        logpT = np.log1p(valid_objs[:, 2])
        eta = valid_objs[:, 3]

        met = X[:, 0]
        logMET = np.log1p(met)

        self.stats['mean_logE'] = logE.mean()
        self.stats['std_logE'] = logE.std() + 1e-6
        self.stats['mean_logpT'] = logpT.mean()
        self.stats['std_logpT'] = logpT.std() + 1e-6
        self.stats['mean_eta'] = eta.mean()
        self.stats['std_eta'] = eta.std() + 1e-6
        self.stats['mean_logMET'] = logMET.mean()
        self.stats['std_logMET'] = logMET.std() + 1e-6

        return self

    def transform(self, X):
        if hasattr(X, "numpy"):
            X = X.cpu().numpy()

        N = X.shape[0]
        objs = X[:, 2:].reshape(N, 18, 5)
        met = X[:, 0] 
        met_phi = X[:, 1]

        pT = objs[:, :, 2]
        # Identify padding
        mask = (pT > 1e-4).astype(np.float32)

        # --- Feature Engineering ---
        # 1. is_real (mask status)
        f0 = mask

        # 2. obj ID (scaled roughly)
        # IDs can be large (PDG ID), scaling makes them robust for NN
        f1 = objs[:, :, 0] * 0.01 

        # 3. logE (standardized)
        f2 = (np.log1p(objs[:, :, 1]) - self.stats['mean_logE']) / self.stats['std_logE']

        # 4. logpT (standardized)
        f3 = (np.log1p(pT) - self.stats['mean_logpT']) / self.stats['std_logpT']

        # 5. eta (standardized)
        f4 = (objs[:, :, 3] - self.stats['mean_eta']) / self.stats['std_eta']

        # 6,7. sin/cos phi
        phi = objs[:, :, 4]
        f5 = np.sin(phi)
        f6 = np.cos(phi)

        # 8. logMET (Global, replicated to all nodes)
        f_met = (np.log1p(met) - self.stats['mean_logMET']) / self.stats['std_logMET']
        f7 = f_met[:, np.newaxis] * np.ones((1, 18))

        # 9. relative angle to MET (Global interaction)
        dphi = phi - met_phi[:, np.newaxis]
        f8 = np.cos(dphi)

        # Stack all features -> (N, 18, 9)
        out = np.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8], axis=2)

        # Apply mask to set padded values strictly to 0 (cleanliness)
        out = out * mask[:, :, np.newaxis]

        return torch.tensor(out, dtype=torch.float32)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is a PyG Data Batch
        in_dim = sample_object.x.shape[1] # Should be 9

        # Graph Transformer Architecture
        hidden_dim = 64
        heads = 4
        dropout = 0.1

        self.conv1 = TransformerConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

        self.conv2 = TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)

        self.conv3 = TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.bn3 = nn.BatchNorm1d(hidden_dim * heads)

        # Global Pooling & MLP Head
        full_dim = 2 * hidden_dim * heads # mean + max
        self.lin1 = nn.Linear(full_dim, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)

        # Global Pooling (Mean + Max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # MLP Head
        x = self.dropout(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)

        return x # Logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 10 
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # One Cycle policy for better convergence in few epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs
    )
    criterion = nn.BCEWithLogitsLoss()

    device = next(model.parameters()).device

    for epoch in range(epochs):
        # Training Phase
        model.train()
        all_tr_preds, all_tr_targets = [], []
        total_tr_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch) 
            # batch.y is Long, BCE needs Float (B, 1)
            target = batch.y.float().unsqueeze(1)

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_tr_loss += loss.item() * batch.num_graphs

            with torch.no_grad():
                all_tr_preds.append(torch.sigmoid(out).cpu().numpy())
                all_tr_targets.append(target.cpu().numpy())

        train_loss = total_tr_loss / len(train_loader.dataset)
        train_auc = roc_auc_score(np.concatenate(all_tr_targets), np.concatenate(all_tr_preds))

        # Validation Phase
        model.eval()
        all_val_preds, all_val_targets = [], []
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                target = batch.y.float().unsqueeze(1)

                loss = criterion(out, target)
                total_val_loss += loss.item() * batch.num_graphs

                all_val_preds.append(torch.sigmoid(out).cpu().numpy())
                all_val_targets.append(target.cpu().numpy())

        val_loss = total_val_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(np.concatenate(all_val_targets), np.concatenate(all_val_preds))

        print(f"Epoch {epoch+1}/{epochs} | Tr Loss: {train_loss:.4f} Tr AUC: {train_auc:.4f} | Va Loss: {val_loss:.4f} Va AUC: {val_auc:.4f}")

    # Return final AUC as accuracy metric
    return model, train_loss, val_loss, train_auc, val_auc
# <end code template>
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

