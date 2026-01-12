
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
import math
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score
import torch_geometric
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.obj_feat_mean = None
        self.obj_feat_std = None
        self.global_feat_mean = None
        self.global_feat_std = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def _extract_objects(self, X_flat):
        batch_size = X_flat.shape[0]
        # Reshape to [batch, 18, 5]
        objects = X_flat[:, 2:].reshape(batch_size, 18, 5)  # [B, 18, 5]
        return objects

    def _compute_pairwise_features(self, objects):
        B, N, _ = objects.shape
        # objects: [B, N, 5] where last 4 are E, pT, eta, phi
        E = objects[..., 1]  # [B, N]
        pT = objects[..., 2]
        eta = objects[..., 3]
        phi = objects[..., 4]

        # Compute px, py, pz
        px = pT * torch.cos(phi)  # [B, N]
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)

        # Prepare for pairwise computation
        E1 = E.unsqueeze(2)  # [B, N, 1]
        E2 = E.unsqueeze(1)  # [B, 1, N]
        px1 = px.unsqueeze(2)
        px2 = px.unsqueeze(1)
        py1 = py.unsqueeze(2)
        py2 = py.unsqueeze(1)
        pz1 = pz.unsqueeze(2)
        pz2 = pz.unsqueeze(1)

        # Invariant mass: m^2 = (E1+E2)^2 - (p1+p2)^2
        E_sum = E1 + E2
        px_sum = px1 + px2
        py_sum = py1 + py2
        pz_sum = pz1 + pz2
        p_sum_sq = px_sum**2 + py_sum**2 + pz_sum**2
        m2 = E_sum**2 - p_sum_sq
        m2 = torch.clamp(m2, min=1e-8)
        inv_mass = torch.sqrt(m2)  # [B, N, N]

        # Delta R
        eta1 = eta.unsqueeze(2)
        eta2 = eta.unsqueeze(1)
        phi1 = phi.unsqueeze(2)
        phi2 = phi.unsqueeze(1)

        dphi = torch.abs(phi1 - phi2)
        dphi = torch.min(dphi, 2*math.pi - dphi)
        deta = eta1 - eta2
        deltaR = torch.sqrt(dphi**2 + deta**2)  # [B, N, N]

        return inv_mass, deltaR

    def fit(self, X, y=None):
        X_tensor = torch.as_tensor(X) if not torch.is_tensor(X) else X
        # Global features (first 2)
        global_feats = X_tensor[:, :2]
        self.global_feat_mean = global_feats.mean(dim=0, keepdim=True)
        self.global_feat_std = global_feats.std(dim=0, keepdim=True) + 1e-8

        # Object features
        objects = self._extract_objects(X_tensor)  # [N_samples, 18, 5]
        # Flatten objects across batch and objects, keep features
        obj_flat = objects.reshape(-1, 5)  # [N_samples*18, 5]
        # Mask for valid objects (obj_id != 0)
        mask = obj_flat[:, 0] != 0
        if mask.any():
            valid_obj = obj_flat[mask, 1:]  # Exclude obj_id
            self.obj_feat_mean = valid_obj.mean(dim=0, keepdim=True)  # [1, 4]
            self.obj_feat_std = valid_obj.std(dim=0, keepdim=True) + 1e-8
        else:
            self.obj_feat_mean = torch.zeros(1, 4)
            self.obj_feat_std = torch.ones(1, 4)
        return self

    def transform(self, X):
        X_tensor = torch.as_tensor(X) if not torch.is_tensor(X) else X
        batch_size = X_tensor.shape[0]

        # Normalize global features
        global_feats = (X_tensor[:, :2] - self.global_feat_mean) / self.global_feat_std

        # Extract objects
        objects = self._extract_objects(X_tensor)  # [B, 18, 5]
        obj_ids = objects[..., 0].long()  # [B, 18]

        # Normalize object kinematic features
        obj_kin = objects[..., 1:]  # [B, 18, 4]
        obj_kin_norm = (obj_kin - self.obj_feat_mean) / self.obj_feat_std

        # Compute pairwise features
        inv_mass, deltaR = self._compute_pairwise_features(objects)  # [B, 18, 18]

        # Build PyG Data objects for each event
        data_list = []
        for i in range(batch_size):
            # Mask for valid objects in this event
            mask = obj_ids[i] != 0
            valid_indices = torch.where(mask)[0]

            if len(valid_indices) == 0:
                # No valid objects (should not happen), create dummy
                x = torch.zeros(1, 4, dtype=torch.float32)
                edge_index = torch.zeros(2, 0, dtype=torch.long)
                edge_attr = torch.zeros(0, 2, dtype=torch.float32)
            else:
                # Node features: normalized kinematics for valid objects
                x = obj_kin_norm[i][valid_indices]  # [num_valid, 4]

                # Create fully connected edges (excluding self-loops)
                num_valid = len(valid_indices)
                rows, cols = [], []
                for a in range(num_valid):
                    for b in range(num_valid):
                        if a != b:
                            rows.append(a)
                            cols.append(b)
                edge_index = torch.tensor([rows, cols], dtype=torch.long)  # [2, E]

                # Edge features: inv_mass and deltaR for each edge
                edge_attr = torch.stack([
                    inv_mass[i][valid_indices[rows], valid_indices[cols]],
                    deltaR[i][valid_indices[rows], valid_indices[cols]]
                ], dim=1)  # [E, 2]

            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([0], dtype=torch.long)  # dummy, will be set later
            )
            data.global_feats = global_feats[i]  # store as attribute
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Sample object is a Data instance from PyG
        node_in = sample_object.x.shape[1]  # 4 kinematic features
        edge_in = sample_object.edge_attr.shape[1] if sample_object.edge_attr is not None else 0  # 2

        # Node feature embedding
        self.node_embed = Linear(node_in, 64)

        # Edge feature embedding
        self.edge_embed = Linear(edge_in, 32) if edge_in > 0 else None

        # GNN layers
        self.gat1 = GATv2Conv(64, 128, edge_dim=32 if edge_in > 0 else None, heads=4, dropout=0.2)
        self.gat2 = GATv2Conv(128*4, 256, edge_dim=32 if edge_in > 0 else None, heads=4, dropout=0.2)
        self.gat3 = GATv2Conv(256*4, 512, edge_dim=32 if edge_in > 0 else None, heads=4, dropout=0.2)

        # Batch norms
        self.bn1 = BatchNorm1d(128*4)
        self.bn2 = BatchNorm1d(256*4)
        self.bn3 = BatchNorm1d(512*4)

        # Global feature processing
        self.global_embed = Linear(2, 64)

        # Readout layers
        self.lin1 = Linear(512*4 + 64, 256)
        self.lin2 = Linear(256, 128)
        self.lin3 = Linear(128, 1)

        # Dropout
        self.dropout = Dropout(0.3)

    def forward(self, batch_x):
        # batch_x is a PyG Batch object
        x, edge_index, edge_attr, batch = batch_x.x, batch_x.edge_index, batch_x.edge_attr, batch_x.batch

        # Node embedding
        x = F.relu(self.node_embed(x))  # [N, 64]

        # Edge embedding
        if edge_attr is not None and self.edge_embed is not None:
            edge_attr = F.relu(self.edge_embed(edge_attr))  # [E, 32]

        # GNN layers
        x = self.gat1(x, edge_index, edge_attr)  # [N, 128*4]
        x = F.relu(self.bn1(x))
        x = self.dropout(x)

        x = self.gat2(x, edge_index, edge_attr)
        x = F.relu(self.bn2(x))
        x = self.dropout(x)

        x = self.gat3(x, edge_index, edge_attr)
        x = F.relu(self.bn3(x))
        x = self.dropout(x)

        # Global pooling
        x_pool = global_mean_pool(x, batch)  # [B, 512*4]

        # Process global features (stored as attribute in each data)
        global_feats = batch_x.global_feats  # [B, 2]
        global_emb = F.relu(self.global_embed(global_feats))  # [B, 64]

        # Concatenate
        x = torch.cat([x_pool, global_emb], dim=1)  # [B, 512*4 + 64]

        # Readout MLP
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.lin3(x)  # [B, 1]

        return x.squeeze(-1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_auc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        for batch in train_loader:
            batch = batch.to(device)
            labels = batch.y.float().squeeze()  # [B]
            optimizer.zero_grad()
            logits = model(batch)  # [B]
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.sigmoid(logits).detach()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        train_loss = total_loss / len(train_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        train_auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())
        train_acc = accuracy_score(all_labels.numpy(), (all_preds > 0.5).numpy())

        # Validation
        model.eval()
        total_val_loss = 0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                labels = batch.y.float().squeeze()
                logits = model(batch)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                preds = torch.sigmoid(logits)
                all_val_preds.append(preds.cpu())
                all_val_labels.append(labels.cpu())

        val_loss = total_val_loss / len(val_loader)
        all_val_preds = torch.cat(all_val_preds)
        all_val_labels = torch.cat(all_val_labels)
        val_auc = roc_auc_score(all_val_labels.numpy(), all_val_preds.numpy())
        val_acc = accuracy_score(all_val_labels.numpy(), (all_val_preds > 0.5).numpy())

        # Update best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()

        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')

    # Load best model
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, train_accs, val_accs

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

