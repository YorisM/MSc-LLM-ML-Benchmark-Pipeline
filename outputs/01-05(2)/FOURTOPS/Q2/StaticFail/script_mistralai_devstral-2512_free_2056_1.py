
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

# ---------- IMPORTS ----------
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.obj_scaler = StandardScaler()
        self.fitted = False

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 256}
        }

    def fit(self, X, y=None):
        # Separate global and object features
        global_feats = X[:, :2]  # E_T_miss, phi_Et_miss
        obj_feats = X[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]

        # Flatten object features for scaling (excluding obj_id which is categorical)
        obj_feats_flat = obj_feats[:, :, 1:].reshape(-1, 4)  # [N*18, 4]

        # Fit scalers
        self.global_scaler.fit(global_feats)
        self.obj_scaler.fit(obj_feats_flat)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted")

        # Process global features
        global_feats = X[:, :2]
        global_scaled = self.global_scaler.transform(global_feats)

        # Process object features
        obj_feats = X[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]
        obj_ids = obj_feats[:, :, 0]  # Keep original obj_ids
        obj_vals = obj_feats[:, :, 1:]  # [N, 18, 4]

        # Scale object features
        obj_vals_flat = obj_vals.reshape(-1, 4)
        obj_vals_scaled = self.obj_scaler.transform(obj_vals_flat).reshape(-1, 18, 4)

        # Reconstruct full object features
        obj_feats_scaled = torch.cat([
            obj_ids.unsqueeze(-1),  # [N, 18, 1]
            torch.from_numpy(obj_vals_scaled)  # [N, 18, 4]
        ], dim=-1)  # [N, 18, 5]

        # Create PyG Data objects
        data_list = []
        for i in range(X.shape[0]):
            # Get non-zero objects (where obj_id != 0)
            mask = obj_feats_scaled[i, :, 0] != 0
            num_objects = mask.sum().item()

            if num_objects == 0:
                # Handle empty events (shouldn't happen but just in case)
                x = torch.zeros(1, 4)  # Just padding
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                # Node features: [E, pT, eta, phi] for each object
                x = obj_feats_scaled[i, mask, 1:]  # [num_objects, 4]

                # Create complete graph (all pairs connected)
                # This is computationally expensive but captures all interactions
                idx = torch.arange(num_objects)
                edge_index = torch.cartesian_prod(idx, idx).t()  # [2, num_objects^2]

                # Add self-loops
                self_loops = torch.stack([idx, idx])
                edge_index = torch.cat([edge_index, self_loops], dim=1)

            # Add global features to each node
            global_vec = torch.from_numpy(global_scaled[i]).repeat(num_objects, 1)
            x = torch.cat([x, global_vec], dim=1)  # [num_objects, 6]

            # Create Data object
            data = Data(
                x=x.float(),
                edge_index=edge_index,
                y=torch.tensor([Y_train[i]] if 'Y_train' in globals() else [0], dtype=torch.long)
            )
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is a PyG Data object
        num_node_features = sample_object.x.shape[1]  # Should be 6 (4 obj + 2 global)

        # Transformer layers for graph processing
        self.conv1 = TransformerConv(
            in_channels=num_node_features,
            out_channels=64,
            heads=4,
            concat=True,
            beta=True
        )
        self.conv2 = TransformerConv(
            in_channels=64 * 4,  # 4 heads * 64
            out_channels=64,
            heads=4,
            concat=True,
            beta=True
        )
        self.conv3 = TransformerConv(
            in_channels=64 * 4,
            out_channels=32,
            heads=2,
            concat=True,
            beta=True
        )

        # Global pooling and classification
        self.lin1 = nn.Linear(32 * 2, 64)  # 2 heads * 32
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 1)

        self.bn1 = nn.BatchNorm1d(64 * 4)
        self.bn2 = nn.BatchNorm1d(64 * 4)
        self.bn3 = nn.BatchNorm1d(32 * 2)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(32)

        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Transformer layers with residual connections
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + x1)  # Residual connection
        x2 = self.dropout(x2)

        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3 + x2)  # Residual connection

        # Global pooling
        x = global_mean_pool(x3, batch)

        # Classification layers
        x = self.lin1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.lin2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.lin3(x)
        return x.squeeze(-1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )

    criterion = nn.BCEWithLogitsLoss()
    best_val_auc = 0
    patience = 10
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data)
            loss = criterion(out, data.y.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            preds = (out > 0).float()
            correct += (preds == data.y).sum().item()
            total += data.num_graphs

        train_loss = total_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y.float())

                val_loss += loss.item() * data.num_graphs
                preds = (out > 0).float()
                correct += (preds == data.y).sum().item()
                total += data.num_graphs

                all_preds.extend(out.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())

        val_loss = val_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
            scheduler.step(val_auc)
        except:
            val_auc = 0

        # Early stopping and best model tracking
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                model.load_state_dict(best_model)
                break

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
    model.load_state_dict(best_model)

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

