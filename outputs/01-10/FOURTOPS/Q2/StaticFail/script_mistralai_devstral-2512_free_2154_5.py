
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

# ---------- IMPORTS ----------
import math
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch.nn import Linear, ReLU, Dropout, BatchNorm1d, Sequential, Sigmoid
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.obj_scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
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

    def fit(self, X, y=None):
        # Separate global and object features
        global_features = X[:, :2]  # E_T_miss, phi_Et_miss
        obj_features = X[:, 2:].reshape(-1, 18, 5)[:, :, 1:]  # Skip obj_id, keep E, pT, eta, phi

        # Flatten object features for scaling
        obj_flat = obj_features.reshape(-1, 4)
        self.global_scaler.fit(global_features)
        self.obj_scaler.fit(obj_flat)
        return self

    def transform(self, X):
        # Apply scaling
        global_features = X[:, :2]
        obj_features = X[:, 2:].reshape(-1, 18, 5)

        # Scale global features
        global_scaled = self.global_scaler.transform(global_features)

        # Extract and scale object features (skip obj_id)
        obj_ids = obj_features[:, :, 0].long()
        obj_kin = obj_features[:, :, 1:].reshape(-1, 4)
        obj_kin_scaled = self.obj_scaler.transform(obj_kin).reshape(-1, 18, 4)

        # Create PyG Data objects
        data_list = []
        for i in range(X.shape[0]):
            # Get non-zero objects (non-padded)
            mask = obj_ids[i] != 0
            num_objects = mask.sum().item()

            if num_objects == 0:
                # Handle empty events (shouldn't happen but just in case)
                x = torch.zeros(1, 4)
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                # Object features (E, pT, eta, phi)
                x = torch.tensor(obj_kin_scaled[i][mask], dtype=torch.float32)

                # Create complete graph edges
                edges = []
                for src in range(num_objects):
                    for dst in range(num_objects):
                        if src != dst:
                            edges.append([src, dst])
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

            # Global features
            global_feat = torch.tensor(global_scaled[i], dtype=torch.float32)

            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                global_feat=global_feat,
                y=torch.tensor([int(y[i])], dtype=torch.long) if y is not None else torch.tensor([0], dtype=torch.long)
            )
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        num_node_features = sample_object.x.shape[1]  # Should be 4 (E, pT, eta, phi)
        num_global_features = sample_object.global_feat.shape[0]  # Should be 2

        # Node feature processing
        self.node_encoder = Sequential(
            Linear(num_node_features, 64),
            ReLU(),
            BatchNorm1d(64),
            Dropout(0.2)
        )

        # Graph layers
        self.conv1 = TransformerConv(64, 64, heads=4, dropout=0.2)
        self.conv2 = TransformerConv(64*4, 64, heads=4, dropout=0.2)
        self.conv3 = TransformerConv(64*4, 64, heads=4, dropout=0.2)

        # Global feature processing
        self.global_encoder = Sequential(
            Linear(num_global_features, 32),
            ReLU(),
            BatchNorm1d(32),
            Dropout(0.2)
        )

        # Combined processing
        self.combined = Sequential(
            Linear(64 + 32, 128),
            ReLU(),
            BatchNorm1d(128),
            Dropout(0.3),
            Linear(128, 64),
            ReLU(),
            BatchNorm1d(64),
            Dropout(0.3),
            Linear(64, 1)
        )

    def forward(self, data):
        # Process node features
        x = self.node_encoder(data.x)

        # Graph convolutions
        x = self.conv1(x, data.edge_index)
        x = ReLU()(x)
        x = self.conv2(x, data.edge_index)
        x = ReLU()(x)
        x = self.conv3(x, data.edge_index)
        x = ReLU()(x)

        # Global pooling
        x = global_mean_pool(x, data.batch)

        # Process global features
        global_feat = self.global_encoder(data.global_feat)

        # Combine features
        combined = torch.cat([x, global_feat], dim=1)
        out = self.combined(combined)

        return out.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_auc = 0
    best_model = None

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data)
            loss = criterion(out, data.y.float())

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.num_graphs
            train_preds.append(out.detach().cpu())
            train_targets.append(data.y.float().detach().cpu())

        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y.float())

                val_loss += loss.item() * data.num_graphs
                val_preds.append(out.detach().cpu())
                val_targets.append(data.y.float().detach().cpu())

        # Calculate metrics
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)

        train_auc = roc_auc_score(train_targets, torch.sigmoid(train_preds))
        val_auc = roc_auc_score(val_targets, torch.sigmoid(val_preds))

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_auc)
        val_acc_history.append(val_auc)

        # Update scheduler
        scheduler.step(val_auc)

        # Early stopping and model saving
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')

    # Load best model
    model.load_state_dict(best_model)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

