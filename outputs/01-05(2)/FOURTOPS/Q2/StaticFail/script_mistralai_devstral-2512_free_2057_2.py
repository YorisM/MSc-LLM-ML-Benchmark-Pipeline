
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
        obj_features = X[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]

        # Flatten object features for scaling
        obj_flat = obj_features.reshape(-1, 5)
        valid_mask = (obj_flat[:, 0] != 0)  # Non-zero object ID
        obj_valid = obj_flat[valid_mask]

        # Fit scalers
        self.global_scaler.fit(global_features)
        if len(obj_valid) > 0:
            self.obj_scaler.fit(obj_valid)
        return self

    def transform(self, X):
        # Apply scaling
        global_features = X[:, :2]
        obj_features = X[:, 2:].reshape(-1, 18, 5)

        # Scale global features
        global_scaled = self.global_scaler.transform(global_features)

        # Scale object features
        obj_flat = obj_features.reshape(-1, 5)
        valid_mask = (obj_flat[:, 0] != 0)
        obj_valid = obj_flat[valid_mask]

        if len(obj_valid) > 0:
            obj_valid_scaled = self.obj_scaler.transform(obj_valid)
            obj_flat[valid_mask] = obj_valid_scaled

        obj_scaled = obj_flat.reshape(-1, 18, 5)

        # Create PyG Data objects
        data_list = []
        for i in range(X.shape[0]):
            # Get non-zero objects
            obj_data = obj_scaled[i]
            obj_mask = obj_data[:, 0] != 0
            obj_data = obj_data[obj_mask]

            if len(obj_data) == 0:
                # Create dummy node if no objects
                obj_data = torch.zeros(1, 5)
            else:
                obj_data = torch.from_numpy(obj_data).float()

            # Create edge indices for complete graph
            num_nodes = obj_data.shape[0]
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
            if num_nodes > 1:
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

            # Add pairwise features
            node_features = obj_data[:, 1:]  # Remove object ID
            edge_features = self._compute_edge_features(node_features, edge_index)

            # Create Data object
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features,
                y=torch.tensor([int(Y_train[i])], dtype=torch.long) if 'Y_train' in globals() else torch.tensor([0], dtype=torch.long),
                global_features=torch.from_numpy(global_scaled[i]).float()
            )
            data_list.append(data)

        return data_list

    def _compute_edge_features(self, node_features, edge_index):
        # Get node features for each edge
        src = node_features[edge_index[0]]
        dst = node_features[edge_index[1]]

        # Compute invariant mass (simplified)
        E1, pt1, eta1, phi1 = src[:, 0], src[:, 1], src[:, 2], src[:, 3]
        E2, pt2, eta2, phi2 = dst[:, 0], dst[:, 1], dst[:, 2], dst[:, 3]

        # Approximate px, py, pz
        px1 = pt1 * torch.cos(phi1)
        py1 = pt1 * torch.sin(phi1)
        pz1 = pt1 * torch.sinh(eta1)

        px2 = pt2 * torch.cos(phi2)
        py2 = pt2 * torch.sin(phi2)
        pz2 = pt2 * torch.sinh(eta2)

        # Invariant mass
        inv_mass = torch.sqrt((E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2)

        # Delta R
        deta = eta1 - eta2
        dphi = torch.abs(phi1 - phi2)
        dphi = torch.min(dphi, 2*math.pi - dphi)
        delta_r = torch.sqrt(deta**2 + dphi**2)

        # Stack features
        edge_features = torch.stack([inv_mass, delta_r], dim=1)
        return edge_features

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract feature dimensions from sample
        num_node_features = sample_object.x.shape[1]
        num_edge_features = sample_object.edge_attr.shape[1] if hasattr(sample_object, 'edge_attr') else 0
        num_global_features = sample_object.global_features.shape[0] if hasattr(sample_object, 'global_features') else 0

        # Node processing
        self.node_encoder = nn.Sequential(
            nn.Linear(num_node_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Edge processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Transformer layers
        self.conv1 = TransformerConv(64, 64, heads=4, edge_dim=32)
        self.conv2 = TransformerConv(64*4, 64, heads=4, edge_dim=32)
        self.conv3 = TransformerConv(64*4, 64, heads=4, edge_dim=32)

        # Global processing
        self.global_encoder = nn.Sequential(
            nn.Linear(num_global_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        # Process nodes
        x = self.node_encoder(data.x)

        # Process edges
        if hasattr(data, 'edge_attr'):
            edge_attr = self.edge_encoder(data.edge_attr)
        else:
            edge_attr = None

        # Transformer layers
        x = F.relu(self.conv1(x, data.edge_index, edge_attr))
        x = F.relu(self.conv2(x, data.edge_index, edge_attr))
        x = F.relu(self.conv3(x, data.edge_index, edge_attr))

        # Global pooling
        x = global_mean_pool(x, data.batch)

        # Process global features
        if hasattr(data, 'global_features'):
            global_feat = self.global_encoder(data.global_features)
            x = torch.cat([x, global_feat], dim=1)

        # Classifier
        return self.classifier(x).squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)
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
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y.float())

                total_loss += loss.item() * data.num_graphs
                preds = (out > 0).float()
                correct += (preds == data.y).sum().item()
                total += data.num_graphs

                all_preds.extend(out.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())

        val_loss = total_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
        except:
            val_auc = 0

        # Early stopping and scheduler
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

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

