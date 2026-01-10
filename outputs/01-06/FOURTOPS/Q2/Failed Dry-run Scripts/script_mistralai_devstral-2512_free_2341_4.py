
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

        # Flatten object features for scaling (excluding obj_id)
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
        obj_ids = obj_feats[:, :, 0]  # [N, 18]
        obj_kin = obj_feats[:, :, 1:]  # [N, 18, 4]

        # Scale object kinematics
        obj_kin_flat = obj_kin.reshape(-1, 4)
        obj_kin_scaled = self.obj_scaler.transform(obj_kin_flat).reshape(-1, 18, 4)

        # Create PyG Data objects
        data_list = []
        for i in range(X.shape[0]):
            # Get non-zero objects (where obj_id != 0)
            mask = obj_ids[i] != 0
            n_objects = mask.sum()

            if n_objects == 0:
                # Handle empty events (shouldn't happen but just in case)
                x = torch.zeros(1, 4)
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                # Node features: [E, pT, eta, phi]
                x = torch.tensor(obj_kin_scaled[i][mask], dtype=torch.float32)

                # Create complete graph edges
                nodes = torch.arange(n_objects)
                edge_index = torch.cartesian_prod(nodes, nodes).t()
                # Remove self-loops
                edge_index = edge_index[:, edge_index[0] != edge_index[1]]

                # Add edge features: delta R and invariant mass
                edge_attr = []
                for src, dst in edge_index.t():
                    # Delta R
                    delta_eta = x[src, 2] - x[dst, 2]
                    delta_phi = x[src, 3] - x[dst, 3]
                    delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))  # handle angle wrapping
                    delta_r = torch.sqrt(delta_eta**2 + delta_phi**2)

                    # Invariant mass (approximate using transverse components)
                    # m^2 = (E1 + E2)^2 - (px1 + px2)^2 - (py1 + py2)^2 - (pz1 + pz2)^2
                    # Approximate using transverse components only
                    e1, pt1, eta1, phi1 = x[src]
                    e2, pt2, eta2, phi2 = x[dst]

                    # Transverse components
                    px1 = pt1 * torch.cos(phi1)
                    py1 = pt1 * torch.sin(phi1)
                    px2 = pt2 * torch.cos(phi2)
                    py2 = pt2 * torch.sin(phi2)

                    # Longitudinal components (approximate)
                    pz1 = pt1 * torch.sinh(eta1)
                    pz2 = pt2 * torch.sinh(eta2)

                    # Invariant mass squared
                    m_sq = (e1 + e2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2
                    m = torch.sqrt(torch.abs(m_sq))

                    edge_attr.append(torch.tensor([delta_r, m], dtype=torch.float32))

                edge_attr = torch.stack(edge_attr).t()  # [2, num_edges]

            # Add global features as node features for a special "global node"
            global_node = torch.tensor(global_scaled[i], dtype=torch.float32).unsqueeze(0)
            x = torch.cat([x, global_node], dim=0)

            # Connect global node to all other nodes
            if n_objects > 0:
                global_edges_src = torch.full((n_objects,), n_objects, dtype=torch.long)
                global_edges_dst = torch.arange(n_objects, dtype=torch.long)
                global_edge_index = torch.stack([global_edges_src, global_edges_dst])

                # Add bidirectional edges
                global_edge_index = torch.cat([
                    global_edge_index,
                    torch.stack([global_edges_dst, global_edges_src])
                ], dim=1)

                # Combine with object edges
                edge_index = torch.cat([edge_index, global_edge_index], dim=1)

                # Add dummy edge features for global edges
                dummy_attr = torch.zeros(2, global_edge_index.shape[1])
                edge_attr = torch.cat([edge_attr, dummy_attr], dim=1)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([0], dtype=torch.long)  # placeholder, will be set in dataset
            )
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Determine input dimensions from sample
        if isinstance(sample_object, Data):
            # PyG mode
            self.node_dim = sample_object.x.shape[1]
            self.edge_dim = sample_object.edge_attr.shape[0] if hasattr(sample_object, 'edge_attr') and sample_object.edge_attr is not None else 0
            self.is_pyg = True
        else:
            # Dense mode (fallback)
            self.input_dim = sample_object.shape[1]
            self.is_pyg = False

        if self.is_pyg:
            # Graph Neural Network architecture
            self.conv1 = TransformerConv(self.node_dim, 64, heads=4, edge_dim=self.edge_dim)
            self.conv2 = TransformerConv(64 * 4, 64, heads=4, edge_dim=self.edge_dim)
            self.conv3 = TransformerConv(64 * 4, 64, heads=4, edge_dim=self.edge_dim)

            self.fc1 = nn.Linear(64 * 4 + 2, 128)  # +2 for global features
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)

            self.bn1 = nn.BatchNorm1d(64 * 4)
            self.bn2 = nn.BatchNorm1d(64 * 4)
            self.bn3 = nn.BatchNorm1d(64 * 4)
            self.bn_fc1 = nn.BatchNorm1d(128)
            self.bn_fc2 = nn.BatchNorm1d(64)

            self.dropout = nn.Dropout(0.3)
        else:
            # Fallback dense architecture
            self.fc1 = nn.Linear(self.input_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, 1)

            self.bn1 = nn.BatchNorm1d(256)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(64)

            self.dropout = nn.Dropout(0.3)

    def forward(self, batch_x):
        if self.is_pyg:
            # PyG forward pass
            x, edge_index, edge_attr, batch = batch_x.x, batch_x.edge_index, batch_x.edge_attr, batch_x.batch

            # Graph convolutions
            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.bn1(x)
            x = self.dropout(x)

            x = self.conv2(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.bn2(x)
            x = self.dropout(x)

            x = self.conv3(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.bn3(x)

            # Global pooling
            x = global_mean_pool(x, batch)

            # Get global features (last node in each graph)
            global_feats = batch_x.x[batch_x.ptr[:-1] - 1]  # last node before each new graph

            # Combine features
            x = torch.cat([x, global_feats], dim=1)

            # Fully connected layers
            x = self.fc1(x)
            x = F.relu(x)
            x = self.bn_fc1(x)
            x = self.dropout(x)

            x = self.fc2(x)
            x = F.relu(x)
            x = self.bn_fc2(x)
            x = self.dropout(x)

            x = self.fc3(x)
            return x.squeeze(-1)
        else:
            # Dense forward pass
            x = batch_x

            x = self.fc1(x)
            x = F.relu(x)
            x = self.bn1(x)
            x = self.dropout(x)

            x = self.fc2(x)
            x = F.relu(x)
            x = self.bn2(x)
            x = self.dropout(x)

            x = self.fc3(x)
            x = F.relu(x)
            x = self.bn3(x)
            x = self.dropout(x)

            x = self.fc4(x)
            return x.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Use focal loss to handle class imbalance
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            return focal_loss.mean()

    criterion = FocalLoss(alpha=0.25, gamma=2).to(device)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # Early stopping
    best_val_auc = 0
    patience = 10
    patience_counter = 0

    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            if isinstance(batch, Data):
                # PyG batch
                batch = batch.to(device)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch.y.float())
            else:
                # Dense batch
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels.float()).sum().item()
            total += labels.size(0)

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
            for batch in val_loader:
                if isinstance(batch, Data):
                    batch = batch.to(device)
                    outputs = model(batch)
                    loss = criterion(outputs, batch.y.float())
                    preds = torch.sigmoid(outputs)
                    labels = batch.y.float()
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float())
                    preds = torch.sigmoid(outputs)

                val_loss += loss.item() * labels.size(0)
                correct += ((preds > 0.5).float() == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
        except:
            val_auc = 0

        # Update scheduler and early stopping
        scheduler.step(val_auc)

        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

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

