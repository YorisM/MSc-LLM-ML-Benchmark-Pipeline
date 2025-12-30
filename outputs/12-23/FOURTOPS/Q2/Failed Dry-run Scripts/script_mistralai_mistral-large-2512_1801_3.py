
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

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
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeomDataLoader
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler_global = RobustScaler()
        self.scaler_obj = RobustScaler()
        self.obj_ids = None
        self.max_objects = 18
        self.global_features = 2
        self.obj_feature_size = 4  # E, pT, eta, phi (obj_id is categorical)

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

            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Extract global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :self.global_features].numpy()  # [N, 2]
        self.scaler_global.fit(global_features)

        # Extract object features (E, pT, eta, phi) for all objects
        obj_features = []
        for i in range(self.max_objects):
            start_idx = self.global_features + i * 5 + 1  # skip obj_id
            end_idx = start_idx + self.obj_feature_size
            obj_features.append(X[:, start_idx:end_idx].numpy())  # [N, 4]

        obj_features = np.concatenate(obj_features, axis=0)  # [N*18, 4]
        self.scaler_obj.fit(obj_features)

        # Store unique object IDs
        obj_ids = []
        for i in range(self.max_objects):
            obj_ids.append(X[:, self.global_features + i * 5].numpy())  # obj_id
        self.obj_ids = np.unique(np.concatenate(obj_ids))

        return self

    def transform(self, X):
        # Transform global features
        global_features = X[:, :self.global_features].numpy()  # [N, 2]
        global_features = self.scaler_global.transform(global_features)

        # Transform object features
        obj_list = []
        for i in range(self.max_objects):
            start_idx = self.global_features + i * 5
            obj_id = X[:, start_idx:start_idx+1]  # [N, 1]
            obj_kinematics = X[:, start_idx+1:start_idx+5]  # [N, 4]

            # Scale kinematics
            obj_kinematics = self.scaler_obj.transform(obj_kinematics.numpy())

            # One-hot encode object ID
            obj_id_onehot = np.zeros((obj_id.shape[0], len(self.obj_ids)))
            for j, id_val in enumerate(self.obj_ids):
                obj_id_onehot[:, j] = (obj_id.numpy() == id_val).squeeze()

            # Combine features
            obj_features = np.concatenate([obj_id_onehot, obj_kinematics], axis=1)  # [N, len(obj_ids)+4]
            obj_list.append(obj_features)

        # Create edge_index for complete graph (all objects connected to each other)
        edge_index = []
        for i in range(self.max_objects):
            for j in range(i+1, self.max_objects):
                edge_index.append([i, j])
                edge_index.append([j, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]

        # Create PyG Data objects
        data_list = []
        for i in range(X.shape[0]):
            # Node features: [num_objects, obj_features]
            x = torch.tensor(np.stack([obj_list[j][i] for j in range(self.max_objects)]), dtype=torch.float32)

            # Global features
            global_feat = torch.tensor(global_features[i], dtype=torch.float32)

            # Pairwise features
            pairwise_features = []
            for j in range(self.max_objects):
                for k in range(j+1, self.max_objects):
                    eta1, phi1 = obj_list[j][i, -2], obj_list[j][i, -1]
                    eta2, phi2 = obj_list[k][i, -2], obj_list[k][i, -1]

                    # Delta R
                    delta_eta = eta1 - eta2
                    delta_phi = (phi1 - phi2 + math.pi) % (2 * math.pi) - math.pi
                    delta_R = math.sqrt(delta_eta**2 + delta_phi**2)

                    # Invariant mass (approximate)
                    pt1, pt2 = obj_list[j][i, -3], obj_list[k][i, -3]
                    m_ij = math.sqrt(2 * pt1 * pt2 * (math.cosh(delta_eta) - math.cos(delta_phi)))

                    pairwise_features.append([delta_R, m_ij])

            pairwise_features = torch.tensor(pairwise_features, dtype=torch.float32)  # [num_edges//2, 2]
            edge_attr = pairwise_features.repeat(2, 1)  # [num_edges, 2]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_feat=global_feat)
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.num_node_features = sample_object.x.shape[1]
        self.num_edge_features = sample_object.edge_attr.shape[1]
        self.global_feat_size = sample_object.global_feat.shape[0]

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(self.num_node_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.num_edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # GNN layers
        self.conv1 = GCNConv(64, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)

        # Global feature processing
        self.global_processor = nn.Sequential(
            nn.Linear(self.global_feat_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, batch_x):
        # batch_x is a Batch object from PyG
        x = batch_x.x  # [num_nodes, node_features]
        edge_index = batch_x.edge_index  # [2, num_edges]
        edge_attr = batch_x.edge_attr  # [num_edges, edge_features]
        batch = batch_x.batch  # [num_nodes]
        global_feat = batch_x.global_feat  # [batch_size, global_features]

        # Encode nodes and edges
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # GNN layers
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)

        # Global pooling
        x = global_mean_pool(x, batch)  # [batch_size, 64]

        # Process global features
        global_feat = self.global_processor(global_feat)  # [batch_size, 32]

        # Combine features
        combined = torch.cat([x, global_feat], dim=1)  # [batch_size, 96]

        # Classifier
        out = self.classifier(combined)  # [batch_size, 1]
        return out.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model = None
    patience = 5
    patience_counter = 0

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch)
            targets = batch.y.float()

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        train_loss.append(total_loss / len(train_loader))
        train_acc.append(correct / total)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                targets = batch.y.float()

                loss = criterion(outputs, targets)
                val_loss_total += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())

        val_loss.append(val_loss_total / len(val_loader))
        val_acc.append(val_correct / val_total)

        # Calculate AUC
        auc = roc_auc_score(all_targets, all_outputs)
        scheduler.step(auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, "
              f"Train Acc: {train_acc[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}, AUC: {auc:.4f}")

        # Early stopping
        if auc > best_auc:
            best_auc = auc
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_loss, val_loss, train_acc, val_acc

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

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

    # Build model
    first_batch = next(iter(train_loader))
    view        = normalise_batch(first_batch, device=device)
    model       = make_model(view.batch_x).to(device)

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
                view = normalise_batch(first_batch, device=device)
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


