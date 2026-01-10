
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
import warnings
warnings.filterwarnings('ignore')


# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.kin_scaler = RobustScaler()
        self.global_scaler = StandardScaler()
        self.obj_id_mapping = {}
        self.obj_id_counter = 0

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Process object kinematic features
        obj_features_list = []
        for i in range(18):
            start_idx = 2 + i*5 + 1  # Skip obj_id, start at E
            if start_idx + 3 < X_np.shape[1]:
                obj_features = X_np[:, start_idx:start_idx+3]  # E, pT, eta
                # Replace zeros (padding) with NaN for robust scaling
                mask = obj_features != 0
                obj_features_masked = np.where(mask, obj_features, np.nan)
                obj_features_list.append(obj_features_masked)

        if obj_features_list:
            all_obj_features = np.concatenate(obj_features_list, axis=0)
            # Fit scaler on non-zero (non-padded) values
            non_nan_mask = ~np.isnan(all_obj_features[:, 0])
            if np.any(non_nan_mask):
                self.kin_scaler.fit(all_obj_features[non_nan_mask])

        # Process global features
        global_features = X_np[:, :2]
        self.global_scaler.fit(global_features)

        # Build object ID mapping
        for i in range(18):
            obj_id_idx = 2 + i*5
            if obj_id_idx < X_np.shape[1]:
                obj_ids = X_np[:, obj_id_idx].astype(int)
                unique_ids = np.unique(obj_ids)
                for uid in unique_ids:
                    if uid != 0 and uid not in self.obj_id_mapping:
                        self.obj_id_mapping[uid] = self.obj_id_counter
                        self.obj_id_counter += 1

        return self

    def transform(self, X):
        X_np = X.numpy() if torch.is_tensor(X) else X
        batch_data = []

        for event_idx in range(X_np.shape[0]):
            event = X_np[event_idx]

            # Extract and scale global features
            global_feats = event[:2].reshape(1, -1)
            global_feats_scaled = self.global_scaler.transform(global_feats).flatten()

            # Process objects
            node_features = []
            node_objects = []

            for obj_idx in range(18):
                start_idx = 2 + obj_idx*5
                if start_idx + 4 >= len(event):
                    break

                obj_id = int(event[start_idx])
                if obj_id == 0:  # Padding
                    continue

                # Kinematic features: E, pT, eta, phi
                kin_feats = event[start_idx+1:start_idx+5].reshape(1, -1)

                # Scale kinematic features (excluding phi initially)
                if kin_feats[0, 0] != 0:  # Non-zero energy
                    kin_scaled = self.kin_scaler.transform(kin_feats[:, :3])
                    # Keep phi as is (cyclic)
                    phi = kin_feats[:, 3:4]
                    kin_scaled_full = np.concatenate([kin_scaled, phi], axis=1)
                else:
                    continue  # Skip padded objects

                # Encode object type
                obj_type_encoded = np.zeros(self.obj_id_counter)
                if obj_id in self.obj_id_mapping:
                    obj_type_encoded[self.obj_id_mapping[obj_id]] = 1.0

                # Create node feature: [scaled_kinematics, obj_type_one_hot, global_feats]
                node_feat = np.concatenate([
                    kin_scaled_full.flatten(),
                    obj_type_encoded,
                    global_feats_scaled
                ])

                node_features.append(node_feat)
                node_objects.append(obj_idx)

            if not node_features:
                # Create dummy node for events with no objects
                dummy_feat = np.zeros(4 + self.obj_id_counter + 2)
                node_features.append(dummy_feat)
                node_objects.append(0)

            node_features = np.array(node_features, dtype=np.float32)

            # Build fully connected graph
            num_nodes = len(node_features)
            if num_nodes > 1:
                edge_index = []
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            edge_index.append([i, j])
                edge_index = np.array(edge_index, dtype=np.int64).T
            else:
                edge_index = np.array([[0], [0]], dtype=np.int64)

            # Create PyG Data object
            data = Data(
                x=torch.from_numpy(node_features).float(),
                edge_index=torch.from_numpy(edge_index).long(),
                y=None  # Will be set by dataset
            )
            batch_data.append(data)

        return batch_data


# Custom Dataset for PyG
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.graphs = pre.transform(X)  # List of Data objects
        self.labels = torch.as_tensor(y).long() if not torch.is_tensor(y) else y.long()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        data.y = self.labels[idx].unsqueeze(0)  # Add batch dimension
        return data


def make_preprocessor():
    return MyPreprocessor()


# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is a Data object from PyG
        node_feat_dim = sample_object.x.shape[1]

        # Graph Attention Network layers
        self.gat1 = GATConv(node_feat_dim, 128, heads=4, dropout=0.2)
        self.gat2 = GATConv(128*4, 64, heads=2, dropout=0.2)
        self.gat3 = GATConv(64*2, 32, heads=1, dropout=0.2)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128*4)
        self.bn2 = nn.BatchNorm1d(64*2)

        # Global pooling and classifier
        self.pool = global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

        # Skip connection projection
        self.skip_proj = nn.Linear(node_feat_dim, 32) if node_feat_dim != 32 else None

    def forward(self, batch_x):
        # PyG Batch object
        x, edge_index, batch = batch_x.x, batch_x.edge_index, batch_x.batch

        # Initial features
        x0 = x

        # GAT layers with residual connections
        x1 = self.gat1(x, edge_index)  # [num_nodes, 128*4]
        x1 = self.bn1(x1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x2 = self.gat2(x1, edge_index)  # [num_nodes, 64*2]
        x2 = self.bn2(x2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)

        x3 = self.gat3(x2, edge_index)  # [num_nodes, 32]
        x3 = F.elu(x3)

        # Global mean pooling
        graph_features = self.pool(x3, batch)  # [batch_size, 32]

        # Optional skip connection from initial features
        if self.skip_proj is not None:
            graph_init = self.pool(x0, batch)
            graph_init_proj = self.skip_proj(graph_init)
            graph_features = graph_features + graph_init_proj

        # Classifier
        logits = self.classifier(graph_features).squeeze(-1)  # [batch_size]

        return logits


def make_model(example_object):
    return BinaryClassifier(example_object)


# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch.y.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == batch.y.float()).sum().item()
            train_total += batch.num_graphs

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = F.binary_cross_entropy_with_logits(logits, batch.y.float())

                val_loss += loss.item() * batch.num_graphs
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == batch.y.float()).sum().item()
                val_total += batch.num_graphs

                probs = torch.sigmoid(logits).cpu().numpy()
                labels = batch.y.cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Calculate AUC using sklearn
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except:
            val_auc = 0.5

        # Update learning rate scheduler based on AUC
        scheduler.step(val_auc)

        # Early stopping based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
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

