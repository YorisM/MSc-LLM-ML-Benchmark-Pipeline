
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: Python 3.12, PyTorch 2.6.0, Torch_Geometric 2.6.1, NumPy 2.2.3, SciPy v1.15.2, SciKit-Learn 1.6.1
import os, sys, pickle, torch, torch_geometric, gc, json, importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
                        
DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv(DATASET["X_train"], dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv(DATASET["Y_train"], dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv(DATASET["X_val"], dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv(DATASET['Y_val'], dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class PairDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
    
        if isinstance(self.x, (tuple, list)) and all(torch.is_tensor(t) for t in self.x):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])

def _make_dataset(x, y):
    custom = globals().get("make_dataset", None)
    if callable(custom):
        ds = custom(x, y)
        if ds is not None:
            return ds
    return PairDataset(x, y)

def make_loaders(X_train, Y_train, X_val, Y_val, *, batch=512, collate_fn=None, loader_cls=None):
    train_ds = _make_dataset(X_train, Y_train)
    val_ds   = _make_dataset(X_val , Y_val)

    if loader_cls is None: 
        loader_cls = DataLoader

    train_ld = loader_cls(train_ds, batch_size=batch, shuffle=True, num_workers=0, 
                        collate_fn=collate_fn)
    val_ld   = loader_cls(val_ds, batch_size=batch, shuffle=False, num_workers=0,
                        collate_fn=collate_fn)

    return train_ld, val_ld

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import EdgeConv, global_add_pool, global_mean_pool
from torch_geometric.utils import k_nearest_neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np
import copy


# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    #    Must implement:
    #   - fit(...)               -> self
    #   - transform(X: ???)      -> ???

    # DATA SPECIFICS
    # Total flat length per event (X_train & X_val): 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    # Indices  7-11 : object 2  ->  obj_2, E_2 , p_T_2 , eta_2 , phi_2
    # ...
    # Indices 88-92 : object 18 ->  obj_18, E_18 , p_T_18 , eta_18 , phi_18
    # Global features       = 2
    # Per-object slice size = 5
    # Max objects encoded   = 18

    # TIPS
    # When modifying data features or feature engineering: annotate tensor size as comments after 
    # each tensor operation to reduce dimension mismatches.

    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module.
    # May allocate NumPy arrays or Torch tensors internally, but:
    # transform() must be deterministic.
    # Store only derived parameters needed for transform i.e. do not store the raw data
    # itself in the preprocessor object.

    def __init__(self, k_neighbors=8):
        self.global_scaler = StandardScaler()
        self.object_scaler = StandardScaler()
        self.obj_id_map = None
        self.num_obj_types = None
        self.k_neighbors = k_neighbors

    def _engineer_features(self, objects_tensor):
        # objects_tensor shape: [N_objects, 5] with columns (obj_id, E, pT, eta, phi)
        # We only engineer features for active objects.
        E = objects_tensor[:, 1]
        pT = objects_tensor[:, 2]
        eta = objects_tensor[:, 3]
        phi = objects_tensor[:, 4]

        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)

        p_squared = px**2 + py**2 + pz**2
        m_squared = E**2 - p_squared
        # Use sign(m^2) * sqrt(|m^2|) to handle tachyonic/massless cases
        mass = torch.sign(m_squared) * torch.sqrt(torch.abs(m_squared))

        # New continuous features: E, pT, eta, phi, px, py, pz, mass
        return torch.stack([E, pT, eta, phi, px, py, pz, mass], dim=1) # [N_objects, 8]


    def fit(self, X, y=None):
        # Fit scalers and object ID mapping. Expects X to be a torch.Tensor
        X_np = X.numpy()

        # Fit global scaler
        self.global_scaler.fit(X_np[:, :2]) # [N_events, 2]

        # Reshape to objects
        objects = X_np[:, 2:].reshape(-1, 18, 5) # [N_events, 18, 5]

        # Filter out padded objects (pT > 0)
        mask = objects[:, :, 2] > 1e-6

        active_objects = torch.from_numpy(objects[mask]) # [N_total_active, 5]

        # Engineer features before fitting the scaler on all active objects
        engineered_features = self._engineer_features(active_objects) # [N_total_active, 8]

        # Fit object scaler on engineered features
        self.object_scaler.fit(engineered_features.numpy())

        # Create object ID mapping
        active_ids = active_objects[:, 0].int().numpy()
        unique_ids = np.unique(active_ids)
        self.obj_id_map = {int(id_): i for i, id_ in enumerate(unique_ids)}
        self.num_obj_types = len(unique_ids)

        return self

    def transform(self, X):
        # Transform data into a list of torch_geometric.Data objects
        data_list = []

        for i in range(X.shape[0]):
            event_flat = X[i]

            # Global features
            global_feats = event_flat[:2].unsqueeze(0) # [1, 2]
            scaled_global_feats = torch.from_numpy(self.global_scaler.transform(global_feats.numpy())).float().squeeze(0) # [2]

            # Object features
            objects = event_flat[2:].reshape(18, 5)  # [18, 5]

            # Filter padding
            mask = objects[:, 2] > 1e-6
            active_objects = objects[mask] # [N_active, 5]

            num_nodes = active_objects.shape[0]

            if num_nodes > 0:
                # Engineer features
                engineered_features = self._engineer_features(active_objects) # [N_active, 8]

                # Scale continous features
                scaled_kinematics = torch.from_numpy(self.object_scaler.transform(engineered_features.numpy())).float()

                # Categorical features (node type)
                raw_ids = active_objects[:, 0].int()
                node_cat_features = torch.tensor([self.obj_id_map.get(id_.item(), -1) for id_ in raw_ids], dtype=torch.long)

                # For k-NN graph, use (eta, phi) coordinates
                pos = active_objects[:, 3:5] # [N_active, 2]

                if num_nodes > 1:
                    edge_index = k_nearest_neighbors(pos, pos, self.k_neighbors, loop=False) # [2, N_edges]
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                # Event with no particles
                scaled_kinematics = torch.empty((0, 8), dtype=torch.float)
                node_cat_features = torch.empty((0,), dtype=torch.long)
                edge_index = torch.empty((2, 0), dtype=torch.long)

            data = Data(x=scaled_kinematics, 
                        edge_index=edge_index, 
                        u=scaled_global_feats, 
                        node_cat_features=node_cat_features)

            data_list.append(data)

        return data_list

    def make_loader_cfg(self):
        # Use PyG's DataLoader for batching graphs
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 256
        }

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Infer feature dimensions from the sample_object (a PyG Batch)
        node_kinematic_dim = sample_object.x.shape[1]    # 8 engineered features
        global_feat_dim = sample_object.u.shape[1]       # 2 global features

        # Heuristic to find num_obj_types from the first batch
        if hasattr(sample_object, 'node_cat_features') and sample_object.node_cat_features.numel() > 0:
            num_obj_types = int(torch.max(sample_object.node_cat_features).item()) + 1
        else: # Handle cases with no categorical features in the first batch
            num_obj_types = 16 # Fallback to a reasonable default

        embedding_dim = 16
        hidden_dim = 96

        # Node feature processing
        self.node_embedding = nn.Embedding(num_obj_types, embedding_dim, padding_idx=-1)
        self.kinematics_mlp = nn.Sequential(
            nn.Linear(node_kinematic_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        node_input_dim = (hidden_dim // 2) + embedding_dim

        # Graph convolution layers (EdgeConv)
        self.conv1 = EdgeConv(nn.Sequential(
            nn.Linear(2 * node_input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        ), aggr='mean')
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = EdgeConv(nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        ), aggr='mean')
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = EdgeConv(nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        ), aggr='mean')
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Classifier Head
        self.classifier_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + global_feat_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        # Deconstruct the PyG Batch object
        x, edge_index, batch, u = data.x, data.edge_index, data.batch, data.u
        node_cat_features = data.node_cat_features

        # 1. Process node features
        kinematic_embed = self.kinematics_mlp(x) # [N_nodes, hidden_dim/2]
        type_embed = self.node_embedding(node_cat_features) # [N_nodes, embedding_dim]

        h = torch.cat([kinematic_embed, type_embed], dim=-1) # [N_nodes, node_input_dim]

        # 2. Graph convolutions with residual connections
        h1 = self.bn1(self.conv1(h, edge_index))
        h1 = F.gelu(h1)

        h2 = self.bn2(self.conv2(h1, edge_index))
        h2 = F.gelu(h1 + h2) # Residual connection

        h3 = self.bn3(self.conv3(h2, edge_index))
        h3 = F.gelu(h2 + h3) # Residual connection

        # 3. Graph pooling
        pooled_add = global_add_pool(h3, batch)   # [batch_size, hidden_dim]
        pooled_mean = global_mean_pool(h3, batch) # [batch_size, hidden_dim]
        pooled = torch.cat([pooled_add, pooled_mean], dim=1) # [batch_size, 2*hidden_dim]

        # 4. Concatenate graph and global features
        combined_features = torch.cat([pooled, u], dim=-1) # [batch_size, 2*hidden_dim + global_feat_dim]

        # 5. Final classification
        logits = self.classifier_mlp(combined_features)  # [batch_size, 1]

        return logits


def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 35
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS 
    # Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    # Must return trained_model, train_loss, val_loss, train_acc, val_acc
    # Implement early-stopping.
    # Forward signature must match.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_auc = -1.0
    best_model_state = None
    patience = 7
    patience_counter = 0

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        for data, y in train_loader:
            data, y = data.to(device), y.to(device)
            optimizer.zero_grad()

            output = model(data)

            y_float = y.float().view(-1, 1)
            loss = loss_fn(output, y_float)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * y.size(0)

            preds = (torch.sigmoid(output) > 0.5).long()
            train_correct += (preds.view(-1) == y).sum().item()
            train_total += y.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_train_acc = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for data, y in val_loader:
                data, y = data.to(device), y.to(device)

                output = model(data)

                y_float = y.float().view(-1, 1)
                loss = loss_fn(output, y_float)

                total_val_loss += loss.item() * y.size(0)

                probs = torch.sigmoid(output)
                preds = (probs > 0.5).long()
                val_correct += (preds.view(-1) == y).sum().item()
                val_total += y.size(0)

                all_preds.extend(probs.view(-1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_acc = val_correct / val_total
        val_auc = roc_auc_score(all_labels, all_preds)

        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        scheduler.step(val_auc)

        # --- Early Stopping ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    # Load best model and return
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

def _import_dotted(path: str):
    mod, name = path.rsplit(".", 1)
    module = importlib.import_module(mod)
    return getattr(module, name)

def _plot(series_train, series_val, name, out_path):
    plt.figure()
    epochs = range(1, len(series_train) + 1)
    plt.plot(epochs, series_train, label=f"Train {name}")
    plt.plot(epochs, series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        X_train, Y_train, X_val, Y_val = X_train[:200], Y_train[:200], X_val[:20], Y_val[:20]
    pre     = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train)
    X_val   = pre.transform(X_val)

    collate = getattr(pre, "_collate_fn", None)
    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val, 
                                            batch      = cfg.get("batch_size", 512), 
                                            collate_fn = collate,
                                            loader_cls = loader_cls)

    # 2. Build model
    first_batch    = next(iter(train_loader))
    example_sample = first_batch[0]
    model          = make_model(example_sample)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. Dry-run safety check
    if dryrun:
        sample, _ = first_batch
        try:
            _ = trained_model(*sample) if isinstance(sample, (tuple, list)) else trained_model(sample)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

        pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
        pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
        pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

        torch.save(trained_model.state_dict(), pth_state)
        with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
        with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

        # 6. Save plots
        _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
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

