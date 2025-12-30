
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
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

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

# ---------- IMPORTS ----------
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.obj_types = None
        self.max_objects = 18
        self.obj_feature_size = 5
        self.global_feature_size = 2

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

            "extra_loader_kwargs": {"follow_batch": ["x"]},

            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Calculate mean and std for normalization
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True)
        self.std[self.std == 0] = 1.0  # Avoid division by zero

        # Extract unique object types for embedding
        obj_columns = X[:, 2::5]  # obj_n columns
        self.obj_types = torch.unique(obj_columns).long()
        self.num_obj_types = len(self.obj_types)

        return self

    def transform(self, X):
        # Normalize data
        X = (X - self.mean) / self.std

        # Convert to PyG Data objects
        data_list = []
        batch_size = X.shape[0]

        for i in range(batch_size):
            event = X[i]

            # Extract global features
            global_features = event[:2]  # E_T_miss, phi_Et_miss

            # Extract object features
            objects = event[2:].reshape(-1, 5)  # [num_objects, 5]
            mask = objects[:, 0] != 0  # obj_id != 0 means valid object
            objects = objects[mask]

            if len(objects) == 0:
                continue  # skip events with no objects

            # Get object features (E, p_T, eta, phi)
            obj_features = objects[:, 1:]  # [num_objects, 4]

            # Create node features: [obj_type_embedding, kinematic_features]
            obj_ids = objects[:, 0].long()
            obj_type_embedding = F.one_hot(obj_ids, num_classes=self.num_obj_types).float()

            # Combine features
            node_features = torch.cat([obj_type_embedding, obj_features], dim=1)  # [num_objects, num_obj_types + 4]

            # Create edge_index for fully connected graph
            num_nodes = node_features.shape[0]
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected graph

            # Calculate pairwise features for edges
            pos = obj_features[:, 1:3]  # eta, phi
            delta_eta = pos[edge_index[0], 0] - pos[edge_index[1], 0]
            delta_phi = pos[edge_index[0], 1] - pos[edge_index[1], 1]
            delta_phi = (delta_phi + math.pi) % (2 * math.pi) - math.pi  # handle periodicity

            delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

            # Invariant mass calculation (approximate)
            energy = obj_features[edge_index[0], 0] + obj_features[edge_index[1], 0]
            pt = torch.sqrt(obj_features[edge_index[0], 1]**2 + obj_features[edge_index[1], 1]**2 +
                           2 * obj_features[edge_index[0], 1] * obj_features[edge_index[1], 1] *
                           torch.cos(delta_phi))
            mass = torch.sqrt(energy**2 - pt**2)

            # Edge features: [delta_R, invariant_mass]
            edge_attr = torch.stack([delta_R, mass], dim=1)

            # Create PyG Data object
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_features=global_features,
                num_nodes=num_nodes
            )
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class ParticleTransformer(nn.Module):
    def __init__(self, num_obj_types, node_feature_size, edge_feature_size=2, hidden_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.num_obj_types = num_obj_types
        self.node_feature_size = node_feature_size
        self.hidden_dim = hidden_dim

        # Node embedding
        self.obj_embedding = nn.Linear(num_obj_types, hidden_dim)
        self.kinematic_embedding = nn.Linear(4, hidden_dim)
        self.node_encoder = nn.Linear(2 * hidden_dim, hidden_dim)

        # Edge embedding
        self.edge_encoder = nn.Linear(edge_feature_size, hidden_dim)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                concat=False,
                beta=True,
                edge_dim=hidden_dim
            ) for _ in range(num_layers)
        ])

        # Global feature processing
        self.global_encoder = nn.Linear(2, hidden_dim)

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, batch_x):
        if isinstance(batch_x, Batch):
            data = batch_x
        else:
            data = batch_x[0] if isinstance(batch_x, list) else batch_x

        # Node features
        obj_emb = self.obj_embedding(data.x[:, :self.num_obj_types])
        kin_emb = self.kinematic_embedding(data.x[:, self.num_obj_types:])
        x = self.node_encoder(torch.cat([obj_emb, kin_emb], dim=1))  # [num_nodes, hidden_dim]

        # Edge features
        edge_attr = self.edge_encoder(data.edge_attr)  # [num_edges, hidden_dim]

        # Global features
        global_feat = self.global_encoder(data.global_features)  # [batch_size, hidden_dim]

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, data.edge_index, edge_attr)
            x = F.relu(x)

        # Global pooling
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(data.num_nodes, dtype=torch.long, device=x.device)
        x_pool = global_mean_pool(x, batch)  # [batch_size, hidden_dim]

        # Combine with global features
        combined = torch.cat([x_pool, global_feat], dim=1)  # [batch_size, 2*hidden_dim]

        # Readout
        logits = self.readout(combined).squeeze(-1)  # [batch_size]

        return logits

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Determine dimensions from sample object
        if isinstance(sample_object, Batch):
            data = sample_object
            num_obj_types = data.x.shape[1] - 4  # obj_type_embedding + 4 kinematic features
            node_feature_size = data.x.shape[1]
            edge_feature_size = data.edge_attr.shape[1] if data.edge_attr is not None else 2
        else:
            # Fallback values if sample_object is not a Batch
            num_obj_types = 10  # reasonable default
            node_feature_size = 14  # num_obj_types + 4 kinematic features
            edge_feature_size = 2

        self.model = ParticleTransformer(
            num_obj_types=num_obj_types,
            node_feature_size=node_feature_size,
            edge_feature_size=edge_feature_size,
            hidden_dim=128,
            num_heads=4,
            num_layers=4
        )

    def forward(self, batch_x):
        return self.model(batch_x)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
def compute_auc(y_true, y_pred):
    if len(torch.unique(y_true)) < 2:
        return 0.5  # return 0.5 if only one class present
    return roc_auc_score(y_true.cpu().numpy(), y_pred.detach().cpu().numpy())

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model = None
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(torch.sigmoid(logits).detach())
            train_targets.append(yb.detach())

        train_loss /= len(train_loader)
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_auc = compute_auc(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                logits = model(xb)
                loss = criterion(logits, yb.float())
                val_loss += loss.item()
                val_preds.append(torch.sigmoid(logits).detach())
                val_targets.append(yb.detach())

        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_auc = compute_auc(val_targets, val_preds)

        # Update learning rate
        scheduler.step(val_auc)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_losses, val_losses, train_aucs, val_aucs

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

