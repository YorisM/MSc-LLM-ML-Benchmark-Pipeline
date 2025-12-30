
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
from sklearn.preprocessing import RobustScaler
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
        self.scaler = RobustScaler()
        self.obj_scalers = [RobustScaler() for _ in range(18)]
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

            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Fit global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :self.global_feature_size]
        self.scaler.fit(global_features)

        # Fit per-object features
        for i in range(self.max_objects):
            start_idx = self.global_feature_size + i * self.obj_feature_size + 1  # skip obj_id
            end_idx = start_idx + 4
            obj_features = X[:, start_idx:end_idx]
            self.obj_scalers[i].fit(obj_features)

        return self

    def transform(self, X):
        # Transform global features
        global_features = X[:, :self.global_feature_size]
        global_features = self.scaler.transform(global_features)

        # Create list to hold all graph data
        data_list = []

        for event in X:
            # Get global features
            et_miss = global_features[data_list.__len__(), 0]
            phi_miss = global_features[data_list.__len__(), 1]

            # Process objects
            objects = []
            obj_ids = []
            for i in range(self.max_objects):
                start_idx = self.global_feature_size + i * self.obj_feature_size
                obj_id = event[start_idx].item()
                if obj_id == 0:  # padding
                    continue

                obj_ids.append(obj_id)
                # Get kinematic features (skip obj_id)
                kin_features = event[start_idx+1:start_idx+5].numpy().reshape(1, -1)
                kin_features = self.obj_scalers[i].transform(kin_features)
                objects.append(kin_features[0])

            if len(objects) == 0:
                continue

            # Convert to tensors
            x = torch.tensor(objects, dtype=torch.float32)  # [num_objects, 4]
            obj_ids = torch.tensor(obj_ids, dtype=torch.long)

            # Create edge_index for fully connected graph
            num_nodes = x.size(0)
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected

            # Compute pairwise features
            pos = x[:, 1:3]  # eta, phi
            delta_eta = pos[edge_index[0], 0] - pos[edge_index[1], 0]
            delta_phi = pos[edge_index[0], 1] - pos[edge_index[1], 1]
            delta_phi = (delta_phi + math.pi) % (2 * math.pi) - math.pi  # handle periodicity
            delta_r = torch.sqrt(delta_eta**2 + delta_phi**2)

            # Compute invariant mass for pairs
            e = x[:, 0]  # energy
            pt = x[:, 1]  # pT
            eta = x[:, 2]  # eta
            phi = x[:, 3]  # phi

            # Get pairs
            e_i = e[edge_index[0]]
            e_j = e[edge_index[1]]
            pt_i = pt[edge_index[0]]
            pt_j = pt[edge_index[1]]
            eta_i = eta[edge_index[0]]
            eta_j = eta[edge_index[1]]
            phi_i = phi[edge_index[0]]
            phi_j = phi[edge_index[1]]

            # Compute invariant mass squared
            m2 = 2 * (e_i * e_j - pt_i * pt_j * (
                torch.cosh(eta_i - eta_j) -
                torch.cos(phi_i - phi_j)
            ))

            # Handle negative values due to numerical precision
            m2 = torch.clamp(m2, min=0)
            m = torch.sqrt(m2)

            # Create edge features
            edge_attr = torch.stack([delta_r, m], dim=1)  # [num_edges, 2]

            # Create graph data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_features=torch.tensor([et_miss, phi_miss], dtype=torch.float32),
                num_nodes=num_nodes
            )
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class ParticleTransformer(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Node feature embedding
        self.node_embed = nn.Linear(4, hidden_dim)

        # Edge feature embedding
        self.edge_embed = nn.Linear(2, hidden_dim)

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
        self.global_embed = nn.Linear(2, hidden_dim)

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, batch):
        # Unpack batch
        x = batch.x  # [total_nodes, 4]
        edge_index = batch.edge_index  # [2, total_edges]
        edge_attr = batch.edge_attr  # [total_edges, 2]
        global_features = batch.global_features  # [batch_size, 2]
        batch_idx = batch.batch  # [total_nodes]

        # Embed node features
        x = self.node_embed(x)  # [total_nodes, hidden_dim]

        # Embed edge features
        edge_attr = self.edge_embed(edge_attr)  # [total_edges, hidden_dim]

        # Embed global features
        global_embed = self.global_embed(global_features)  # [batch_size, hidden_dim]

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)

        # Global pooling
        x_pool = global_mean_pool(x, batch_idx)  # [batch_size, hidden_dim]

        # Combine with global features
        combined = torch.cat([x_pool, global_embed], dim=1)  # [batch_size, hidden_dim * 2]

        # Readout
        out = self.readout(combined)  # [batch_size, 1]

        return out.squeeze(1)

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.model = ParticleTransformer()

    def forward(self, batch_x):
        return self.model(batch_x)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_auc = 0.0
    best_model = None
    patience = 5
    patience_counter = 0

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
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch)
            targets = batch.y.float()

            loss = F.binary_cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(targets.detach().cpu().numpy())

        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_targets, train_preds)
        train_losses.append(train_loss)
        train_aucs.append(train_auc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                targets = batch.y.float()

                loss = F.binary_cross_entropy(outputs, targets)
                val_loss += loss.item()
                val_preds.extend(outputs.detach().cpu().numpy())
                val_targets.extend(targets.detach().cpu().numpy())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping based on validation AUC
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
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

    return model, train_losses, val_losses, train_aucs, val_aucs

EPOCHS = 30

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


