
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
        x = self.X[idx]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x, self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

# -------------------------- START OF LLM BLOCK ------------------------------
# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
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
        self.scaler_global = StandardScaler()
        self.scaler_objects = StandardScaler()
        self.scaler_pairwise = StandardScaler()

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "collate": "ragged_xy",
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Extract and preprocess for scaling
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        batch_features = self._extract_features(X_np[:10000])  # Use subset for fitting
        global_feats, object_feats, pairwise_feats = batch_features

        # Fit scalers
        self.scaler_global.fit(global_feats)
        self.scaler_objects.fit(object_feats)
        self.scaler_pairwise.fit(pairwise_feats)
        return self

    def _extract_features(self, X):
        """Extract features for scaling"""
        batch_size = X.shape[0]
        global_features = []
        object_features = []
        pairwise_features = []

        for i in range(batch_size):
            # Global features
            E_T_miss = X[i, 0]
            phi_Et_miss = X[i, 1]
            global_features.append([E_T_miss, math.cos(phi_Et_miss), math.sin(phi_Et_miss)])

            # Object features
            objects = X[i, 2:].reshape(18, 5)  # [18, 5]
            mask = objects[:, 0] != 0  # obj_id != 0 indicates real object

            if mask.sum() > 0:
                real_objects = objects[mask]  # [n_real, 5]
                obj_ids = real_objects[:, 0]
                energies = real_objects[:, 1]
                pTs = real_objects[:, 2]
                etas = real_objects[:, 3]
                phis = real_objects[:, 4]

                # Kinematic features
                for j in range(len(real_objects)):
                    obj_features = [
                        energies[j], pTs[j], etas[j], 
                        math.cos(phis[j]), math.sin(phis[j]),
                        obj_ids[j]
                    ]
                    object_features.append(obj_features)

                # Pairwise features for real objects
                n_real = len(real_objects)
                for j in range(n_real):
                    for k in range(j+1, n_real):
                        # DeltaR
                        delta_eta = etas[j] - etas[k]
                        delta_phi = phis[j] - phis[k]
                        delta_phi = (delta_phi + math.pi) % (2*math.pi) - math.pi
                        deltaR = math.sqrt(delta_eta**2 + delta_phi**2)

                        # Invariant mass approximation using pT and eta
                        pT_sum = pTs[j] + pTs[k]
                        eta_diff = abs(etas[j] - etas[k])
                        m_inv = pT_sum * math.cosh(eta_diff/2)  # Approximation

                        pairwise_features.append([deltaR, m_inv])

        return (
            np.array(global_features),
            np.array(object_features),
            np.array(pairwise_features)
        )

    def transform(self, X):
        """Transform batch of events to graph representation"""
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        batch_size = X_np.shape[0]
        data_list = []

        for i in range(batch_size):
            # Extract and preprocess global features
            E_T_miss = X_np[i, 0]
            phi_Et_miss = X_np[i, 1]
            global_feat = np.array([[E_T_miss, math.cos(phi_Et_miss), math.sin(phi_Et_miss)]])
            global_feat = self.scaler_global.transform(global_feat)[0]

            # Extract objects
            objects = X_np[i, 2:].reshape(18, 5)  # [18, 5]
            mask = objects[:, 0] != 0  # obj_id != 0 indicates real object

            if mask.sum() == 0:
                # Handle empty events
                x = torch.zeros((1, 6), dtype=torch.float32)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 2), dtype=torch.float32)
            else:
                real_objects = objects[mask]  # [n_real, 5]
                n_real = len(real_objects)

                obj_ids = real_objects[:, 0]
                energies = real_objects[:, 1]
                pTs = real_objects[:, 2]
                etas = real_objects[:, 3]
                phis = real_objects[:, 4]

                # Preprocess object features
                obj_features = []
                for j in range(n_real):
                    features = np.array([
                        energies[j], pTs[j], etas[j],
                        math.cos(phis[j]), math.sin(phis[j]),
                        obj_ids[j]
                    ]).reshape(1, -1)
                    scaled = self.scaler_objects.transform(features)[0]
                    obj_features.append(scaled)

                x = torch.tensor(obj_features, dtype=torch.float32)  # [n_real, 6]

                # Create edge index and edge attributes for pairwise features
                edge_indices = []
                edge_attrs = []

                for j in range(n_real):
                    for k in range(j+1, n_real):
                        # DeltaR
                        delta_eta = etas[j] - etas[k]
                        delta_phi = phis[j] - phis[k]
                        delta_phi = (delta_phi + math.pi) % (2*math.pi) - math.pi
                        deltaR = math.sqrt(delta_eta**2 + delta_phi**2)

                        # Invariant mass approximation
                        pT_sum = pTs[j] + pTs[k]
                        eta_diff = abs(etas[j] - etas[k])
                        m_inv = pT_sum * math.cosh(eta_diff/2)

                        features = np.array([[deltaR, m_inv]])
                        scaled = self.scaler_pairwise.transform(features)[0]

                        # Add both directions for undirected graph
                        edge_indices.append([j, k])
                        edge_indices.append([k, j])
                        edge_attrs.append(scaled)
                        edge_attrs.append(scaled)

                edge_index = torch.tensor(edge_indices, dtype=torch.long).t() if edge_indices else torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float32) if edge_attrs else torch.zeros((0, 2), dtype=torch.float32)

            # Create PyG Data object
            data = Data(
                x=x,  # [n_real, 6]
                edge_index=edge_index,  # [2, n_edges]
                edge_attr=edge_attr,  # [n_edges, 2]
                global_features=torch.tensor(global_feat, dtype=torch.float32).unsqueeze(0),  # [1, 3]
                y=torch.zeros(1, dtype=torch.long)  # placeholder
            )
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        node_dim = sample_object.x.shape[1] if hasattr(sample_object, 'x') else 6
        edge_dim = sample_object.edge_attr.shape[1] if hasattr(sample_object, 'edge_attr') and sample_object.edge_attr.numel() > 0 else 2
        global_dim = sample_object.global_features.shape[1] if hasattr(sample_object, 'global_features') else 3

        hidden_dim = 256
        self.hidden_dim = hidden_dim

        # GNN Layers with attention
        self.gat1 = GATConv(node_dim, hidden_dim, edge_dim=edge_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, edge_dim=edge_dim, heads=4, concat=True)
        self.gat3 = GATConv(hidden_dim * 4, hidden_dim, edge_dim=edge_dim, heads=4, concat=False)

        # Transformer encoder for global context
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=512,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=2)

        # Global feature processing
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, batch_x):
        if isinstance(batch_x, list):
            batch = Batch.from_data_list(batch_x)
        else:
            batch = batch_x

        x, edge_index, edge_attr, batch_vector = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        # GNN processing
        x = F.gelu(self.gat1(x, edge_index, edge_attr))  # [n_nodes, hidden_dim*4]
        x = F.gelu(self.gat2(x, edge_index, edge_attr))  # [n_nodes, hidden_dim*4]
        x = F.gelu(self.gat3(x, edge_index, edge_attr))  # [n_nodes, hidden_dim]

        # Graph-level pooling
        graph_emb = global_mean_pool(x, batch_vector)  # [batch_size, hidden_dim]

        # Process global features
        global_feats = torch.cat([data.global_features for data in batch_x], dim=0)
        global_emb = self.global_proj(global_feats)  # [batch_size, hidden_dim]

        # Transformer to capture interactions
        combined = torch.stack([graph_emb, global_emb], dim=1)  # [batch_size, 2, hidden_dim]
        combined = self.transformer(combined)  # [batch_size, 2, hidden_dim]
        combined = combined.mean(dim=1)  # [batch_size, hidden_dim]

        # Final classification
        out = self.classifier(combined)  # [batch_size, 1]
        return out.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 100
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 15

    # For AUC calculation
    from sklearn.metrics import roc_auc_score

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.float())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.extend(yb.cpu().numpy())

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                outputs = model(xb)
                loss = criterion(outputs, yb.float())

                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(yb.cpu().numpy())

        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)

        train_auc = roc_auc_score(train_labels, train_preds)
        val_auc = roc_auc_score(val_labels, val_preds)

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_auc)
        val_accs.append(val_auc)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

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

