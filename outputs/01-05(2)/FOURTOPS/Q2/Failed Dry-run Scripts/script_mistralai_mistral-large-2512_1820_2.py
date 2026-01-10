
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
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_max_pool
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.max_objects = 18
        self.obj_feature_size = 5
        self.global_feature_size = 2
        self.total_features = self.global_feature_size + self.max_objects * self.obj_feature_size

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

            "eval_overrides": {"shuffle": False,
                               "batch_size": 256}
        }

    def fit(self, X, y=None):
        # Extract global features (E_T_miss, phi_Et_miss) and object features separately
        global_features = X[:, :2].numpy()
        self.scaler.fit(global_features)
        return self

    def transform(self, X):
        # X shape: [N, 92]
        N = X.shape[0]

        # Scale global features
        global_features = X[:, :2].numpy()
        global_features = self.scaler.transform(global_features)
        global_features = torch.from_numpy(global_features).float()

        # Process object features
        object_features = X[:, 2:].reshape(N, self.max_objects, self.obj_feature_size)  # [N, 18, 5]

        # Create graph data objects
        data_list = []
        for i in range(N):
            # Get object features for this event
            obj_feats = object_features[i]  # [18, 5]

            # Filter out zero-padded objects (obj_id == 0)
            mask = obj_feats[:, 0] != 0
            valid_obj_feats = obj_feats[mask]  # [num_valid_objects, 5]

            if valid_obj_feats.shape[0] == 0:
                # Handle case with no valid objects (shouldn't happen in practice)
                valid_obj_feats = obj_feats[:1]  # Take first object as fallback

            # Create node features: [E, p_T, eta, phi] (skip obj_id)
            node_features = valid_obj_feats[:, 1:]  # [num_nodes, 4]

            # Create edge_index (fully connected graph)
            num_nodes = node_features.shape[0]
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Undirected graph

            # Compute pairwise features for edges
            src, dst = edge_index
            src_feats = node_features[src]  # [num_edges, 4]
            dst_feats = node_features[dst]  # [num_edges, 4]

            # Energy and momentum components
            E_src, pt_src, eta_src, phi_src = src_feats.unbind(-1)
            E_dst, pt_dst, eta_dst, phi_dst = dst_feats.unbind(-1)

            # Compute delta R
            delta_eta = eta_src - eta_dst
            delta_phi = (phi_src - phi_dst + math.pi) % (2 * math.pi) - math.pi
            delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

            # Compute invariant mass (approximate)
            # m_ij = sqrt((E_i + E_j)^2 - (p_i + p_j)^2)
            # Using transverse momentum approximation
            px_src = pt_src * torch.cos(phi_src)
            py_src = pt_src * torch.sin(phi_src)
            px_dst = pt_dst * torch.cos(phi_dst)
            py_dst = pt_dst * torch.sin(phi_dst)

            E_sum = E_src + E_dst
            px_sum = px_src + px_dst
            py_sum = py_src + py_dst
            pt_sum = torch.sqrt(px_sum**2 + py_sum**2)

            # Approximate invariant mass (neglecting p_z)
            m_ij = torch.sqrt(E_sum**2 - pt_sum**2)

            # Edge features: [delta_R, m_ij]
            edge_attr = torch.stack([delta_R, m_ij], dim=1)  # [num_edges, 2]

            # Create PyG Data object
            data = Data(
                x=node_features,  # [num_nodes, 4]
                edge_index=edge_index,  # [2, num_edges]
                edge_attr=edge_attr,  # [num_edges, 2]
                global_features=global_features[i].unsqueeze(0),  # [1, 2]
                num_nodes=num_nodes
            )
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class ParticleTransformer(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_layers=4, dropout=0.1):
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
                concat=True,
                beta=True,
                dropout=dropout,
                edge_dim=hidden_dim
            ) for _ in range(num_layers)
        ])

        # Layer normalization and dropout
        self.norm = nn.LayerNorm(hidden_dim * num_heads)
        self.dropout = nn.Dropout(dropout)

        # Global feature processing
        self.global_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_heads + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        # batch is a PyG Batch object
        x = batch.x  # [total_nodes, 4]
        edge_index = batch.edge_index  # [2, total_edges]
        edge_attr = batch.edge_attr  # [total_edges, 2]
        batch_idx = batch.batch  # [total_nodes]
        global_feats = batch.global_features  # [batch_size, 2]

        # Embed node features
        h = self.node_embed(x)  # [total_nodes, hidden_dim]

        # Embed edge features
        edge_attr = self.edge_embed(edge_attr)  # [total_edges, hidden_dim]

        # Transformer layers
        for layer in self.transformer_layers:
            h = layer(h, edge_index, edge_attr)
            h = self.norm(h)
            h = F.relu(h)
            h = self.dropout(h)

        # Global pooling
        h_pooled = global_max_pool(h, batch_idx)  # [batch_size, hidden_dim * num_heads]

        # Process global features
        global_h = self.global_mlp(global_feats)  # [batch_size, hidden_dim]

        # Concatenate pooled node features and global features
        combined = torch.cat([h_pooled, global_h], dim=1)  # [batch_size, hidden_dim * num_heads + hidden_dim]

        # Classification
        logits = self.classifier(combined)  # [batch_size, 1]

        return logits.squeeze(-1)

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.model = ParticleTransformer(
            hidden_dim=128,
            num_heads=4,
            num_layers=4,
            dropout=0.1
        )

    def forward(self, batch):
        return self.model(batch)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=False)

    best_auc = 0.0
    best_model_state = None
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch)
            targets = batch.y.float()

            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).detach().cpu())
            all_targets.append(targets.detach().cpu())

        # Calculate training metrics
        train_loss = total_loss / len(train_loader)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        train_auc = roc_auc_score(all_targets.numpy(), all_preds.numpy())

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                targets = batch.y.float()

                loss = F.binary_cross_entropy_with_logits(logits, targets)
                val_loss += loss.item()

                val_preds.append(torch.sigmoid(logits).cpu())
                val_targets.append(targets.cpu())

        val_loss = val_loss / len(val_loader)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_auc = roc_auc_score(val_targets.numpy(), val_preds.numpy())

        # Update learning rate
        scheduler.step(val_auc)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        # Early stopping and model checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping if no improvement for 5 epochs
        if epoch - np.argmax(val_aucs) >= 5:
            print("Early stopping triggered")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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

