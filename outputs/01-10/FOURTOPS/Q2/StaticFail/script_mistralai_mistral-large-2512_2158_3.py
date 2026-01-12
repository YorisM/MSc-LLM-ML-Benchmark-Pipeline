
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import add_self_loops
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.obj_ids = None
        self.max_objects = 18
        self.global_features = 2
        self.per_object_features = 4  # E, pT, eta, phi (obj_id is categorical)

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

            "eval_overrides": {"shuffle": False,
                               "batch_size": 512}
        }

    def fit(self, X, y=None):
        # Extract global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :2].numpy()
        self.scaler.fit(global_features)

        # Extract object features (E, pT, eta, phi) for all objects
        object_features = []
        for i in range(self.max_objects):
            start_idx = 2 + i * 5 + 1  # skip obj_id
            end_idx = start_idx + 4
            object_features.append(X[:, start_idx:end_idx].numpy())

        object_features = np.concatenate(object_features, axis=0)
        self.object_scaler = RobustScaler().fit(object_features)

        # Store unique object IDs
        obj_ids = []
        for i in range(self.max_objects):
            obj_ids.append(X[:, 2 + i * 5].numpy())
        self.obj_ids = np.unique(np.concatenate(obj_ids))

        return self

    def transform(self, X):
        # Scale global features
        global_features = X[:, :2].numpy()
        global_features = self.scaler.transform(global_features)

        # Scale object features and create graph data
        data_list = []
        batch_size = X.shape[0]

        for i in range(batch_size):
            # Get event data
            event = X[i]

            # Get global features
            global_feat = global_features[i]

            # Process objects
            x_list = []
            obj_ids = []
            for j in range(self.max_objects):
                start_idx = 2 + j * 5
                obj_id = event[start_idx].item()
                if obj_id == 0:  # padding
                    continue

                # Get kinematic features
                kinematic_feat = event[start_idx+1:start_idx+5].numpy().reshape(1, -1)
                kinematic_feat = self.object_scaler.transform(kinematic_feat)
                x_list.append(kinematic_feat[0])
                obj_ids.append(obj_id)

            if len(x_list) == 0:
                continue  # skip empty events

            x = torch.tensor(np.array(x_list), dtype=torch.float32)  # [num_objects, 4]

            # Create complete graph (fully connected)
            num_nodes = x.size(0)
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected

            # Add self-loops
            edge_index = add_self_loops(edge_index, num_nodes=num_nodes)[0]

            # Compute pairwise features for edges
            edge_attr = []
            for src, dst in edge_index.t():
                if src == dst:  # self-loop
                    edge_attr.append(torch.zeros(2))
                    continue

                # Get 4-vectors
                p_src = x[src]
                p_dst = x[dst]

                # Compute delta R
                delta_eta = p_src[2] - p_dst[2]
                delta_phi = (p_src[3] - p_dst[3] + math.pi) % (2 * math.pi) - math.pi
                delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

                # Compute invariant mass (approximate)
                E_sum = p_src[0] + p_dst[0]
                pT_sum = torch.sqrt((p_src[1] * torch.cos(p_src[3]) + p_dst[1] * torch.cos(p_dst[3]))**2 +
                                   (p_src[1] * torch.sin(p_src[3]) + p_dst[1] * torch.sin(p_dst[3]))**2)
                m_inv = torch.sqrt(E_sum**2 - pT_sum**2)

                edge_attr.append(torch.tensor([delta_R, m_inv]))

            edge_attr = torch.stack(edge_attr)  # [num_edges, 2]

            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        global_feat=torch.tensor(global_feat, dtype=torch.float32),
                        y=torch.tensor([0 if y is None else y[i].item()], dtype=torch.long))

            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class ParticleTransformer(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Node embedding
        self.node_encoder = nn.Linear(4, hidden_dim)

        # Edge embedding
        self.edge_encoder = nn.Linear(2, hidden_dim)

        # Global feature embedding
        self.global_encoder = nn.Linear(2, hidden_dim)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                           concat=True, beta=True, edge_dim=hidden_dim)
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Output layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Encode nodes and edges
        x = self.node_encoder(x)  # [num_nodes, hidden_dim]
        edge_attr = self.edge_encoder(edge_attr)  # [num_edges, hidden_dim]

        # Encode global features
        global_feat = self.global_encoder(data.global_feat)  # [batch_size, hidden_dim]

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, edge_index, edge_attr)
            x = self.norm(x)
            x = F.relu(x)

        # Global pooling
        x_pool = global_mean_pool(x, batch)  # [batch_size, hidden_dim]

        # Concatenate with global features
        x_combined = torch.cat([x_pool, global_feat], dim=1)  # [batch_size, hidden_dim * 2]

        # Output
        out = self.mlp(x_combined)  # [batch_size, 1]
        return out.squeeze(-1)

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.model = ParticleTransformer(hidden_dim=128, num_heads=4, num_layers=4)

    def forward(self, batch):
        return self.model(batch)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    best_model = None
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)
            loss = criterion(out, batch.y.float())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.num_graphs
            preds = (torch.sigmoid(out) > 0.5).float()
            epoch_correct += (preds == batch.y.float()).sum().item()
            epoch_total += batch.num_graphs

        train_loss.append(epoch_loss / epoch_total)
        train_acc.append(epoch_correct / epoch_total)

        # Validation
        model.eval()
        val_loss_epoch = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y.float())

                val_loss_epoch += loss.item() * batch.num_graphs
                preds = (torch.sigmoid(out) > 0.5).float()
                val_correct += (preds == batch.y.float()).sum().item()
                val_total += batch.num_graphs

                all_preds.extend(torch.sigmoid(out).cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        val_loss.append(val_loss_epoch / val_total)
        val_acc.append(val_correct / val_total)

        # Calculate AUC
        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, "
              f"Train Acc: {train_acc[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}, AUC: {auc:.4f}")

        # Early stopping and model saving
        scheduler.step(auc)
        if auc > best_auc:
            best_auc = auc
            best_model = model.state_dict()

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_loss, val_loss, train_acc, val_acc

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

