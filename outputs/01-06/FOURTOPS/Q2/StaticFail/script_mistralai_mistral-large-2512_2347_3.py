
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
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
        # Extract global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :2].numpy()
        self.scaler.fit(global_features)
        return self

    def transform(self, X):
        # X shape: [N, 92]
        N = X.shape[0]

        # Scale global features
        global_features = X[:, :2].numpy()
        global_features_scaled = self.scaler.transform(global_features)
        global_features_tensor = torch.from_numpy(global_features_scaled).float()

        # Process object features
        object_features = X[:, 2:].reshape(N, self.max_objects, self.obj_feature_size)  # [N, 18, 5]

        # Extract object types and kinematic features
        obj_types = object_features[:, :, 0].long()  # [N, 18]
        kinematic_features = object_features[:, :, 1:]  # [N, 18, 4]

        # Create mask for valid objects (non-zero energy)
        valid_mask = (kinematic_features[:, :, 0] > 0)  # [N, 18]

        # Create PyG Data objects
        data_list = []
        for i in range(N):
            # Get valid objects for this event
            valid_idx = valid_mask[i].nonzero(as_tuple=True)[0]
            num_valid = len(valid_idx)

            if num_valid == 0:
                # Handle edge case with no valid objects
                x = torch.zeros((1, 16), dtype=torch.float32)
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                # Get kinematic features for valid objects
                kin_valid = kinematic_features[i, valid_idx]  # [num_valid, 4]

                # Create node features: [E, pT, eta, phi, E^2, pT^2, eta^2, phi^2, log(E), log(pT)]
                E = kin_valid[:, 0]
                pT = kin_valid[:, 1]
                eta = kin_valid[:, 2]
                phi = kin_valid[:, 3]

                node_features = torch.cat([
                    kin_valid,
                    E.unsqueeze(1)**2,
                    pT.unsqueeze(1)**2,
                    eta.unsqueeze(1)**2,
                    phi.unsqueeze(1)**2,
                    torch.log(E.unsqueeze(1) + 1e-8),
                    torch.log(pT.unsqueeze(1) + 1e-8)
                ], dim=1)  # [num_valid, 12]

                # Add pairwise features
                num_nodes = node_features.shape[0]
                pairwise_features = []

                for j in range(num_nodes):
                    for k in range(j+1, num_nodes):
                        # Invariant mass
                        E_j, E_k = E[j], E[k]
                        px_j = pT[j] * torch.cos(phi[j])
                        py_j = pT[j] * torch.sin(phi[j])
                        pz_j = pT[j] * torch.sinh(eta[j])
                        px_k = pT[k] * torch.cos(phi[k])
                        py_k = pT[k] * torch.sin(phi[k])
                        pz_k = pT[k] * torch.sinh(eta[k])

                        E_tot = E_j + E_k
                        px_tot = px_j + px_k
                        py_tot = py_j + py_k
                        pz_tot = pz_j + pz_k

                        m_inv_sq = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
                        m_inv = torch.sqrt(torch.clamp(m_inv_sq, min=0))

                        # Delta R
                        delta_eta = eta[j] - eta[k]
                        delta_phi = torch.min(
                            torch.abs(phi[j] - phi[k]),
                            2 * math.pi - torch.abs(phi[j] - phi[k])
                        )
                        delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

                        pairwise_features.append(torch.tensor([
                            m_inv, delta_R,
                            E_j * E_k, pT[j] * pT[k],
                            torch.abs(eta[j] - eta[k]),
                            torch.abs(phi[j] - phi[k])
                        ]))

                if pairwise_features:
                    pairwise_features = torch.stack(pairwise_features)  # [num_pairs, 6]
                    # Create edge_index for fully connected graph
                    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
                    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # [2, 2*num_pairs]
                    edge_attr = torch.cat([
                        pairwise_features,
                        pairwise_features
                    ], dim=0)  # [2*num_pairs, 6]
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_attr = torch.empty((0, 6), dtype=torch.float32)

                # Add global features to each node
                global_expanded = global_features_tensor[i].unsqueeze(0).expand(num_valid, -1)
                x = torch.cat([node_features, global_expanded], dim=1)  # [num_valid, 14]

            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([0 if y is None else y[i]], dtype=torch.long)
            )
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Node feature size: 14 (12 kinematic + 2 global)
        # Edge feature size: 6
        self.node_encoder = nn.Sequential(
            nn.Linear(14, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        self.gat1 = GATv2Conv(64, 64, heads=4, edge_dim=32, dropout=0.1)
        self.gat2 = GATv2Conv(64 * 4, 64, heads=4, edge_dim=32, dropout=0.1)
        self.gat3 = GATv2Conv(64 * 4, 64, edge_dim=32, dropout=0.1)

        self.readout = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, batch):
        # batch is a PyG Batch object
        x = batch.x  # [num_nodes, 14]
        edge_index = batch.edge_index  # [2, num_edges]
        edge_attr = batch.edge_attr  # [num_edges, 6]
        batch_idx = batch.batch  # [num_nodes]

        # Encode node and edge features
        x = self.node_encoder(x)  # [num_nodes, 64]
        edge_attr = self.edge_encoder(edge_attr)  # [num_edges, 32]

        # GAT layers
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = F.relu(self.gat2(x, edge_index, edge_attr))
        x = F.relu(self.gat3(x, edge_index, edge_attr))

        # Global pooling
        x_mean = global_mean_pool(x, batch_idx)  # [batch_size, 64]
        x_max = global_max_pool(x, batch_idx)  # [batch_size, 64]
        x = torch.cat([x_mean, x_max], dim=1)  # [batch_size, 128]

        # Readout
        logits = self.readout(x)  # [batch_size, 1]
        return logits.squeeze(1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_auc = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch.y.float())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(torch.sigmoid(logits).detach().cpu())
            train_targets.append(batch.y.detach().cpu())

        train_loss /= len(train_loader)
        train_preds = torch.cat(train_preds).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_auc = roc_auc_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = F.binary_cross_entropy_with_logits(logits, batch.y.float())

                val_loss += loss.item()
                val_preds.append(torch.sigmoid(logits).detach().cpu())
                val_targets.append(batch.y.detach().cpu())

        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_auc = roc_auc_score(val_targets, val_preds)

        # Update scheduler
        scheduler.step(val_auc)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Learning rate too small, stopping early")
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

