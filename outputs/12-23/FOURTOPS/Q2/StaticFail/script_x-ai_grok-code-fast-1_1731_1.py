
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
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()  # For global features ET_miss and phiET_miss, but keep object params intact

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
        # Extract global features for scaling
        global_features = X[:, :2].numpy()
        self.scaler.fit(global_features)
        return self

    def transform(self, X):
        data_list = []
        for event in X:
            event = event.numpy()
            global_feat = self.scaler.transform(event[:2].reshape(1, -1)).ravel()  # [2]
            nodes = []
            node_features = []
            edges = []
            edge_features = []

            for i in range(18):
                start = 2 + i * 5
                obj_id = int(event[start])
                if obj_id == 0:  # Assume padding
                    continue
                E, pt, eta, phi = event[start+1:start+5]
                nodes.append(i)
                # Node features: obj_id (one-hot? but simplistic, just numerical), E, pt, eta, phi -> [5]
                # Obj_id is categorical, but I'll keep as is, later model can handle
                node_features.append([obj_id, E, pt, eta, phi])

            # Add global node
            nodes.append(len(nodes))
            node_features.append([0, 0, 0, global_feat[0], global_feat[1]])  # Dummy, then globals as eta-like, phi
            global_idx = len(nodes) - 1

            # Build edges: full connected among objects, and each object to global
            num_obj = len(nodes) - 1  # excluding global
            for i in range(num_obj):
                for j in range(i+1, num_obj):
                    edges.append([i,j])
                    # Compute m_ij and delta R
                    E1, pt1, eta1, phi1 = node_features[i][1:]
                    E2, pt2, eta2, phi2 = node_features[j][1:]
                    # Reconstruct momenta
                    px1 = pt1 * np.cos(phi1)
                    py1 = pt1 * np.sin(phi1)
                    pz1 = pt1 * np.sinh(eta1)
                    px2 = pt2 * np.cos(phi2)
                    py2 = pt2 * np.sin(phi2)
                    pz2 = pt2 * np.sinh(eta2)
                    # Invariant mass: m^2 = (E1+E2)^2 - (px1+px2)^2 - (py1+py2)^2 - (pz1+pz2)^2
                    m2_ij = (E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2
                    m_ij = np.sqrt(np.maximum(0, m2_ij)) / 1000  # Scale to GeV or something, but arbitrary
                    delta_eta = eta1 - eta2
                    delta_phi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))  # Wrap around
                    delta_R = np.sqrt(delta_eta**2 + delta_phi**2)
                    edge_features.append([m_ij, delta_R])
                # Edge to global
                edges.append([i, global_idx])
                edge_features.append([0, 0])  # Dummy edge features to global
                edges.append([global_idx, i])
                edge_features.append([0, 0])

            if nodes:
                node_features = torch.tensor(node_features, dtype=torch.float32)  # [num_nodes, 5]
                edges = torch.tensor(edges, dtype=torch.long).t()  # [2, num_edges]
                edge_features = torch.tensor(edge_features, dtype=torch.float32)  # [num_edges, 2]
                data = Data(x=node_features, edge_index=edges, edge_attr=edge_features)
                data_list.append(data)
            else:
                # Empty, but shouldn't happen
                data_list.append(Data(x=torch.empty(1,5), edge_index=torch.empty(2,0), edge_attr=torch.empty(0,2)))

        return data_list  # List of Data

# ---------- MODEL DEFINITION ----------
class EdgeAwareMPNN(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, heads=1):
        super().__init__(aggr='mean')
        self.conv1 = nn.Linear(in_channels + edge_dim, out_channels)
        self.conv2 = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        msg = torch.cat([x[row], edge_attr], dim=1)  # [num_edges, in+edge_dim]
        msg = self.conv1(msg).relu()
        out = self.message(msg, col=col)
        out = self.conv2(out).relu()
        return out

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        num_node_features = sample_object.x.shape[1]  # 5
        num_edge_features = sample_object.edge_attr.shape[1]  # 2
        self.gnn1 = EdgeAwareMPNN(num_node_features, 64, num_edge_features)
        self.gnn2 = EdgeAwareMPNN(64, 128, num_edge_features)
        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.gnn1(x, edge_index, edge_attr)
        x = self.gnn2(x, edge_index, edge_attr)
        x = global_mean_pool(x, data.batch) if hasattr(data, 'batch') else x.mean(dim=0, keepdim=True)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return torch.sigmoid(x)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    best_val_auc = 0.0
    patience = 5
    counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out.squeeze(), data.y.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pred = (out > 0.5).long()
            epoch_acc += (pred.squeeze() == data.y).sum().item()
        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_loader.dataset)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out.squeeze(), data.y.float())
                epoch_val_loss += loss.item()
                pred = (out > 0.5).long()
                epoch_val_acc += (pred.squeeze() == data.y).sum().item()
                all_preds.append(out.cpu())
                all_targets.append(data.y.cpu())
        epoch_val_loss /= len(val_loader)
        epoch_val_acc /= len(val_loader.dataset)
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        # Compute AUC
        from sklearn.metrics import roc_auc_score
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        val_auc = roc_auc_score(targets, preds)

        scheduler.step()

        # Early stopping based on val AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
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


