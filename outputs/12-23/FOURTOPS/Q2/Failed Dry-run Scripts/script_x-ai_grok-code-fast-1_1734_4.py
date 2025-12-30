
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
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import numpy as np


# -------- CUSTOM DATASET --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.data_list = pre.transform(X, y)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx], self.data_list[idx].y


# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.obj_slice = 5  # obj_id, E, pT, eta, phi
        self.global_features = 2  # E_T_miss, phi_E_T_miss
        self.max_objects = 18

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
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
        return self

    def transform(self, X, y=None):
        data_list = []
        for i in range(X.shape[0]):
            event_x = X[i]  # shape [92]
            event_y = y[i] if y is not None else 0
            global_feat = event_x[:self.global_features]  # [2]

            nodes = []
            for j in range(self.max_objects):
                start = self.global_features + j * self.obj_slice
                obj_feat = event_x[start:start + self.obj_slice]  # [5]: obj_id, E, pT, eta, phi
                # Check if valid object: obj_id > 0 and energy > 0
                if obj_feat[0] > 0 and obj_feat[1] > 0:
                    p_x = obj_feat[2] * torch.cos(obj_feat[4])
                    p_y = obj_feat[2] * torch.sin(obj_feat[4])
                    p_z = obj_feat[2] * torch.sinh(obj_feat[3])
                    # Node features: obj_id, E, pT, eta, phi, p_x, p_y, p_z  (8 dims)
                    nodes.append(torch.cat([obj_feat, torch.tensor([p_x, p_y, p_z])]))
            nodes = torch.stack(nodes) if nodes else torch.empty(0, 8)  # shape [num_nodes, 8]

            # Create fully connected edges if nodes exist
            if len(nodes) > 0:
                num_nodes = len(nodes)
                edge_index = []
                edge_attr = []
                for u in range(num_nodes):
                    for v in range(u + 1, num_nodes):
                        # Compute delta R
                        eta_u, phi_u = nodes[u, 3], nodes[u, 4]
                        eta_v, phi_v = nodes[v, 3], nodes[v, 4]
                        delta_phi = phi_u - phi_v
                        delta_phi = torch.remainder(delta_phi + torch.pi, 2 * torch.pi) - torch.pi  # handle wrap
                        delta_r = torch.sqrt((eta_u - eta_v)**2 + delta_phi**2)
                        # Compute m_ij
                        e_u, px_u, py_u, pz_u = nodes[u, 1], nodes[u, 5], nodes[u, 6], nodes[u, 7]
                        e_v, px_v, py_v, pz_v = nodes[v, 1], nodes[v, 5], nodes[v, 6], nodes[v, 7]
                        m_ij = torch.sqrt((e_u + e_v)**2 - (px_u + px_v)**2 - (py_u + py_v)**2 - (pz_u + pz_v)**2)
                        m_ij = torch.clamp(m_ij, 0, float('inf'))  # ensure non-negative
                        edge_attr.append([m_ij, delta_r])
                        edge_index.append([u, v])
                        edge_index.append([v, u])  # undirected

                if edge_index:
                    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
                    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                else:
                    edge_index = torch.empty(2, 0, dtype=torch.long)
                    edge_attr = torch.empty(0, 2, dtype=torch.float)
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)
                edge_attr = torch.empty(0, 2, dtype=torch.float)

            data = Data(
                x=nodes,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_feat=global_feat,
                y=event_y
            )
            data_list.append(data)
        return data_list


def make_preprocessor():
    return MyPreprocessor()


# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, example_object):
        super().__init__()
        self.in_dim = 8  # node features
        self.edge_dim = 2  # edge features
        self.global_dim = 2  # global features
        self.hidden_dim = 64

        self.conv1 = GINConv(nn.Linear(self.in_dim, self.hidden_dim), edge_dim=self.edge_dim)
        self.conv2 = GINConv(nn.Linear(self.hidden_dim, self.hidden_dim), edge_dim=self.edge_dim)
        self.global_pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.global_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        x, edge_index, edge_attr, global_feat, batch = batch_x.x, batch_x.edge_index, batch_x.edge_attr, batch_x.global_feat, batch_x.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.global_pool(x, batch)  # shape [batch_size, hidden_dim]
        x = torch.cat([x, global_feat], dim=1)  # concat global
        return self.mlp(x).squeeze()


def make_model(example_object):
    return BinaryClassifier(example_object)


# ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=False)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_state = None
    patience = 5
    counter = 0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.to(torch.float))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_y.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            train_correct += (preds == batch_y).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.to(torch.float))
                val_loss += loss.item() * batch_y.size(0)
                preds = torch.sigmoid(outputs)
                preds_binary = (preds > 0.5).long()
                val_correct += (preds_binary == batch_y).sum().item()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        val_auc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.5

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break  # early stop

    # Load best model
    model.load_state_dict(best_model_state)
    trained_model = model

    return trained_model, train_losses[-1], val_losses[-1], train_accs[-1], val_accs[-1]

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


