
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
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
import numpy as np

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.obj_id_max = 20  # assume object ids up to 20 or so
        self.emb_dim = 8
        self.object_id_embedding = nn.Embedding(self.obj_id_max + 1, self.emb_dim)

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # previous, but will be overridden for PyG
            "dataset_builder": "llm_script:CustomDataset",  # use custom to return Data
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        # No fitting needed for now
        return self

    def transform(self, X):
        # X: [N, 92]
        data_list = []
        for i in range(X.size(0)):
            event = X[i]
            et_miss = event[0]
            phi_et_miss = event[1]
            nodes = []
            momenta = []
            for j in range(18):
                start = 2 + j * 5
                obj_id = event[start].item()
                if obj_id == 0:  # padding
                    continue
                E_ = event[start + 1].item()
                pT = event[start + 2].item()
                eta = event[start + 3].item()
                phi = event[start + 4].item()
                node_feat = torch.tensor([obj_id, E_, pT, eta, phi, et_miss.item(), phi_et_miss.item()])
                nodes.append(node_feat)
                momenta.append({'E': E_, 'pT': pT, 'eta': eta, 'phi': phi})
            if not nodes:
                # dummy node if no objects, unlikely
                dummy = torch.zeros(7)
                x = dummy.unsqueeze(0)  # [1,7]
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # self loop
                edge_attr = torch.tensor([[0., 0., 0.]], dtype=torch.float)  # dummy
            else:
                num_nodes = len(nodes)
                x = torch.stack(nodes)  # [num_nodes,7]
                # full interconnect, skip self
                edges = [(a, b) for a in range(num_nodes) for b in range(num_nodes) if a != b]
                edge_index = torch.tensor(edges, dtype=torch.long).t()
                edge_attrs = []
                for a, b in edges:
                    p1 = momenta[a]
                    p2 = momenta[b]
                    delta_eta = p1['eta'] - p2['eta']
                    delta_phi = ((p1['phi'] - p2['phi'] + np.pi) % (2 * np.pi)) - np.pi
                    # m_ij
                    e1, e2 = p1['E'], p2['E']
                    pt1, pt2 = p1['pT'], p2['pT']
                    eta1, eta2 = p1['eta'], p2['eta']
                    phi1, phi2 = p1['phi'], p2['phi']
                    pz1 = pt1 * np.sinh(eta1)
                    px1 = pt1 * np.cos(phi1)
                    py1 = pt1 * np.sin(phi1)
                    pz2 = pt2 * np.sinh(eta2)
                    px2 = pt2 * np.cos(phi2)
                    py2 = pt2 * np.sin(phi2)
                    E_total = e1 + e2
                    px_total = px1 + px2
                    py_total = py1 + py2
                    pz_total = pz1 + pz2
                    p_total2 = px_total**2 + py_total**2 + pz_total**2
                    m_ij = np.sqrt(max(0, E_total**2 - p_total2))
                    edge_attrs.append([delta_eta, delta_phi, m_ij])
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            # y will be added in dataset, but here we return the Data without y
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)
        return data_list  # list of Data

class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events  # X is now list of Data
        self.X = X  # already preprocessed list of Data
        self.y = torch.as_tensor(y).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        data = self.X[idx]
        data.y = self.y[idx]  # set label
        return data

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.obj_id_max = 20
        self.emb_dim = 8
        self.object_id_embedding = nn.Embedding(self.obj_id_max + 1, self.emb_dim)
        hidden = 64
        node_base_dim = 6  # E,pT,eta,phi,et_miss,phi_et_miss
        full_node_dim = self.emb_dim + node_base_dim  # 14
        self.edge_dim = 3  # delta_eta, delta_phi, m_ij
        self.batch_norm1 = nn.BatchNorm1d(full_node_dim)
        self.conv1 = GATv2Conv(full_node_dim, hidden, edge_dim=self.edge_dim, heads=4, dropout=0.2, concat=True, v2=False)
        self.batch_norm2 = nn.BatchNorm1d(hidden * 4)
        self.conv2 = GATv2Conv(hidden*4, hidden, edge_dim=self.edge_dim, heads=4, dropout=0.2, concat=True, v2=False)
        self.pool = global_mean_pool
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden * 4, 1)

    def forward(self, batch_x):
        # G.x : [total_nodes, 7]
        x = batch_x.x
        edge_index = batch_x.edge_index
        edge_attr = batch_x.edge_attr

        obj_id = x[:, 0].long().clamp(0, self.obj_id_max)  # safety
        emb = self.object_id_embedding(obj_id)
        obj_features = x[:, 1:]  # [nodes,6]
        x_emb = torch.cat([emb, obj_features], dim=1)  # [nodes,14]

        x_emb = self.batch_norm1(x_emb)
        x_emb = torch.relu(x_emb)
        x_emb = self.conv1(x_emb, edge_index, edge_attr)
        x_emb = self.batch_norm2(x_emb)
        x_emb = torch.relu(x_emb)
        x_emb = self.conv2(x_emb, edge_index, edge_attr)
        x_emb = torch.relu(x_emb)

        pooled = self.pool(x_emb, batch_x.batch)
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

EPOCHS = 25
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)  # for VIC AUC-ish, but we have loss

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float('inf')
    patience = 10
    trigger = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs.squeeze(), data.y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.y.size(0)
            preds = torch.sigmoid(outputs).squeeze() > 0.5
            correct += (preds == data.y).sum().item()
            total += data.y.size(0)
        train_loss /= total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs.squeeze(), data.y.float())
                val_loss += loss.item() * data.y.size(0)
                preds = torch.sigmoid(outputs).squeeze() > 0.5
                correct += (preds == data.y).sum().item()
                total += data.y.size(0)
        val_loss /= total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)  # for loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger = 0
            # save best model, but since no save, keep training
        else:
            trigger += 1
            if trigger >= patience:
                break

    return model, train_losses, val_losses, train_accs, val_accs

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

