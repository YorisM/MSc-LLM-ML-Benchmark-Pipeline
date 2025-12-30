
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GIN, global_mean_pool
from torch_geometric.data import Data
from torch.utils.data import Dataset as TorchDataset
from sklearn.preprocessing import StandardScaler

#  -------- CUSTOM DATASET  --------
class CustomDataset(TorchDataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X)  # list of Data
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = self.X[idx]
        data.y = self.y[idx]  # set label in Data
        return data

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Fit scaler on flattened features, but for now, skip global scaling
        valid_nodes = []
        for row in X:
            for i in range(18):
                start = 2 + i * 5
                obj_id = row[start].item()
                if obj_id != 0:
                    E, pT, eta, phi = row[start+1:start+5].tolist()
                    valid_nodes.append([E, pT, eta, phi])
        if valid_nodes:
            self.scaler.fit(valid_nodes)
        return self

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

    def transform(self, X):
        data_list = []
        for row in X:
            global_et_miss = row[0].item()
            global_phi_miss = row[1].item()
            nodes = []
            for i in range(18):
                start = 2 + i * 5
                obj_id = row[start].item()
                if obj_id != 0:
                    E, pT, eta, phi = row[start+1:start+5].tolist()
                    nodes.append([E, pT, eta, phi])
            if not nodes:
                # Empty graph, skip or handle
                continue
            node_features = torch.tensor(nodes, dtype=torch.float32)
            node_features = torch.tensor(self.scaler.transform(node_features.numpy()), dtype=torch.float32)
            num_nodes = node_features.shape[0]
            edge_list = []
            edge_attr = []
            pi = 3.1415926535
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    # Compute px, py, pz
                    pxi, pyi, pzi = pT * torch.cos(phi), pT * torch.sin(phi), pT * torch.sinh(eta)
                    pxi, pyi, pzi = node_features[i,1]*torch.cos(node_features[i,3]), node_features[i,1]*torch.sin(node_features[i,3]), node_features[i,1]*torch.sinh(node_features[i,2])
                    pxj, pyj, pzj = node_features[j,1]*torch.cos(node_features[j,3]), node_features[j,1]*torch.sin(node_features[j,3]), node_features[j,1]*torch.sinh(node_features[j,2])
                    Ei, Ej = node_features[i,0], node_features[j,0]
                    m_ij_sq = (Ei + Ej)**2 - (pxi + pxj)**2 - (pyi + pyj)**2 - (pzi + pzj)**2
                    m_ij = torch.sqrt(torch.clamp(m_ij_sq, min=0))
                    dphi = torch.atan2(torch.sin(node_features[i,3] - node_features[j,3]), torch.cos(node_features[i,3] - node_features[j,3]))
                    dR = torch.sqrt((node_features[i,2] - node_features[j,2]) ** 2 + dphi ** 2)
                    edge_list.append([i, j])
                    edge_list.append([j, i])
                    edge_attr.append([dR, m_ij])
                    edge_attr.append([dR, m_ij])
            edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.empty(2,0, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.empty(0,2, dtype=torch.float32)
            data_list.append(Data(
                x=node_features,  # [num_nodes, 4]
                edge_index=edge_index,  # [2, num_edges]
                edge_attr=edge_attr,  # [num_edges, 2]
                global_feat=torch.tensor([global_et_miss, global_phi_miss], dtype=torch.float32)  # [2]
            ))
        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.conv1 = GIN(4, 64, eps=0.0)
        self.conv2 = GIN(64, 128, eps=0.0)
        self.pool = global_mean_pool
        self.linear = nn.Linear(128 + 2, 1)  # 128 from graph emb, 2 from global

    def forward(self, batch_x):
        # batch_x is Batch
        x = F.relu(self.conv1(batch_x.x, batch_x.edge_index, batch_x.edge_attr))  # [num_nodes, 64]
        x = F.relu(self.conv2(x, batch_x.edge_index, batch_x.edge_attr))  # [num_nodes, 128]
        graph_emb = self.pool(x, batch_x.batch)  # [batch, 128]

        # Global feat: batch.global_feat is concatenated [batch*2]
        global_emb = []
        for i in range(batch_x.num_graphs):
            start = i * 2
            global_emb.append(batch_x.global_feat[start:start+2])
        global_emb = torch.stack(global_emb)  # [batch, 2]

        combined = torch.cat([graph_emb, global_emb], dim=1)  # [batch, 128+2]
        out = self.linear(combined)  # [batch, 1]
        return out.squeeze(-1)  # [batch]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss, correct_train, total_train = 0, 0, 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            batch_x = view.batch_x
            out = model(batch_x)
            loss = criterion(out, view.batch_y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            preds = (torch.sigmoid(out) > 0.5).float()
            correct_train += (preds == view.batch_y).sum().item()
            total_train += view.batch_y.size(0)
        train_loss.append(epoch_train_loss / len(train_loader))
        train_acc.append(correct_train / total_train)

        model.eval()
        epoch_val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                batch_x = view.batch_x
                out = model(batch_x)
                loss = criterion(out, view.batch_y.float())
                epoch_val_loss += loss.item()
                preds = (torch.sigmoid(out) > 0.5).float()
                correct_val += (preds == view.batch_y).sum().item()
                total_val += view.batch_y.size(0)
        val_loss.append(epoch_val_loss / len(val_loader))
        val_acc.append(correct_val / total_val)
        print(f"Epoch {epoch+1}: Train Loss {train_loss[-1]:.4f}, Val Loss {val_loss[-1]:.4f}, Train Acc {train_acc[-1]:.4f}, Val Acc {val_acc[-1]:.4f}")

    return model, train_loss, val_loss, train_acc, val_acc

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

