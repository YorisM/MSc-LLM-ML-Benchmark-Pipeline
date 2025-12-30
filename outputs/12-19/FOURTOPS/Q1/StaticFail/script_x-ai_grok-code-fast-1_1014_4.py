
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
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data
from torch.nn import Linear, Sequential
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

# -------- (OPTIONAL) CUSTOM DATASET  --------
def make_dataset(events, pre, train: bool = True, **kwargs):
    X, y = events
    self.X = X
    self.y = y
    return CustomDataset(X, y)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.obj_means = None
        self.obj_stds = None
        self.obj_ids = None
        self.glob_means = None
        self.glob_stds = None
        self.obj_id_to_idx = None

    def make_loader_cfg(self):
        return {
            "dataset_builder": "llm_script:make_dataset",
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # collate must be builtin string or None (torch default collate / PyG)
            "collate": None,

            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Extract statistics

        # Global features: indices 0,1
        glob = X[:, :2].numpy()
        self.glob_means = glob.mean(axis=0)
        self.glob_stds = glob.std(axis=0) + 1e-8

        # Object features: for each object slot, if obj_id != 0, collect features
        obj_ids = set()
        obj_features = []
        for i in range(18):
            start = 2 + i*5
            obj_slice = X[:, start:start+5].numpy()  # obj_id, E, pT, eta, phi
            # Assume valid if obj_id > 0
            valid = obj_slice[:, 0] > 0
            if valid.any():
                obj_ids.update(obj_slice[valid, 0].astype(int).tolist())
                # Collect E, pT, eta, phi for normalization
                valid_features = obj_slice[valid, 1:].reshape(-1, 4)
                obj_features.append(valid_features)
        self.obj_ids = sorted(list(obj_ids))
        self.obj_id_to_idx = {oid: idx for idx, oid in enumerate(self.obj_ids)}

        if obj_features:
            all_obj_features = np.concatenate(obj_features, axis=0)
            self.obj_means = all_obj_features.mean(axis=0)
            self.obj_stds = all_obj_features.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        data_list = []
        for idx in range(X.shape[0]):
            event = X[idx].numpy()
            # Global
            glob = (event[:2] - self.glob_means) / self.glob_stds
            # Objects
            nodes = []
            for i in range(18):
                start = 2 + i*5
                obj_slice = event[start:start+5]
                obj_id, E, pT, eta, phi = obj_slice
                if obj_id > 0:
                    # One-hot obj_id
                    one_hot = np.zeros(len(self.obj_ids))
                    if int(obj_id) in self.obj_id_to_idx:
                        one_hot[self.obj_id_to_idx[int(obj_id)]] = 1
                    # Normalize continuous
                    cont = (np.array([E, pT, eta, phi]) - self.obj_means) / self.obj_stds
                    # Node: one_hot + cont, shape (len(obj_ids) + 4,)
                    node_feat = np.concatenate([one_hot, cont])
                    nodes.append(node_feat)
            if nodes:
                x = torch.tensor(np.stack(nodes), dtype=torch.float32)
                num_nodes = x.shape[0]
                # Fully connected edges
                edge_index = torch.combinations(torch.arange(num_nodes), r=2, with_replacement=False).t()
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # bidirectional
                # Glob can be added later in model or not
                data = Data(x=x, edge_index=edge_index, glob=torch.tensor(glob, dtype=torch.float32))
                data_list.append(data)
            else:
                # If no objects, maybe create a dummy with zeros
                dummy_x = torch.zeros((1, len(self.obj_ids) + 4), dtype=torch.float32)
                dummy_edge = torch.empty((2, 0), dtype=torch.long)
                data = Data(x=dummy_x, edge_index=dummy_edge, glob=torch.tensor(glob, dtype=torch.float32))
                data_list.append(data)
        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_batch):
        super().__init__()
        self.glob_dim = 2  # Et_miss, phi
        self.node_dim = sample_batch.x.shape[1]
        assert self.node_dim == 10  # assuming 6 obj_ids + 4

        self.conv1 = torch_geometric.nn.GCNConv(self.node_dim, 64)
        self.conv2 = torch_geometric.nn.GCNConv(64, 128)
        self.norm1 = nn.BatchNorm1d(64)
        self.norm2 = nn.BatchNorm1d(128)
        self.pool = torch_geometric.nn.global_sum_pool
        self.glob_lin = Linear(self.glob_dim, 32)
        self.classifier = Sequential(
            Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            Linear(64, 1)
        )

    def forward(self, batch):
        x, edge_index, glob = batch.x, batch.edge_index, batch.glob if hasattr(batch, 'glob') else None
        if glob is None:
            glob = torch.zeros(batch.batch.shape[0], self.glob_dim, device=x.device).mean(dim=0).expand(x.shape[0]//batch.num_graphs, -1)  # placeholder

        # GNN
        x = self.conv1(x, edge_index).relu()
        x = self.norm1(x)
        x = self.conv2(x, edge_index).relu()
        x = self.norm2(x)

        # Pool per event
        batch_size = batch.batch.max().item() + 1 if batch.batch is not None else 1
        graph_x = self.pool(x, batch.batch)  # (batch_size, 128)

        # Glob features per batch
        if hasattr(batch, 'glob'):
            glob_processed = self.glob_lin(batch.glob)  # (batch_size, 32) if glob is batched per graph
        else:
            glob_processed = torch.zeros(batch_size, 32, device=graph_x.device)

        combined = torch.cat([graph_x, glob_processed], dim=1)
        out = self.classifier(combined)
        return out

def make_model(example_batch):
    return BinaryClassifier(example_batch)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    best_val_loss = float('inf')
    patience = 10
    counter = 0

    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        total = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch).squeeze()
            loss = criterion(outputs, batch.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            preds = (torch.sigmoid(outputs) > 0.5).int()
            total_acc += (preds == batch.y).sum().item()
            total += batch.num_graphs
        train_loss = total_loss / total
        train_acc = total_acc / total
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        model.eval()
        total_loss = 0
        total_acc = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch).squeeze()
                loss = criterion(outputs, batch.y.float())
                total_loss += loss.item() * batch.num_graphs
                preds = (torch.sigmoid(outputs) > 0.5).int()
                total_acc += (preds == batch.y).sum().item()
                total += batch.num_graphs
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        val_loss = total_loss / total
        val_acc = total_acc / total
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            counter = 0
            best_model = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_model if 'best_model' in locals() else model.state_dict())
    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

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


