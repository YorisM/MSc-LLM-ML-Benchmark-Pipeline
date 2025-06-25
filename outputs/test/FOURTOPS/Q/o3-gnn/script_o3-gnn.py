# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: Python 3.12, PyTorch 2.6.0, Torch_Geometric 2.6.1, NumPy 2.2.3, SciPy v1.15.2, SciKit-Learn 1.6.1
import os, sys, pickle, torch, torch_geometric, gc, json, importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

def load_data():
    X_train = pd.read_csv("./challenges/FOURTOPS/data/X_train.csv", dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv("./challenges/FOURTOPS/data/Y_train.csv", dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv("./challenges/FOURTOPS/data/X_val.csv", dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv("./challenges/FOURTOPS/data/Y_val.csv", dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class PairDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
    
        if isinstance(self.x, (tuple, list)) and all(torch.is_tensor(t) for t in self.x):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])

def _make_dataset(x, y):
    custom = globals().get("make_dataset", None)
    if callable(custom):
        ds = custom(x, y)
        if ds is not None:
            return ds
    return PairDataset(x, y)

def make_loaders(X_train, Y_train, X_val, Y_val, *, batch=512, collate_fn=None, loader_cls=None):
    train_ds = _make_dataset(X_train, Y_train)
    val_ds   = _make_dataset(X_val , Y_val)

    if loader_cls is None: 
        loader_cls = DataLoader

    train_ld = loader_cls(train_ds, batch_size=batch, shuffle=True, num_workers=0, 
                        collate_fn=collate_fn)
    val_ld   = loader_cls(val_ds, batch_size=batch, shuffle=False, num_workers=0,
                        collate_fn=collate_fn)

    return train_ld, val_ld

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

# 0. ---------- IMPORTS ----------
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    """
    Build a *star* graph per event:
      • one global “missing-ET” node (index 0, features 2)
      • 18 object nodes (features 5 each)
      • edges: global ↔ every object  (undirected)
    Node-feature dim = 5 (pad MET node to 5).
    """

    def fit(self, X, y=None):
        return self

    # build one torch_geometric.data.Data
    def _to_graph(self, x_row: torch.Tensor) -> Data:
        global_feat = torch.zeros(5, dtype=x_row.dtype)
        global_feat[:2] = x_row[:2]                     # (5,)
        obj = x_row[2:].view(18, 5)                     # (18,5)
        x = torch.cat([global_feat.unsqueeze(0), obj], dim=0)  # (19,5)

        src = []
        dst = []
        for i in range(1, 19):          # connect global (0) <-> object i
            src += [0, i];  dst += [i, 0]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    def transform(self, X: torch.Tensor):
        data_list = [self._to_graph(row) for row in X]   # len == N
        return data_list                                 # list[Data]

    # ------------- loader spec for evaluator ------------------
    def make_loader_cfg(self):
        return {
            "loader_class": "torch_geometric.loader.DataLoader",
            "collate_fn":   None,         # GeoLoader has built-in collate
            "batch_size":   256,
            "shuffle":      False,
            "num_workers":  0
        }

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_graph: Data):
        super().__init__()
        in_dim = sample_graph.x.size(-1)   # 5
        hidden = 64
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.cls   = nn.Linear(hidden, 1)

    def forward(self, data: Batch):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x,       data.edge_index))
        x = global_mean_pool(x, data.batch)    # (batch, hidden)
        return self.cls(x).squeeze(-1)         # (batch,)

def make_model(example_sample):
    return BinaryClassifier(example_sample)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 5
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()

    def _epoch(loader, train):
        if train: model.train()
        else:     model.eval()
        tot_loss = tot_corr = tot = 0
        with torch.enable_grad() if train else torch.no_grad():
            for batch, y in loader:
                batch = batch.to(device)
                y = y.float().to(device)
                logit = model(batch)
                loss  = crit(logit, y)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
                tot_loss += loss.item() * y.size(0)
                tot_corr += (logit.sigmoid() > 0.5).eq(y.bool()).sum().item()
                tot += y.size(0)
        return tot_loss / tot, tot_corr / tot

    train_loss = []; val_loss = []; train_acc = []; val_acc = []
    best = float("inf"); patience = 4
    for ep in range(epochs):
        tl, ta = _epoch(train_loader, True)
        vl, va = _epoch(val_loader,   False)
        train_loss.append(tl); train_acc.append(ta)
        val_loss.append(vl);   val_acc.append(va)
        if vl < best: best, wait = vl, patience
        else:
            wait -= 1
            if wait == 0: break
    return model, train_loss, val_loss, train_acc, val_acc

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

def _import_dotted(path: str):
    mod, name = path.rsplit(".", 1)
    module = importlib.import_module(mod)
    return getattr(module, name)

def _plot(series_train, series_val, name, out_path):
    plt.figure()
    epochs = range(1, len(series_train) + 1)
    plt.plot(epochs, series_train, label=f"Train {name}")
    plt.plot(epochs, series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        X_train, Y_train, X_val, Y_val = X_train[:200], Y_train[:200], X_val[:20], Y_val[:20]
    pre     = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train)
    X_val   = pre.transform(X_val)

    collate = getattr(pre, "_collate_fn", None)
    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val, 
                                            batch      = cfg.get("batch_size", 512), 
                                            collate_fn = collate,
                                            loader_cls = loader_cls)

    # 2. Build model
    first_batch    = next(iter(train_loader))
    example_sample = first_batch[0]
    model          = make_model(example_sample)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. Dry-run safety check
    if dryrun:
        sample, _ = first_batch
        try:
            _ = trained_model(*sample) if isinstance(sample, (tuple, list)) else trained_model(sample)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

        pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
        pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
        pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

        torch.save(trained_model.state_dict(), pth_state)
        with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
        with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

        # 6. Save plots
        _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
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