
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data/train"
TAG      = "REDVID_10-50_linear_frac0.05"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def split_X_y(evt):
    X = np.column_stack([
        evt["hit_r"].astype(np.float32),
        evt["hit_theta"].astype(np.float32),
        evt["hit_z"].astype(np.float32),
        evt["layer_id"].astype(np.float32)
    ])
    y = evt["track_id"].astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, labels = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, labels)

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# <start code template>
# ---------- IMPORTS ----------
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch.utils.data import Dataset
from sklearn.neighbors import kneighbors_graph

#  -------- CUSTOM DATASET  --------
class CustomDataset(Dataset):
    REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
    def __init__(self, events, pre, train=True):
        self.events = events
        self.pre = pre
        self.train = train

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        evt = self.events[idx]
        X, y = self.pre.transform(evt["hit_r"]), evt["track_id"]  # pre transform X
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()
        data = self.build_graph(X, y)
        return data

    def build_graph(self, X, y):
        # X [N, 4]: r, theta, z, layer_id
        N = X.shape[0]
        pos = torch.zeros(N, 3, dtype=torch.float)
        pos[:, 0] = X[:, 0] * torch.cos(X[:, 1])
        pos[:, 1] = X[:, 0] * torch.sin(X[:, 1])
        pos[:, 2] = X[:, 2]
        k = min(6, N - 1)  # k-nearest
        adj = kneighbors_graph(pos.numpy(), n_neighbors=k, mode='connectivity', include_self=False)
        row, col = adj.nonzero()
        edge_index = torch.tensor(np.column_stack([row, col]), dtype=torch.long).t()
        data = Data(x=X, y=y, pos=pos, edge_index=edge_index)
        return data

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.stats = {}

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, events):
        # events: list of dicts with keys hit_r etc.
        rs = []; thetas = []; zs = []; layers = []
        for evt in events:
            rs.extend(evt["hit_r"])
            thetas.extend(evt["hit_theta"])
            zs.extend(evt["hit_z"])
            layers.extend(evt["layer_id"])
        rs = np.array(rs); thetas = np.array(thetas); zs = np.array(zs); layers = np.array(layers)
        self.stats['r_mean'] = rs.mean(); self.stats['r_std'] = rs.std() + 1e-6
        self.stats['theta_min'] = thetas.min(); self.stats['theta_max'] = thetas.max()
        self.stats['z_mean'] = zs.mean(); self.stats['z_std'] = zs.std() + 1e-6
        self.stats['layer_min'] = layers.min(); self.stats['layer_max'] = layers.max()
        return self

    def transform(self, evt):
        # evt: dict
        r = np.array(evt["hit_r"]); theta = np.array(evt["hit_theta"]); z = np.array(evt["hit_z"]); layer = np.array(evt["layer_id"])
        r = (r - self.stats['r_mean']) / self.stats['r_std']
        theta = (theta - self.stats['theta_min']) / (self.stats['theta_max'] - self.stats['theta_min'] + 1e-6)
        z = (z - self.stats['z_mean']) / self.stats['z_std']
        layer = (layer - self.stats['layer_min']) / (self.stats['layer_max'] - self.stats['layer_min'] + 1e-6)
        X = np.column_stack([r, theta, z, layer]).astype(np.float32)
        return X

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        F = example_batch_x.x.shape[1]  # [sum N, 4]
        self.strategy = "inference"
        self.conv1 = SAGEConv(F, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = SAGEConv(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.cls = nn.Linear(64, 100)  # classes 0 to 99

    def set_training_mode(self, mode):
        self.strategy = mode

    def forward(self, batch):
        x = batch.x  # [sum N, 4]
        edge_index = batch.edge_index  # [2, E]
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        logits = self.cls(x)  # [sum N, 100]
        if self.strategy == "training":
            return logits
        else:
            return torch.argmax(logits, dim=1)  # [sum N] int64

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    model.set_training_mode("training")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_loss = float('inf')
    patience = 3; counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss_total = 0; train_correct = 0; train_total = 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb = view.batch_x
            yb = view.batch_y
            out = model(xb)  # logits [sum N, 100]
            loss = F.cross_entropy(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            preds = out.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.numel()
        scheduler.step()
        tr_acc = train_correct / train_total
        tr_loss = train_loss_total / len(train_loader)
        model.eval()
        val_loss_total = 0; val_correct = 0; val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb = view.batch_x
                yb = view.batch_y
                out = model(xb)
                loss = F.cross_entropy(out, yb)
                val_loss_total += loss.item()
                preds = out.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.numel()
        va_acc = val_correct / val_total
        va_loss = val_loss_total / len(val_loader)
        if va_loss < best_loss:
            best_loss = va_loss; counter = 0
        else:
            counter += 1
        if counter >= patience:
            break
    model.set_training_mode("inference")
    return model, tr_loss, va_loss, tr_acc, va_acc

# IMPORTANT: DO NOT execute the pipeline here - the harness will do that.
# <end code template>

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    Xs = [split_X_y(evt)[0] for evt in raw_train]
    pre = make_preprocessor().fit(Xs)

    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, raw_train, pre, train=True)
    val_ds       = build_dataset(spec, raw_val,   pre, train=False)
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
                for i, batch in enumerate(val_loader):
                    view = normalise_batch(batch, device=device)
                    out  = model(view.batch_x)
                    assert_label_output(view.batch_x, out, allow_noise_label=True)
                    if i >= 4: # loop over 4 batches
                        break
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

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

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

