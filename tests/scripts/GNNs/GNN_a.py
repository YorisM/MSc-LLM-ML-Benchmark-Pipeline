"""
docker run --rm --gpus all --network none --read-only --cap-drop ALL --security-opt seccomp=docker/seccomp_profile.json --tmpfs /tmp:rw,noexec,nosuid --tmpfs /dev/shm:rw -e DRYRUN_TIMEOUT_S=3600 -e TRAIN_TIMEOUT_S=99999 -e EVAL_TIMEOUT_S=3600 -e PYTHONPATH=/workspace -e FOURTOPS_DISABLE_NORMALISE_BATCH=1 -w /workspace -v C:\Users\yoris\projects\Benchmark\utils\llm_io.py:/workspace/utils/llm_io.py:ro -v C:\Users\yoris\projects\Benchmark\utils\loaderspec.py:/workspace/utils/loaderspec.py:ro -v C:\Users\yoris\projects\Benchmark\utils\suffix_utils.py:/workspace/utils/suffix_utils.py:ro -v C:/Users/yoris/projects/Benchmark:/workspace:rw -v C:\Users\yoris\projects\Benchmark\challenges\FOURTOPS\data\train:/data/train:ro -v C:/Users/yoris/projects/Benchmark/outputs:/workspace/out:rw --entrypoint /usr/local/bin/train.sh llm-sandbox:latest "tests/scripts/GNNs/GNN_a.py" --dryrun
"""

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
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATv2Conv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class PygFourTopsDataset(Dataset):
    """
    Converts each flat FOURTOPS event (92,) into a PyG Data object with:
      - 18 nodes (objects), each node feature: [obj_id, E, pT, eta, phi, present_flag]  -> (6,)
      - graph-level features: [met, met_phi] stored as data.u -> (2,)
      - label stored as data.y -> shape (1,)
      - fully-connected edges among present nodes (no self-loops)
    """
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X
        self.y = y

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        x = self.X[idx]  # (92,)
        y = self.y[idx]  # scalar

        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.long)

        # Global features
        met = x[0:1]      # (1,)
        met_phi = x[1:2]  # (1,)
        u = torch.cat([met, met_phi], dim=0).to(torch.float32)  # (2,)

        # Objects: 18 * 5 = 90 entries starting at index 2
        obj = x[2:].view(18, 5)  # (18,5) -> [obj_id, E, pT, eta, phi]
        obj_id = obj[:, 0:1]     # (18,1)
        kin = obj[:, 1:5]        # (18,4)

        # present flag: object id != 0 (padding)
        present = (obj_id.squeeze(-1) != 0).to(torch.float32).view(18, 1)  # (18,1)
        node_x = torch.cat([obj_id.to(torch.float32), kin.to(torch.float32), present], dim=1)  # (18,6)

        # Build edges among present nodes only (fully connected, no self loops)
        present_idx = torch.nonzero(present.squeeze(-1) > 0.5, as_tuple=False).view(-1)  # (Np,)
        if present_idx.numel() <= 1:
            edge_index = torch.empty((2, 0), dtype=torch.long)  # (2,0)
        else:
            ii = present_idx.repeat_interleave(present_idx.numel())
            jj = present_idx.repeat(present_idx.numel())
            mask = ii != jj
            edge_index = torch.stack([ii[mask], jj[mask]], dim=0)  # (2,E)

        data = Data(x=node_x, edge_index=edge_index)
        data.u = u.view(1, 2) # (1,2) so Batch.u becomes (B,2)
        data.y = y.view(1)    # (1,)
        return data


# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    """
    Model A: "GCN + global mean pool + concat graph globals"
    Preprocessing: standardize continuous kinematics (E, pT, eta, phi, met) with
    per-feature mean/std computed over training data (torch-only, picklable).
    """
    def __init__(self):
        self.mean_: Optional[torch.Tensor] = None
        self.std_: Optional[torch.Tensor] = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:PygFourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,  # required for PyG loader
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # X: (N,92) float32 tensor
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float32)

        # Continuous fields: met(0), met_phi(1), then for each object: E,pT,eta,phi (skip obj_id)
        met = X[:, 0:2]  # (N,2)
        obj = X[:, 2:].view(-1, 18, 5)                # (N,18,5)
        cont = obj[:, :, 1:5].reshape(-1, 4)          # (N*18,4)

        met_rep = met.repeat_interleave(18, dim=0)    # (N*18,2)
        feats = torch.cat([met_rep, cont], dim=1)     # ✅ (N*18,6) = [met,met_phi,E,pT,eta,phi]

        mean = feats.mean(dim=0)
        std = feats.std(dim=0).clamp_min(1e-6)
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            return X
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float32)

        X = X.clone()

        # Scale met & met_phi
        X[:, 0:2] = (X[:, 0:2] - self.mean_[0:2]) / self.std_[0:2]  # (N,2)

        # Scale object kinematics
        obj = X[:, 2:].view(-1, 18, 5)  # (N,18,5)
        cont = obj[:, :, 1:5]           # (N,18,4)
        cont = (cont - self.mean_[2:6].view(1, 1, 4)) / self.std_[2:6].view(1, 1, 4)  # (N,18,4)
        obj[:, :, 1:5] = cont
        X[:, 2:] = obj.view(-1, 90)
        return X


def make_preprocessor():
    return MyPreprocessor()


# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    """
    Model A: GCN on object graph, pool, concat globals -> MLP -> logits
    Output: logits shape (B,) (float)
    """
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is expected to be a PyG Batch/Data
        in_dim = int(sample_object.x.shape[-1])  # 6
        self.hidden = 64

        self.conv1 = GCNConv(in_dim, self.hidden)
        self.conv2 = GCNConv(self.hidden, self.hidden)
        self.lin_u = nn.Linear(2, self.hidden)

        self.head = nn.Sequential(
            nn.Linear(self.hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, batch_x):
        # batch_x: PyG Batch
        x = batch_x.x          # (sum_nodes, 6)
        ei = batch_x.edge_index
        b = getattr(batch_x, "batch", None)
        if b is None:
            b = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        h = F.relu(self.conv1(x, ei))   # (sum_nodes, 64)
        h = F.relu(self.conv2(h, ei))   # (sum_nodes, 64)
        g = global_mean_pool(h, b)      # (B,64)

        # graph globals
        u = getattr(batch_x, "u", None)
        B = int(g.shape[0])

        if u is None:
            u = torch.zeros((B, 2), device=g.device)
        elif torch.is_tensor(u):
            if u.ndim == 1:
                if u.numel() == 2:
                    u = u.view(1, 2).expand(B, 2)              # (2,) -> (B,2)
                elif u.numel() == B * 2:
                    u = u.view(B, 2)                           # (B*2,) -> (B,2)
                else:
                    raise RuntimeError(f"Unexpected u shape: {tuple(u.shape)} for B={B}")
            elif u.ndim == 2:
                if tuple(u.shape) == (1, 2):
                    u = u.expand(B, 2)                         # (1,2) -> (B,2)
                elif tuple(u.shape) != (B, 2):
                    raise RuntimeError(f"Unexpected u shape: {tuple(u.shape)} for B={B}")
            else:
                raise RuntimeError(f"Unexpected u ndim: {u.ndim}")

        gu = self.lin_u(u)   # (B,2) -> (B,hid)
        z = torch.cat([g, gu], dim=1)   # (B,128)
        logits = self.head(z).view(-1)  # (B,)
        return logits


def make_model(example_object):
    return BinaryClassifier(example_object)


# ---------- MODEL TRAINING ----------
EPOCHS = 5
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)

    best_val = float("inf")
    best_state = None
    patience = 3
    bad = 0

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for ep in range(int(epochs)):
        model.train()
        n, correct, total_loss = 0, 0, 0.0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            # Labels may be in Data.y or provided separately; ensure tensor (B,)
            if yb is None:
                yb = getattr(xb, "y", None)
            if torch.is_tensor(yb) and yb.ndim == 2 and yb.shape[1] == 1:
                yb = yb.view(-1)
            if torch.is_tensor(yb) and yb.ndim == 1 and yb.dtype != torch.float32:
                yb_f = yb.float()
            else:
                yb_f = yb

            opt.zero_grad(set_to_none=True)
            logits = model(xb)                    # (B,)
            loss = F.binary_cross_entropy_with_logits(logits, yb_f)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += float(loss.item()) * int(yb_f.shape[0])
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += int((preds == yb.long()).sum().item())
            n += int(yb.shape[0])

        tr_loss = total_loss / max(1, n)
        tr_acc = correct / max(1, n)
        train_loss.append(tr_loss)
        train_acc.append(tr_acc)

        model.eval()
        n, correct, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                if yb is None:
                    yb = getattr(xb, "y", None)
                if torch.is_tensor(yb) and yb.ndim == 2 and yb.shape[1] == 1:
                    yb = yb.view(-1)
                yb_f = yb.float() if torch.is_tensor(yb) else yb

                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb_f)

                total_loss += float(loss.item()) * int(yb.shape[0])
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += int((preds == yb.long()).sum().item())
                n += int(yb.shape[0])

        va_loss = total_loss / max(1, n)
        va_acc = correct / max(1, n)
        val_loss.append(va_loss)
        val_acc.append(va_acc)

        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    trained_model = model
    return trained_model, train_loss, val_loss, train_acc, val_acc

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