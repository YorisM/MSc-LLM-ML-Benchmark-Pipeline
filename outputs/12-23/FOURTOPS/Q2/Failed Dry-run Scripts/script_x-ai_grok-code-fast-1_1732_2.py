
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
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric import nn as gnn
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    # REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataSet"
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        event = self.X[idx]
        # parse
        global_feat = event[0:2]  # ET_miss, phi
        objs = []
        for i in range(18):
            start = 2 + i * 5
            obj_id = event[start].item()
            if obj_id != 0:
                e, pt, eta, phi = event[start + 1:start + 5]
                objs.append((e, pt, eta, phi))
        n_objs = len(objs)
        if n_objs == 0:
            # handle, add dummy
            e, pt, eta, phi = torch.tensor([0., 0., 0., 0.])
            objs.append((e, pt, eta, phi))
            n_objs = 1
        # nodes
        node_features = torch.stack(objs)  # (n_objs, 4)
        # add global to each node
        global_node = global_feat.repeat(n_objs, 1)  # (n_objs, 2)
        x = torch.cat([node_features, global_node], dim=-1)  # (n_objs, 6)
        # edges: all pairs i<j
        pairs = []
        pair_attrs = []
        for i in range(n_objs):
            for j in range(i + 1, n_objs):
                ei, pti, etai, phii = objs[i]
                ej, ptj, etaj, phij = objs[j]
                # invariant mass (in GeV, approx)
                pxi = pti * torch.cos(phii)
                pyi = pti * torch.sin(phii)
                pzi = pti * torch.sinh(etai)
                pxj = ptj * torch.cos(phij)
                pyj = ptj * torch.sin(phij)
                pzj = ptj * torch.sinh(etaj)
                e_tot = ei + ej
                px_tot = pxi + pxj
                py_tot = pyi + pyj
                pz_tot = pzi + pzj
                mass2 = e_tot ** 2 - px_tot ** 2 - py_tot ** 2 - pz_tot ** 2
                mass = torch.sqrt(torch.maximum(mass2 / 1e6 ** 2, torch.tensor(0.0)))  # MeV to GeV, approx
                # delta R
                dr = torch.sqrt((etai - etaj) ** 2 + (phii - phij) ** 2)
                pair_attrs.append([mass, dr])
                pairs.append([i, j])
        if pairs:
            edge_attr = torch.tensor(pair_attrs)  # (n_pairs, 2)
            edge_index = torch.tensor(pairs).t().long()  # (2, n_pairs)
        else:
            edge_attr = torch.empty(0, 2)
            edge_index = torch.empty(2, 0).long()
        # Data object
        data = Data(x=x.float(), edge_index=edge_index, edge_attr=edge_attr.float(), y=self.y[idx].long())
        return data

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit() 
    #   - transform()

    # DATA SPECIFICS
    # Total flat length per event (X_train & X_val): 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    # Indices  7-11 : object 2  ->  obj_2, E_2 , p_T_2 , eta_2 , phi_2
    # ...
    # Indices 87-91 : object 18 ->  obj_18, E_18 , p_T_18 , eta_18 , phi_18
    # Global features       = 2
    # Per-object slice size = 5
    # Max objects encoded   = 18

    # TIPS
    # When modifying data features or feature engineering: annotate tensor size as comments after 
    # each tensor operation to reduce dimension mismatches.

    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module.
    # May allocate NumPy arrays or Torch tensors internally, but:
    # transform() must be deterministic.
    # Store only derived parameters needed for transform i.e. do not store the raw data
    # itself in the preprocessor object.

    def __init__(self):
        self.col_means = None
        self.col_stds = None

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:CustomDataset",   # default harness dataset
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
        self.col_means = torch.mean(X, dim=0)
        self.col_stds = torch.std(X, dim=0, unbiased=False) + 1e-6
        return self

    def transform(self, X):
        return (X - self.col_means) / self.col_stds  # normalize per column

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is the Batch from first batch, x.shape[1] = 6
        self.conv1 = gnn.GCNConv(in_channels=6, out_channels=64)
        self.conv2 = gnn.GCNConv(64, 128)
        self.conv3 = gnn.GCNConv(128, 128)
        self.pool = gnn.global_mean_pool
        self.fc = nn.Linear(128, 1)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.pool(x, batch.batch)
        return self.fc(x)  # logit for BCEWithLogits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20   # <LLM: adjust if you wish>
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS 
    #   Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #   Must return trained_model, train_loss, val_loss, train_acc, val_acc
    #   Use CUDA - torch.cuda.is_available()
    #   Implement early-stopping.
    #   Forward signature must match.

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    tr_loss = [] ; va_loss = [] ; tr_acc = [] ; va_acc = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits.squeeze(), batch.y.float())
            loss.backward()
            optimizer.step()
            preds = (logits.sigmoid() > 0.5).float()
            acc = (preds.squeeze() == batch.y).float().mean()
            epoch_train_loss += loss.item()
            epoch_train_acc += acc.item()
        epoch_train_loss /= len(train_loader)
        epoch_train_acc /= len(train_loader)
        tr_loss.append(epoch_train_loss)
        tr_acc.append(epoch_train_acc)

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits.squeeze(), batch.y.float())
                preds = (logits.sigmoid() > 0.5).float()
                acc = (preds.squeeze() == batch.y).float().mean()
                epoch_val_loss += loss.item()
                epoch_val_acc += acc.item()
        epoch_val_loss /= len(val_loader)
        epoch_val_acc /= len(val_loader)
        va_loss.append(epoch_val_loss)
        va_acc.append(epoch_val_acc)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    return model, tr_loss, va_loss, tr_acc, va_acc

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


