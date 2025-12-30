
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
import torch_geometric.nn as gnn
import torch_geometric.data as data
import math
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# class CustomDataset(Dataset):
#   REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
#    def __init__(self, events, pre, train: bool = True, **kwargs):
#        X, y = events
#        self.X = pre.transform(X) if pre is not None else X
#        self.y = y
#    def __len__(self):
#        return int(self.y.shape[0])
#    def __getitem__(self, idx):
#        return self.X[idx], self.y[idx]

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    # TIPS
    #   - When modifying data features or feature engineering: annotate tensor size as comments after 
    #   - each tensor operation to reduce dimension mismatches.

    # DATA SPECIFICS
    #    Total flat length per event (X_train & X_val): 92
    #    Index  0 :  missing-ET magnitude  (E_T_miss)
    #    Index  1 :  missing-ET azimuth    (phi_Et_miss)
    #    Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    #    Indices  7-11 : object 2  ->  obj_2, E_2 , p_T_2 , eta_2 , phi_2
    #    ...
    #    Indices 87-91 : object 18 ->  obj_18, E_18 , p_T_18 , eta_18 , phi_18
    #    Global features       = 2
    #    Per-object slice size = 5
    #    Max objects encoded   = 18

    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        pass

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one:
            "collate": None, # (or "ragged_xy" or "identity" - If loader_class is torch_geometric.loader:DataLoader, set "collate": None.)

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # <LLM: Extract statistics for transform>
        return self

    def transform(self, X):
        # Extract objects, skip padded with obj_id == 0, create graph per event
        datas = []
        for event_idx in range(X.shape[0]):
            event = X[event_idx]
            missing_et = event[0]
            phi_met = event[1]
            globals = torch.tensor([missing_et, phi_met])
            objects = []
            for i in range(18):
                start = 2 + i * 5
                obj_id = event[start]
                if obj_id == 0:
                    break
                E, pT, eta, phi = event[start+1:start+5]
                objects.append([obj_id, E, pT, eta, phi])
            num_obj = len(objects)
            if num_obj < 2:
                # Skip events with fewer than 2 objects, as pairs needed; unlikely in this dataset
                continue
            node_features = torch.stack([torch.tensor(o) for o in objects])  # [num_obj, 5]
            if num_obj < 18:
                # Zero-pad to max size if needed, but since we break at zero, optional
                pass
            eta = node_features[:, 3]
            phi = node_features[:, 4]
            E_tensor = node_features[:, 1]
            pT_tensor = node_features[:, 2]
            # Compute combinations for undirected edges (bidirectional)
            combo_pairs = torch.combinations(torch.arange(num_obj), r=2)  # [num_combos, 2], num_combos = num_obj*(num_obj-1)//2
            row = torch.cat([combo_pairs[:, 0], combo_pairs[:, 1]])
            col = torch.cat([combo_pairs[:, 1], combo_pairs[:, 0]])
            edge_index = torch.stack([row, col])  # [2, num_edges], num_edges = num_combos * 2
            edge_attr_list = []
            for pair_idx in range(combo_pairs.shape[0]):
                i, j = combo_pairs[pair_idx]
                eta_i, phi_i = eta[i], phi[i]
                eta_j, phi_j = eta[j], phi[j]
                deltaR = torch.sqrt((eta_i - eta_j)**2 + (phi_i - phi_j)**2)
                E1, pT1 = E_tensor[i], pT_tensor[i]
                E2, pT2 = E_tensor[j], pT_tensor[j]
                phi1, eta1 = phi[i], eta[i]
                phi2, eta2 = phi[j], eta[j]
                px1 = pT1 * torch.cos(phi1)
                py1 = pT1 * torch.sin(phi1)
                pz1 = pT1 * torch.sinh(eta1)
                px2 = pT2 * torch.cos(phi2)
                py2 = pT2 * torch.sin(phi2)
                pz2 = pT2 * torch.sinh(eta2)
                total_E = E1 + E2
                total_px = px1 + px2
                total_py = py1 + py2
                total_pz = pz1 + pz2
                m_ij_sq = total_E**2 - (total_px**2 + total_py**2 + total_pz**2)
                m_ij = torch.sqrt(torch.clamp(m_ij_sq, min=0.0))  # Ensure no negative sqrt
                edge_attr_list.append([deltaR, m_ij])
                edge_attr_list.append([deltaR, m_ij])  # For both directions
            edge_attr = torch.tensor(edge_attr_list)  # [num_edges, 2]
            datas.append(data.Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, globals=globals))
        return datas  # List of Data objects

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
# Model batch contract:
#   Your DataLoader batch is NOT guaranteed to be a single Tensor.
#   Depending your dataset/loader choice, a batch can be:
#      - (X, y) tuple OR [X, y] list  (common for default PyTorch/PyG collation)
#      - ragged: X is list[Tensor] and y is list[Tensor] (one Tensor per event)
#      - multi-input: (X1, X2, ..., y) OR [X1, X2, ..., y]
#      - dict-like: {"x": X, "y": y} (or inputs/labels variants)
#      - PyG: torch_geometric.data.Data or torch_geometric.data.Batch
#
# ALWAYS adapt the raw batch using:
#     view = normalise_batch(batch, device=device)
#
# normalise_batch returns a BatchView with:
#   view.batch_x : the model inputs (Tensor / list[Tensor] / tuple / dict / PyG Batch)
#   view.batch_y : labels if present, else None
#
# IMPORTANT: normalise_batch(..., device=device) moves ALL contained tensors to device (recursively). Do NOT call .to(device) on the raw batch object.

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        num_node_features = sample_object.x.shape[1]  # 5 (obj_id, E, pT, eta, phi)
        self.obj_emb = nn.Embedding(20, 8)  # Assume obj_id up to 20, embed to 8 dims
        node_dim = num_node_features - 1 + 8  # obj_id replaced with emb + others
        self.gnn = gnn.GATConv(node_dim, 128, edge_dim=2, heads=4, concat=False, dropout=0.1)  # GAT with edge features
        self.pool = gnn.global_max_pool  # Use max pool for better discriminative power
        hidden_dim = 128 + 2  # pooled + globals
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, batch_x):
        # batch_x is Batch object
        obj_ids = batch_x.x[:, 0].long()
        obj_emb = self.obj_emb(obj_ids)
        node_x = torch.cat([batch_x.x[:, 1:], obj_emb], dim=1)  # Concat after obj_id, [total_nodes, node_dim]
        x = self.gnn(node_x, batch_x.edge_index, batch_x.edge_attr)  # [total_nodes, 128]
        pooled = self.pool(x, batch_x.batch)  # [batch_size, 128]
        concat = torch.cat([pooled, batch_x.globals], dim=1)  # [batch_size, 130]
        out = self.mlp(concat)  # [batch_size, 1]
        return out.squeeze()  # [batch_size] logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20   # Increased epochs for better training
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    #   - Must return: trained_model, train_loss, val_loss, train_acc, val_acc
    #   - Do NOT:
    #       - pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #       - batch = batch.to(device)
    #       - xb, yb = batch
    #       - for xb, yb in loader: ...

    # Canonical batch handling (use this inside every loop):
    # for batch in train_loader:
    #     view = normalise_batch(batch, device=device)
    #     xb, yb = view.batch_x, view.batch_y
    #     out = model(xb)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    patience = 5
    no_improve = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * yb.size(0)
            preds = torch.sigmoid(out) > 0.5
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                out = model(xb)
                loss = criterion(out, yb.float())
                val_loss += loss.item() * yb.size(0)
                preds = torch.sigmoid(out) > 0.5
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        scheduler.step()
        # Early stopping based on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return model, train_losses[-1] if train_losses else None, val_losses[-1] if val_losses else None, train_accs[-1] if train_accs else None, val_accs[-1] if val_accs else None

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

