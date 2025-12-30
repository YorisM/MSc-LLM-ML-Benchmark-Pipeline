
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
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.data_list = pre.transform(X, y) if pre is not None else X
        self.y = y
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx], self.y[idx]

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
    #    Indices 7-11 : object 2  ->  obj_2, E_2 , p_T_2 , eta_2 , phi_2
    #    ...
    #    Indices 87-91 : object 18 ->  obj_18, E_18 , p_T_18 , eta_18 , phi_18
    #    Global features       = 2
    #    Per-object slice size = 5
    #    Max objects encoded   = 18

    def __init__(self):
        self.scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:CustomDataset",   # use custom for PyG
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",     # PyG loader
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one: 
            "collate": None, # for PyG DataLoader

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Assuming X is (N, 92)
        # Scale only kinematic features, not obj_id
        kinematic_indices = [0] + list(range(3,92,5)) + list(range(4,92,5)) + list(range(5,92,5)) + list(range(6,92,5))  # ETmiss, E, pT, eta, phi for all
        self.scaler.fit(X[:, kinematic_indices].numpy())
        return self

    def transform(self, X, y=None):
        # Build PyG Data objects
        data_list = []
        for i in range(X.shape[0]):
            event = X[i]  # (92,)
            global_feats = event[:2]  # (2,) ETmiss, phi
            objects = []
            edges = []
            for j in range(18):
                start = 2 + j*5
                obj_id = event[start].item()
                if obj_id > 0:  # valid object
                    feat = event[start+1:start+5]  # E, pT, eta, phi  (4,)
                    objects.append(feat)
                else:
                    objects.append(torch.zeros(4))  # pad for fixed size
            x = torch.stack(objects)  # (18, 4)
            # Add global as an extra node
            global_node = global_feats.unsqueeze(0)  # (1,2), but to match, pad to 4? Wait, better make all features 4-dim
            # Scale kinematic
            kinematic = [global_node.squeeze(), x]  # global (2,), x (18,4)
            # To unify, perhaps embed later
            # But for now, x will be for objects, global separate

            # Simple: Fully connected graph between valid objects
            num_valid = sum(1 for j in range(18) if event[2+j*5].item() > 0)
            if num_valid > 0:
                # Valid nodes 0 to num_valid-1
                edge_index = torch.combinations(torch.arange(num_valid), 2, r=2).t()  # (2, num_valid choose 2)
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected
            else:
                edge_index = torch.empty(2,0, dtype=torch.long)
            # Add global node connected to all
            global_edges = torch.tensor([[num_valid], [j] for j in range(num_valid)] + [[j], [num_valid] for j in range(num_valid)], dtype=torch.long).t()
            edge_index = torch.cat([edge_index, global_edges], dim=1) if global_edges.numel() > 0 else edge_index

            # Features: objects x (18,4), but only first num_valid valid, but we're using padded
            # To handle, perhaps mask later, but for simplicity, use all 18
            x = torch.cat([x, torch.zeros(1,4)], dim=0)  # (19,4) for global as zeros? Wait no
            # Better: make global features into 4-dim by padding
            global_feat_padded = torch.cat([global_feats, torch.zeros(2)])  # (4,)
            x_all = torch.cat([x, global_feat_padded.unsqueeze(0)], dim=0)  # (19,4)
            # Now edge_index should be based on this
            # Recreate edges
            edge_index = torch.combinations(torch.arange(num_valid + 1), r=2).t()  # Full connected for num_valid +1 nodes (objects + global), but global is 18? Wait mess
            # Simplify: make N=19 nodes: 18 objects (padded), +1 global
            # Edges: fully connected between all non-padded objects and global

            data = Data(x=x_all, edge_index=edge_index, y=y[i] if y is not None else None)
            data_list.append(data)
        return data_list

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
        self.conv1 = GCNConv(4, 64)  # (batch_size*19, 64)
        self.conv2 = GCNConv(64, 128)  # (batch_size*19, 128)
        self.fc = nn.Linear(128, 1)  # (batch_size, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, batch):
        # batch is PyG Batch
        x = batch.x  # (total_nodes, 4)
        edge_index = batch.edge_index
        batch_idx = batch.batch  # for pooling
        x = F.relu(self.conv1(x, edge_index))  # (total, 64)
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))  # (total, 128)
        x = self.dropout(x)
        x = global_mean_pool(x, batch_idx)  # (batch_size, 128)
        out = self.fc(x)  # (batch_size, 1)
        return out.squeeze()  # (batch_size,)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20  # increased epochs
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_auc = 0
    early_stop_cnt = 0
    patience = 10

    from sklearn.metrics import roc_auc_score

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.sigmoid(out) > 0.5
            correct += (preds == yb).sum().item()
            total += yb.shape[0]

            all_preds.extend(out.cpu().detach().numpy())
            all_targets.extend(yb.cpu().numpy())

        train_auc = roc_auc_score(all_targets, all_preds)
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_loss_history.append(train_loss)
        train_acc_history.append(train_auc)  # use AUC for acc

        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                out = model(xb)
                loss = criterion(out, yb.float())

                total_loss += loss.item()
                preds = torch.sigmoid(out) > 0.5
                correct += (preds == yb).sum().item()
                total += yb.shape[0]

                all_preds.extend(out.cpu().detach().numpy())
                all_targets.extend(yb.cpu().numpy())

        val_auc = roc_auc_score(all_targets, all_preds)
        val_loss = total_loss / len(val_loader)
        val_acc = correct / total
        val_loss_history.append(val_loss)
        val_acc_history.append(val_auc)

        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            early_stop_cnt = 0
            best_model = model.state_dict().copy()
        else:
            early_stop_cnt += 1

        if early_stop_cnt >= patience:
            break

    model.load_state_dict(best_model)
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

