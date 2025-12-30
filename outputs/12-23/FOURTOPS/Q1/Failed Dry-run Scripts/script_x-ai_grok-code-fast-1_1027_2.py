
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
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.data import Data

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    # REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataSet"
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.data_list = pre.transform(X) if pre is not None else X
        self.y = y
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx], self.y[idx]

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
        # <LLM: Define and initialize any stateful components here>
        pass

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:CustomDataset",       # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",  # Updated to use PyG DataLoader
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one: 
            "collate": None,  # Set to None for PyG DataLoader

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # No fitting needed for graph construction
        return self

    def transform(self, X):
        # X is a torch tensor of shape [N, 92]
        # Build list of torch_geometric.data.Data objects
        data_list = []
        for event in X:  # event shape [92]
            # Extract global features
            et_miss = event[0]  # scalar
            phi_et_miss = event[1]  # scalar

            # Extract objects
            node_features = []  # list of [E, p_T, eta, phi]
            nodes_positions = []  # for potential edge computation, but not used here
            num_objects = 0
            for i in range(18):
                start = 2 + i * 5
                obj_id = event[start].item()
                if obj_id == 0:  # padding
                    break
                E = event[start + 1]
                p_T = event[start + 2]
                eta = event[start + 3]
                phi = event[start + 4]
                node_features.append([E, p_T, eta, phi])  # shape [num_objects, 4]
                nodes_positions.append((eta, phi))
                num_objects += 1

            if num_objects == 0:
                # Edge case: no objects, add dummy graph?
                # For completeness, create a graph with only global node
                x = torch.tensor([[et_miss, phi_et_miss, 0.0, 0.0]], dtype=torch.float)  # [1, 4]
                edge_index = torch.empty(2, 0, dtype=torch.long)  # No edges
                data = Data(x=x, edge_index=edge_index)
                data_list.append(data)
                continue

            # Build edges: fully connected undirected graph
            edges = []
            for i in range(num_objects):
                for j in range(num_objects):
                    if i != j:
                        edges.append([i, j])
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t()  # shape [2, num_edges]
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)

            # Node features including objects, shape [num_objects, 4]
            x = torch.tensor(node_features, dtype=torch.float)

            # Add global node: connect fully to all object nodes
            global_features = [et_miss.item(), phi_et_miss.item(), 0.0, 0.0]  # shape [4]
            x = torch.cat([x, torch.tensor([global_features], dtype=torch.float)], dim=0)  # [num_objects+1, 4]
            global_idx = num_objects

            # Add edges from global to all objects
            for i in range(num_objects):
                edges.append([i, global_idx])
                edges.append([global_idx, i])
            if edges:
                new_edges = torch.tensor(edges[len(edge_index[0]) // 2:] if edge_index.shape[1] > 0 else edges, dtype=torch.long).t()  # Avoid duplicates
                edge_index = torch.cat([edge_index, new_edges], dim=1) if edge_index.shape[1] > 0 else new_edges  # [2, total_edges]

            data = Data(x=x, edge_index=edge_index)  # x: [num_nodes, 4], edge_index: [2, num_edges]
            data_list.append(data)

        return data_list  # list of Data

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        in_dim = sample_object.num_node_features  # 4 from transform
        self.conv1 = torch_geometric.nn.GCNConv(in_dim, 128)
        self.conv2 = torch_geometric.nn.GCNConv(128, 128)
        self.pool = torch_geometric.nn.global_mean_pool
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)

    def forward(self, batch_x):
        x, edge_index, batch = batch_x.x, batch_x.edge_index, batch_x.batch  # x: [num_nodes, 4], batch: [num_nodes]
        x = self.conv1(x, edge_index)  # [num_nodes, 128]
        x = F.elu(x)
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)  # [num_nodes, 128]
        x = F.elu(x)
        x = self.pool(x, batch)  # [batch_size, 128]
        x = self.fc1(x)  # [batch_size, 64]
        x = F.elu(x)
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch_size, 32]
        x = F.elu(x)
        x = self.batch_norm3(x)
        x = self.dropout(x)
        x = self.fc3(x).squeeze()  # [batch_size]
        return x

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50  # Increased for better convergence
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS 
    #   Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #   Must return trained_model, train_loss, val_loss, train_acc, val_acc
    #   Use CUDA - torch.cuda.is_available()
    #   Implement early-stopping.
    #   Forward signature must match.

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience = 10
    best_model_state = None

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            data, y = batch
            data = data.to(device)
            y = y.to(device).float()
            optimizer.zero_grad()
            output = model(data)  # [batch_size]
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
            probs = torch.sigmoid(output)
            pred = (probs > 0.5).long()
            train_correct += (pred == y.long()).sum().item()
            train_total += data.num_graphs

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                data, y = batch
                data = data.to(device)
                y = y.to(device).float()
                output = model(data)
                loss = criterion(output, y)
                val_loss += loss.item() * data.num_graphs
                probs = torch.sigmoid(output)
                pred = (probs > 0.5).long()
                val_correct += (pred == y.long()).sum().item()
                val_total += data.num_graphs

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, train_accs, val_accs

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.

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
    def _unpack_xy_strict(batch):
        # Strict FOURTOPS assumption: default collate returns (X, y) or [X, y]
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
            if torch.is_tensor(x) and torch.is_tensor(y):
                return x, y
        raise TypeError(
            f"Expected FOURTOPS batch as (X, y) tensors, got {type(batch)} with value structure: {repr(type(batch))}"
        )

    first_batch = next(iter(train_loader))

    if os.environ.get("FOURTOPS_DISABLE_NORMALISE_BATCH", "0") == "1":
        batch_x, batch_y = _unpack_xy_strict(first_batch)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        model   = make_model(batch_x).to(device)
    else:
        view  = normalise_batch(first_batch, device=device)
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


