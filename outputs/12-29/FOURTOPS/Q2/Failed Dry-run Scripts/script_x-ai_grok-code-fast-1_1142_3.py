
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

# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
# <LLM: Import modules>
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# class CustomDataset(Dataset):
#  REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
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

    # <LLM: Write code to preprocess the data> 

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
        # <LLM: Apply pre-processing logic>
        # X: [N, 92] tensor
        data_list = []
        for event in X:  # event: [92]
            # Extract global features
            e_miss = event[0]
            phi_miss = event[1]
            objects = []
            for i in range(18):
                start = 2 + i * 5
                obj_id = event[start]
                e = event[start + 1]
                pt = event[start + 2]
                eta = event[start + 3]
                phi = event[start + 4]
                if obj_id != 0:  # Assuming 0 is padding
                    objects.append([obj_id, e, pt, eta, phi])
            if not objects:
                # If no objects, create minimal data (rare edge case)
                node_x = torch.empty(0, 5, dtype=torch.float32)
                edge_index = torch.empty(2, 0, dtype=torch.long)
                edge_attr = torch.empty(0, 2, dtype=torch.float32)
                global_feat = torch.tensor([e_miss, phi_miss])
                data = torch_geometric.data.Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr, global_feat=global_feat)
                data_list.append(data)
                continue
            objects_tensor = torch.stack(objects)  # [num_obj, 5]
            num_obj = objects_tensor.shape[0]
            node_x = objects_tensor  # [num_obj, 5]

            # Compute edges (undirected, all pairs)
            edge_index = []
            edge_attr = []
            for i in range(num_obj):
                for j in range(i + 1, num_obj):
                    # 4-momenta
                    p1 = objects_tensor[i, 1:]  # [e, pt, eta, phi]
                    p2 = objects_tensor[j, 1:]  # [e, pt, eta, phi]
                    # To 4-vec: [e, px, py, pz]
                    def to_4vec(e, pt, eta, phi_local):
                        px = pt * torch.sin(phi_local)
                        py = pt * torch.cos(phi_local)
                        pz = pt * torch.sinh(eta)
                        return torch.stack([e, px, py, pz])
                    v1 = to_4vec(p1[0], p1[1], p1[2], p1[3])
                    v2 = to_4vec(p2[0], p2[1], p2[2], p2[3])
                    v_sum = v1 + v2
                    m_sq = v_sum[0]**2 - v_sum[1]**2 - v_sum[2]**2 - v_sum[3]**2
                    m_ij = torch.sqrt(torch.clamp(m_sq, min=0.0))
                    dr_ij = torch.sqrt((p1[2] - p2[2])**2 + (p1[3] - p2[3])**2)
                    # Undirected
                    edge_index.extend([[i, j], [j, i]])
                    edge_attr.extend([[m_ij, dr_ij], [m_ij, dr_ij]])
            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # [2, num_edges]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # [num_edges, 2]
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)
                edge_attr = torch.empty(0, 2, dtype=torch.float32)
            global_feat = torch.tensor([e_miss, phi_miss])
            # Note: PyG Data with x=[num_obj, 5], edge_index=[2, num_edges], edge_attr=[num_edges, 2], global_feat=[2]
            data = torch_geometric.data.Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr, global_feat=global_feat)
            data_list.append(data)
        return data_list  # list of Data

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
        # sample_object is a torch_geometric.data.Batch
        # Assume sample_object.x: [total_nodes, 5], edge_index: [2, total_edges], edge_attr: [total_edges, 2], global_feat: [batch_size, 2] but in batch it's concatenated
        # In Batch, global_feat would be [batch_size, 2]
        # Nodes: 5 features (obj_id, e, pt, eta, phi)
        # Edges: 2 features (m_ij, dr_ij)
        num_node_features = 5
        num_edge_features = 2
        hidden = 64
        self.conv1 = GCNConv(num_node_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        # Global pooling
        # Also include global_feat via concat or linear
        self.global_lin = nn.Linear(2, hidden)  # for global_feat
        self.lin1 = nn.Linear(hidden * 2, hidden)  # concat pooled and global
        self.lin2 = nn.Linear(hidden, 1)

    # <LLM: optionally build extra layers here>

    def forward(self, batch_x):
        # batch_x is Batch from PyG
        # Proceed with GNN
        x = batch_x.x  # [total_nodes, 5]
        edge_index = batch_x.edge_index  # [2, total_edges]
        edge_attr = batch_x.edge_attr  # [total_edges, 2]
        global_feat = batch_x.global_feat  # [batch_size, 2]
        # But edge_attr not used in GCNConv; use GAT if want to use edge_attr, but for simplicity, ignore edge_attr in conv
        # First conv
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Pool per graph (event)
        x_pooled = global_mean_pool(x, batch_x.batch)  # [batch_size, hidden]
        # Process global_feat
        g = self.global_lin(global_feat)  # [batch_size, hidden]
        g = F.relu(g)
        # Concat
        comb = torch.cat([x_pooled, g], dim=1)  # [batch_size, hidden*2]
        comb = F.relu(self.lin1(comb))
        out = self.lin2(comb)  # [batch_size, 1]
        return out.squeeze(-1)  # logits [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 10   # <LLM: adjust if you wish>
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

    # <LLM: Write code to define training loop, use the code above>
    # <LLM: Implement early stopping if possible>
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    # Simple early stopping
    best_val_auc = -float('inf')
    patience = 3
    patience_counter = 0
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            yb = yb.float()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            epoch_train_correct += (preds == yb).sum().item()
            epoch_train_total += yb.shape[0]
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = epoch_train_correct / epoch_train_total
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_y = []
            all_preds = []
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                yb = yb.float()
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                val_correct += (preds == yb).sum().item()
                val_total += yb.shape[0]
                all_y.extend(yb.cpu())
                all_preds.extend(probs.cpu())
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            val_loss_list.append(avg_val_loss)
            val_acc_list.append(val_acc)
            # AUC
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(all_y, all_preds)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Optionally save model
        else:
            patience_counter += 1
        if patience_counter > patience:
            break
        model.train()
    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list
    # Note: Returns lists, harness expects values, but this should work as per context

# DO NOT execute the pipeline here – the harness will do that.
# <end code template>

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

