
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops

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
        X2 = pre.transform(X) if pre is not None else X
        if not torch.is_tensor(X2):
            X2 = torch.as_tensor(X2)
        self.X = X2.float()
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        self.y = y.long()
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
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GNNConv, global_mean_pool
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    # REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
    def __init__(self, events, pre, train: bool = True, **kwargs):
        objects, global_ = events
        self.objects = objects  # [n, 18, 5]
        self.global_ = global_  # [n, 2]
        self.y = pre.y if hasattr(pre, 'y') else None  # Wait, no: y is part of events? Wait, in code below.
        # Actually, events is (X2, y), but in harness, build_dataset(spec, (X, y), pre, train)
        # In code: train_ds = build_dataset(spec, (X_train, Y_train), pre, train=True)
        # And spec["dataset_builder"] = "llm_script:CustomDataset"
        # build_dataset calls CustomDataset((X_train, Y_train), pre, train=True)
        X2, y = events
        objects, global_ = pre.transform(X2)
        self.objects = objects  # [n, 18, 5]
        self.global_ = global_  # [n, 2]
        self.y = y.long() if torch.is_tensor(y) else torch.as_tensor(y).long()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        obj = self.objects[idx]  # [18, 5]
        glob = self.global_[idx]  # [2]
        label = self.y[idx]
        # Filter out padded objects: mask where obj_id != 0 and E > 0
        mask = (obj[:, 0] != 0) & (obj[:, 1] > 0)
        nodes = obj[mask].float()  # [num_nodes, 5], [obj_id, E, pT, eta, phi]
        num_nodes = nodes.shape[0]
        if num_nodes < 1:
            # Handle empty: create a dummy node with global features
            nodes = torch.zeros(1, 5)  # dummy node, features all 0
            num_nodes = 1
        # Build node features: use obj_id as float, and kin
        x = nodes  # [num_nodes, 5]
        # Build fully connected edges
        edge_index = torch.empty(2, num_nodes * num_nodes, dtype=torch.long)
        edge_attr = torch.empty(num_nodes * num_nodes, 2, dtype=torch.float32)  # m_ij, delta_r
        k = 0
        for ii in range(num_nodes):
            for jj in range(num_nodes):
                edge_index[0, k] = ii
                edge_index[1, k] = jj
                # Compute pairwise features
                eta1, phi1 = nodes[ii, 3], nodes[ii, 4]
                eta2, phi2 = nodes[jj, 3], nodes[jj, 4]
                deta = eta1 - eta2
                dphi = phi1 - phi2
                dphi = (dphi + math.pi) % (2 * math.pi) - math.pi
                delta_r = torch.sqrt(deta**2 + dphi**2)
                # Invariant mass
                E1, pT1 = nodes[ii, 1], nodes[ii, 2]
                E2, pT2 = nodes[jj, 1], nodes[jj, 2]
                px1 = pT1 * torch.cos(phi1)
                py1 = pT1 * torch.sin(phi1)
                pz1 = pT1 * torch.sinh(eta1)
                px2 = pT2 * torch.cos(phi2)
                py2 = pT2 * torch.sin(phi2)
                pz2 = pT2 * torch.sinh(eta2)
                E_sum = E1 + E2
                px_sum = px1 + px2
                py_sum = py1 + py2
                pz_sum = pz1 + pz2
                m_ij = torch.sqrt(E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2))
                edge_attr[k] = torch.tensor([m_ij, delta_r])
                k += 1
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label.unsqueeze(0))  # y as [1]
        return data

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
        # LoaderSpec-first: evaluator rebuilds loaders from this. Configure as you please.
        return {
            "dataset_builder": "llm_script:CustomDataset",   # custom dataset for PyG graphs
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",     # PyG DataLoader
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed.
            "collate": None,

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False, 
                                "batch_size": 512} # Or whatever you want
        }

    def fit(self, X, y=None):
        # <LLM: Extract statistics for transform>
        return self

    def transform(self, X):
        # <LLM: Apply pre-processing logic>
        # Reshape to objects and global
        global_feat = X[:, :2]  # [n_events, 2]
        objects = X[:, 2:].reshape(X.shape[0], 18, 5)  # [n_events, 18, 5]
        return objects, global_feat  # return tuple of tensors

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
# MODEL I/O BATCH CONTRACT (CHOOSE ONE LANE)
# You MUST choose exactly one of the two supported input lanes and keep it consistent:
#
# --- LANE A: Torch dense batch (default) ---
# Loader:
#   - loader_class: "torch.utils.data:DataLoader"
#   - collate: None
# Batch from DataLoader:
#   (Xb, yb) where
#     Xb: FloatTensor[B, F]
#     yb: LongTensor[B] (or [B,1])
# Model forward:
#   out = model(Xb)
#   out must be FloatTensor[B] or FloatTensor[B,1] (logits or probabilities)
#
# --- LANE B: PyTorch Geometric (PyG) graphs ---
# Loader:
#   - loader_class: "torch_geometric.loader:DataLoader"
#   - collate: None
# Dataset samples MUST be torch_geometric.data.Data with at least:
#   data.x : FloatTensor[N_i, F]
#   data.edge_index : LongTensor[2, E_i]   (or equivalent; your model can build edges too)
#   data.y : LongTensor[1]                (GRAPH-LEVEL label for the event!)
# Batch from DataLoader:
#   G : torch_geometric.data.Batch (has G.x, G.edge_index, G.batch, and G.y)
# Model forward:
#   out = model(G)
#   out must be FloatTensor[num_graphs] or FloatTensor[num_graphs,1] (logits or probabilities)
#
# Any other batch shapes are NOT supported.

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Input: num node features = 5, edge_attr = 2
        self.conv1 = GNNConv(in_channels=5, out_channels=64, num_layers=1, edge_attr_channels=2)
        self.conv2 = GNNConv(in_channels=64, out_channels=128, num_layers=1, edge_attr_channels=2)
        self.pool = global_mean_pool
        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 1)

    # <LLM: optionally build extra layers here>

    def forward(self, G):
        # IMPORTANT output must be logits/probabilities per event
        # Batch G
        x, edge_index, edge_attr, batch = G.x, G.edge_index, G.edge_attr, G.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.pool(x, batch)  # [batch_size, 128]
        x = self.lin1(x)
        x = F.relu(x)
        out = self.lin2(x).squeeze(-1)  # [batch_size]
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 10   # <LLM: adjust if you wish>
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    #   - Must return: trained_model, train_loss, val_loss, train_acc, val_acc
    #   - Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).

    # <LLM: Write code to define training loop, use the code above>
    # <LLM: Implement early stopping if possible>
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    best_auc = 0
    best_model_state = None
    patience = 5
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)  # [batch_size]
            loss = criterion(out, batch.y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(out.detach().cpu().numpy())
            train_labels.extend(batch.y.cpu().numpy())
        train_loss /= len(train_loader)

        # Accuracy
        train_acc = ((torch.tensor(train_preds) > 0).float() == torch.tensor(train_labels)).float().mean().item()

        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y.float())
                val_loss += loss.item()
                val_preds.extend(out.detach().cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())
        val_loss /= len(val_loader)
        val_acc = ((torch.tensor(val_preds) > 0).float() == torch.tensor(val_labels)).float().mean().item()

        # AUC
        try:
            auc = roc_auc_score(val_labels, val_preds)
        except ValueError:
            auc = 0  # if only one class

        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

    model.load_state_dict(best_model_state)
    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

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

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane_fourtops(spec, first_batch)
    view = make_view_by_lane_fourtops(mode, first_batch, device)

    # Build model
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
                mode = detect_and_assert_lane_fourtops(spec, first_batch)
                view = make_view_by_lane_fourtops(mode, first_batch, device)
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

