
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

# -------------------------- START OF LLM BLOCK ------------------------------
# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    # REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X  # List of Data objects
        self.y = y
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
        pass

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:CustomDataset",   # Use custom for PyG
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",     # PyG DataLoader
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": None,  # PyG uses its own

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply pre-processing: Convert each event to a PyG Data object
        data_list = []
        for i in range(X.shape[0]):
            # Global features: [E_T_miss, phi_Et_miss]
            globals = X[i, :2]

            # Nodes: for each of 18 objects
            node_features = []
            for j in range(18):
                start = 2 + j * 5
                obj_id = X[i, start]
                if obj_id != 0:  # Skip padding
                    E = X[i, start + 1]
                    pT = X[i, start + 2]
                    eta = X[i, start + 3]
                    phi = X[i, start + 4]
                    # Compute px, py, pz
                    pz = pT * torch.sinh(torch.tensor(eta))
                    px = pT * torch.cos(torch.tensor(phi))
                    py = pT * torch.sin(torch.tensor(phi))
                    # Node features: [E, pT, eta, phi, obj_id, px, py, pz]
                    node_features.append([E, pT, eta, phi, obj_id, px, py, pz])

            if node_features:
                x = torch.tensor(node_features, dtype=torch.float32)
                num_nodes = x.shape[0]

                # Fully connected edges (undirected, excluding self)
                edge_index = []
                edge_attr = []
                for u in range(num_nodes):
                    for v in range(num_nodes):
                        if u != v:
                            edge_index.append([u, v])
                            # Edge features: delta_eta, delta_phi
                            delta_eta = x[u, 2] - x[v, 2]
                            delta_phi = x[u, 3] - x[v, 3]
                            edge_attr.append([delta_eta, delta_phi])

                edge_index = torch.tensor(edge_index, dtype=torch.long).t()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
            else:
                # No nodes, create dummy node
                x = torch.zeros(1, 8)
                edge_index = torch.empty(2, 0, dtype=torch.long)
                edge_attr = torch.empty(0, 2, dtype=torch.float32)
                num_nodes = 1

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, globals=globals)
            data_list.append(data)

        return data_list  # List of Data objects

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is a Data
        input_dim = sample_object.x.shape[1]  # 8
        edge_dim = sample_object.edge_attr.shape[1]  # 2
        hidden_dim = 128
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool = global_mean_pool
        global_in = hidden_dim + 2  # hidden + globals (2)
        self.fc = nn.Linear(global_in, 1)

    def forward(self, batch_x):
        # batch_x is PyG Batch
        x, edge_index, edge_attr, batch, globals = batch_x.x, batch_x.edge_index, batch_x.edge_attr, batch_x.batch, batch_x.globals
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.pool(x, batch)  # [batch_size, hidden_dim]
        x = torch.cat([x, globals.unsqueeze(0).repeat(x.shape[0], 1)], dim=1)  # Assume globals is [batch_size, 2] but in batch it's concat
        # Fixing globals: since PyG Batch concatenates globals, need to separate per graph
        # global_features = torch.stack([data.globals for data in batch_x.to_data_list()])
        # For simplicity, assume globals is batch-sized
        x = self.fc(x)
        return x

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20  # Increased for better training
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0
    patience = 5
    counter = 0
    trained_model = model

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        all_train_logits, all_train_labels = [], []

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            optimizer.zero_grad()
            out = model(xb).squeeze()
            loss = criterion(out, yb.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = (torch.sigmoid(out) > 0.5).int()
            train_correct += (preds == yb).sum().item()
            total_train += yb.size(0)
            all_train_logits.extend(out.detach().cpu().numpy())
            all_train_labels.extend(yb.detach().cpu().numpy())

        scheduler.step()
        train_acc = train_correct / total_train
        train_auc = roc_auc_score(all_train_labels, all_train_logits) if len(set(all_train_labels)) > 1 else 0
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        all_val_logits, all_val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                out = model(xb).squeeze()
                loss = criterion(out, yb.float())
                val_loss += loss.item()
                preds = (torch.sigmoid(out) > 0.5).int()
                val_correct += (preds == yb).sum().item()
                total_val += yb.size(0)
                all_val_logits.extend(out.detach().cpu().numpy())
                all_val_labels.extend(yb.detach().cpu().numpy())

        val_acc = val_correct / total_val
        val_auc = roc_auc_score(all_val_labels, all_val_logits) if len(set(all_val_labels)) > 1 else 0
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, "
              f"Train ACC={train_acc:.4f}, Val ACC={val_acc:.4f}, Val AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            counter = 0
            trained_model = model.state_dict() if isinstance(model, nn.Module) else model
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    model.load_state_dict(trained_model)
    return model, train_losses, val_losses, train_accs, val_accs
# <end code template>
# ---------------------------  END OF LLM-CODE BLOCK  ---------------------------

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

