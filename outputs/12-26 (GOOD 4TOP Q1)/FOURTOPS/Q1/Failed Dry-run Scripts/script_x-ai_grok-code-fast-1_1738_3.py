
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
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader as PyGDataLoader
# <LLM: Import modules>

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

    def __init__(self):
        self.means = None
        self.stds = None
        self.global_means = None
        self.global_stds = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",     # use PyG DataLoader for graphs
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        X_np = X.numpy()
        all_x = []  # Collect all non-zero object features across all events
        for i in range(X_np.shape[0]):
            x = X_np[i]
            for j in range(18):
                start = 2 + j * 5
                obj_id = x[start]
                E = x[start + 1]
                pT = x[start + 2]
                eta = x[start + 3]
                phi = x[start + 4]
                if E == 0 and pT == 0:  # Skip zero-padded
                    continue
                all_x.append([obj_id, E, pT, eta, phi])
        all_x = np.array(all_x)
        if len(all_x) > 0:
            self.means = np.mean(all_x, axis=0)
            self.stds = np.std(all_x, axis=0) + 1e-8
        else:
            self.means = np.zeros(5)
            self.stds = np.ones(5)
        # Globals
        globals_ = X_np[:, :2]
        self.global_means = np.mean(globals_, axis=0)
        self.global_stds = np.std(globals_, axis=0) + 1e-8
        return self

    def transform(self, X):
        data_list = []
        X_np = X.numpy()
        for i in range(X_np.shape[0]):
            x = X_np[i]
            E_T_miss = x[0]
            phi_miss = x[1]
            nodes = []
            for j in range(18):
                start = 2 + j * 5
                obj_id = x[start]
                E = x[start + 1]
                pT = x[start + 2]
                eta = x[start + 3]
                phi = x[start + 4]
                if E == 0 and pT == 0:  # Zero-padded
                    continue
                nodes.append([obj_id, E, pT, eta, phi])
            if len(nodes) == 0:
                nodes = [[0, 0, 0, 0, 0]]  # Dummy node
            node_x = torch.tensor(nodes, dtype=torch.float32)  # [len(nodes), 5]
            if self.means is not None:
                node_x = (node_x - self.means.unsqueeze(0)) / self.stds.unsqueeze(0)  # Normalize
            n = node_x.shape[0]
            # Fully connected graph
            edge_list = []
            for a in range(n):
                for b in range(n):
                    if a != b:
                        edge_list.extend([[a, b], [b, a]])
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()  # [2, num_edges]
            else:
                edge_index = torch.tensor([], dtype=torch.long).view(2, 0)
            u = torch.tensor([E_T_miss, phi_miss], dtype=torch.float32)  # [2]
            if self.global_means is not None:
                u = (u - self.global_means) / self.global_stds
            data = Data(x=node_x, edge_index=edge_index, u=u)
            data_list.append(data)
        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is a PyG Batch, sample_object.x.shape[1] == 5 (normalized features)
        input_dim = sample_object.x.shape[1]
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 256)
        self.pool = global_mean_pool
        self.fc1 = nn.Linear(256 + 2, 128)  # 256 from pool + 2 globals
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, batch):
        x = batch.x  # Node features [total_nodes, 5]
        edge_index = batch.edge_index  # Edges
        u = batch.u  # Globals [batch_size, 2]
        batch_index = batch.batch  # For pooling [total_nodes,]
        x = self.conv1(x, edge_index)  # [total_nodes, 128]
        x = nn.ReLU()(x)
        x = self.conv2(x, edge_index)  # [total_nodes, 256]
        x = nn.ReLU()(x)
        pooled = self.pool(x, batch_index)  # [batch_size, 256]
        combined = torch.cat([pooled, u], dim=1)  # [batch_size, 256+2=258]
        x = self.fc1(combined)  # [batch_size, 128]
        x = nn.ReLU()(x)
        x = self.fc2(x)  # [batch_size, 64]
        x = nn.ReLU()(x)
        out = self.fc3(x).squeeze(-1)  # [batch_size,]
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 15  # Increased for better convergence
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # Added weight_decay for regularization
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Decay LR
    best_val_loss = float('inf')
    patience = 5
    no_improve = 0
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb = view.batch_x
            yb = view.batch_y.float()
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.num_graphs
            pred = (torch.sigmoid(out) > 0.5).float()  # For accuracy
            acc = (pred == yb).float().mean().item()
            total_acc += acc * xb.num_graphs
            count += xb.num_graphs
        train_loss = total_loss / count
        train_acc = total_acc / count
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb = view.batch_x
                yb = view.batch_y.float()
                out = model(xb)
                loss = criterion(out, yb)
                total_loss += loss.item() * xb.num_graphs
                pred = (torch.sigmoid(out) > 0.5).float()
                acc = (pred == yb).float().mean().item()
                total_acc += acc * xb.num_graphs
                count += xb.num_graphs
        val_loss = total_loss / count
        val_acc = total_acc / count
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        scheduler.step()
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

# DO NOT execute the pipeline here – the harness will do that.
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

