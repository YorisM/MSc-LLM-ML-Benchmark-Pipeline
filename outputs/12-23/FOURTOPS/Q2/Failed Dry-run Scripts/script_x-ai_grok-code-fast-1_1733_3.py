
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
# <LLM: Import modules>
import torch_geometric
from torch_geometric.data import Data
import math

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
        batch_data = []
        for event in X:
            # event: torch.Tensor of shape [92]
            E_miss = event[0]  # scalar
            phi_miss = event[1]  # scalar
            nodes = []
            for i in range(18):
                start = 2 + i * 5
                obj_id_int = int(event[start])  # int, could be 0 for padded
                E = event[start + 1]
                pT = event[start + 2]
                eta = event[start + 3]
                phi = event[start + 4]
                if E <= 0.0:  # assume padded if E <= 0
                    continue
                # Compute 4-momentum components
                px = pT * torch.cos(phi)
                py = pT * torch.sin(phi)
                pz = pT * torch.sinh(eta)
                # Node features: [E, px, py, pz, eta, phi, E_miss, phi_miss] -> shape [8]
                node_feat = torch.tensor([E, px, py, pz, eta, phi, E_miss, phi_miss], dtype=torch.float32)
                nodes.append(node_feat)
            if not nodes:
                # Empty event? Add a dummy node with zero features
                dummy = torch.zeros(8, dtype=torch.float32)
                nodes.append(dummy)
            nodes = torch.stack(nodes)  # shape [n_objs, 8]
            # Compute edges: fully connected, excluding self-loops
            n = nodes.shape[0]
            edge_list = []
            edge_attr_list = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # 4-momenta
                        vi = nodes[i, :4]  # [E, px, py, pz] shape [4]
                        vj = nodes[j, :4]  # [E, px, py, pz] shape [4]
                        total_E = vi[0] + vj[0]
                        total_px = vi[1] + vj[1]
                        total_py = vi[2] + vj[2]
                        total_pz = vi[3] + vj[3]
                        m_ij_sq = total_E**2 - (total_px**2 + total_py**2 + total_pz**2)
                        m_ij = torch.sqrt(torch.clamp(m_ij_sq, min=0.0))
                        delta_eta = nodes[i, 4] - nodes[j, 4]
                        delta_phi_raw = nodes[i, 5] - nodes[j, 5]
                        # Correct delta_phi to [-pi, pi]
                        delta_phi = torch.atan2(torch.sin(delta_phi_raw), torch.cos(delta_phi_raw))
                        delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)
                        edge_list.append([i, j])
                        edge_attr_list.append(torch.tensor([m_ij, delta_R], dtype=torch.float32))
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()  # shape [2, num_edges]
                edge_attr = torch.stack(edge_attr_list)  # shape [num_edges, 2]
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)
                edge_attr = torch.empty(0, 2, dtype=torch.float32)
            # Create Data object
            data = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr)
            batch_data.append(data)
        # batch_data is a list of Data objects, length = len(X)
        return batch_data

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # <LLM: Define and initialize any stateful components here>
        # sample_object: a Data object with x, etc.
        dim_input = sample_object.x.size(1)  # 8
        edge_dim_input = sample_object.edge_attr.size(1) if sample_object.edge_attr.numel() > 0 else 2  # 2
        self.conv1 = torch_geometric.nn.TransformerConv(dim_input, 128, edge_dim=edge_dim_input, heads=8, dropout=0.1)
        self.conv2 = torch_geometric.nn.TransformerConv(128, 128, edge_dim=edge_dim_input, heads=8, dropout=0.1)
        self.pool = torch_geometric.nn.global_mean_pool
        self.fc = nn.Linear(128, 1)

    # <LLM: optionally build extra layers here>

    def forward(self, batch_data):
        # <LLM: Define your model's forward pass here>
        # batch_data is a Batch from PyG DataLoader: has x, edge_index, edge_attr, batch, etc.
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.pool(x, batch_data.batch)  # batch_data.batch is the batch assignment
        x = self.fc(x)
        return x.squeeze(-1)  # for binary classification logits, shape [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 10   # <LLM: adjust if you wish>
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS 
    #   Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #   Must return trained_model, train_loss, val_loss, train_acc, val_acc
    #   Use CUDA - torch.cuda.is_available()
    #   Implement early-stopping.
    #   Forward signature must match.

    # <LLM: Write code to define training loop>
    # <LLM: Implement early stopping if possible>
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience = 5
    counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            y = batch.y.float()
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * y.size(0)
            preds = (torch.sigmoid(out) > 0.5).long()
            total_train_correct += (preds == batch.y).sum().item()
            total_train_samples += y.size(0)
        scheduler.step()
        train_loss_epoch = total_train_loss / total_train_samples
        train_acc_epoch = total_train_correct / total_train_samples
        train_losses.append(train_loss_epoch)
        train_accs.append(train_acc_epoch)

        model.eval()
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                y = batch.y.float()
                loss = loss_fn(out, y)
                total_val_loss += loss.item() * y.size(0)
                preds = (torch.sigmoid(out) > 0.5).long()
                total_val_correct += (preds == batch.y).sum().item()
                total_val_samples += y.size(0)
        val_loss_epoch = total_val_loss / total_val_samples
        val_acc_epoch = total_val_correct / total_val_samples
        val_losses.append(val_loss_epoch)
        val_accs.append(val_acc_epoch)

        # Early stopping
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    return model, train_losses, val_losses, train_accs, val_accs

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
# <end code template>

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


