
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

# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
# <LLM: Import modules>
import torch_geometric
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# def make_dataset(events, pre, train: bool, **kwargs):
#   REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:make_dataset"
#   k = kwargs.get("k", 16)
#   <LLM: Insert custom dataset logic here>
#   return CustomDataset(events, pre, train=train, k=k)

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
        self.means = None
        self.stds = None

    def make_loader_cfg(self):
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",     # or torch.utils.data:DataLoader
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # collate must be builtin string or None (torch default collate / PyG)
            "collate": None,

            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # <LLM: Extract statistics for transform>
        self.means = torch.mean(X, dim=0)
        self.stds = torch.std(X, dim=0)
        self.stds = torch.where(self.stds == 0, torch.ones_like(self.stds), self.stds)
        return self

    def transform(self, X):
        # <LLM: Apply pre-processing logic>
        if self.means is not None:
            X = (X - self.means) / self.stds  # [N,92]

        N = X.size(0)
        data_list = []
        for i in range(N):
            event = X[i]  # [92]
            global_et = event[:2]  # [2]
            objects = event[2:].reshape(18, 5)  # [18,5]

            # Create 19 nodes: 18 objects + 1 global
            node_features = torch.zeros(19, 5)  # [19,5]
            # Global node: obj_id = -1, E_T_miss, 0, 0, phi_E_t_miss
            node_features[0, 0] = -1
            node_features[0, 1] = global_et[0]  # E_T_miss
            node_features[0, 2] = 0  # p_T
            node_features[0, 3] = 0  # eta
            node_features[0, 4] = global_et[1]  # phi
            # Object nodes: [obj_id, E, p_T, eta, phi]
            node_features[1:19] = objects  # [18,5]

            # Positions: [eta, phi] for geometric edges (if needed)
            pos = torch.zeros(19, 2)  # [19,2]
            pos[0] = torch.tensor([0, global_et[1]])  # arbitrary eta=0 for global
            pos[1:19] = objects[:, 3:5]  # [18,2] eta, phi

            # Fully connected edges: all pairs including self? but no self-loops
            num_nodes = 19
            edge_list = []
            for src in range(num_nodes):
                for dst in range(num_nodes):
                    if src != dst:
                        edge_list.append([src, dst])
            edge_index = torch.tensor(edge_list, dtype=torch.long).T.contiguous()  # [2, 19*18]

            data = Data(x=node_features, edge_index=edge_index, pos=pos)  # pos for potential use
            data_list.append(data)

        return data_list  # list of Data objects, length N

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # <LLM: Define and initialize any stateful components here>
        self.conv1 = torch_geometric.nn.GCNConv(5, 128)
        self.conv2 = torch_geometric.nn.GCNConv(128, 128)
        self.conv3 = torch_geometric.nn.GCNConv(128, 128)
        self.pool = torch_geometric.nn.global_mean_pool
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    # <LLM: optionally build extra layers here>

    def forward(self, data):
        # <LLM: Define your model's forward pass here>
        x, edge_index, batch = data.x, data.edge_index, data.batch  # batch is added by collate
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index).relu()
        x = self.dropout(x)
        x = self.pool(x, batch)  # [batch_size, 128]
        x = self.fc1(x).relu()
        x = self.fc2(x)  # [batch_size, 1]
        return x.squeeze(-1)  # [batch_size]

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

    # <LLM: Write code to define training loop>
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    best_val_auc = 0.0
    patience = 5
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        tl = 0
        correct = 0
        total = 0
        all_logits = []
        all_labels = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.float())
            loss.backward()
            optimizer.step()
            tl += loss.item()
            probs = torch.sigmoid(logits)
            all_logits.extend(logits.detach().cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            preds = (probs > 0.5).float()
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)
        train_loss.append(tl / len(train_loader))
        train_acc.append(correct / total)
        train_auc = roc_auc_score(all_labels, all_logits)

        model.eval()
        vl = 0
        correct = 0
        total = 0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y.float())
                vl += loss.item()
                probs = torch.sigmoid(logits)
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                preds = (probs > 0.5).float()
                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)
        val_auc = roc_auc_score(all_labels, all_logits)
        val_loss.append(vl / len(val_loader))
        val_acc.append(correct / total)
        scheduler.step(val_auc)  # step on AUC

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    if best_model_state:
        trained_model.load_state_dict(best_model_state)
    trained_model = model

    return trained_model, train_loss, val_loss, train_acc, val_acc

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


