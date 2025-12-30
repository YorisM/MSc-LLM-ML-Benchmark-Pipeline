
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
# <LLM: Import modules>
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        self.means = None
        self.stds = None

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     # or torch_geometric.loader:DataLoader
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
        # X is torch.Tensor [N, 92]
        # Skip object ids (indices 2,7,...,87) since they are categorical and sparse, keep only kinematic 4-vec per object and global
        # For normalization: features 0-1 (global), then 3-6 (skip 2), 8-11 (skip7), etc.
        # But to simplify, normalize all numeric features: E_T_miss, phi_E_t_miss, and for each object: E, p_T, eta, phi (skip obj_id)
        # So skip every 5th starting from 2: indices 2,7,12,...,87
        mask = torch.ones(92, dtype=bool)
        for i in range(2, 92, 5):
            mask[i] = False
        self.means = X[:, mask].mean(dim=0)  # [90] since 92-2 skipped =90? Wait 92 total, 2 global +36 per obj (4*18=72, total 74? Wait no:
        # Wait: feats: 0: ET, 1:phi_ET, then repeat obj_id (ignore), E, pT, eta, phi for 18 objs: 4*18=72, total 2+72=74, but 92? Wait mistake.
        # Wait, object 1: indices 2:obj1, 3:E1,4:pT1,5:eta1,6:phi1 — 5 items
        # No: "obj_n, E_n, p_Tn, eta_n, phi_n" — 5 per object, 18 objs *5 =90 +2=92 yes.
        # So per event: 2 global + 18*5 =92
        # To normalize: skip obj_n which is 1 every 5 starting at 2.
        # mask: skip indices where (idx-2)%5 ==0 and idx>=2
        # Yes, as above, 2,7,12,...,87 — 18 indices, so mask.sum()=92-18=74 features to normalize.
        # But computing means for selected.
        X_selected = X[:, mask]  # [N, 74]
        self.means = X_selected.mean(dim=0)
        self.stds = X_selected.std(dim=0)
        return self

    def transform(self, X):
        # <LLM: Apply pre-processing logic>
        # X [N,92]
        mask = torch.ones(92, dtype=bool)
        for i in range(2, 92, 5):
            mask[i] = False
        X_selected = X[:, mask]  # [N,74]
        X_norm = (X_selected - self.means) / (self.stds + 1e-8)
        # Now, put back: create full [N,92], but replace non-skipped with norm, and skipped remain 0 (obj_id)
        X_out = X.clone()  # [N,92]
        X_out[:, mask] = X_norm  # [N,74] into the masked positions
        return X_out  # [N,92], normalized kinematics, obj_id untouched

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
        # sample_object is the batch_x from first batch, which is X [batch,92]
        input_dim = 92  # flat features
        # Use a simple MLP for classification
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    # <LLM: optionally build extra layers here>

    def forward(self, batch_x):
        # IMPORTANT output must be logits/probabilities per event
        # batch_x [batch_size, 92]
        return self.net(batch_x).squeeze(-1)  # [batch_size] logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30   # <LLM: adjust if you wish>
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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0
        all_train_preds = []
        all_train_labels = []
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            optimizer.zero_grad()
            logits = model(xb)
            probs = torch.sigmoid(logits)
            loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            preds = (probs > 0.5).long()
            correct_train += (preds == yb).sum().item()
            total_train += yb.numel()

            all_train_preds.extend(probs.detach().cpu().numpy())
            all_train_labels.extend(yb.cpu().numpy())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / total_train if total_train > 0 else 0
        train_auc = roc_auc_score(all_train_labels, all_train_preds) if len(set(all_train_labels)) > 1 else 0

        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                logits = model(xb)
                probs = torch.sigmoid(logits)
                loss = criterion(logits, yb.float())

                epoch_val_loss += loss.item()
                preds = (probs > 0.5).long()
                correct_val += (preds == yb).sum().item()
                total_val += yb.numel()

                all_val_preds.extend(probs.cpu().numpy())
                all_val_labels.extend(yb.cpu().numpy())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_acc = correct_val / total_val if total_val > 0 else 0
        val_auc = roc_auc_score(all_val_labels, all_val_preds) if len(set(all_val_labels)) > 1 else 0

        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Train ACC={train_acc:.4f}, Val ACC={val_acc:.4f}, Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}")

        # Early stopping based on val_loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # <LLM: Implement early stopping if possible>
    return model, train_losses, val_losses, train_accs, val_accs

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

