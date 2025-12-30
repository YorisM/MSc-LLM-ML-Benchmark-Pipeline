
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

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
        self.means = None
        self.stds = None

    def make_loader_cfg(self):
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # collate must be builtin string or None (torch default collate / PyG)
            "collate": None,

            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Extract statistics for transform
        data = X.numpy()
        eps = 1e-6
        self.means = np.mean(data, axis=0, keepdims=True)  # [1, 92]
        self.stds = np.std(data, axis=0, keepdims=True) + eps  # [1, 92]
        return self

    def transform(self, X):
        # Apply pre-processing logic: standard scaling
        data = (X.numpy() - self.means) / self.stds
        return torch.tensor(data, dtype=torch.float32)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is view.batch_x, assumed shape [batch, 92]
        global_size = 2
        obj_size = 5  # obj_id (treated as float), E, p_T, eta, phi
        max_objs = 18
        lstm_hidden = 128
        self.lstm = nn.LSTM(obj_size, lstm_hidden, batch_first=True, num_layers=2, dropout=0.3, bidirectional=True)
        # Output: forward + backward = 256
        self.global_proj = nn.Linear(global_size, lstm_hidden * 2)  # match bidirectional output
        self.dropout = nn.Dropout(0.4)
        self.attention = nn.Sequential(  # Add attention for better feature aggregation
            nn.Linear(lstm_hidden * 2, lstm_hidden * 2),
            nn.Tanh(),
            nn.Linear(lstm_hidden * 2, max_objs),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + lstm_hidden * 2, 256),  # obj_emb + global_emb
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x: [batch, 92]
        global_feat = batch_x[:, :2]  # [batch, 2]
        obj_feat = batch_x[:, 2:].view(batch_x.size(0), 18, 5)  # [batch, 18, 5]
        # LSTM on obj sequence
        lstm_out, _ = self.lstm(obj_feat)  # lstm_out: [batch, 18, hidden*2=256]
        weights = self.attention(lstm_out).unsqueeze(-1)  # [batch, 18, 1]
        obj_emb = (lstm_out * weights).sum(dim=1)  # [batch, 256] weighted sum
        global_emb = self.global_proj(global_feat)  # [batch, 256]
        combined = torch.cat([obj_emb, global_emb], dim=1)  # [batch, 512]
        combined = self.dropout(combined)
        out = self.fc(combined)  # [batch, 1]
        return out.squeeze()  # [batch]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30   # Adjusted for more epochs
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS 
    #   Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #   Must return trained_model, train_loss, val_loss, train_acc, val_acc
    #   Use CUDA - torch.cuda.is_available()
    #   Implement early-stopping.
    #   Forward signature must match.

    optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.7)
    criterion = nn.BCEWithLogitsLoss()
    # Early stopping
    best_val_loss = float('inf')
    patience = 7
    wait = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            epoch_train_correct += (preds == (batch_y > 0.5)).sum().item()
            epoch_train_total += batch_y.size(0)
        scheduler.step()
        epoch_train_loss /= len(train_loader)
        epoch_train_acc = epoch_train_correct / epoch_train_total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                epoch_val_correct += (preds == (batch_y > 0.5)).sum().item()
                epoch_val_total += batch_y.size(0)
        epoch_val_loss /= len(val_loader)
        epoch_val_acc = epoch_val_correct / epoch_val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            wait = 0
            best_model_state = model.state_dict()
        else:
            wait += 1
        if wait >= patience:
            break
    if best_model_state is not None:
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


