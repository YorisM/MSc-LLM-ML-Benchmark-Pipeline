
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: Python 3.12, PyTorch 2.6.0, Torch_Geometric 2.6.1, NumPy 2.2.3, SciPy v1.15.2, SciKit-Learn 1.6.1
import os, sys, pickle, torch, torch_geometric, gc, json, importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
                        
DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv(DATASET["X_train"], dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv(DATASET["Y_train"], dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv(DATASET["X_val"], dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv(DATASET['Y_val'], dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class PairDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
    
        if isinstance(self.x, (tuple, list)) and all(torch.is_tensor(t) for t in self.x):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])

def _make_dataset(x, y):
    custom = globals().get("make_dataset", None)
    if callable(custom):
        ds = custom(x, y)
        if ds is not None:
            return ds
    return PairDataset(x, y)

def make_loaders(X_train, Y_train, X_val, Y_val, *, batch=512, collate_fn=None, loader_cls=None):
    train_ds = _make_dataset(X_train, Y_train)
    val_ds   = _make_dataset(X_val , Y_val)

    if loader_cls is None: 
        loader_cls = DataLoader

    train_ld = loader_cls(train_ds, batch_size=batch, shuffle=True, num_workers=0, 
                        collate_fn=collate_fn)
    val_ld   = loader_cls(val_ds, batch_size=batch, shuffle=False, num_workers=0,
                        collate_fn=collate_fn)

    return train_ld, val_ld

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

# 0. ---------- IMPORTS ----------
from sklearn.metrics import roc_auc_score
import math
# (torch, nn, numpy, DataLoader already imported in wrapper)

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    """
    1. Standard-scales every raw feature (length 92).
    2. Adds 6 physics-motivated aggregated features:
       [n_objects, ΣE, ΣpT, max(pT), mean(pT), MET-ΣpT] → length 6
    Output feature length: 98
    """
    def __init__(self):
        # Will be filled in fit()
        self.mean_raw  = None   # shape [92]
        self.std_raw   = None
        self.mean_agg  = None   # shape [6]
        self.std_agg   = None

    # ---------- internal helpers ----------
    @staticmethod
    def _compute_agg(X):                        # X: (N, 92)
        """Return aggregated physics features tensor (N, 6)"""
        # Slicing indices for all objects
        obj_ids = X[:, 2::5]                   # shape (N, 18)
        energies = X[:, 3::5]                  # shape (N, 18)
        pTs      = X[:, 4::5]                  # shape (N, 18)

        n_objects = torch.count_nonzero(obj_ids, dim=1).float()        # (N,)
        sum_E     = energies.sum(dim=1)                                # (N,)
        sum_pT    = pTs.sum(dim=1)                                     # (N,)
        max_pT, _ = pTs.max(dim=1)                                     # (N,)
        mean_pT   = sum_pT / torch.clamp(n_objects, min=1.0)           # (N,)
        met       = X[:, 0]                                            # (N,)
        diff_met_sumPt = met - sum_pT                                  # (N,)

        agg = torch.stack([n_objects, sum_E, sum_pT,
                           max_pT, mean_pT, diff_met_sumPt], dim=1)    # (N, 6)
        return agg

    # ---------- Stats ----------
    def fit(self, X, y=None):
        # Expect torch.Tensor
        X = X.float()
        self.mean_raw = X.mean(dim=0)                                   # (92,)
        self.std_raw  = X.std (dim=0).clamp(min=1e-6)                   # (92,)

        agg = self._compute_agg(X)                                      # (N,6)
        self.mean_agg = agg.mean(dim=0)                                 # (6,)
        self.std_agg  = agg.std (dim=0).clamp(min=1e-6)                 # (6,)
        return self

    # ---------- Transformation ----------
    def transform(self, X):
        X = X.float()
        X_std  = (X - self.mean_raw) / self.std_raw                     # (N,92)

        agg    = self._compute_agg(X)                                   # (N,6)
        agg_std = (agg - self.mean_agg) / self.std_agg                  # (N,6)

        out = torch.cat([X_std, agg_std], dim=1)                        # (N,98)
        return out

    # Optional (not needed but shows how to change batch size)
    def make_loader_cfg(self):
        return {"batch_size": 1024}

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        """
        sample_object: first batch returned by loader – used to infer input dim
        """
        super().__init__()
        in_features = sample_object.shape[-1]          # 98
        hidden1, hidden2, hidden3 = 256, 128, 64

        self.net = nn.Sequential(                      # input: (B,in_features)
            nn.Linear(in_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden3, 1)                      # output: (B,1)
        )

        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):                              # x: (B,98)
        logits = self.net(x)                           # (B,1)
        return logits.squeeze(-1)                      # (B,)

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.5, patience=2)

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist,  val_acc_hist  = [], []

    best_auc = -math.inf
    patience, patience_cntr = 5, 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)                             # (B,)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            preds = (torch.sigmoid(logits) > 0.5)
            correct += (preds == yb.bool()).sum().item()
            total   += yb.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        # ---- Validation ----
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        val_probs, val_targets = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                logits = model(xb)
                loss = criterion(logits, yb)

                val_running_loss += loss.item() * yb.size(0)
                probs = torch.sigmoid(logits)

                val_probs.append(probs.detach().cpu())
                val_targets.append(yb.detach().cpu())

                preds = (probs > 0.5)
                val_correct += (preds == yb.bool()).sum().item()
                val_total   += yb.size(0)

        val_loss = val_running_loss / val_total
        val_acc  = val_correct / val_total
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        val_probs   = torch.cat(val_probs).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_auc = roc_auc_score(val_targets, val_probs)

        # Scheduler & early stopping on AUC
        scheduler.step(val_auc)
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cntr = 0
        else:
            patience_cntr += 1
            if patience_cntr >= patience:
                break  # early stop

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

def _import_dotted(path: str):
    mod, name = path.rsplit(".", 1)
    module = importlib.import_module(mod)
    return getattr(module, name)

def _plot(series_train, series_val, name, out_path):
    plt.figure()
    epochs = range(1, len(series_train) + 1)
    plt.plot(epochs, series_train, label=f"Train {name}")
    plt.plot(epochs, series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        X_train, Y_train, X_val, Y_val = X_train[:200], Y_train[:200], X_val[:20], Y_val[:20]
    pre     = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train)
    X_val   = pre.transform(X_val)

    collate = getattr(pre, "_collate_fn", None)
    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val, 
                                            batch      = cfg.get("batch_size", 512), 
                                            collate_fn = collate,
                                            loader_cls = loader_cls)

    # 2. Build model
    first_batch    = next(iter(train_loader))
    example_sample = first_batch[0]
    model          = make_model(example_sample)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. Dry-run safety check
    if dryrun:
        sample, _ = first_batch
        try:
            _ = trained_model(*sample) if isinstance(sample, (tuple, list)) else trained_model(sample)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

        pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
        pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
        pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

        torch.save(trained_model.state_dict(), pth_state)
        with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
        with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

        # 6. Save plots
        _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
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

