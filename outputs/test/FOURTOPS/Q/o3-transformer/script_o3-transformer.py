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

def load_data():
    X_train = pd.read_csv("./challenges/FOURTOPS/data/X_train.csv", dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv("./challenges/FOURTOPS/data/Y_train.csv", dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv("./challenges/FOURTOPS/data/X_val.csv", dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv("./challenges/FOURTOPS/data/Y_val.csv", dtype=np.int64).to_numpy(copy=False).ravel()

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
import torch
import torch.nn.functional as F
from torch import nn

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    """
    Reshape flat (N, 92) → sequence (N, L=19, F=5):
      • token 0  = two global MET features padded with zeros
      • tokens 1-18 = per-object 5-feature slices
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):                 # nothing to learn
        return self

    def transform(self, X: torch.Tensor):
        B = X.size(0)
        # split global & objects ------------------------------------------------
        globals_ = X[:, :2]                           # (B,2)
        objs     = X[:, 2:].view(B, 18, 5)            # (B,18,5)
        # pad global token to 5 dims
        g_pad = torch.zeros(B, 1, 5, dtype=X.dtype, device=X.device)
        g_pad[:, :, :2] = globals_.unsqueeze(1)
        seq = torch.cat([g_pad, objs], dim=1)         # (B,19,5)
        return seq                                    # single tensor

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_tensor: torch.Tensor):
        super().__init__()
        L, F = sample_tensor.shape[-2], sample_tensor.shape[-1]   # 19,5
        d_model = 64
        self.input_proj  = nn.Linear(F, d_model)
        encoder_layer    = nn.TransformerEncoderLayer(
                               d_model=d_model, nhead=4, dim_feedforward=128,
                               batch_first=True)
        self.encoder     = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.cls_head    = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.cls_token_id = 0        # first token is “global”

    def forward(self, x):            # x: (B,19,5)
        x = self.input_proj(x)       # (B,19,64)
        h = self.encoder(x)          # (B,19,64)
        cls = h[:, self.cls_token_id]# (B,64)
        return self.cls_head(cls).squeeze(-1)  # (B,)

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 5
def train_model(model, train_loader, val_loader, epochs):
    opt  = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    crit = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def _run(loader, train):
        if train: model.train()
        else:     model.eval()
        tot_loss = tot_correct = tot = 0
        with torch.enable_grad() if train else torch.no_grad():
            for x, y in loader:
                x = x.to(device, dtype=torch.float32)
                x, y = x.to(device), y.float().to(device)
                logits = model(x)
                loss = crit(logits, y)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
                tot_loss += loss.item() * y.size(0)
                tot_correct += (logits.sigmoid() > 0.5).eq(y.bool()).sum().item()
                tot += y.size(0)
        return tot_loss / tot, tot_correct / tot

    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    best = float("inf"); patience = 3
    for epoch in range(epochs):
        tl, ta = _run(train_loader, True)
        vl, va = _run(val_loader, False)
        train_loss.append(tl); val_loss.append(vl)
        train_acc.append(ta);  val_acc.append(va)
        if vl < best: best, wait = vl, patience
        else:
            wait -= 1
            if wait == 0: break
    return model, train_loss, val_loss, train_acc, val_acc

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