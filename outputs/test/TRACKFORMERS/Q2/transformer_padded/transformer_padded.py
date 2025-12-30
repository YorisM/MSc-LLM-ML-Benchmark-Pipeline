# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.llm_io import normalise_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data/train"
TAG      = "REDVID_10-50_linear_frac0.05"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def _split_X_y(evt):
    X = np.column_stack((evt["hit_r"].astype(np.float32),
                        evt["hit_theta"].astype(np.float32),
                        evt["hit_z"].astype(np.float32),
                        evt["layer_id"].astype(np.float32)))
    y = evt["track_id"].astype(np.int32)
    return (torch.from_numpy(X),torch.from_numpy(y))

def _make_dataset(events, pre, *, train: bool):
    custom = globals().get("make_dataset", None)
    if callable(custom):
        ds = custom(events, pre, train=train)
        if ds is not None:
            return ds
    return EventDataset(events, pre, train=train)

def make_loaders(raw_train, raw_val, pre, *, batch=512,
                 collate_fn=None, loader_cls=None, workers=0):
    train_ds  = _make_dataset(raw_train, pre, train=True)
    val_ds    = _make_dataset(raw_val,  pre, train=False)

    if loader_cls is None:
        loader_cls = DataLoader

    pin = (device.type == "cuda")
    train_ld = loader_cls(train_ds, batch_size=batch, shuffle=True,
                        num_workers=workers, collate_fn=collate_fn,
                        pin_memory=pin, persistent_workers=(workers > 0))
    val_ld   = loader_cls(val_ds,   batch_size=batch, shuffle=False,
                        num_workers=workers, collate_fn=collate_fn,
                        pin_memory=pin, persistent_workers=(workers > 0))
    return train_ld, val_ld
    
class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, track_id = _split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, track_id)

def _ragged(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    # batch[i] = (hits_i, track_id_i)      <- shapes: (N_i, F), (N_i)
    return batch

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# 0. ---------- IMPORTS ----------
import torch
from torch import nn

# 1.2 ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self, max_hits: int = 256):
        self.max_hits = int(max_hits)

    def fit(self, data):
        return self

    def transform(self, data):
        return data

    @staticmethod
    def _collate_fn(batch: list):
        xs = [xy[0] for xy in batch]   # each [Ni, F]
        ys = [xy[1] for xy in batch]   # each [Ni]

        B = len(xs)
        if B == 0:
            Xpad = torch.empty(0, 0, 0)
            mask = torch.empty(0, 0, dtype=torch.bool)
            return ((Xpad, mask), ys)

        MAX_HITS = 256
        lens = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
        N = min(int(lens.max().item()), MAX_HITS)
        F = int(xs[0].size(1))

        Xpad = xs[0].new_zeros((B, N, F))            # [B,N,F]
        mask = torch.zeros((B, N), dtype=torch.bool) # [B,N] bool

        for i, x in enumerate(xs):
            n = min(int(x.size(0)), N)
            Xpad[i, :n] = x[:n]
            mask[i, :n] = True

        out = ((Xpad, mask), ys)

        # Hard guardrails so this never silently regresses again
        assert isinstance(out, tuple) and len(out) == 2
        assert isinstance(out[0], tuple) and len(out[0]) == 2
        assert torch.is_tensor(out[0][0]) and out[0][0].dim() == 3
        assert torch.is_tensor(out[0][1]) and out[0][1].dtype == torch.bool and out[0][1].dim() == 2
        assert out[0][0].shape[:2] == out[0][1].shape

        return out


    def make_loader_cfg(self):
        return {
            "loader_class": "torch.utils.data.DataLoader",
            "batch_size": 32,   # transformer is heavier
            "num_workers": 0,
            "pin_memory": True,
        }

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        Xpad, _mask = example_batch_x
        Fin = int(Xpad.shape[-1])
        D = 32

        self.in_proj = nn.Linear(Fin, D)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.head = nn.Linear(D, 1)

    def forward(self, batch_x):
        Xpad, mask = batch_x
        mask = mask.to(dtype=torch.bool, device=Xpad.device)

        x = self.in_proj(Xpad)              # [B,N,D]
        key_padding = ~mask                 # bool [B,N], True where PAD
        x = self.encoder(x, src_key_padding_mask=key_padding)  # [B,N,D]

        logits = self.head(x).squeeze(-1)   # [B,N]
        return logits.masked_select(mask)   # [sumNi]

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 1

def train_model(model, train_loader, val_loader, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr_loss, va_loss, tr_acc, va_acc = [], [], [], []

    for _ in range(int(epochs)):
        model.train()
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            out = model(view.batch_x)
            loss = (out**2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_loss.append(float(loss.detach().cpu()))
            tr_acc.append(0.0)
            break  # smoke-test: only 1 batch

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                out = model(view.batch_x)
                loss = (out**2).mean()

                va_loss.append(float(loss.detach().cpu()))
                va_acc.append(0.0)
                break  # smoke-test: only 1 batch

    return model, tr_loss, va_loss, tr_acc, va_acc


# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

def _import_dotted(path: str):
    mod, name = path.rsplit(".", 1)
    module = importlib.import_module(mod)
    return getattr(module, name)

def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    pre = make_preprocessor().fit(raw_train)

    collate = getattr(pre, "_collate_fn", None)
    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None

    train_loader, val_loader = make_loaders(raw_train, raw_val, pre,
                                            batch = cfg.get("batch_size", 128),
                                            collate_fn = collate or _ragged,
                                            loader_cls = loader_cls,
                                            workers    = cfg.get("num_workers", 0))
    
    print("[DEBUG] train_loader.collate_fn:", getattr(train_loader, "collate_fn", None))

    # 2. Build model
    first_batch = next(iter(train_loader))
    view        = normalise_batch(first_batch, device=device)
    model       = make_model(view.batch_x)
    model       = model.to(device)

    print("[DEBUG] collate from preproc:", getattr(pre, "_collate_fn", None))
    print("[DEBUG] loader_cls:", loader_cls)
    print("[DEBUG] first_batch type:", type(first_batch))
    print("[DEBUG] normalise meta:", view.meta)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single reduced forward pass
    if dryrun:
        try:
            batch = first_batch
            view  = normalise_batch(batch, device=device)
            with torch.no_grad():
                _ = trained_model(view.batch_x)
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