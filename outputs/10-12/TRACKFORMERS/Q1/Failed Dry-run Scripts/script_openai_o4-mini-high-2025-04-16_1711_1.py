
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data"
TAG      = "10_50_linear_frac0.05"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"REDVID_{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def _split_X_y(evt):
    X = np.column_stack((evt["hit_r"],
                        evt["hit_theta"],
                        evt["hit_z"],
                        evt["layer_id"]))
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

import numpy as np
import torch
from torch import nn

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        px = self.parent[x]
        if px != x:
            self.parent[x] = self.find(px)
        return self.parent[x]
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        else:
            self.parent[yroot] = xroot
            if self.rank[xroot] == self.rank[yroot]:
                self.rank[xroot] += 1

class MyPreprocessor:
    def __init__(self):
        self.theta_th = None
        self.z_th = None

    def _raw_reshape(self, data):
        return data

    def make_loader_cfg(self):
        return None

    def fit(self, data):
        dtheta_list = []
        dz_list = []
        for evt in data:
            theta = np.array(evt["hit_theta"], dtype=float)       # [N_hits]
            z = np.array(evt["hit_z"], dtype=float)               # [N_hits]
            layer_id = np.array(evt["layer_id"], dtype=int)       # [N_hits]
            track_id = np.array(evt["track_id"], dtype=int)       # [N_hits]
            for t in np.unique(track_id):
                mask = (track_id == t)
                layers = layer_id[mask]
                if layers.size <= 1:
                    continue
                theta_t = theta[mask]
                z_t = z[mask]
                order = np.argsort(layers)
                layers_s = layers[order]
                theta_s = theta_t[order]
                z_s = z_t[order]
                for k in range(len(layers_s) - 1):
                    if layers_s[k+1] - layers_s[k] == 1:
                        d_theta = abs(theta_s[k+1] - theta_s[k])
                        if d_theta > np.pi:
                            d_theta = 2 * np.pi - d_theta
                        dtheta_list.append(d_theta)
                        dz_list.append(abs(z_s[k+1] - z_s[k]))
        if len(dtheta_list) > 0:
            th_theta = np.percentile(dtheta_list, 90) * 1.5
            th_z = np.percentile(dz_list, 90) * 1.5
        else:
            th_theta = 0.05
            th_z = 5.0
        self.theta_th = float(th_theta)
        self.z_th = float(th_z)
        global THETA_TH, Z_TH
        THETA_TH = self.theta_th
        Z_TH = self.z_th
        return self

    def transform(self, data):
        return data

def make_preprocessor():
    return MyPreprocessor()

class HitClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.theta_th = globals().get("THETA_TH", 0.05)
        self.z_th = globals().get("Z_TH", 5.0)

    def forward(self, batch):
        preds = []
        for ev in batch:
            if isinstance(ev, (list, tuple)):
                X = ev[0]
            else:
                X = ev
            x_cpu = X.detach().cpu().numpy()                   # [N, 4]
            theta = x_cpu[:, 1]                                # [N]
            z = x_cpu[:, 2]                                    # [N]
            layer = x_cpu[:, 3].astype(int)                    # [N]
            N = layer.shape[0]
            dsu = DSU(N)
            unique_layers = np.unique(layer)
            for L in unique_layers:
                nextL = L + 1
                if nextL not in unique_layers:
                    continue
                idxL = np.where(layer == L)[0]
                idxNext = np.where(layer == nextL)[0]
                for i in idxL:
                    dt = np.abs(theta[idxNext] - theta[i])
                    dt = np.where(dt > np.pi, 2 * np.pi - dt, dt)
                    dz_arr = np.abs(z[idxNext] - z[i])
                    cond = (dt <= self.theta_th) & (dz_arr <= self.z_th)
                    for j_idx in np.where(cond)[0]:
                        j = idxNext[j_idx]
                        dsu.union(int(i), int(j))
            labels_np = np.zeros(N, dtype=int)
            root_map = {}
            curr_label = 0
            for idx in range(N):
                r = dsu.find(idx)
                if r not in root_map:
                    root_map[r] = curr_label
                    curr_label += 1
                labels_np[idx] = root_map[r]
            labels = torch.from_numpy(labels_np).to(X.device)
            preds.append(labels)
        return preds

def make_model(in_features):
    return HitClassifier(in_features)

EPOCHS = 10

def train_model(model, train_loader, val_loader, epochs):
    return model, [], [], [], []

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

    # 2. Build model
    first_batch    = next(iter(train_loader))
    example_sample = first_batch[0]
    model          = make_model(example_sample)
    model          = model.to(device)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single toy forward pass
    if dryrun:
        try:
            _ = trained_model(first_batch)
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


