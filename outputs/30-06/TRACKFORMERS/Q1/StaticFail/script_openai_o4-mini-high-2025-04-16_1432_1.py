
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.7.1, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

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

    train_ld = loader_cls(train_ds, batch_size=batch, shuffle=True,
                          num_workers=workers, collate_fn=collate_fn)
    val_ld   = loader_cls(val_ds,   batch_size=batch, shuffle=False,
                          num_workers=workers, collate_fn=collate_fn)
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
import numpy as np
import torch.nn.functional as F

# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.n_layers = None

    def _raw_reshape(self, data):
        return data

    def make_loader_cfg(self):
        return None

    def fit(self, events):
        # Compute mean and std of (r,theta,z) and max layer id
        total = np.zeros(3, dtype=np.float64)
        total_sq = np.zeros(3, dtype=np.float64)
        count = 0
        max_layer = 0
        for evt in events:
            # evt["hit_r","hit_theta","hit_z"] stacked into [N,3]
            coords = np.column_stack((evt["hit_r"], evt["hit_theta"], evt["hit_z"]))  # [N_i,3]
            total += coords.sum(axis=0)
            total_sq += (coords**2).sum(axis=0)
            count += coords.shape[0]
            max_layer = max(max_layer, int(evt["layer_id"].max()))
        mean = total / count  # [3]
        var = total_sq / count - mean**2  # [3]
        std = np.sqrt(var + 1e-8)  # [3]
        # store as torch tensors
        self.mean = torch.from_numpy(mean).float()  # [3]
        self.std = torch.from_numpy(std).float()    # [3]
        self.n_layers = max_layer + 1
        return self

    def transform(self, data):
        # data: torch.Tensor [N,4]
        rtz = data[:, 0:3]                             # [N,3]
        layer = data[:, 3].long()                      # [N]
        # normalize continuous features
        rtz_norm = (rtz - self.mean.unsqueeze(0).to(rtz.device)) / self.std.unsqueeze(0).to(rtz.device)  # [N,3]
        # one-hot encode layer_id
        layer_onehot = F.one_hot(layer, num_classes=self.n_layers).float()  # [N,n_layers]
        # concatenate
        features = torch.cat([rtz_norm, layer_onehot], dim=1)  # [N, 3+n_layers]
        return features

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
from torch import nn

class HitClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        hidden_dim = 128
        num_classes = 50
        # MLP encoder
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        # classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, X):
        # X: [total_hits, in_features]
        return self.layers(X)  # [total_hits, hidden_dim]

    def logits(self, X):
        # X: [total_hits, in_features]
        h = self.encode(X)     # [total_hits, hidden_dim]
        return self.classifier(h)  # [total_hits, num_classes]

    def forward(self, batch):
        # batch: list of (X_i, y_i) pairs
        xs, _ = zip(*batch)
        lengths = [x.shape[0] for x in xs]             # [batch_size]
        X_all = torch.cat(xs, dim=0)                   # [sum(N_i), in_features]
        logits = self.logits(X_all)                    # [sum(N_i), num_classes]
        preds = logits.argmax(dim=1)                   # [sum(N_i)]
        preds_list = list(torch.split(preds, lengths)) # list of [N_i]
        return preds_list

def make_model(in_features):
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    best_val_loss = float("inf")
    best_state = None
    patience = 5
    counter = 0

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch in train_loader:
            xs, ys = zip(*batch)
            X_all = torch.cat(xs, dim=0).to(device)   # [sum(N_i), in_features]
            y_all = torch.cat(ys, dim=0).to(device)   # [sum(N_i)]
            optimizer.zero_grad()
            logits = model.logits(X_all)              # [sum(N_i), num_classes]
            loss = criterion(logits, y_all)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)              # [sum(N_i)]
            total_correct += (preds == y_all).sum().item()
            total_samples += y_all.size(0)
            total_loss += loss.item() * y_all.size(0)
        avg_train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                xs, ys = zip(*batch)
                X_all = torch.cat(xs, dim=0).to(device)  # [sum(N_i), in_features]
                y_all = torch.cat(ys, dim=0).to(device)  # [sum(N_i)]
                logits = model.logits(X_all)             # [sum(N_i), num_classes]
                loss = criterion(logits, y_all)
                preds = logits.argmax(dim=1)             # [sum(N_i)]
                total_val_correct += (preds == y_all).sum().item()
                total_val_samples += y_all.size(0)
                total_val_loss += loss.item() * y_all.size(0)
        avg_val_loss = total_val_loss / total_val_samples
        val_acc = total_val_correct / total_val_samples
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(val_acc)

        # Scheduler step
        scheduler.step(avg_val_loss)
        # Early-stopping
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

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
    hits0, _       = first_batch[0]
    in_features    = hits0.shape[-1]                   
    model          = make_model(in_features)

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


