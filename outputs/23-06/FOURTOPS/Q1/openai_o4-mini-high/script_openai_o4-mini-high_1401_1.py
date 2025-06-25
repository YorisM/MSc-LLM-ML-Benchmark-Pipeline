
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
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None

    def _raw_reshape(self, X):
        return X

    def _extract_features(self, X):
        X = self._raw_reshape(X)
        X = X.contiguous()
        N = X.shape[0]
        # raw shape (N,92)
        # global features
        et_miss = X[:, 0:1]                       # (N,1)
        phi_miss = X[:, 1]                        # (N,)
        phi_miss_sin = torch.sin(phi_miss)        # (N,)
        phi_miss_cos = torch.cos(phi_miss)        # (N,)
        global_feats = torch.cat(
            [et_miss, phi_miss_sin.unsqueeze(1), phi_miss_cos.unsqueeze(1)], dim=1
        )                                         # (N,3)
        # object features
        objs = X[:, 2:].view(N, 18, 5)            # (N,18,5)
        obj_id = objs[:, :, 0]                    # (N,18)
        E = objs[:, :, 1]                         # (N,18)
        pT = objs[:, :, 2]                        # (N,18)
        eta = objs[:, :, 3]                       # (N,18)
        phi = objs[:, :, 4]                       # (N,18)
        mask = (obj_id > 0).float()               # (N,18)
        phi_sin = torch.sin(phi) * mask           # (N,18)
        phi_cos = torch.cos(phi) * mask           # (N,18)
        # stack features: obj_id, E, pT, eta, phi_sin, phi_cos
        obj_feats = torch.stack(
            [obj_id, E, pT, eta, phi_sin, phi_cos], dim=2
        )                                         # (N,18,6)
        obj_feats_flat = obj_feats.view(N, 18 * 6) # (N,108)
        X_feats = torch.cat([global_feats, obj_feats_flat], dim=1) # (N,111)
        return X_feats

    def make_loader_cfg(self):
        return None

    def fit(self, X, y=None):
        feats = self._extract_features(X)
        self.mean = feats.mean(dim=0)
        self.std = feats.std(dim=0, unbiased=False)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        feats = self._extract_features(X)
        feats_norm = (feats - self.mean) / self.std
        return feats_norm

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        if isinstance(sample_object, torch.Tensor):
            input_dim = sample_object.shape[1]
        else:
            raise ValueError("Expected sample_object to be a Tensor")
        hidden_dim1 = 256
        hidden_dim2 = 128
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits.squeeze(1)

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 10

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    criterion = nn.BCEWithLogitsLoss()
    best_val_auc = 0.0
    best_state = None
    patience = 5
    counter = 0
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        y_true_train = []
        y_score_train = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float()
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
            probs = torch.sigmoid(logits)
            y_true_train.append(y_batch.detach().cpu())
            y_score_train.append(probs.detach().cpu())
        epoch_loss_train = running_loss / len(train_loader.dataset)
        y_true_train = torch.cat(y_true_train).numpy()
        y_score_train = torch.cat(y_score_train).numpy()
        train_auc = roc_auc_score(y_true_train, y_score_train)
        train_losses.append(epoch_loss_train)
        train_aucs.append(train_auc)

        model.eval()
        running_loss_val = 0.0
        y_true_val = []
        y_score_val = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).float()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                running_loss_val += loss.item() * x_batch.size(0)
                probs = torch.sigmoid(logits)
                y_true_val.append(y_batch.cpu())
                y_score_val.append(probs.cpu())
        epoch_loss_val = running_loss_val / len(val_loader.dataset)
        y_true_val = torch.cat(y_true_val).numpy()
        y_score_val = torch.cat(y_score_val).numpy()
        val_auc = roc_auc_score(y_true_val, y_score_val)
        val_losses.append(epoch_loss_val)
        val_aucs.append(val_auc)

        scheduler.step(val_auc)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses, train_aucs, val_aucs

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

