
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
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.means = None
        self.stds = None

    def _raw_reshape(self, X):
        return X

    def fit(self, X, y=None):
        # X: (N,92)
        X_r = self._raw_reshape(X)
        glob_mag = X_r[:, 0:1]                  # (N,1)
        glob_phi = X_r[:, 1:2]                  # (N,1)
        sin_glob = torch.sin(glob_phi)          # (N,1)
        cos_glob = torch.cos(glob_phi)          # (N,1)
        N = X_r.size(0)
        objs = X_r[:, 2:].view(N, 18, 5)        # (N,18,5)
        obj_id  = objs[:, :, 0:1]               # (N,18,1)
        obj_E   = objs[:, :, 1:2]               # (N,18,1)
        obj_pT  = objs[:, :, 2:3]               # (N,18,1)
        obj_eta = objs[:, :, 3:4]               # (N,18,1)
        obj_phi = objs[:, :, 4:5]               # (N,18,1)
        sin_obj = torch.sin(obj_phi)            # (N,18,1)
        cos_obj = torch.cos(obj_phi)            # (N,18,1)
        obj_feat = torch.cat([obj_id, obj_E, obj_pT, obj_eta, sin_obj, cos_obj], dim=2)  # (N,18,6)
        obj_feat = obj_feat.view(N, 18*6)       # (N,108)
        numeric = torch.cat([glob_mag, sin_glob, cos_glob, obj_feat], dim=1)  # (N,111)
        self.means = numeric.mean(dim=0)        # (111,)
        self.stds = numeric.std(dim=0)          # (111,)
        self.stds[self.stds < 1e-6] = 1.0
        return self

    def transform(self, X):
        X_r = self._raw_reshape(X)
        glob_mag = X_r[:, 0:1]                  
        glob_phi = X_r[:, 1:2]                  
        sin_glob = torch.sin(glob_phi)          
        cos_glob = torch.cos(glob_phi)          
        N = X_r.size(0)
        objs = X_r[:, 2:].view(N, 18, 5)        
        obj_id  = objs[:, :, 0:1]               
        obj_E   = objs[:, :, 1:2]               
        obj_pT  = objs[:, :, 2:3]               
        obj_eta = objs[:, :, 3:4]               
        obj_phi = objs[:, :, 4:5]               
        sin_obj = torch.sin(obj_phi)            
        cos_obj = torch.cos(obj_phi)            
        obj_feat = torch.cat([obj_id, obj_E, obj_pT, obj_eta, sin_obj, cos_obj], dim=2)
        obj_feat = obj_feat.view(N, 18*6)       
        numeric = torch.cat([glob_mag, sin_glob, cos_glob, obj_feat], dim=1)
        numeric = (numeric - self.means) / self.stds
        return numeric

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        sample = sample_object[0] if isinstance(sample_object, (tuple, list)) else sample_object
        input_dim = sample.shape[-1]
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        out = self.model(x)    # (batch,1)
        return out.view(-1)    # (batch,)

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 10

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val_auc = 0.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience = 3
    counter = 0
    train_loss_list, val_loss_list = [], []
    train_auc_list, val_auc_list = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_targets, train_preds = [], []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)                        # (batch,)
            loss = criterion(logits, y_batch.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                train_preds.append(probs.cpu())
                train_targets.append(y_batch.cpu())
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_loss_list.append(epoch_train_loss)
        preds = torch.cat(train_preds).numpy()
        targets = torch.cat(train_targets).numpy()
        train_auc = roc_auc_score(targets, preds)
        train_auc_list.append(train_auc)
        model.eval()
        val_running_loss = 0.0
        val_targets, val_preds = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch.float())
                val_running_loss += loss.item() * x_batch.size(0)
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu())
                val_targets.append(y_batch.cpu())
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_loss_list.append(epoch_val_loss)
        val_pred = torch.cat(val_preds).numpy()
        val_true = torch.cat(val_targets).numpy()
        val_auc = roc_auc_score(val_true, val_pred)
        val_auc_list.append(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    model.load_state_dict(best_state)
    return model, train_loss_list, val_loss_list, train_auc_list, val_auc_list

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

