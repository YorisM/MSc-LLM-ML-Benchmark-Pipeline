
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.met_log_mean = None
        self.met_log_std = None
        self.E_log_mean = None
        self.E_log_std = None
        self.pT_log_mean = None
        self.pT_log_std = None
        self.eta_mean = None
        self.eta_std = None
        self.obj_types = None
        self.obj_type_map = None
        self.n_types = None

    def _raw_reshape(self, X):
        return X

    def fit(self, X, y=None):
        # X: [n,92]
        met = X[:,0]
        met_log = torch.log10(met + 1)
        self.met_log_mean = met_log.mean().item()
        self.met_log_std = met_log.std(unbiased=False).item()
        E = X[:,3:92:5]
        E_log = torch.log10(E + 1)
        self.E_log_mean = E_log.mean().item()
        self.E_log_std = E_log.std(unbiased=False).item()
        pT = X[:,4:92:5]
        pT_log = torch.log10(pT + 1)
        self.pT_log_mean = pT_log.mean().item()
        self.pT_log_std = pT_log.std(unbiased=False).item()
        eta = X[:,5:92:5]
        self.eta_mean = eta.mean().item()
        self.eta_std = eta.std(unbiased=False).item()
        obj_ids = X[:,2:92:5].long().flatten()
        unique_ids = torch.unique(obj_ids).tolist()
        unique_ids.sort()
        self.obj_types = unique_ids
        self.obj_type_map = {orig: idx for idx, orig in enumerate(unique_ids)}
        self.n_types = len(unique_ids)
        return self

    def transform(self, X):
        X = self._raw_reshape(X)
        # global features
        met = X[:,0]  # [n]
        met_log = torch.log10(met + 1)
        met_norm = (met_log - self.met_log_mean) / self.met_log_std
        phi = X[:,1]  # [n]
        phi_sin = torch.sin(phi)
        phi_cos = torch.cos(phi)
        global_feats = torch.stack([met_norm, phi_sin, phi_cos], dim=1)  # [n,3]
        # object ids
        obj_ids_orig = X[:,2:92:5].long()  # [n,18]
        obj_ids = torch.zeros_like(obj_ids_orig)
        for orig, idx in self.obj_type_map.items():
            obj_ids[obj_ids_orig == orig] = idx
        one_hot = F.one_hot(obj_ids, num_classes=self.n_types).float()  # [n,18,n_types]
        # continuous object features
        E = X[:,3:92:5]  # [n,18]
        E_log = torch.log10(E + 1)
        E_norm = (E_log - self.E_log_mean) / self.E_log_std
        pT = X[:,4:92:5]
        pT_log = torch.log10(pT + 1)
        pT_norm = (pT_log - self.pT_log_mean) / self.pT_log_std
        eta = X[:,5:92:5]
        eta_norm = (eta - self.eta_mean) / self.eta_std
        phi_obj = X[:,6:92:5]
        phi_sin_obj = torch.sin(phi_obj)
        phi_cos_obj = torch.cos(phi_obj)
        cont_feats = torch.stack([E_norm, pT_norm, eta_norm, phi_sin_obj, phi_cos_obj], dim=2)  # [n,18,5]
        object_feats = torch.cat([one_hot, cont_feats], dim=2)  # [n,18,n_types+5]
        mask = (obj_ids_orig != 0).float()  # [n,18]
        return (global_feats, object_feats, mask)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        global_feats, object_feats, mask = sample_object
        global_dim = global_feats.shape[1]
        obj_feat_dim = object_feats.shape[2]
        self.object_mlp = nn.Sequential(
            nn.Linear(obj_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        combined_dim = 64*2 + 16
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, global_feats, object_feats, mask):
        B, M, F = object_feats.shape
        x = object_feats.view(B*M, F)  # [B*M,F]
        x = self.object_mlp(x)  # [B*M,64]
        x = x.view(B, M, -1)  # [B,18,64]
        mask_exp = mask.unsqueeze(-1)  # [B,18,1]
        x_masked = x * mask_exp  # [B,18,64]
        obj_sum = x_masked.sum(dim=1)  # [B,64]
        x_masked_max = x.masked_fill(mask_exp == 0, -1e9)
        obj_max, _ = x_masked_max.max(dim=1)  # [B,64]
        g = self.global_mlp(global_feats)  # [B,16]
        combined = torch.cat([obj_sum, obj_max, g], dim=1)  # [B,144]
        logits = self.classifier(combined).squeeze(1)  # [B]
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 10

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    best_val_auc = 0.0
    best_state = None
    patience_cnt = 6
    counter = 0
    train_loss_history = []
    val_loss_history = []
    train_auc_history = []
    val_auc_history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        preds = []
        trues = []
        total = 0
        for x, y in train_loader:
            global_feats, object_feats, mask = x
            global_feats = global_feats.to(device)
            object_feats = object_feats.to(device)
            mask = mask.to(device)
            y = y.float().to(device)
            logits = model(global_feats, object_feats, mask)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
            total += y.size(0)
        train_loss = running_loss / total
        preds_all = np.concatenate(preds)
        trues_all = np.concatenate(trues)
        train_auc = roc_auc_score(trues_all, preds_all)
        model.eval()
        running_loss_val = 0.0
        preds = []
        trues = []
        total_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                global_feats, object_feats, mask = x
                global_feats = global_feats.to(device)
                object_feats = object_feats.to(device)
                mask = mask.to(device)
                y = y.float().to(device)
                logits = model(global_feats, object_feats, mask)
                loss = criterion(logits, y)
                running_loss_val += loss.item() * y.size(0)
                preds.append(torch.sigmoid(logits).cpu().numpy())
                trues.append(y.cpu().numpy())
                total_val += y.size(0)
        val_loss = running_loss_val / total_val
        preds_all = np.concatenate(preds)
        trues_all = np.concatenate(trues)
        val_auc = roc_auc_score(trues_all, preds_all)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_auc_history.append(train_auc)
        val_auc_history.append(val_auc)
        scheduler.step(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience_cnt:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_loss_history, val_loss_history, train_auc_history, val_auc_history

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

