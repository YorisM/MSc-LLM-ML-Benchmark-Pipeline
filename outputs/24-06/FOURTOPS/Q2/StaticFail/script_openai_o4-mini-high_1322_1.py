
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
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.id_map = None
        self.num_obj_types = None

    def _raw_reshape(self, X):
        return X

    def make_loader_cfg(self):
        return None

    def fit(self, X, y=None):
        # X: [n,92]
        obj_ids = X[:, 2::5].long()  # [n,18]
        unique_ids = torch.unique(obj_ids)  # sorted unique
        # map pad id=0 to index 0
        self.id_map = {0: 0}
        real_ids = [int(i) for i in unique_ids.tolist() if int(i) != 0]
        for idx, pid in enumerate(real_ids, start=1):
            self.id_map[pid] = idx
        self.num_obj_types = len(real_ids) + 1
        return self

    def transform(self, X):
        # X: [n,92]
        # Global features
        E_miss = X[:, 0:1]  # [n,1]
        phi_miss = X[:, 1:2]  # [n,1]
        g0 = torch.log1p(E_miss)  # [n,1]
        sin_phi_miss = torch.sin(phi_miss)  # [n,1]
        cos_phi_miss = torch.cos(phi_miss)  # [n,1]
        global_feats = torch.cat([g0, sin_phi_miss, cos_phi_miss], dim=1)  # [n,3]

        # Object IDs and features
        raw_ids = X[:, 2::5].long()  # [n,18]
        # map raw_ids to indices
        id_indices = torch.zeros_like(raw_ids)  # [n,18]
        for raw_id, idx in self.id_map.items():
            id_indices[raw_ids == raw_id] = idx
        mask = (id_indices != 0).unsqueeze(-1)  # [n,18,1]

        # one-hot encode ids
        id_onehot = F.one_hot(id_indices, num_classes=self.num_obj_types).float()  # [n,18,num_obj_types]
        id_onehot = id_onehot * mask  # zero out pad ones

        # Kinematic features
        E = X[:, 3::5]  # [n,18]
        pT = X[:, 4::5]  # [n,18]
        eta = X[:, 5::5]  # [n,18]
        phi = X[:, 6::5]  # [n,18]
        logE = torch.log1p(E)  # [n,18]
        logpT = torch.log1p(pT)  # [n,18]
        sin_phi = torch.sin(phi)  # [n,18]
        cos_phi = torch.cos(phi)  # [n,18]
        kin_feats = torch.stack([logE, logpT, eta, sin_phi, cos_phi], dim=2)  # [n,18,5]
        kin_feats = kin_feats * mask  # zero out pad

        # Combine id and kin features
        object_feats = torch.cat([id_onehot, kin_feats], dim=2)  # [n,18,num_obj_types+5]

        return (global_feats, object_feats)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object: tuple (global_feats, object_feats), shapes ([batch,3], [batch,18,D])
        global_dim = sample_object[0].size(1)
        num_objects = sample_object[1].size(1)
        obj_feat_dim = sample_object[1].size(2)
        # per-object MLP
        self.obj_mlp = nn.Sequential(
            nn.Linear(obj_feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        # classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(32 + global_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, global_feats, object_feats):
        # global_feats: [batch,3], object_feats: [batch,18,D]
        b, n, d = object_feats.shape
        x = object_feats.view(b * n, d)  # [b*n, d]
        obj_emb = self.obj_mlp(x)  # [b*n,32]
        obj_emb = obj_emb.view(b, n, -1)  # [b,18,32]
        agg = obj_emb.sum(dim=1)  # [b,32]
        combined = torch.cat([agg, global_feats], dim=1)  # [b,32+3]
        out = self.classifier(combined)  # [b,1]
        return out.squeeze(1)  # [b]

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    best_val_auc = 0.0
    best_state = None
    patience, counter = 5, 0
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for x, y in train_loader:
            global_feats, object_feats = x
            global_feats = global_feats.to(device)
            object_feats = object_feats.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(global_feats, object_feats)  # [b]
            loss = criterion(logits, y.float())
            loss.backward()
            optimizer.step()
            preds = (torch.sigmoid(logits) > 0.5).long()
            total_correct += preds.eq(y).sum().item()
            total_samples += y.size(0)
            total_loss += loss.item() * y.size(0)
        train_loss_epoch = total_loss / total_samples
        train_acc_epoch = total_correct / total_samples
        train_loss.append(train_loss_epoch)
        train_acc.append(train_acc_epoch)
        model.eval()
        val_total_loss, val_total_correct, val_total = 0.0, 0, 0
        all_labels, all_scores = [], []
        with torch.no_grad():
            for x, y in val_loader:
                global_feats, object_feats = x
                global_feats = global_feats.to(device)
                object_feats = object_feats.to(device)
                y = y.to(device)
                logits = model(global_feats, object_feats)
                loss = criterion(logits, y.float())
                preds = (torch.sigmoid(logits) > 0.5).long()
                val_total_correct += preds.eq(y).sum().item()
                val_total += y.size(0)
                val_total_loss += loss.item() * y.size(0)
                all_labels.append(y.cpu())
                all_scores.append(torch.sigmoid(logits).cpu())
        val_loss_epoch = val_total_loss / val_total
        val_acc_epoch = val_total_correct / val_total
        labels_cat = torch.cat(all_labels).numpy()
        scores_cat = torch.cat(all_scores).numpy()
        try:
            val_auc = roc_auc_score(labels_cat, scores_cat)
        except:
            val_auc = 0.0
        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)
        scheduler.step(val_loss_epoch)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu()
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

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

