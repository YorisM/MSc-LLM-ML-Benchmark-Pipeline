
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

import math
import itertools
from sklearn.metrics import roc_auc_score

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.k = 4
        self.pairs = list(itertools.combinations(range(self.k), 2))

    def _raw_reshape(self, X):
        return X

    def make_loader_cfg(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X: [N,92]
        N = X.shape[0]
        # global features
        global_feats = X[:, :2]  # [N,2]
        # per-object features
        objs = X[:, 2:].reshape(N, 18, 5)  # [N,18,5]
        ids = objs[:, :, 0]    # [N,18]
        E = objs[:, :, 1]      # [N,18]
        pT = objs[:, :, 2]     # [N,18]
        eta = objs[:, :, 3]    # [N,18]
        phi = objs[:, :, 4]    # [N,18]
        # select top-k by pT
        idx = torch.argsort(pT, dim=1, descending=True)  # [N,18]
        id_sorted = torch.gather(ids, 1, idx)    # [N,18]
        E_sorted = torch.gather(E, 1, idx)       # [N,18]
        pT_sorted = torch.gather(pT, 1, idx)     # [N,18]
        eta_sorted = torch.gather(eta, 1, idx)   # [N,18]
        phi_sorted = torch.gather(phi, 1, idx)   # [N,18]
        # take top-k objects
        id_k = id_sorted[:, :self.k]             # [N,k]
        E_k = E_sorted[:, :self.k]               # [N,k]
        pT_k = pT_sorted[:, :self.k]             # [N,k]
        eta_k = eta_sorted[:, :self.k]           # [N,k]
        phi_k = phi_sorted[:, :self.k]           # [N,k]
        # flatten per-object features
        feats_k = torch.stack([id_k, E_k, pT_k, eta_k, phi_k], dim=2)  # [N,k,5]
        feats_k_flat = feats_k.reshape(N, self.k * 5)  # [N, k*5]
        # precompute momentum components for invariant mass
        px_k = pT_k * torch.cos(phi_k)            # [N,k]
        py_k = pT_k * torch.sin(phi_k)            # [N,k]
        pz_k = pT_k * torch.sinh(eta_k)           # [N,k]
        # compute pairwise features
        pair_feats = []
        for i, j in self.pairs:
            # angular distance
            dphi = phi_k[:, i] - phi_k[:, j]      # [N]
            dphi = torch.remainder(dphi + math.pi, 2 * math.pi) - math.pi  # [N]
            deta = eta_k[:, i] - eta_k[:, j]      # [N]
            dr = torch.sqrt(deta * deta + dphi * dphi)  # [N]
            # invariant mass
            E_sum = E_k[:, i] + E_k[:, j]         # [N]
            px_sum = px_k[:, i] + px_k[:, j]      # [N]
            py_sum = py_k[:, i] + py_k[:, j]      # [N]
            pz_sum = pz_k[:, i] + pz_k[:, j]      # [N]
            m2 = E_sum * E_sum - (px_sum * px_sum + py_sum * py_sum + pz_sum * pz_sum)  # [N]
            m2 = torch.clamp(m2, min=0.0)
            m = torch.sqrt(m2)                    # [N]
            pair_feats.append(dr)
            pair_feats.append(m)
        pair_feats = torch.stack(pair_feats, dim=1)  # [N, 2*C(k,2)]
        # concatenate all features
        X_out = torch.cat([global_feats, feats_k_flat, pair_feats], dim=1)  # [N, 2 + k*5 + 2*C(k,2)]
        return X_out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[-1]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [batch, input_dim]
        return self.net(x)

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []
    best_val_auc = 0.0
    patience_counter = 0
    patience = 3
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x).squeeze(1)            # [batch]
            loss = criterion(out, y.float())
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * y.size(0)
            preds = (torch.sigmoid(out) > 0.5)
            train_correct += preds.eq(y.bool()).sum().item()
            train_total += y.size(0)
        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_targets = []
        val_scores = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                out = model(x).squeeze(1)        # [batch]
                loss = criterion(out, y.float())
                val_loss_sum += loss.item() * y.size(0)
                probs = torch.sigmoid(out)
                preds = (probs > 0.5)
                val_correct += preds.eq(y.bool()).sum().item()
                val_total += y.size(0)
                val_targets.append(y.cpu())
                val_scores.append(probs.cpu())
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        val_targets = torch.cat(val_targets).numpy()
        val_scores = torch.cat(val_scores).numpy()
        val_auc = roc_auc_score(val_targets, val_scores)

        scheduler.step(val_auc)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

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

