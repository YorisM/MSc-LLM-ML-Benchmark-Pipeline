
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import copy

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.fitted = False

    def _raw_reshape(self, X):
        return X

    def make_loader_cfg(self):
        return None

    def fit(self, X, y=None):
        X_raw = X  # [N, 92]
        N = X_raw.shape[0]
        raw_objs = X_raw[:, 2:].contiguous().view(N, 18, 5)  # [N, 18, 5]
        obj_types = raw_objs[:, :, 0].long()  # [N, 18]
        unique_codes = torch.unique(obj_types[obj_types != 0])
        unique_codes_sorted, _ = torch.sort(unique_codes)
        self.unique_codes = unique_codes_sorted.tolist()
        max_code = int(obj_types.max().item())
        self.max_code = max_code
        mapping_arr = torch.zeros(max_code + 1, dtype=torch.long)
        for idx, code in enumerate(unique_codes_sorted):
            mapping_arr[code] = idx + 1
        self.mapping_arr = mapping_arr  # [max_code+1]
        self.num_types = len(unique_codes_sorted)
        cont_raw = raw_objs[:, :, 1:4]  # [N, 18, 3]
        mask = obj_types != 0  # [N, 18]
        cont_flat = cont_raw[mask]  # [n_valid, 3]
        self.obj_cont_means = cont_flat.mean(dim=0)  # [3]
        self.obj_cont_stds = cont_flat.std(dim=0, unbiased=False)  # [3]
        global_cont = X_raw[:, 0]  # [N]
        self.global_mean = global_cont.mean().item()
        self.global_std = global_cont.std(unbiased=False).item()
        self.fitted = True
        return self

    def transform(self, X):
        assert self.fitted, "Preprocessor must be fitted before calling transform"
        X_raw = X  # [N, 92]
        N = X_raw.shape[0]
        global_et = X_raw[:, 0]  # [N]
        phi_et =# 0. ---------- IMPORTS ----------
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import copy

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.fitted = False

    def _raw_reshape(self, X):
        return X

    def make_loader_cfg(self):
        return None

    def fit(self, X, y=None):
        X_raw = X  # [N, 92]
        N = X_raw.shape[0]
        raw_objs = X_raw[:, 2:].contiguous().view(N, 18, 5)  # [N, 18, 5]
        obj_types = raw_objs[:, :, 0].long()  # [N, 18]
        unique_codes = torch.unique(obj_types[obj_types != 0])
        unique_codes_sorted, _ = torch.sort(unique_codes)
        self.unique_codes = unique_codes_sorted.tolist()
        max_code = int(obj_types.max().item())
        self.max_code = max_code
        mapping_arr = torch.zeros(max_code + 1, dtype=torch.long)
        for idx, code in enumerate(unique_codes_sorted):
            mapping_arr[code] = idx + 1
        self.mapping_arr = mapping_arr  # [max_code+1]
        self.num_types = len(unique_codes_sorted)
        cont_raw = raw_objs[:, :, 1:4]  # [N, 18, 3]
        mask = obj_types != 0  # [N, 18]
        cont_flat = cont_raw[mask]  # [n_valid, 3]
        self.obj_cont_means = cont_flat.mean(dim=0)  # [3]
        self.obj_cont_stds = cont_flat.std(dim=0, unbiased=False)  # [3]
        global_cont = X_raw[:, 0]  # [N]
        self.global_mean = global_cont.mean().item()
        self.global_std = global_cont.std(unbiased=False).item()
        self.fitted = True
        return self

    def transform(self, X):
        assert self.fitted, "Preprocessor must be fitted before calling transform"
        X_raw = X  # [N, 92]
        N = X_raw.shape[0]
        global_et = X_raw[:, 0]  # [N]
        phi_et = X_raw[:, 1]  # [N]
        global_et_norm = (global_et - self.global_mean) / (self.global_std + 1e-8)  # [N]
        sin_phi_et = torch.sin(phi_et)  # [N]
        cos_phi_et = torch.cos(phi_et)  # [N]
        global_feats = torch.stack([global_et_norm, sin_phi_et, cos_phi_et], dim=1)  # [N, 3]
        raw_objs = X_raw[:, 2:].contiguous().view(N, 18, 5)  # [N, 18, 5]
        obj_types = raw_objs[:, :, 0].long()  # [N, 18]
        cont_raw = raw_objs[:, :, 1:4]  # [N, 18, 3]
        phi = raw_objs[:, :, 4]  # [N, 18]
        sin_phi = torch.sin(phi)  # [N, 18]
        cos_phi = torch.cos(phi)  # [N, 18]
        cont = torch.cat([cont_raw, sin_phi.unsqueeze(-1), cos_phi.unsqueeze(-1)], dim=2)  # [N, 18, 5]
        cont_norm = cont.clone()  # [N, 18, 5]
        cont_norm[:, :, 0] = (cont[:, :, 0] - self.obj_cont_means[0]) / (self.obj_cont_stds[0] + 1e-8)  # [N, 18]
        cont_norm[:, :, 1] = (cont[:, :, 1] - self.obj_cont_means[1]) / (self.obj_cont_stds[1] + 1e-8)  # [N, 18]
        cont_norm[:, :, 2] = (cont[:, :, 2] - self.obj_cont_means[2]) / (self.obj_cont_stds[2] + 1e-8)  # [N, 18]
        cont = cont_norm  # [N, 18, 5]
        mapping_arr = self.mapping_arr  # [max_code+1]
        new_idxs = mapping_arr[obj_types]  # [N, 18]
        one_hot = F.one_hot(new_idxs, num_classes=self.num_types + 1)  # [N, 18, num_types+1]
        type_onehot = one_hot[:, :, 1:].float()  # [N, 18, num_types]
        obj_feats = torch.cat([cont, type_onehot], dim=2)  # [N, 18, 5+num_types]
        mask = (new_idxs != 0).unsqueeze(-1).float()  # [N, 18, 1]
        obj_feats = obj_feats * mask  # [N, 18, 5+num_types]
        return (global_feats, obj_feats)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample):
        super().__init__()
        if isinstance(sample, (tuple, list)):
            global_sample, obj_sample = sample
        else:
            global_sample = sample
            obj_sample = None
        self.global_dim = global_sample.shape[1]
        if obj_sample is not None:
            self.num_objects = obj_sample.shape[1]
            self.obj_feat_dim = obj_sample.shape[2]
        else:
            self.num_objects = 0
            self.obj_feat_dim = 0
        hidden_dim = 128
        self.hidden_dim = hidden_dim
        self.phi = nn.Sequential(
            nn.Linear(self.obj_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.rho = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.global_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, global_feats, obj_feats):
        B = global_feats.shape[0]
        x = obj_feats  # [B, M, D]
        M = x.shape[1]
        D = x.shape[2]
        x = x.view(B * M, D)  # [B*M, D]
        x = self.phi(x)  # [B*M, hidden_dim]
        x = x.view(B, M, self.hidden_dim)  # [B, M, hidden_dim]
        x_sum = x.sum(dim=1)  # [B, hidden_dim]
        x_max = x.max(dim=1).values  # [B, hidden_dim]
        x_global = torch.cat([x_sum, x_max, global_feats], dim=1)  # [B, 2*hidden_dim+global_dim]
        out = self.rho(x_global)  # [B, 1]
        return out.view(-1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_auc = 0.0
    patience = 3
    epochs_no_improve = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, labels in train_loader:
            optimizer.zero_grad()
            if isinstance(data, (tuple, list)):
                inputs = [d.to(device) for d in data]
                logits = model(*inputs)
            else:
                inputs = data.to(device)
                logits = model(inputs)
            labels = labels.to(device).float()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for data, labels in val_loader:
                if isinstance(data, (tuple, list)):
                    inputs = [d.to(device) for d in data]
                    logits = model(*inputs)
                else:
                    inputs = data.to(device)
                    logits = model(inputs)
                labels = labels.to(device).float()
                loss = criterion(logits, labels)
                val_running_loss += loss.item() * labels.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())
        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()
        val_auc = roc_auc_score(all_labels, all_logits)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_accs, val_accs

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

