
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

import copy
from sklearn.metrics import roc_auc_score

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_feature_count = 2
        self.n_objects = 18
        self.obj_dim = 5
        self.kin_mean = None
        self.kin_std = None
        self.max_obj_id = None

    def _raw_reshape(self, X):
        return X

    def make_loader_cfg(self):
        return None

    def fit(self, X, y=None):
        # X: [N,92]
        object_data = X[:, self.global_feature_count:].reshape(-1, self.n_objects, self.obj_dim)  # [N,18,5]
        obj_ids = object_data[:, :, 0]  # [N,18]
        mask = obj_ids > 0  # [N,18]
        kin_feats = object_data[:, :, 1:]  # [N,18,4]
        kin_flat = kin_feats.reshape(-1, kin_feats.shape[-1])  # [N*18,4]
        mask_flat = mask.reshape(-1)  # [N*18]
        valid_kin = kin_flat[mask_flat]  # [num_valid,4]
        self.kin_mean = valid_kin.mean(dim=0)  # [4]
        self.kin_std = valid_kin.std(dim=0)  # [4]
        self.kin_std[self.kin_std < 1e-6] = 1.0
        self.max_obj_id = int(obj_ids.max().item())
        return self

    def transform(self, X):
        # X: [N,92]
        # Global features: ET_miss and phi
        log_et = torch.log1p(X[:, 0:1])  # [N,1]
        phi = X[:, 1:2]  # [N,1]
        phi_sin = torch.sin(phi)  # [N,1]
        phi_cos = torch.cos(phi)  # [N,1]
        global_feats = torch.cat([log_et, phi_sin, phi_cos], dim=1)  # [N,3]
        object_data = X[:, self.global_feature_count:].reshape(-1, self.n_objects, self.obj_dim)  # [N,18,5]
        obj_ids = object_data[:, :, 0].to(torch.long)  # [N,18]
        kin_feats = object_data[:, :, 1:]  # [N,18,4]
        kin_feats_norm = (kin_feats - self.kin_mean) / self.kin_std  # [N,18,4]
        mask = (obj_ids > 0).unsqueeze(-1).float()  # [N,18,1]
        kin_feats_norm = kin_feats_norm * mask  # [N,18,4]
        max_ids = torch.full((X.shape[0],), self.max_obj_id, dtype=torch.long)  # [N]
        return (obj_ids, kin_feats_norm, global_feats, max_ids)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        obj_ids, kin_feats, global_feats, max_ids = sample_object
        self.n_objects = obj_ids.shape[1]
        self.kin_dim = kin_feats.shape[2]
        self.global_dim = global_feats.shape[1]
        self.num_obj_types = int(max_ids.max().item()) + 1
        self.id_emb_dim = 4
        self.obj_emb = nn.Embedding(self.num_obj_types, self.id_emb_dim, padding_idx=0)
        self.kin_proj = nn.Linear(self.kin_dim, self.id_emb_dim)
        self.hidden_dim = 128
        self.input_dim = self.n_objects * (self.id_emb_dim * 2) + self.global_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, obj_ids, kin_feats, global_feats, max_ids=None):
        # obj_ids: [B,18], kin_feats: [B,18,4], global_feats: [B,3]
        id_emb = self.obj_emb(obj_ids)  # [B,18,4]
        kin_proj = self.kin_proj(kin_feats)  # [B,18,4]
        obj_feat = torch.cat([id_emb, kin_proj], dim=-1)  # [B,18,8]
        obj_flat = obj_feat.view(obj_feat.size(0), -1)  # [B,144]
        x = torch.cat([obj_flat, global_feats], dim=1)  # [B,147]
        logits = self.classifier(x).squeeze(1)  # [B]
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_val_auc = 0.0
    best_model_wts = {k: v.clone() for k, v in model.state_dict().items()}
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    patience, counter = 3, 0
    for epoch in range(epochs):
        model.train()
        train_loss_sum, correct, total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            if isinstance(batch_x, (tuple, list)):
                inputs = [t.to(device) for t in batch_x]
                logits = model(*inputs)
            else:
                inputs = batch_x.to(device)
                logits = model(inputs)
            labels = batch_y.to(device)
            loss = criterion(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = train_loss_sum / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        # Validation
        model.eval()
        val_loss_sum, correct_v, total_v = 0.0, 0, 0
        all_labels, all_probs = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                if isinstance(batch_x, (tuple, list)):
                    inputs = [t.to(device) for t in batch_x]
                    logits = model(*inputs)
                else:
                    inputs = batch_x.to(device)
                    logits = model(inputs)
                labels = batch_y.to(device)
                loss = criterion(logits, labels.float())
                val_loss_sum += loss.item() * labels.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                correct_v += (preds == labels).sum().item()
                total_v += labels.size(0)
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
        val_loss = val_loss_sum / total_v
        val_acc = correct_v / total_v
        val_auc = roc_auc_score(all_labels, all_probs)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        scheduler.step(val_auc)
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_model_wts = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
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

