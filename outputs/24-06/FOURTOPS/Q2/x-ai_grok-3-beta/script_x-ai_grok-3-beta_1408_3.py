
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
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
from torch.nn import functional as F

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.per_obj_mean = None
        self.per_obj_std = None
        self.max_objs = 18
        self.obj_features = 4  # E, p_T, eta, phi

    def _raw_reshape(self, X):
        # Reshape to separate global features and per-object features
        # X shape: [N, 92]
        batch_size = X.shape[0]
        global_feats = X[:, :2]  # E_T_miss, phi_Et_miss
        obj_data = X[:, 2:].view(batch_size, self.max_objs, 5)  # Shape: [N, 18, 5]
        obj_feats = obj_data[:, :, 1:]  # Drop object ID, keep E, p_T, eta, phi; Shape: [N, 18, 4]
        obj_mask = (obj_data[:, :, 0] != 0).float()  # Mask for valid objects; Shape: [N, 18]
        return global_feats, obj_feats, obj_mask

    def fit(self, X, y=None):
        global_feats, obj_feats, obj_mask = self._raw_reshape(X)

        # Compute mean and std for global features
        self.global_mean = global_feats.mean(dim=0, keepdim=True)  # Shape: [1, 2]
        self.global_std = global_feats.std(dim=0, keepdim=True) + 1e-8  # Shape: [1, 2]

        # Compute mean and std for object features, considering only valid objects
        valid_objs = obj_feats * obj_mask.unsqueeze(-1)  # Shape: [N, 18, 4]
        n_valid = obj_mask.sum(dim=1, keepdim=True).clamp(min=1)  # Shape: [N, 1]
        self.per_obj_mean = valid_objs.sum(dim=1).mean(dim=0, keepdim=True)  # Shape: [1, 4]
        self.per_obj_std = valid_objs.std(dim=1).mean(dim=0, keepdim=True) + 1e-8  # Shape: [1, 4]
        return self

    def transform(self, X):
        global_feats, obj_feats, obj_mask = self._raw_reshape(X)

        # Normalize global features
        global_feats = (global_feats - self.global_mean) / self.global_std  # Shape: [N, 2]

        # Normalize object features
        obj_feats = (obj_feats - self.per_obj_mean.view(1, 1, -1)) / self.per_obj_std.view(1, 1, -1)  # Shape: [N, 18, 4]

        return global_feats, obj_feats, obj_mask

    def make_loader_cfg(self):
        return None

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.global_dim = 2
        self.obj_dim = 4
        self.max_objs = 18
        self.hidden_dim = 128

        # Object encoder (process each object's features)
        self.obj_encoder = nn.Sequential(
            nn.Linear(self.obj_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Attention mechanism for objects
        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # Final classifier
        total_dim = 32 + 32  # global encoded dim + object aggregated dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, *data):
        global_feats, obj_feats, obj_mask = data

        # Encode global features
        global_encoded = self.global_encoder(global_feats)  # Shape: [N, 32]

        # Encode object features
        batch_size = obj_feats.shape[0]
        obj_encoded = self.obj_encoder(obj_feats.view(-1, self.obj_dim))  # Shape: [N*18, 32]
        obj_encoded = obj_encoded.view(batch_size, self.max_objs, -1)  # Shape: [N, 18, 32]

        # Apply attention to objects
        attn_weights = self.attention(obj_encoded)  # Shape: [N, 18, 1]
        attn_weights = attn_weights.masked_fill(obj_mask.unsqueeze(-1) == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)  # Shape: [N, 18, 1]
        obj_aggregated = torch.sum(obj_encoded * attn_weights, dim=1)  # Shape: [N, 32]

        # Combine global and object features
        combined = torch.cat([global_encoded, obj_aggregated], dim=-1)  # Shape: [N, 64]

        # Final classification
        logits = self.classifier(combined)  # Shape: [N, 1]
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    patience = 10
    patience_counter = 0

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    val_auc = []

    for epoch in range(epochs):
        # Training
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for batch in train_loader:
            data, labels = batch
            global_feats, obj_feats, obj_mask = [d.to(device) for d in data]
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(global_feats, obj_feats, obj_mask).squeeze(-1)  # Shape: [batch_size]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            tr_correct += (preds == labels).sum().item()
            tr_total += labels.size(0)

        avg_tr_loss = tr_loss / len(train_loader)
        avg_tr_acc = tr_correct / tr_total

        # Validation
        model.eval()
        va_loss = 0.0
        va_correct = 0
        va_total = 0
        va_preds = []
        va_labels = []

        with torch.no_grad():
            for batch in val_loader:
                data, labels = batch
                global_feats, obj_feats, obj_mask = [d.to(device) for d in data]
                labels = labels.to(device).float()

                outputs = model(global_feats, obj_feats, obj_mask).squeeze(-1)
                loss = criterion(outputs, labels)

                va_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                va_correct += (preds == labels).sum().item()
                va_total += labels.size(0)

                va_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                va_labels.extend(labels.cpu().numpy())

        avg_va_loss = va_loss / len(val_loader)
        avg_va_acc = va_correct / va_total
        auc_score = roc_auc_score(va_labels, va_preds)

        train_loss.append(avg_tr_loss)
        val_loss.append(avg_va_loss)
        train_acc.append(avg_tr_acc)
        val_acc.append(avg_va_acc)
        val_auc.append(auc_score)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_tr_loss:.4f} | Val Loss: {avg_va_loss:.4f} | "
              f"Train Acc: {avg_tr_acc:.4f} | Val Acc: {avg_va_acc:.4f} | Val AUC: {auc_score:.4f}")

        # Early stopping based on AUC
        if auc_score > best_val_auc:
            best_val_auc = auc_score
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model)
                break

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

