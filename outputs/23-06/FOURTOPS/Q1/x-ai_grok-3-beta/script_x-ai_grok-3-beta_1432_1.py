
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
import numpy as np

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.per_obj_mean = None
        self.per_obj_std = None

    def _raw_reshape(self, X):
        # Reshape the input tensor to separate global and per-object features
        global_features = X[:, :2]  # Shape: [N, 2]
        obj_features = X[:, 2:].reshape(X.shape[0], 18, 5)  # Shape: [N, 18, 5]
        return global_features, obj_features

    def fit(self, X, y=None):
        global_features, obj_features = self._raw_reshape(X)

        # Compute mean and std for global features (missing ET magnitude and phi)
        self.global_mean = torch.mean(global_features, dim=0)  # Shape: [2]
        self.global_std = torch.std(global_features, dim=0) + 1e-6  # Shape: [2]

        # Compute mean and std for per-object features
        obj_mask = obj_features[:, :, 0] != 0  # Shape: [N, 18]
        obj_features_masked = obj_features[obj_mask]  # Shape: [N_valid, 5]
        if obj_features_masked.shape[0] > 0:
            self.per_obj_mean = torch.mean(obj_features_masked, dim=0)  # Shape: [5]
            self.per_obj_std = torch.std(obj_features_masked, dim=0) + 1e-6  # Shape: [5]
        else:
            self.per_obj_mean = torch.zeros(5)
            self.per_obj_std = torch.ones(5)
        return self

    def transform(self, X):
        global_features, obj_features = self._raw_reshape(X)  # Shape: [N, 2], [N, 18, 5]

        # Normalize global features
        global_features = (global_features - self.global_mean) / self.global_std  # Shape: [N, 2]

        # Normalize per-object features (only where object exists)
        obj_mask = obj_features[:, :, 0] != 0  # Shape: [N, 18]
        obj_features_expanded = obj_features.clone()  # Shape: [N, 18, 5]
        obj_features_expanded[obj_mask] = (obj_features[obj_mask] - self.per_obj_mean) / self.per_obj_std

        return global_features, obj_features_expanded

    def make_loader_cfg(self):
        return None

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.global_fc = nn.Linear(2, 16)  # For missing ET features
        self.obj_fc = nn.Linear(5, 16)  # For per-object features
        self.obj_attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        self.combined_fc1 = nn.Linear(16 + 16, 64)
        self.combined_fc2 = nn.Linear(64, 32)
        self.combined_fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(32)

    def forward(self, *data):
        global_features, obj_features = data  # global: [N, 2], obj: [N, 18, 5]

        # Process global features
        global_emb = F.relu(self.global_fc(global_features))  # Shape: [N, 16]

        # Process per-object features
        obj_emb = F.relu(self.obj_fc(obj_features))  # Shape: [N, 18, 16]

        # Apply attention to object embeddings
        attn_mask = (obj_features[:, :, 0] == 0)  # Shape: [N, 18], True for padding
        attn_output, _ = self.obj_attention(obj_emb, obj_emb, obj_emb, key_padding_mask=attn_mask)  # Shape: [N, 18, 16]
        obj_pooled = torch.mean(attn_output, dim=1)  # Shape: [N, 16]

        # Combine global and object features
        combined = torch.cat([global_emb, obj_pooled], dim=1)  # Shape: [N, 32]
        x = F.relu(self.batch_norm1(self.combined_fc1(combined)))  # Shape: [N, 64]
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.combined_fc2(x)))  # Shape: [N, 32]
        x = self.dropout(x)
        x = self.combined_fc3(x)  # Shape: [N, 1]
        return x

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_auc = -float('inf')
    patience = 8
    early_stop_counter = 0

    train_loss_history = []
    val_loss_history = []
    train_auc_history = []
    val_auc_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        for batch in train_loader:
            data, labels = batch
            global_feat, obj_feat = data[0].to(device), data[1].to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(global_feat, obj_feat).squeeze()  # Shape: [N]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_auc = roc_auc_score(train_labels, train_preds)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                data, labels = batch
                global_feat, obj_feat = data[0].to(device), data[1].to(device)
                labels = labels.to(device).float()
                outputs = model(global_feat, obj_feat).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_auc = roc_auc_score(val_labels, val_preds)

        scheduler.step(epoch_val_auc)

        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)
        train_auc_history.append(epoch_train_auc)
        val_auc_history.append(epoch_val_auc)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Train AUC: {epoch_train_auc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val AUC: {epoch_val_auc:.4f}')

        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break

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

