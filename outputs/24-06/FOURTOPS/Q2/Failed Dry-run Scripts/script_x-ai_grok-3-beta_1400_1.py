
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
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler_global = StandardScaler()
        self.scaler_obj = StandardScaler()
        self.max_objects = 18
        self.obj_features = 4  # E, p_T, eta, phi (excluding obj identifier)

    def _raw_reshape(self, X):           
        return X  # Shape: [N, 92]

    def fit(self, X, y=None):
        # Extract global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :2]  # Shape: [N, 2]
        self.scaler_global.fit(global_features)

        # Extract object features for scaling (skip obj identifier)
        obj_data = []
        for i in range(self.max_objects):
            start_idx = 2 + i * 5 + 1  # Skip obj identifier
            end_idx = start_idx + self.obj_features
            obj_features = X[:, start_idx:end_idx]  # Shape: [N, 4]
            obj_data.append(obj_features)
        obj_data = np.stack(obj_data, axis=1)  # Shape: [N, max_objects, 4]
        obj_data_reshaped = obj_data.reshape(-1, self.obj_features)  # Shape: [N*max_objects, 4]
        self.scaler_obj.fit(obj_data_reshaped)
        return self

    def transform(self, X):
        N = X.shape[0]
        # Transform global features
        global_features = X[:, :2]  # Shape: [N, 2]
        global_features_scaled = self.scaler_global.transform(global_features)  # Shape: [N, 2]

        # Transform object features
        obj_data_scaled = []
        for i in range(self.max_objects):
            start_idx = 2 + i * 5 + 1  # Skip obj identifier
            end_idx = start_idx + self.obj_features
            obj_features = X[:, start_idx:end_idx]  # Shape: [N, 4]
            obj_data_scaled.append(obj_features)
        obj_data = np.stack(obj_data_scaled, axis=1)  # Shape: [N, max_objects, 4]
        obj_data_reshaped = obj_data.reshape(-1, self.obj_features)  # Shape: [N*max_objects, 4]
        obj_data_scaled = self.scaler_obj.transform(obj_data_reshaped)  # Shape: [N*max_objects, 4]
        obj_data_scaled = obj_data_scaled.reshape(N, self.max_objects, self.obj_features)  # Shape: [N, 18, 4]

        # Return as torch tensors
        global_tensor = torch.tensor(global_features_scaled, dtype=torch.float32)  # Shape: [N, 2]
        obj_tensor = torch.tensor(obj_data_scaled, dtype=torch.float32)  # Shape: [N, 18, 4]
        return global_tensor, obj_tensor

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
        self.max_objects = 18
        self.obj_features = 4
        self.global_features = 2

        # Object feature processing with attention
        self.obj_fc1 = nn.Linear(self.obj_features, 64)
        self.obj_fc2 = nn.Linear(64, 32)
        self.attention = nn.Linear(32, 1)

        # Global feature processing
        self.global_fc1 = nn.Linear(self.global_features, 16)
        self.global_fc2 = nn.Linear(16, 8)

        # Combined features processing
        self.combined_fc1 = nn.Linear(32 * self.max_objects + 8, 256)
        self.combined_fc2 = nn.Linear(256, 128)
        self.combined_fc3 = nn.Linear(128, 64)
        self.combined_fc4 = nn.Linear(64, 1)

    def forward(self, *data):
        global_data, obj_data = data  # global_data: [B, 2], obj_data: [B, 18, 4]
        batch_size = global_data.shape[0]

        # Process object features with attention
        obj_x = F.relu(self.obj_fc1(obj_data))  # Shape: [B, 18, 64]
        obj_x = F.relu(self.obj_fc2(obj_x))  # Shape: [B, 18, 32]
        attn_weights = self.attention(obj_x)  # Shape: [B, 18, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # Shape: [B, 18, 1]
        obj_context = torch.sum(attn_weights * obj_x, dim=1)  # Shape: [B, 32]
        obj_context = obj_context.view(batch_size, -1)  # Shape: [B, 32]

        # Process global features
        global_x = F.relu(self.global_fc1(global_data))  # Shape: [B, 16]
        global_x = F.relu(self.global_fc2(global_x))  # Shape: [B, 8]

        # Combine features
        x = torch.cat([obj_context, global_x], dim=-1)  # Shape: [B, 32 + 8]
        x = F.relu(self.combined_fc1(x))  # Shape: [B, 256]
        x = F.dropout(x, 0.3, training=self.training)
        x = F.relu(self.combined_fc2(x))  # Shape: [B, 128]
        x = F.dropout(x, 0.2, training=self.training)
        x = F.relu(self.combined_fc3(x))  # Shape: [B, 64]
        x = self.combined_fc4(x)  # Shape: [B, 1]
        return x

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    patience = 10
    epochs_no_improve = 0

    train_loss_history = []
    val_loss_history = []
    train_auc_history = []
    val_auc_history = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        for batch in train_loader:
            (global_data, obj_data), labels = batch
            global_data = global_data.to(device)
            obj_data = obj_data.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(global_data, obj_data).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                (global_data, obj_data), labels = batch
                global_data = global_data.to(device)
                obj_data = obj_data.to(device)
                labels = labels.to(device).float()

                outputs = model(global_data, obj_data).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)

        scheduler.step(val_auc)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_auc_history.append(train_auc)
        val_auc_history.append(val_auc)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

    return (model, train_loss_history, val_loss_history, 
            train_auc_history, val_auc_history)

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

