
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # Initialize stateful components for feature scaling
        self.scaler_global = StandardScaler()  # For global features (E_T_miss, phi_Et_miss)
        self.scaler_objects = StandardScaler()  # For object kinematic features (E, p_T, eta, phi)

    def _raw_reshape(self, X):
        # Reshape input data to separate global features and object features
        # Input shape: [N, 92]
        global_features = X[:, :2]  # Shape: [N, 2]
        object_features = X[:, 2:].reshape(-1, 18, 5)  # Shape: [N, 18, 5] (18 objects, 5 features each)
        return global_features, object_features

    def fit(self, X, y=None):
        # Extract statistics for scaling
        global_features, object_features = self._raw_reshape(X)

        # Fit scaler for global features
        self.scaler_global.fit(global_features)

        # Fit scaler for object features (only non-zero padded data)
        mask = object_features[:, :, 0] != 0  # Mask for non-padded objects (obj_id != 0)
        valid_features = object_features[mask, 1:]  # Shape: [num_valid_objects, 4]
        if valid_features.shape[0] > 0:  # Ensure there are valid features to fit
            self.scaler_objects.fit(valid_features)
        return self

    def transform(self, X):
        # Apply preprocessing logic
        global_features, object_features = self._raw_reshape(X)  # Shapes: [N, 2], [N, 18, 5]

        # Scale global features
        global_scaled = torch.tensor(self.scaler_global.transform(global_features), 
                                     dtype=torch.float32)  # Shape: [N, 2]

        # Scale object features (only for non-padded objects)
        object_scaled = object_features.clone()  # Shape: [N, 18, 5]
        mask = object_features[:, :, 0] != 0  # Shape: [N, 18]
        valid_features = object_features[mask, 1:]  # Shape: [num_valid, 4]
        if valid_features.shape[0] > 0:
            scaled_valid = torch.tensor(self.scaler_objects.transform(valid_features), 
                                        dtype=torch.float32)  # Shape: [num_valid, 4]
            object_scaled[mask, 1:] = scaled_valid

        # Compute additional features (e.g., number of objects per event)
        num_objects = torch.sum(mask, dim=1).unsqueeze(-1)  # Shape: [N, 1]

        # Concatenate global features with computed features
        global_enriched = torch.cat([global_scaled, num_objects], dim=1)  # Shape: [N, 3]

        return global_enriched, object_scaled  # Return tuple of tensors

    def make_loader_cfg(self):
        # Custom loader configuration
        return {
            "loader_class": "torch.utils.data.DataLoader",
            "batch_size": 256,
            "shuffle": False,
            "num_workers": 0
        }

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Unpack sample object (global features and object features)
        sample_global, sample_objects = sample_object
        global_input_dim = sample_global.shape[-1]  # Should be 3 (E_T_miss, phi, num_objects)
        object_feature_dim = sample_objects.shape[-1] - 1  # Should be 4 (E, p_T, eta, phi)
        max_objects = sample_objects.shape[-2]  # Should be 18

        # Define layers for processing object features
        self.object_mlp = nn.Sequential(
            nn.Linear(object_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )  # Output per object: 16

        # Attention mechanism for aggregating object features
        self.attention = nn.Linear(16, 1)  # Compute attention scores

        # Define layers for global features
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )  # Output for global: 8

        # Final classifier combining object and global features
        combined_dim = 16 + 8  # Object aggregated (16) + global (8)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, *data):
        # Unpack input data
        global_features, object_features = data  # Shapes: [batch, 3], [batch, 18, 5]

        # Process object features
        obj_mask = object_features[:, :, 0] != 0  # Shape: [batch, 18]
        obj_feats = object_features[:, :, 1:]  # Shape: [batch, 18, 4]
        obj_processed = self.object_mlp(obj_feats)  # Shape: [batch, 18, 16]

        # Apply attention to aggregate object features
        attn_scores = self.attention(obj_processed)  # Shape: [batch, 18, 1]
        attn_weights = torch.softmax(attn_scores + (~obj_mask.unsqueeze(-1) * -1e9), dim=1)  # Mask padded objects
        obj_aggregated = torch.sum(obj_processed * attn_weights, dim=1)  # Shape: [batch, 16]

        # Process global features
        global_processed = self.global_mlp(global_features)  # Shape: [batch, 8]

        # Combine features for final classification
        combined = torch.cat([obj_aggregated, global_processed], dim=1)  # Shape: [batch, 24]
        output = self.classifier(combined)  # Shape: [batch, 1]
        return output

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=1e-3)

    best_val_auc = -float('inf')
    early_stopping_patience = 10
    early_stopping_counter = 0
    best_model_state = None

    train_loss_history = []
    val_loss_history = []
    train_auc_history = []
    val_auc_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            data, targets = batch
            global_feats, obj_feats = data
            global_feats = global_feats.to(device)
            obj_feats = obj_feats.to(device)
            targets = targets.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(global_feats, obj_feats)  # Shape: [batch, 1]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy().flatten())
            train_labels.extend(targets.cpu().numpy().flatten())

        epoch_train_loss = running_loss / len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)
        train_loss_history.append(epoch_train_loss)
        train_auc_history.append(train_auc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                data, targets = batch
                global_feats, obj_feats = data
                global_feats = global_feats.to(device)
                obj_feats = obj_feats.to(device)
                targets = targets.to(device).float().view(-1, 1)

                outputs = model(global_feats, obj_feats)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_labels.extend(targets.cpu().numpy().flatten())

        epoch_val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_loss_history.append(epoch_val_loss)
        val_auc_history.append(val_auc)

        # Scheduler step based on validation AUC
        scheduler.step(val_auc)

        # Early stopping based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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

