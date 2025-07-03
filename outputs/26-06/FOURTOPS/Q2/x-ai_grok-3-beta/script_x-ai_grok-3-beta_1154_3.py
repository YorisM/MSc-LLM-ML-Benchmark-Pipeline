
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
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import math

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    # DATA SPECIFICS
    # Total flat length per event (X_train & X_val): 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    # Indices  7-11 : object 2  ->  obj_2, E_2 , p_T_2 , eta_2 , phi_2
    # ...
    # Indices 88-92 : object 18 ->  obj_18, E_18 , p_T_18 , eta_18 , phi_18
    # Global features       = 2
    # Per-object slice size = 5
    # Max objects encoded   = 18

    def __init__(self):
        self.global_mean = 0.0
        self.global_std = 1.0
        self.object_means = np.zeros((18, 4))  # Means for E, p_T, eta, phi for each object
        self.object_stds = np.ones((18, 4))    # Stds for E, p_T, eta, phi for each object
        self.max_objects = 18
        self.per_obj_size = 5

    def _raw_reshape(self, X):
        # Reshape the flat tensor to separate global and object features
        # X shape: [N, 92]
        N = X.shape[0]
        global_feats = X[:, :2]  # Shape: [N, 2]
        obj_feats = X[:, 2:].reshape(N, self.max_objects, self.per_obj_size)  # Shape: [N, 18, 5]
        return global_feats, obj_feats

    def make_loader_cfg(self):
        return None

    def fit(self, X, y=None):
        # Extract statistics for normalization
        global_feats, obj_feats = self._raw_reshape(X)

        # Compute mean and std for global features (E_T_miss and phi_Et_miss)
        self.global_mean = global_feats.mean(dim=0)  # Shape: [2]
        self.global_std = global_feats.std(dim=0) + 1e-6  # Shape: [2]

        # Compute mean and std for object features (only for E, p_T, eta, phi, ignoring obj_id)
        N = obj_feats.shape[0]
        for i in range(self.max_objects):
            # Extract kinematic features (indices 1:5, i.e., E, p_T, eta, phi)
            feats = obj_feats[:, i, 1:]  # Shape: [N, 4]
            mask = feats[:, 0] != 0  # Assuming E=0 indicates padding
            if mask.sum() > 0:
                self.object_means[i] = feats[mask].mean(dim=0).numpy()  # Shape: [4]
                self.object_stds[i] = feats[mask].std(dim=0).numpy() + 1e-6  # Shape: [4]

        return self

    def transform(self, X):
        # Apply normalization and compute pairwise features
        global_feats, obj_feats = self._raw_reshape(X)
        N = X.shape[0]

        # Normalize global features
        global_feats = (global_feats - self.global_mean) / self.global_std  # Shape: [N, 2]

        # Normalize object features
        norm_obj_feats = torch.zeros_like(obj_feats)  # Shape: [N, 18, 5]
        for i in range(self.max_objects):
            norm_obj_feats[:, i, 0] = obj_feats[:, i, 0]  # Keep obj_id as is
            norm_obj_feats[:, i, 1:] = (obj_feats[:, i, 1:] - torch.tensor(self.object_means[i])) / torch.tensor(self.object_stds[i])  # Shape: [N, 4]

        # Compute pairwise features (invariant mass and delta R)
        pairwise_masses = torch.zeros(N, self.max_objects, self.max_objects)  # Shape: [N, 18, 18]
        pairwise_dR = torch.zeros(N, self.max_objects, self.max_objects)  # Shape: [N, 18, 18]

        for i in range(self.max_objects):
            for j in range(i + 1, self.max_objects):
                E_i = obj_feats[:, i, 1]  # Energy of object i
                E_j = obj_feats[:, j, 1]  # Energy of object j
                pT_i = obj_feats[:, i, 2]
                pT_j = obj_feats[:, j, 2]
                eta_i = obj_feats[:, i, 3]
                eta_j = obj_feats[:, j, 3]
                phi_i = obj_feats[:, i, 4]
                phi_j = obj_feats[:, j, 4]

                # Compute invariant mass (simplified, assuming massless particles for computational simplicity)
                mass_ij = torch.sqrt(2 * pT_i * pT_j * (torch.cosh(eta_i - eta_j) - torch.cos(phi_i - phi_j)))  # Shape: [N]
                pairwise_masses[:, i, j] = mass_ij
                pairwise_masses[:, j, i] = mass_ij

                # Compute delta R
                d_eta = eta_i - eta_j
                d_phi = phi_i - phi_j
                dR_ij = torch.sqrt(d_eta**2 + d_phi**2)  # Shape: [N]
                pairwise_dR[:, i, j] = dR_ij
                pairwise_dR[:, j, i] = dR_ij

        return global_feats, norm_obj_feats, pairwise_masses, pairwise_dR
        # Returns: global_feats [N, 2], norm_obj_feats [N, 18, 5], pairwise_masses [N, 18, 18], pairwise_dR [N, 18, 18]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Unpack sample object to get shapes
        global_feats, obj_feats, pairwise_masses, pairwise_dR = sample_object
        self.max_objects = obj_feats.shape[1]  # 18
        self.obj_feat_dim = obj_feats.shape[2] - 1  # 4 (E, p_T, eta, phi)

        # Embedding for object features
        self.obj_embed = nn.Sequential(
            nn.Linear(self.obj_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Transformer encoder for object features
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Global features processing
        self.global_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        # Pairwise features processing (invariant mass and delta R)
        self.pairwise_fc = nn.Sequential(
            nn.Linear(2, 32),  # 2 for mass and dR
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * self.max_objects + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, *data):
        global_feats, obj_feats, pairwise_masses, pairwise_dR = data
        N = global_feats.shape[0]

        # Process object features
        obj_kin = obj_feats[:, :, 1:]  # Shape: [N, 18, 4]
        obj_emb = self.obj_embed(obj_kin)  # Shape: [N, 18, 128]
        obj_emb = obj_emb.permute(1, 0, 2)  # Shape: [18, N, 128] for transformer
        obj_out = self.transformer_encoder(obj_emb)  # Shape: [18, N, 128]
        obj_out = obj_out.permute(1, 0, 2).reshape(N, -1)  # Shape: [N, 18*128]

        # Process global features
        global_out = self.global_fc(global_feats)  # Shape: [N, 64]

        # Concatenate outputs
        x = torch.cat([obj_out, global_out], dim=-1)  # Shape: [N, 18*128 + 64]

        # Output logits
        out = self.classifier(x)  # Shape: [N, 1]
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_auc = 0.0
    early_stopping_patience = 5
    early_stopping_counter = 0

    tr_loss_history = []
    va_loss_history = []
    tr_acc_history = []
    va_acc_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        tr_loss = 0.0
        tr_preds = []
        tr_labels = []
        for batch in train_loader:
            data, labels = batch
            data = [d.to(device) for d in data]
            labels = labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(*data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            tr_labels.append(labels.detach().cpu().numpy())

        tr_loss /= len(train_loader)
        tr_preds = np.concatenate(tr_preds)
        tr_labels = np.concatenate(tr_labels)
        tr_acc = accuracy_score(tr_labels, (tr_preds > 0.5).astype(int))
        tr_loss_history.append(tr_loss)
        tr_acc_history.append(tr_acc)

        # Validation phase
        model.eval()
        va_loss = 0.0
        va_preds = []
        va_labels = []
        with torch.no_grad():
            for batch in val_loader:
                data, labels = batch
                data = [d.to(device) for d in data]
                labels = labels.to(device).float().view(-1, 1)
                outputs = model(*data)
                loss = criterion(outputs, labels)
                va_loss += loss.item()
                va_preds.append(torch.sigmoid(outputs).cpu().numpy())
                va_labels.append(labels.cpu().numpy())

        va_loss /= len(val_loader)
        va_preds = np.concatenate(va_preds)
        va_labels = np.concatenate(va_labels)
        va_acc = accuracy_score(va_labels, (va_preds > 0.5).astype(int))
        va_auc = roc_auc_score(va_labels, va_preds)
        va_loss_history.append(va_loss)
        va_acc_history.append(va_acc)

        # Scheduler step based on validation AUC
        scheduler.step(va_auc)

        # Early stopping
        if va_auc > best_val_auc:
            best_val_auc = va_auc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return model, tr_loss_history, va_loss_history, tr_acc_history, va_acc_history

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

