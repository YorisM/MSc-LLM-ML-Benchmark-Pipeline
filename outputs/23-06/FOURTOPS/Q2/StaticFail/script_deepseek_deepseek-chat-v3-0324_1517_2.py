
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
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None

    def _raw_reshape(self, X):
        return X

    def _get_objects(self, X):
        objects = []
        met = X[..., :2]  # Missing ET features [2]
        for i in range(2, 87, 5):
            obj_slice = X[..., i:i+5]
            mask = (obj_slice[..., 0] != 0)  # obj_id != 0
            objects.append(obj_slice)
        stacked = torch.stack(objects, dim=1)  # [batch, 17, 5]
        return met, stacked

    def _calculate_pairwise_features(self, objects):
        batch_size, num_objects, _ = objects.shape

        # Get pt, eta, phi for all objects [batch, num_objects]
        pt = objects[..., 2]
        eta = objects[..., 3]
        phi = objects[..., 4]

        # Calculate delta R between all pairs
        eta1 = eta.unsqueeze(2)  # [batch, num_objects, 1]
        eta2 = eta.unsqueeze(1)  # [batch, 1, num_objects]
        phi1 = phi.unsqueeze(2)
        phi2 = phi.unsqueeze(1)

        delta_eta = eta1 - eta2
        delta_phi = phi1 - phi2
        delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))  # Handle periodicity

        delta_r = torch.sqrt(delta_eta**2 + delta_phi**2)  # [batch, num_objects, num_objects]

        # Calculate invariant mass between all pairs
        e = objects[..., 1]  # Energy
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)

        e1 = e.unsqueeze(2)
        e2 = e.unsqueeze(1)
        px1 = px.unsqueeze(2)
        px2 = px.unsqueeze(1)
        py1 = py.unsqueeze(2)
        py2 = py.unsqueeze(1)
        pz1 = pz.unsqueeze(2)
        pz2 = pz.unsqueeze(1)

        invariant_mass = torch.sqrt(
            (e1 + e2)**2 - 
            (px1 + px2)**2 - 
            (py1 + py2)**2 - 
            (pz1 + pz2)**2
        )  # [batch, num_objects, num_objects]

        return delta_r, invariant_mass

    def fit(self, X, y=None):
        met, objects = self._get_objects(X)

        # Calculate mean and std for normalization
        flat_objects = objects[objects[..., 0] != 0]  # Filter out padding
        self.global_mean = torch.mean(flat_objects[..., 1:], dim=0)  # Skip obj_id
        self.global_std = torch.std(flat_objects[..., 1:], dim=0)
        self.global_std[self.global_std == 0] = 1.0  # Avoid division by zero

        return self

    def transform(self, X):
        batch_size = X.shape[0]

        # Get MET and objects
        met, objects = self._get_objects(X)  # met: [batch, 2], objects: [batch, 17, 5]

        # Normalize object features (skip obj_id)
        objects_norm = objects.clone()
        objects_norm[..., 1:] = (objects_norm[..., 1:] - self.global_mean) / self.global_std

        # Calculate pairwise features
        delta_r, invariant_mass = self._calculate_pairwise_features(objects_norm)

        # Flatten pairwise features (upper triangular without diagonal)
        triu_mask = torch.triu(torch.ones(17, 17), diagonal=1).bool().to(X.device)
        delta_r_flat = delta_r[:, triu_mask]  # [batch, 136]
        invariant_mass_flat = invariant_mass[:, triu_mask]  # [batch, 136]

        # Get object-level features (mean, std, max, min)
        pt = objects_norm[..., 2]  # [batch, 17]
        pt_mean = torch.mean(pt, dim=1, keepdim=True)  # [batch, 1]
        pt_std = torch.std(pt, dim=1, keepdim=True)
        pt_max = torch.max(pt, dim=1, keepdim=True)[0]
        pt_min = torch.min(pt, dim=1, keepdim=True)[0]
        pt_features = torch.cat([pt_mean, pt_std, pt_max, pt_min], dim=1)

        # Combine all features
        combined = torch.cat([
            met,  # [batch, 2]
            objects_norm.reshape(batch_size, -1),  # [batch, 85]
            delta_r_flat,  # [batch, 136]
            invariant_mass_flat,  # [batch, 136]
            pt_features  # [batch, 4]
        ], dim=1)  # Total: 2 + 85 + 136 + 136 + 4 = 363 features

        return combined.float()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_size = sample_object.shape[-1]  # 363

        self.main = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.main(x)

EPOCHS = 50

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_auc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        # Training loop
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.float().to(device)
            optimizer.zero_grad()

            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).long()
            correct += (predicted == batch_y.long()).sum().item()
            total += batch_y.size(0)

        train_loss = epoch_loss / total
        train_acc = correct / total

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.float().to(device)

                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).long()
                correct += (predicted == batch_y.long()).sum().item()
                total += batch_y.size(0)

                all_outputs.append(outputs.cpu())
                all_targets.append(batch_y.cpu())

        val_loss = val_loss / total
        val_acc = correct / total

        # Calculate AUC
        all_outputs = torch.cat(all_outputs).numpy()
        all_targets = torch.cat(all_targets).numpy()
        auc = roc_auc_score(all_targets, all_outputs)

        # Update scheduler
        scheduler.step(auc)

        # Early stopping
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_model.pt')

        if epoch > 10 and auc < best_auc - 0.01:  # Tolerate 1% drop
            print(f'Early stopping at epoch {epoch}')
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}: Loss: {train_loss:.4f} / {val_loss:.4f}, '
              f'Acc: {train_acc:.4f} / {val_acc:.4f}, AUC: {auc:.4f}')

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))

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

