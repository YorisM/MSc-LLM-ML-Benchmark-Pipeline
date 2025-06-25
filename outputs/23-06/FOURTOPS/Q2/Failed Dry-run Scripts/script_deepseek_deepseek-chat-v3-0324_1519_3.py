
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.max_objects = 18
        self.obj_features = 5  # obj_id, E, pT, eta, phi
        self.global_features = 2  # Et_miss, phi_Et_miss
        self.total_features = self.global_features + self.max_objects * self.obj_features

    def _extract_objects(self, X):
        batch_size = X.shape[0]
        objects = X[:, 2:].reshape(batch_size, self.max_objects, self.obj_features)
        valid_objects = (objects[..., 0] != 0)  # obj_id != 0 indicates valid object
        return objects, valid_objects

    def _compute_pairwise_features(self, objects, valid_objects):
        batch_size = objects.shape[0]
        all_mass = []
        all_delta_r = []

        for i in range(self.max_objects):
            for j in range(i+1, self.max_objects):
                mask = valid_objects[:, i] & valid_objects[:, j]

                # Get four-vectors (E, pT, eta, phi)
                obj_i = objects[:, i]
                obj_j = objects[:, j]

                # Compute invariant mass
                px_i = obj_i[:, 2] * torch.cos(obj_i[:, 4])  # pT*cos(phi)
                py_i = obj_i[:, 2] * torch.sin(obj_i[:, 4])  # pT*sin(phi)
                pz_i = obj_i[:, 2] * torch.sinh(obj_i[:, 3]) # pT*sinh(eta)
                ei = obj_i[:, 1]  # E

                px_j = obj_j[:, 2] * torch.cos(obj_j[:, 4])
                py_j = obj_j[:, 2] * torch.sin(obj_j[:, 4])
                pz_j = obj_j[:, 2] * torch.sinh(obj_j[:, 3])
                ej = obj_j[:, 1]

                px_sum = px_i + px_j
                py_sum = py_i + py_j
                pz_sum = pz_i + pz_j
                e_sum = ei + ej

                mass = torch.sqrt(e_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2))

                # Compute delta R
                deta = obj_i[:, 3] - obj_j[:, 3]
                dphi = torch.atan2(torch.sin(obj_i[:, 4]-obj_j[:, 4]), 
                                  torch.cos(obj_i[:, 4]-obj_j[:, 4]))
                delta_r = torch.sqrt(deta**2 + dphi**2)

                # Apply mask
                mass[~mask] = 0
                delta_r[~mask] = 0

                all_mass.append(mass.unsqueeze(1))
                all_delta_r.append(delta_r.unsqueeze(1))

        pairwise_mass = torch.cat(all_mass, dim=1)  # [batch, n_pairs]
        pairwise_dr = torch.cat(all_delta_r, dim=1) # [batch, n_pairs]

        return pairwise_mass, pairwise_dr

    def _combine_features(self, X, objects, valid_objects, pairwise_mass, pairwise_dr):
        batch_size = X.shape[0]

        # Global features (Et miss, phi Et miss)
        global_feats = X[:, :2]

        # Flatten objects and keep only valid ones
        flat_objects = objects.reshape(batch_size, -1)

        # Normalize features
        norm_objects = flat_objects.clone()
        for i in range(self.max_objects):
            start = i * self.obj_features + 2  # Skip obj id
            norm_objects[:, start] = torch.log1p(norm_objects[:, start]/1000)  # E in GeV
            norm_objects[:, start+1] = torch.log1p(norm_objects[:, start+1]/1000)  # pT in GeV

        # Create mask for real objects
        obj_mask = valid_objects.float().unsqueeze(-1)

        # Combine all features
        processed = torch.cat([
            global_feats,
            norm_objects,
            pairwise_mass,
            pairwise_dr
        ], dim=1)

        return processed

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if torch.is_tensor(X):
            X = X.clone()
        else:
            X = torch.tensor(X, dtype=torch.float32)

        objects, valid_objects = self._extract_objects(X)
        pairwise_mass, pairwise_dr = self._compute_pairwise_features(objects, valid_objects)
        processed = self._combine_features(X, objects, valid_objects, pairwise_mass, pairwise_dr)
        return processed

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Sample object shape will be [batch_size, n_features]
        n_features = sample_object.shape[1] if len(sample_object.shape) > 1 else sample_object.shape[0]

        self.model = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))

def make_model(example_object):
    return BinaryClassifier(example_object)

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=False)

    best_auc = 0
    best_model = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0
        all_targets_train = []
        all_outputs_train = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.float().to(device), targets.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_train += (predicted == targets).sum().item()
            total_train += targets.size(0)

            all_targets_train.extend(targets.cpu().numpy())
            all_outputs_train.extend(outputs.detach().cpu().numpy())

        train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_auc = roc_auc_score(all_targets_train, all_outputs_train)

        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        all_targets_val = []
        all_outputs_val = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.float().to(device), targets.float().to(device)
                outputs = model(inputs).squeeze()

                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                correct_val += (predicted == targets).sum().item()
                total_val += targets.size(0)

                all_targets_val.extend(targets.cpu().numpy())
                all_outputs_val.extend(outputs.cpu().numpy())

        val_loss = epoch_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_auc = roc_auc_score(all_targets_val, all_outputs_val)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        print("-" * 50)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()

    # Load best model
    model.load_state_dict(best_model)
    return model, train_losses, val_losses, train_accs, val_accs

EPOCHS = 30

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

