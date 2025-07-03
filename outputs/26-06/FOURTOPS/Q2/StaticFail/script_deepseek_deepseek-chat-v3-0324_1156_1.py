
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
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler_mean = None
        self.scaler_std = None

    def _raw_reshape(self, X):
        return X

    def _extract_features(self, X):
        # Global features (2): E_T_miss, phi_Et_miss
        global_features = X[:, :2]

        # Object features (18 objects max, 5 features each)
        object_features = X[:, 2:].reshape(X.shape[0], -1, 5)

        # Remove padding (obj_id = 0)
        mask = object_features[:, :, 0] != 0  # [batch, max_objects]
        masked_objects = object_features * mask.unsqueeze(-1).float()

        # Compute mean values per object feature
        n_objects = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
        means = masked_objects.sum(dim=1) / n_objects  # [batch, 5]

        # Compute pairwise features
        batch_size, max_objects, _ = object_features.shape

        # Prepare object pairs
        obj_pairs = []
        for i in range(max_objects):
            for j in range(i+1, max_objects):
                # Only consider real objects (non-padding)
                pair_mask = mask[:, i] & mask[:, j]
                if pair_mask.any():
                    obj1 = object_features[:, i]
                    obj2 = object_features[:, j]

                    # Delta R
                    deta = obj1[:, 3] - obj2[:, 3]
                    dphi = torch.remainder(obj1[:, 4] - obj2[:, 4] + np.pi, 2*np.pi) - np.pi
                    delta_r = torch.sqrt(deta**2 + dphi**2)

                    # Invariant mass
                    px1 = obj1[:, 2] * torch.cos(obj1[:, 4])
                    py1 = obj1[:, 2] * torch.sin(obj1[:, 4])
                    pz1 = obj1[:, 2] * torch.sinh(obj1[:, 3])

                    px2 = obj2[:, 2] * torch.cos(obj2[:, 4])
                    py2 = obj2[:, 2] * torch.sin(obj2[:, 4])
                    pz2 = obj2[:, 2] * torch.sinh(obj2[:, 3])

                    p1 = torch.sqrt(px1**2 + py1**2 + pz1**2)
                    p2 = torch.sqrt(px2**2 + py2**2 + pz2**2)

                    e1 = obj1[:, 1]
                    e2 = obj2[:, 1]
                    m = torch.sqrt(2*(e1*e2 - (px1*px2 + py1*py2 + pz1*pz2)))

                    pair_features = torch.stack([delta_r, m], dim=1)
                    obj_pairs.append(pair_features)

        if obj_pairs:
            pairwise_features = torch.cat(obj_pairs, dim=1)  # [batch, n_pairs*2]
            pairwise_means = pairwise_features.mean(dim=1, keepdim=True)
        else:
            pairwise_features = torch.zeros(X.shape[0], 0)
            pairwise_means = torch.zeros(X.shape[0], 1)

        # Combine all features
        features = torch.cat([
            global_features,
            means,  # mean object features
            pairwise_means,  # mean pairwise features
        ], dim=1)

        return features

    def fit(self, X, y=None):
        features = self._extract_features(X)
        self.scaler_mean = features.mean(dim=0)
        self.scaler_std = features.std(dim=0) + 1e-8
        return self

    def transform(self, X):
        features = self._extract_features(X)
        return (features - self.scaler_mean) / self.scaler_std

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[-1]

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=4
        )

        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Add sequence dimension for transformer
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, features]

        x = self.transformer(x)  # [batch, 1, features]
        x = x.squeeze(1)  # [batch, features]
        return self.head(x)

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    best_model = None

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for X, y in train_loader:
            X = X.to(device)
            y = y.float().to(device)

            optimizer.zero_grad()
            outputs = model(X).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.extend(y.detach().cpu().numpy())

        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_targets, train_preds)
        train_losses.append(train_loss)
        train_accs.append(train_auc)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.float().to(device)

                outputs = model(X).squeeze()
                loss = criterion(outputs, y)

                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_losses.append(val_loss)
        val_accs.append(val_auc)

        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()

        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')

        # Early stopping if no improvement for 5 epochs
        if epoch >= 5 and val_auc < np.max(val_accs[-5:]):
            print('Early stopping')
            break

    if best_model is not None:
        model.load_state_dict(best_model)

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

