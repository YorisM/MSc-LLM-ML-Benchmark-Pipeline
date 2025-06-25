
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
import math

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.max_objects = 18
        self.per_object_size = 5

    def _raw_reshape(self, X):
        # Reshape to separate global features and object-wise features
        batch_size = X.shape[0]
        global_features = X[:, :2]  # Shape: (batch_size, 2)
        object_features = X[:, 2:].view(batch_size, self.max_objects, self.per_object_size)  # Shape: (batch_size, 18, 5)
        return global_features, object_features

    def fit(self, X, y=None):
        global_features, object_features = self._raw_reshape(X)

        # Compute mean and std for global features
        self.global_mean = global_features.mean(dim=0, keepdim=True)  # Shape: (1, 2)
        self.global_std = global_features.std(dim=0, keepdim=True) + 1e-6  # Shape: (1, 2)
        return self

    def transform(self, X):
        global_features, object_features = self._raw_reshape(X)  # global: (batch_size, 2), object: (batch_size, 18, 5)

        # Normalize global features
        global_features = (global_features - self.global_mean) / self.global_std  # Shape: (batch_size, 2)

        # Compute pairwise features (invariant mass and angular distance)
        batch_size = X.shape[0]
        pairwise_masses = []
        pairwise_dR = []
        for i in range(self.max_objects):
            for j in range(i + 1, self.max_objects):
                # Extract kinematics for object i and j
                E_i = object_features[:, i, 1]  # Shape: (batch_size,)
                pt_i = object_features[:, i, 2]
                eta_i = object_features[:, i, 3]
                phi_i = object_features[:, i, 4]

                E_j = object_features[:, j, 1]
                pt_j = object_features[:, j, 2]
                eta_j = object_features[:, j, 3]
                phi_j = object_features[:, j, 4]

                # Compute invariant mass (simplified, assuming massless particles)
                mass_ij = torch.sqrt(2 * pt_i * pt_j * (torch.cosh(eta_i - eta_j) - torch.cos(phi_i - phi_j)))  # Shape: (batch_size,)
                mass_ij = mass_ij.unsqueeze(-1)  # Shape: (batch_size, 1)

                # Compute angular distance delta R
                d_eta = eta_i - eta_j
                d_phi = phi_i - phi_j
                d_phi = torch.where(d_phi > math.pi, d_phi - 2 * math.pi, d_phi)
                d_phi = torch.where(d_phi < -math.pi, d_phi + 2 * math.pi, d_phi)
                dR_ij = torch.sqrt(d_eta**2 + d_phi**2)  # Shape: (batch_size,)
                dR_ij = dR_ij.unsqueeze(-1)  # Shape: (batch_size, 1)

                pairwise_masses.append(mass_ij)
                pairwise_dR.append(dR_ij)

        # Concatenate pairwise features
        pairwise_masses = torch.cat(pairwise_masses, dim=-1)  # Shape: (batch_size, num_pairs)
        pairwise_dR = torch.cat(pairwise_dR, dim=-1)  # Shape: (batch_size, num_pairs)

        # Flatten object features for input to model
        object_features_flat = object_features.view(batch_size, -1)  # Shape: (batch_size, 18*5)

        # Combine all features
        combined_features = torch.cat([global_features, object_features_flat, pairwise_masses, pairwise_dR], dim=-1)  # Shape: (batch_size, 2 + 90 + num_pairs*2)
        return combined_features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def make_loader_cfg(self):
        return None

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[-1]  # Dynamically get input dimension from preprocessed data
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))  # Shape: (batch_size, 512)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))  # Shape: (batch_size, 256)
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))  # Shape: (batch_size, 128)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))  # Shape: (batch_size, 1)
        return x

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    best_val_auc = 0.0
    patience = 5
    counter = 0
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    val_auc_list = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        train_loss.append(avg_train_loss)
        train_acc.append(train_accuracy)

        # Validation
        model.eval()
        val_total_loss = 0.0
        correct_val = 0
        total_val = 0
        val_outputs_all = []
        val_labels_all = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                val_total_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
                val_outputs_all.extend(outputs.cpu().numpy())
                val_labels_all.extend(batch_y.cpu().numpy())
        avg_val_loss = val_total_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_auc = roc_auc_score(val_labels_all, val_outputs_all)
        val_loss.append(avg_val_loss)
        val_acc.append(val_accuracy)
        val_auc_list.append(val_auc)

        # Early stopping based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))
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

