
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

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_means = None
        self.global_stds = None
        self.obj_means = None
        self.obj_stds = None
        self.max_objects = 18
        self.obj_slice_size = 5

    def _raw_reshape(self, X):
        return X  # Return as is, shape: [N, 92]

    def fit(self, X, y=None):
        # Extract global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :2]
        self.global_means = torch.mean(global_features, dim=0)
        self.global_stds = torch.std(global_features, dim=0)
        self.global_stds[self.global_stds == 0] = 1.0

        # Extract object features
        obj_features = X[:, 2:].reshape(-1, self.max_objects, self.obj_slice_size)  # Shape: [N, 18, 5]
        obj_features_valid = obj_features[obj_features[:, :, 0] != 0]  # Filter out padding
        if obj_features_valid.size(0) > 0:
            self.obj_means = torch.mean(obj_features_valid, dim=0)
            self.obj_stds = torch.std(obj_features_valid, dim=0)
            self.obj_stds[self.obj_stds == 0] = 1.0
        else:
            self.obj_means = torch.zeros(self.obj_slice_size)
            self.obj_stds = torch.ones(self.obj_slice_size)
        return self

    def transform(self, X):
        # Standardize global features
        global_features = X[:, :2]
        global_features = (global_features - self.global_means) / self.global_stds  # Shape: [N, 2]

        # Reshape object features
        obj_features = X[:, 2:].reshape(-1, self.max_objects, self.obj_slice_size)  # Shape: [N, 18, 5]

        # Standardize object features
        obj_features = (obj_features - self.obj_means) / self.obj_stds  # Shape: [N, 18, 5]

        # Calculate pairwise features (invariant mass and delta R)
        pairwise_masses = []
        pairwise_deltaRs = []
        for i in range(self.max_objects):
            for j in range(i + 1, self.max_objects):
                # Extract kinematics for pair
                pt1, eta1, phi1 = obj_features[:, i, 2:5]  # p_T, eta, phi for obj i
                pt2, eta2, phi2 = obj_features[:, j, 2:5]  # p_T, eta, phi for obj j
                E1 = obj_features[:, i, 1]  # Energy for obj i
                E2 = obj_features[:, j, 1]  # Energy for obj j

                # Compute invariant mass approximation (simplified)
                mass = torch.sqrt(torch.abs(2 * pt1 * pt2 * (1 - torch.cos(phi1 - phi2)) + (E1 + E2) ** 2))
                pairwise_masses.append(mass.unsqueeze(-1))  # Shape: [N, 1]

                # Compute delta R
                delta_eta = eta1 - eta2
                delta_phi = torch.abs(phi1 - phi2)
                delta_phi = torch.min(delta_phi, 2 * np.pi - delta_phi)
                delta_R = torch.sqrt(delta_eta ** 2 + delta_phi ** 2)
                pairwise_deltaRs.append(delta_R.unsqueeze(-1))  # Shape: [N, 1]

        pairwise_features = torch.cat(pairwise_masses + pairwise_deltaRs, dim=-1)  # Shape: [N, num_pairs * 2]
        num_pairs = len(pairwise_masses)
        expected_pairwise_features = num_pairs * 2  # invariant masses + delta Rs

        # Check if pairwise_features matches expected shape
        if pairwise_features.shape[-1] != expected_pairwise_features:
            # If mismatch, create a tensor of zeros with the expected shape
            pairwise_features = torch.zeros(X.shape[0], expected_pairwise_features, device=X.device)

        # Flatten object features
        obj_features_flat = obj_features.reshape(-1, self.max_objects * self.obj_slice_size)  # Shape: [N, 90]

        # Concatenate all features
        transformed_X = torch.cat([global_features, obj_features_flat, pairwise_features], dim=-1)  # Shape: [N, 2 + 90 + num_pairs*2]
        return transformed_X

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
        input_dim = sample_object.shape[-1]  # Dynamic input dimension based on preprocessed features
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_auc = 0.0
    patience = 5
    trigger_times = 0
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    val_auc = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct / total
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        val_outputs_all = []
        val_labels_all = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                outputs = model(batch_x).squeeze()
                val_running_loss += criterion(outputs, batch_y).item()
                predicted = (outputs >= 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                val_outputs_all.append(outputs.cpu().numpy())
                val_labels_all.append(batch_y.cpu().numpy())
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = correct / total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        # Compute AUC
        val_outputs_all = np.concatenate(val_outputs_all)
        val_labels_all = np.concatenate(val_labels_all)
        epoch_val_auc = roc_auc_score(val_labels_all, val_outputs_all)
        val_auc.append(epoch_val_auc)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val AUC: {epoch_val_auc:.4f}")

        # Early stopping based on AUC
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered")
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

