
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.feature_mean = None
        self.feature_std = None

    def _raw_reshape(self, X):           
        return X

    def fit(self, X, y=None):
        # Compute normalization statistics
        self.feature_mean = X.mean(dim=0)
        self.feature_std = X.std(dim=0) + 1e-8
        return self

    def transform(self, X):
        # Normalize all original features
        X_norm = (X - self.feature_mean) / self.feature_std

        batch_size = X.shape[0]

        # Parse missing ET features
        met_mag = X[:, 0]  # [batch_size]
        met_phi = X[:, 1]  # [batch_size]

        # Parse object features: reshape to [batch_size, 18, 5]
        obj_data = X[:, 2:].reshape(batch_size, 18, 5)  # [batch_size, 18, 5]

        # Extract object properties
        obj_energies = obj_data[:, :, 1]  # [batch_size, 18]
        obj_pt = obj_data[:, :, 2]        # [batch_size, 18]
        obj_eta = obj_data[:, :, 3]       # [batch_size, 18]
        obj_phi = obj_data[:, :, 4]       # [batch_size, 18]

        # Create mask for valid objects (energy > 0)
        obj_mask = obj_energies > 0  # [batch_size, 18]

        # === Basic aggregate features ===
        n_objects = obj_mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]

        # Masked aggregations
        masked_energies = torch.where(obj_mask, obj_energies, torch.zeros_like(obj_energies))
        masked_pt = torch.where(obj_mask, obj_pt, torch.zeros_like(obj_pt))

        total_energy = masked_energies.sum(dim=1, keepdim=True)  # [batch_size, 1]
        total_pt = masked_pt.sum(dim=1, keepdim=True)            # [batch_size, 1]

        # Avoid division by zero
        mean_energy = torch.where(n_objects > 0, total_energy / n_objects, torch.zeros_like(total_energy))
        mean_pt = torch.where(n_objects > 0, total_pt / n_objects, torch.zeros_like(total_pt))

        max_energy = masked_energies.max(dim=1, keepdim=True)[0]  # [batch_size, 1]
        max_pt = masked_pt.max(dim=1, keepdim=True)[0]            # [batch_size, 1]

        # === Pairwise angular distance features for top objects ===
        # Get top 6 objects by pT
        top_k = 6
        _, top_indices = torch.topk(masked_pt, k=min(top_k, masked_pt.shape[1]), dim=1)  # [batch_size, k]

        pairwise_features = []
        k = min(top_k, masked_pt.shape[1])

        for i in range(k):
            for j in range(i+1, k):
                # Initialize with zeros
                delta_R = torch.zeros(batch_size, device=X.device)

                # Only compute for valid pairs
                valid_samples = (top_indices.shape[1] > max(i, j))
                if valid_samples:
                    batch_idx = torch.arange(batch_size, device=X.device)
                    idx_i = top_indices[:, i]
                    idx_j = top_indices[:, j]

                    # Check if both objects are valid
                    mask_i = obj_mask[batch_idx, idx_i]
                    mask_j = obj_mask[batch_idx, idx_j]
                    pair_valid = mask_i & mask_j

                    # Compute delta R for valid pairs
                    eta_i = obj_eta[batch_idx, idx_i]
                    phi_i = obj_phi[batch_idx, idx_i]
                    eta_j = obj_eta[batch_idx, idx_j]
                    phi_j = obj_phi[batch_idx, idx_j]

                    deta = eta_i - eta_j
                    dphi = phi_i - phi_j
                    # Handle phi wraparound
                    dphi = torch.remainder(dphi + math.pi, 2*math.pi) - math.pi
                    delta_R_computed = torch.sqrt(deta**2 + dphi**2)

                    # Only keep valid pairs
                    delta_R = torch.where(pair_valid, delta_R_computed, delta_R)

                pairwise_features.append(delta_R.unsqueeze(1))  # [batch_size, 1]

        # Combine all features
        engineered_features = [
            n_objects,     # [batch_size, 1]
            total_energy,  # [batch_size, 1] 
            total_pt,      # [batch_size, 1]
            mean_energy,   # [batch_size, 1]
            mean_pt,       # [batch_size, 1]
            max_energy,    # [batch_size, 1]
            max_pt         # [batch_size, 1]
        ]
        engineered_features.extend(pairwise_features)  # Add pairwise features

        engineered_tensor = torch.cat(engineered_features, dim=1)  # [batch_size, num_features]

        # Combine normalized original features with engineered features
        combined_features = torch.cat([
            X_norm,             # [batch_size, 92]
            engineered_tensor   # [batch_size, 7 + 15] = [batch_size, 22]
        ], dim=1)  # [batch_size, 114]

        return combined_features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        if isinstance(sample_object, (tuple, list)):
            input_dim = sample_object[0].shape[-1]
        else:
            input_dim = sample_object.shape[-1]

        self.input_norm = nn.LayerNorm(input_dim)

        # Deep network with batch normalization and residual connections
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]

        x = self.input_norm(x)  # [batch_size, input_dim]

        # Forward pass with residual connections
        h1 = F.relu(self.bn1(self.fc1(x)))      # [batch_size, 512]
        h1 = self.dropout(h1)

        h2 = F.relu(self.bn2(self.fc2(h1)))     # [batch_size, 512]
        h2 = self.dropout(h2 + h1)  # residual connection

        h3 = F.relu(self.bn3(self.fc3(h2)))     # [batch_size, 256]
        h3 = self.dropout(h3)

        h4 = F.relu(self.bn4(self.fc4(h3)))     # [batch_size, 128]
        h4 = self.dropout(h4)

        h5 = F.relu(self.bn5(self.fc5(h4)))     # [batch_size, 64]
        h5 = self.dropout(h5)

        logits = self.fc_out(h5)  # [batch_size, 1]

        return logits.squeeze(-1)  # [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_auc = 0
    patience_counter = 0
    patience = 8

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_probs = []
        train_targets = []

        for batch_x, batch_y in train_loader:
            if isinstance(batch_x, (tuple, list)):
                batch_x = batch_x[0]

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

            train_probs.extend(probs.detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                if isinstance(batch_x, (tuple, list)):
                    batch_x = batch_x[0]

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

                val_probs.extend(probs.detach().cpu().numpy())
                val_targets.extend(batch_y.detach().cpu().numpy())

        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        train_auc = roc_auc_score(train_targets, train_probs)
        val_auc = roc_auc_score(val_targets, val_probs)

        train_loss_history.append(train_loss_avg)
        val_loss_history.append(val_loss_avg)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')

        # Learning rate scheduling
        scheduler.step(val_auc)

        # Early stopping based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            model.load_state_dict(best_model_state)
            break

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

