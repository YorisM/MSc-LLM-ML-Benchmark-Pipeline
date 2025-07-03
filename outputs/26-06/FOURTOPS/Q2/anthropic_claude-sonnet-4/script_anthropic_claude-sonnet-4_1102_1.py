
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.7.1, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0,, SciKit-Learn 1.7.0
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

def make_loaders(X_train, Y_train, X_val, Y_val, *, batch=512, collate_fn=None, loader_cls=None, workers=0):
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
from sklearn.preprocessing import RobustScaler

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.fitted = False

    def _raw_reshape(self, X):           
        return X

    def fit(self, X, y=None):
        enhanced_features = self._extract_physics_features(X)
        self.scaler.fit(enhanced_features)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        enhanced_features = self._extract_physics_features(X)
        scaled_features = torch.tensor(self.scaler.transform(enhanced_features), dtype=torch.float32)
        return scaled_features

    def _extract_physics_features(self, X):
        # X shape: [N, 92]
        X_np = X.numpy() if torch.is_tensor(X) else X
        batch_size = X_np.shape[0]

        # Parse the data structure
        met = X_np[:, 0]          # Missing ET magnitude
        met_phi = X_np[:, 1]      # Missing ET phi

        # Reshape object data: [N, 18, 5] where 5 = [obj_id, E, pT, eta, phi]
        obj_data = X_np[:, 2:].reshape(batch_size, 18, 5)

        # Extract object properties
        E = obj_data[:, :, 1]    # Energy
        pt = obj_data[:, :, 2]   # Transverse momentum
        eta = obj_data[:, :, 3]  # Pseudorapidity
        phi = obj_data[:, :, 4]  # Azimuthal angle

        # Valid object mask (non-zero energy)
        valid = E > 0

        # Start with global features
        features = [met, met_phi]

        # Event-level features
        n_obj = valid.sum(axis=1)
        ht = np.where(valid, pt, 0).sum(axis=1)  # Scalar sum of pT
        max_pt = np.where(valid, pt, 0).max(axis=1)

        features.extend([n_obj, ht, max_pt])

        # Leading object kinematics
        for i in range(6):  # Top 6 objects
            features.extend([
                np.where(valid[:, i], pt[:, i], 0),
                np.where(valid[:, i], eta[:, i], 0),
                np.where(valid[:, i], phi[:, i], 0)
            ])

        # Pairwise physics features (focus on leading objects)
        for i in range(4):
            for j in range(i+1, 4):
                valid_pair = valid[:, i] & valid[:, j]

                # Invariant mass
                px_i = pt[:, i] * np.cos(phi[:, i])
                py_i = pt[:, i] * np.sin(phi[:, i])
                pz_i = pt[:, i] * np.sinh(eta[:, i])

                px_j = pt[:, j] * np.cos(phi[:, j])
                py_j = pt[:, j] * np.sin(phi[:, j])
                pz_j = pt[:, j] * np.sinh(eta[:, j])

                E_tot = E[:, i] + E[:, j]
                px_tot = px_i + px_j
                py_tot = py_i + py_j
                pz_tot = pz_i + pz_j

                m_inv_sq = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
                m_inv = np.sqrt(np.maximum(m_inv_sq, 0))

                # Angular separation
                deta = eta[:, i] - eta[:, j]
                dphi = phi[:, i] - phi[:, j]
                dphi = np.arctan2(np.sin(dphi), np.cos(dphi))  # Wrap to [-pi, pi]
                dr = np.sqrt(deta**2 + dphi**2)

                # pT ratio
                pt_ratio = np.divide(pt[:, i], pt[:, j], 
                                   out=np.ones_like(pt[:, i]), 
                                   where=(pt[:, j] > 0))

                features.extend([
                    np.where(valid_pair, m_inv, 0),
                    np.where(valid_pair, dr, 0),
                    np.where(valid_pair, pt_ratio, 1)
                ])

        # Convert to array
        feature_array = np.column_stack(features)
        return feature_array

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Get input dimension
        if isinstance(sample_object, tuple):
            input_dim = sample_object[0].shape[-1]
        else:
            input_dim = sample_object.shape[-1]

        # Network architecture optimized for physics data
        self.network = nn.Sequential(
            # Input normalization
            nn.BatchNorm1d(input_dim),

            # First block
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),

            # Second block
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            # Third block  
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Fourth block
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Output
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        return self.network(x).squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    # Early stopping
    best_auc = 0.0
    patience = 8
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_probs = []
        train_labels = []

        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            logits = model(batch_data)
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).long()
            train_correct += (predicted == batch_labels).sum().item()
            train_total += batch_labels.size(0)

            train_probs.extend(probs.detach().cpu().numpy())
            train_labels.extend(batch_labels.cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                logits = model(batch_data)
                loss = F.binary_cross_entropy_with_logits(logits, batch_labels.float())

                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                predicted = (probs > 0.5).long()
                val_correct += (predicted == batch_labels).sum().item()
                val_total += batch_labels.size(0)

                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_labels, val_probs)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
              f'Val AUC: {val_auc:.4f}')

        # Learning rate scheduling
        scheduler.step()

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1} with best AUC: {best_auc:.4f}')
            break

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
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:20]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre     = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train)
    X_val   = pre.transform(X_val)

    collate = getattr(pre, "_collate_fn", None)
    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val, 
                                            batch      = cfg.get("batch_size", 512), 
                                            collate_fn = collate,
                                            loader_cls = loader_cls,
                                            workers    = cfg.get("num_workers", 0))

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


