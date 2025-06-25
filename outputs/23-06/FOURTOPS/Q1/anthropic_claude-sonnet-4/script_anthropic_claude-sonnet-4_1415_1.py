
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
from sklearn.preprocessing import StandardScaler

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def _raw_reshape(self, X):           
        return X

    def make_loader_cfg(self):
        return None

    def _extract_physics_features(self, X):
        """Extract physics-inspired features from the raw data"""
        # X shape: [N, 92]
        N = X.shape[0]

        # Extract global features
        met_mag = X[:, 0]  # [N]
        met_phi = X[:, 1]  # [N]

        # Extract object features - reshape to [N, 18, 5]
        object_data = X[:, 2:].reshape(N, 18, 5)  # [N, 18, 5]
        obj_ids = object_data[:, :, 0]  # [N, 18]
        obj_E = object_data[:, :, 1]    # [N, 18]
        obj_pt = object_data[:, :, 2]   # [N, 18]
        obj_eta = object_data[:, :, 3]  # [N, 18]
        obj_phi = object_data[:, :, 4]  # [N, 18]

        # Create mask for valid objects (non-zero)
        valid_mask = (obj_ids != 0)  # [N, 18]

        # Feature 1: Number of objects
        n_objects = valid_mask.sum(axis=1)  # [N]

        # Feature 2: Total transverse momentum
        total_pt = (obj_pt * valid_mask).sum(axis=1)  # [N]

        # Feature 3: Total energy
        total_energy = (obj_E * valid_mask).sum(axis=1)  # [N]

        # Feature 4: Leading object pT (highest pT)
        masked_pt = obj_pt * valid_mask + (1-valid_mask) * (-1e6)
        leading_pt = np.where(valid_mask.any(axis=1), 
                            np.max(masked_pt, axis=1), 
                            0)  # [N]

        # Feature 5: MET significance (MET / sqrt(sum pT))
        met_sig = np.where(total_pt > 0, met_mag / np.sqrt(total_pt + 1e-6), 0)  # [N]

        # Feature 6: Average eta (centrality)
        valid_count = valid_mask.sum(axis=1) + 1e-6
        avg_eta = (obj_eta * valid_mask).sum(axis=1) / valid_count  # [N]

        # Feature 7: Eta spread
        eta_spread = np.where(n_objects > 1,
                            np.sqrt(((obj_eta - avg_eta[:, np.newaxis])**2 * valid_mask).sum(axis=1) / valid_count),
                            0)  # [N]

        # Combine physics features
        physics_features = np.column_stack([
            n_objects, total_pt, total_energy, leading_pt, 
            met_sig, avg_eta, eta_spread
        ])  # [N, 7]

        return physics_features

    def fit(self, X, y=None):
        # X is a torch tensor [N, 92]
        if torch.is_tensor(X):
            X_np = X.numpy()
        else:
            X_np = X

        # Extract physics features
        physics_features = self._extract_physics_features(X_np)  # [N, 7]

        # Combine original features with physics features
        X_combined = np.concatenate([X_np, physics_features], axis=1)  # [N, 99]

        # Fit scaler on combined features
        self.scaler.fit(X_combined)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        if torch.is_tensor(X):
            X_np = X.numpy()
        else:
            X_np = X

        # Extract physics features
        physics_features = self._extract_physics_features(X_np)  # [N, 7]

        # Combine original features with physics features
        X_combined = np.concatenate([X_np, physics_features], axis=1)  # [N, 99]

        # Apply standard scaling
        X_scaled = self.scaler.transform(X_combined)  # [N, 99]

        return torch.from_numpy(X_scaled.astype(np.float32))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Determine input size from sample
        if isinstance(sample_object, (tuple, list)):
            input_size = sample_object[0].shape[-1]
        else:
            input_size = sample_object.shape[-1]

        # Deep MLP optimized for particle physics data
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, input_size]
        return self.classifier(x).squeeze(-1)  # [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.7)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_auc = 0
    patience_counter = 0
    patience = 10
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_probs = []
        train_targets = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)  # [batch_size]
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            train_correct += (predicted == batch_y.float()).sum().item()
            train_total += batch_y.size(0)

            train_probs.extend(probs.detach().cpu().numpy())
            train_targets.extend(batch_y.cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.float())

                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                val_correct += (predicted == batch_y.float()).sum().item()
                val_total += batch_y.size(0)

                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        train_auc = roc_auc_score(train_targets, train_probs)
        val_auc = roc_auc_score(val_targets, val_probs)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_auc)

        # Early stopping based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Load best model if we found one
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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

