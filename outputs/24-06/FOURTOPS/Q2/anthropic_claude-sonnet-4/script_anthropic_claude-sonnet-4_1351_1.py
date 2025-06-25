
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

    def fit(self, X, y=None):
        # Create physics-inspired features
        X_enhanced = self._create_physics_features(X)
        self.scaler.fit(X_enhanced.numpy())
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Preprocessor not fitted")

        X_enhanced = self._create_physics_features(X)
        X_scaled = torch.tensor(self.scaler.transform(X_enhanced.numpy()), dtype=torch.float32)
        return X_scaled

    def _create_physics_features(self, X):
        batch_size = X.shape[0]

        # Extract basic features
        met_mag = X[:, 0]  # Missing ET magnitude
        met_phi = X[:, 1]  # Missing ET phi

        # Extract object data: reshape to [N, 18, 5]
        object_data = X[:, 2:].reshape(batch_size, 18, 5)
        object_ids = object_data[:, :, 0]
        object_energies = object_data[:, :, 1]
        object_pts = object_data[:, :, 2]
        object_etas = object_data[:, :, 3]
        object_phis = object_data[:, :, 4]

        # Mask for valid objects
        valid_mask = object_ids != 0  # [N, 18]

        # Basic counting and energy features
        n_objects = valid_mask.sum(dim=1).float()
        total_energy = (object_energies * valid_mask.float()).sum(dim=1)
        total_pt = (object_pts * valid_mask.float()).sum(dim=1)

        # Safe division for averages
        avg_energy = total_energy / torch.clamp(n_objects, min=1)
        avg_pt = total_pt / torch.clamp(n_objects, min=1)

        # High-pT object counting (multiple thresholds)
        high_pt_50 = ((object_pts > 50.0) & valid_mask).sum(dim=1).float()
        high_pt_100 = ((object_pts > 100.0) & valid_mask).sum(dim=1).float()
        high_pt_200 = ((object_pts > 200.0) & valid_mask).sum(dim=1).float()

        # Energy thresholds
        high_e_100 = ((object_energies > 100.0) & valid_mask).sum(dim=1).float()
        high_e_200 = ((object_energies > 200.0) & valid_mask).sum(dim=1).float()

        # MET significance
        met_significance = met_mag / torch.sqrt(torch.clamp(total_pt, min=1.0))

        # Object centrality (in eta)
        central_mask = (torch.abs(object_etas) < 2.5) & valid_mask
        n_central = central_mask.sum(dim=1).float()
        forward_mask = (object_etas > 2.5) & valid_mask
        n_forward = forward_mask.sum(dim=1).float()

        # Energy fractions
        leading_pt = torch.zeros(batch_size)
        subleading_pt = torch.zeros(batch_size)
        for i in range(batch_size):
            valid_pts = object_pts[i][valid_mask[i]]
            if len(valid_pts) > 0:
                sorted_pts, _ = torch.sort(valid_pts, descending=True)
                leading_pt[i] = sorted_pts[0]
                if len(sorted_pts) > 1:
                    subleading_pt[i] = sorted_pts[1]

        # Vector sum of momenta
        px_sum = torch.zeros(batch_size)
        py_sum = torch.zeros(batch_size)
        for i in range(batch_size):
            if valid_mask[i].sum() > 0:
                valid_pts = object_pts[i][valid_mask[i]]
                valid_phis = object_phis[i][valid_mask[i]]
                px_sum[i] = (valid_pts * torch.cos(valid_phis)).sum()
                py_sum[i] = (valid_pts * torch.sin(valid_phis)).sum()

        vector_pt_sum = torch.sqrt(px_sum**2 + py_sum**2)
        pt_balance = vector_pt_sum / torch.clamp(total_pt, min=1e-6)

        # Eta and phi spreads
        eta_spread = torch.zeros(batch_size)
        phi_spread = torch.zeros(batch_size)
        for i in range(batch_size):
            if valid_mask[i].sum() > 1:
                valid_etas = object_etas[i][valid_mask[i]]
                valid_phis = object_phis[i][valid_mask[i]]
                eta_spread[i] = valid_etas.max() - valid_etas.min()
                # Handle phi periodicity
                phi_diff = valid_phis.unsqueeze(0) - valid_phis.unsqueeze(1)
                phi_diff = torch.atan2(torch.sin(phi_diff), torch.cos(phi_diff))
                phi_spread[i] = torch.abs(phi_diff).max()

        # Combine physics features
        physics_features = torch.stack([
            met_mag, met_phi, n_objects, total_energy, total_pt,
            avg_energy, avg_pt, high_pt_50, high_pt_100, high_pt_200,
            high_e_100, high_e_200, met_significance, n_central, n_forward,
            leading_pt, subleading_pt, vector_pt_sum, pt_balance,
            eta_spread, phi_spread
        ], dim=1)  # [N, 21]

        # Keep original features as well
        original_features = X  # [N, 92]

        # Concatenate enhanced features with original
        enhanced_features = torch.cat([physics_features, original_features], dim=1)  # [N, 113]

        return enhanced_features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        input_size = sample_object.shape[-1]

        self.network = nn.Sequential(
            # First layer - larger for complex feature interactions
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Second layer
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Third layer
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Fourth layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Fifth layer
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Output layer
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 40
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss and optimizer - using label smoothing for better generalization
    class LabelSmoothingBCE(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing

        def forward(self, input, target):
            target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
            return F.binary_cross_entropy_with_logits(input, target_smooth)

    criterion = LabelSmoothingBCE(smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 12

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs.squeeze(), batch_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.sigmoid(outputs.squeeze()) > 0.5
            train_correct += (predictions == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device).float()

                outputs = model(batch_data)
                loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), batch_labels)
                val_loss += loss.item()

                predictions = torch.sigmoid(outputs.squeeze()) > 0.5
                val_correct += (predictions == batch_labels).sum().item()
                val_total += batch_labels.size(0)

        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Step scheduler
        scheduler.step()

        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
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

