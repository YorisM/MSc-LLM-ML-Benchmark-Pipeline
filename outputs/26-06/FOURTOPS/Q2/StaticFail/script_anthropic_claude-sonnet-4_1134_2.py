
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
        self.object_mean = None
        self.object_std = None
        self.pairwise_mean = None
        self.pairwise_std = None

    def _parse_input(self, X):
        # X shape: [batch_size, 92]
        batch_size = X.shape[0]

        # Extract global features (missing ET)
        global_features = X[:, :2]  # [batch_size, 2]

        # Extract object features 
        # 18 objects * 5 features = 90, indices 2-91
        object_data = X[:, 2:].reshape(batch_size, 18, 5)  # [batch_size, 18, 5]

        return global_features, object_data

    def _compute_high_level_features(self, global_features, object_data):
        # Compute additional high-level features
        batch_size = global_features.shape[0]

        # Missing ET features
        met_magnitude = global_features[:, 0]  # [batch_size]
        met_phi = global_features[:, 1]  # [batch_size]

        # Object features
        E = object_data[:, :, 1]    # [batch_size, 18]
        p_T = object_data[:, :, 2]  # [batch_size, 18]  
        eta = object_data[:, :, 3]  # [batch_size, 18]
        phi = object_data[:, :, 4]  # [batch_size, 18]

        # Valid object mask
        valid_mask = object_data[:, :, 0] != 0  # [batch_size, 18]

        # Number of objects per event
        n_objects = valid_mask.sum(dim=1).float()  # [batch_size]

        # Total energy and pT
        total_energy = (E * valid_mask.float()).sum(dim=1)  # [batch_size]
        total_pt = (p_T * valid_mask.float()).sum(dim=1)  # [batch_size]

        # Average pT and energy of valid objects
        avg_pt = total_pt / torch.clamp(n_objects, min=1)  # [batch_size]
        avg_energy = total_energy / torch.clamp(n_objects, min=1)  # [batch_size]

        # Max pT and energy
        max_pt = (p_T * valid_mask.float() + (~valid_mask).float() * (-1e6)).max(dim=1)[0]  # [batch_size]
        max_energy = (E * valid_mask.float() + (~valid_mask).float() * (-1e6)).max(dim=1)[0]  # [batch_size]

        # Scalar sum of pT (HT)
        HT = total_pt  # [batch_size]

        # Central vs forward objects (|eta| < 2.5 vs |eta| >= 2.5)
        central_mask = (torch.abs(eta) < 2.5) & valid_mask  # [batch_size, 18]
        forward_mask = (torch.abs(eta) >= 2.5) & valid_mask  # [batch_size, 18]

        n_central = central_mask.sum(dim=1).float()  # [batch_size]
        n_forward = forward_mask.sum(dim=1).float()  # [batch_size]

        high_level_features = torch.stack([
            n_objects, total_energy, total_pt, avg_pt, avg_energy,
            max_pt, max_energy, HT, n_central, n_forward,
            met_magnitude, met_phi
        ], dim=1)  # [batch_size, 12]

        return high_level_features

    def _compute_pairwise_features(self, objects):
        # objects shape: [batch_size, max_objects, 5] where 5 = [obj_id, E, p_T, eta, phi]
        batch_size, max_objects, _ = objects.shape

        # Extract relevant features
        E = objects[:, :, 1]    # [batch_size, max_objects]
        p_T = objects[:, :, 2]  # [batch_size, max_objects]  
        eta = objects[:, :, 3]  # [batch_size, max_objects]
        phi = objects[:, :, 4]  # [batch_size, max_objects]

        # Create masks for valid objects
        valid_mask = objects[:, :, 0] != 0  # [batch_size, max_objects]

        pairwise_features = []

        # Only compute for first 8 objects to limit feature explosion
        max_particles = min(8, max_objects)

        for i in range(max_particles):
            for j in range(i+1, max_particles):
                # Extract features for particles i and j
                E_i, E_j = E[:, i], E[:, j]
                pT_i, pT_j = p_T[:, i], p_T[:, j]
                eta_i, eta_j = eta[:, i], eta[:, j]
                phi_i, phi_j = phi[:, i], phi[:, j]

                # Check if both particles are valid
                pair_valid = valid_mask[:, i] & valid_mask[:, j]  # [batch_size]

                # Calculate momentum components
                px_i = pT_i * torch.cos(phi_i)
                py_i = pT_i * torch.sin(phi_i)
                pz_i = pT_i * torch.sinh(eta_i)

                px_j = pT_j * torch.cos(phi_j)
                py_j = pT_j * torch.sin(phi_j)
                pz_j = pT_j * torch.sinh(eta_j)

                # Invariant mass
                E_sum = E_i + E_j
                px_sum = px_i + px_j
                py_sum = py_i + py_j
                pz_sum = pz_i + pz_j
                p_sum_sq = px_sum**2 + py_sum**2 + pz_sum**2

                m_inv_sq = E_sum**2 - p_sum_sq
                m_inv = torch.sqrt(torch.clamp(m_inv_sq, min=0))  # [batch_size]

                # Angular distance
                delta_eta = eta_i - eta_j
                delta_phi = phi_i - phi_j
                # Handle phi wraparound
                delta_phi = torch.where(delta_phi > math.pi, delta_phi - 2*math.pi, delta_phi)
                delta_phi = torch.where(delta_phi < -math.pi, delta_phi + 2*math.pi, delta_phi)
                delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)  # [batch_size]

                # Set invalid pairs to zero
                m_inv = torch.where(pair_valid, m_inv, torch.zeros_like(m_inv))
                delta_R = torch.where(pair_valid, delta_R, torch.zeros_like(delta_R))

                pairwise_features.append(m_inv.unsqueeze(1))  # [batch_size, 1]
                pairwise_features.append(delta_R.unsqueeze(1))  # [batch_size, 1]

        if pairwise_features:
            return torch.cat(pairwise_features, dim=1)  # [batch_size, num_pairs*2]
        else:
            return torch.zeros(batch_size, 0)

    def fit(self, X, y=None):
        # Parse input
        global_features, object_data = self._parse_input(X)

        # Compute high-level features
        high_level_features = self._compute_high_level_features(global_features, object_data)

        # Fit global features normalization
        self.global_mean = high_level_features.mean(dim=0)
        self.global_std = high_level_features.std(dim=0) + 1e-8

        # Fit object features normalization (excluding obj_id)
        # Only use valid objects for statistics
        valid_mask = object_data[:, :, 0] != 0  # [batch_size, 18]
        valid_object_features = object_data[:, :, 1:][valid_mask]  # [num_valid, 4]

        if len(valid_object_features) > 0:
            self.object_mean = valid_object_features.mean(dim=0)
            self.object_std = valid_object_features.std(dim=0) + 1e-8
        else:
            self.object_mean = torch.zeros(4)
            self.object_std = torch.ones(4)

        # Fit pairwise features normalization
        pairwise_features = self._compute_pairwise_features(object_data)
        if pairwise_features.shape[1] > 0:
            valid_pairwise = pairwise_features[pairwise_features != 0]
            if len(valid_pairwise) > 0:
                self.pairwise_mean = valid_pairwise.mean()
                self.pairwise_std = valid_pairwise.std() + 1e-8
            else:
                self.pairwise_mean = torch.tensor(0.0)
                self.pairwise_std = torch.tensor(1.0)
        else:
            self.pairwise_mean = torch.tensor(0.0)
            self.pairwise_std = torch.tensor(1.0)

        return self

    def transform(self, X):
        # Parse input
        global_features, object_data = self._parse_input(X)  # [batch_size, 2], [batch_size, 18, 5]

        # Compute high-level features
        high_level_features = self._compute_high_level_features(global_features, object_data)  # [batch_size, 12]

        # Normalize high-level features
        if self.global_mean is not None:
            high_level_norm = (high_level_features - self.global_mean) / self.global_std  # [batch_size, 12]
        else:
            high_level_norm = high_level_features

        # Normalize object features
        object_norm = object_data.clone()
        valid_mask = object_data[:, :, 0] != 0  # [batch_size, 18]

        if self.object_mean is not None:
            # Normalize E, pT, eta, phi for valid objects
            for i in range(1, 5):  # indices 1,2,3,4 = E, pT, eta, phi
                object_norm[:, :, i] = torch.where(
                    valid_mask,
                    (object_data[:, :, i] - self.object_mean[i-1]) / self.object_std[i-1],
                    torch.zeros_like(object_data[:, :, i])
                )

        # Compute and normalize pairwise features
        pairwise_features = self._compute_pairwise_features(object_data)  # [batch_size, num_pairs*2]

        if pairwise_features.shape[1] > 0 and self.pairwise_mean is not None:
            pairwise_norm = torch.where(
                pairwise_features != 0,
                (pairwise_features - self.pairwise_mean) / self.pairwise_std,
                torch.zeros_like(pairwise_features)
            )
        else:
            pairwise_norm = pairwise_features

        # Flatten object features
        object_flat = object_norm.reshape(object_norm.shape[0], -1)  # [batch_size, 18*5]

        # Combine all features
        if pairwise_norm.shape[1] > 0:
            combined = torch.cat([high_level_norm, object_flat, pairwise_norm], dim=1)
        else:
            combined = torch.cat([high_level_norm, object_flat], dim=1)

        return combined

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

        self.input_size = input_size

        # Deep neural network with skip connections
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: [batch_size, input_size]
        x1 = self.layer1(x)  # [batch_size, 512]
        x2 = self.layer2(x1)  # [batch_size, 256]
        x3 = self.layer3(x2)  # [batch_size, 128]
        x4 = self.layer4(x3)  # [batch_size, 64]

        logits = self.output_layer(x4)  # [batch_size, 1]
        return logits.squeeze(-1)  # [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)

    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_auc = 0
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()

            if isinstance(batch_x, (tuple, list)):
                outputs = model(*batch_x)
            else:
                outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.float().to(device)

                if isinstance(batch_x, (tuple, list)):
                    outputs = model(*batch_x)
                else:
                    outputs = model(batch_x)

                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)

                # Store for AUC calculation
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Calculate AUC
        val_auc = roc_auc_score(val_true, val_preds)

        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')

        # Learning rate scheduling
        scheduler.step(val_auc)

        # Early stopping based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Load best model
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

