
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
from torch import optim
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import math

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.object_mean = None
        self.object_std = None
        self.num_objects = 18
        self.object_features = 5
        self.global_features = 2

    def _raw_reshape(self, X):
        # Reshape from (batch_size, 92) to (batch_size, total_features)
        # where total_features = global_features (2) + num_objects * object_features (18*5=90)
        return X.reshape(-1, self.global_features + self.num_objects * self.object_features)

    def _get_objects(self, X):
        # Extract objects from reshaped X
        # shape: (batch_size, num_objects, object_features)
        return X[:, self.global_features:].reshape(-1, self.num_objects, self.object_features)

    def _compute_pairwise_features(self, objects):
        # Compute pairwise invariant mass and delta R
        # objects shape: (batch_size, num_objects, 5)
        # where each object has (obj_id, E, pT, eta, phi)
        batch_size = objects.shape[0]
        num_objects = objects.shape[1]

        # Precompute px, py, pz for each particle
        pt = objects[:, :, 2]  # pT
        eta = objects[:, :, 3] # eta
        phi = objects[:, :, 4] # phi
        E = objects[:, :, 1]   # E

        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)

        # Initialize tensors for pairwise features
        m_inv = torch.zeros((batch_size, num_objects, num_objects))
        delta_r = torch.zeros((batch_size, num_objects, num_objects))

        for i in range(num_objects):
            for j in range(i+1, num_objects):
                # Compute invariant mass
                p_i = torch.stack((E[:, i], px[:, i], py[:, i], pz[:, i]), dim=1)
                p_j = torch.stack((E[:, j], px[:, j], py[:, j], pz[:, j]), dim=1)
                p_sum = p_i + p_j
                m2 = (p_sum[:, 0]**2 - p_sum[:, 1]**2 - p_sum[:, 2]**2 - p_sum[:, 3]**2).clamp(min=0)
                m_inv[:, i, j] = torch.sqrt(m2)
                m_inv[:, j, i] = m_inv[:, i, j]

                # Compute delta R
                deta = eta[:, i] - eta[:, j]
                dphi = phi[:, i] - phi[:, j]
                dphi = torch.where(dphi > math.pi, dphi - 2*math.pi, 
                                 torch.where(dphi < -math.pi, dphi + 2*math.pi, dphi))
                delta_r[:, i, j] = torch.sqrt(deta**2 + dphi**2)
                delta_r[:, j, i] = delta_r[:, i, j]

        return m_inv, delta_r

    def fit(self, X, y=None):
        X_reshaped = self._raw_reshape(X)

        # Compute global feature statistics
        self.global_mean = X_reshaped[:, :self.global_features].mean(dim=0)
        self.global_std = X_reshaped[:, :self.global_features].std(dim=0)

        # Compute object feature statistics
        objects = self._get_objects(X_reshaped)
        self.object_mean = objects.mean(dim=(0, 1))  # (object_features,)
        self.object_std = objects.std(dim=(0, 1))

        return self

    def transform(self, X):
        X_reshaped = self._raw_reshape(X)
        batch_size = X_reshaped.shape[0]

        # Normalize global features
        X_global = (X_reshaped[:, :self.global_features] - self.global_mean) / (self.global_std + 1e-8)

        # Normalize object features
        objects = self._get_objects(X_reshaped)
        objects_norm = (objects - self.object_mean) / (self.object_std + 1e-8)

        # Compute pairwise features
        m_inv, delta_r = self._compute_pairwise_features(objects_norm)

        # Combine all features: global + objects + pairwise
        # Flatten pairwise features
        pairwise_features = torch.cat([
            m_inv.flatten(start_dim=1), 
            delta_r.flatten(start_dim=1)
        ], dim=1)

        # Combine all features
        objects_flattened = objects_norm.flatten(start_dim=1)
        X_transformed = torch.cat([X_global, objects_flattened, pairwise_features], dim=1)

        # Add some engineered features
        pt_sum = objects[:, :, 2].sum(dim=1)  # Sum of pT
        e_sum = objects[:, :, 1].sum(dim=1)   # Sum of energy
        eta_max = torch.max(torch.abs(objects[:, :, 3]), dim=1)[0]  # Maximum |eta|

        engineered_features = torch.stack([pt_sum, e_sum, eta_max], dim=1)
        X_transformed = torch.cat([X_transformed, engineered_features], dim=1)

        return X_transformed

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_size = sample_object.size(1)

        # Define transformer-based architecture
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True
        )

        # Project input to transformer dimension
        self.input_proj = nn.Linear(input_size, 128)

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=4)

        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Project inputs
        x = self.input_proj(x)  # (batch_size, seq_len=1, 128)

        # Transformer expects sequence dimension
        x = x.unsqueeze(1)  # (batch_size, 1, 128)

        # Transformer processing
        x = self.transformer(x)

        # Classifier
        x = x.squeeze(1)  # (batch_size, 128)
        return self.classifier(x)

# 3. ---------- MODEL TRAINING ----------
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        y_true_train = []
        y_pred_train = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(outputs.detach().cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_auc = roc_auc_score(y_true_train, y_pred_train)

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)

                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(outputs.cpu().numpy())

        val_epoch_loss = val_running_loss / val_total
        val_epoch_acc = val_correct / val_total
        val_epoch_auc = roc_auc_score(y_true_val, y_pred_val)

        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)

        # Update learning rate
        scheduler.step(val_epoch_loss)

        # Early stopping check
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Train AUC: {epoch_auc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f} | Val AUC: {val_epoch_auc:.4f}")
        print("-" * 60)

    # Load best model
    model.load_state_dict(best_model_state)

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

