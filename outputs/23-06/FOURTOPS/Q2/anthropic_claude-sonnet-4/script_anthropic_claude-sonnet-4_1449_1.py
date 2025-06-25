
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

from sklearn.metrics import roc_auc_score
import torch.optim as optim

class MyPreprocessor:
    def __init__(self):
        self.stats = {}

    def _raw_reshape(self, X):           
        return X

    def _parse_objects(self, X):
        # X: [batch_size, 92]
        # First 2 features: missing ET magnitude and phi
        global_features = X[:, :2]  # [batch_size, 2]

        # Remaining 90 features: 18 objects * 5 features each
        object_features = X[:, 2:].reshape(X.shape[0], 18, 5)  # [batch_size, 18, 5]

        return global_features, object_features

    def _compute_pairwise_features(self, objects):
        # objects: [batch_size, 18, 5] -> [obj_id, E, p_T, eta, phi]
        batch_size = objects.shape[0]

        E = objects[:, :, 1]     # [batch_size, 18]
        p_T = objects[:, :, 2]   # [batch_size, 18]
        eta = objects[:, :, 3]   # [batch_size, 18]
        phi = objects[:, :, 4]   # [batch_size, 18]

        # Compute px, py, pz
        px = p_T * torch.cos(phi)
        py = p_T * torch.sin(phi)
        pz = p_T * torch.sinh(eta)

        # Valid objects (non-zero energy)
        valid = E > 0  # [batch_size, 18]

        invariant_masses = []
        angular_distances = []

        for i in range(18):
            for j in range(i + 1, 18):
                # Invariant mass
                E_sum = E[:, i] + E[:, j]
                px_sum = px[:, i] + px[:, j]
                py_sum = py[:, i] + py[:, j]
                pz_sum = pz[:, i] + pz[:, j]

                m_sq = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
                m_inv = torch.sqrt(torch.clamp(m_sq, min=0))

                # Angular distance
                d_eta = eta[:, i] - eta[:, j]
                d_phi = phi[:, i] - phi[:, j]
                d_phi = torch.remainder(d_phi + torch.pi, 2*torch.pi) - torch.pi
                d_R = torch.sqrt(d_eta**2 + d_phi**2)

                # Mask invalid pairs
                pair_valid = valid[:, i] & valid[:, j]
                m_inv = torch.where(pair_valid, m_inv, torch.zeros_like(m_inv))
                d_R = torch.where(pair_valid, d_R, torch.zeros_like(d_R))

                invariant_masses.append(m_inv)
                angular_distances.append(d_R)

        # Concatenate all pairwise features
        pairwise_features = torch.stack(invariant_masses + angular_distances, dim=1)
        return pairwise_features  # [batch_size, 306]

    def fit(self, X, y=None):
        global_features, object_features = self._parse_objects(X)

        # Statistics for global features
        self.stats['global_mean'] = global_features.mean(dim=0)
        self.stats['global_std'] = global_features.std(dim=0) + 1e-8

        # Statistics for object features (flatten first)
        obj_flat = object_features.reshape(object_features.shape[0], -1)
        self.stats['object_mean'] = obj_flat.mean(dim=0)
        self.stats['object_std'] = obj_flat.std(dim=0) + 1e-8

        # Statistics for pairwise features
        pairwise_features = self._compute_pairwise_features(object_features)

        # For pairwise features, compute stats only on non-zero values
        pairwise_means = torch.zeros(pairwise_features.shape[1])
        pairwise_stds = torch.ones(pairwise_features.shape[1])

        for i in range(pairwise_features.shape[1]):
            nonzero_vals = pairwise_features[:, i][pairwise_features[:, i] != 0]
            if len(nonzero_vals) > 1:
                pairwise_means[i] = nonzero_vals.mean()
                pairwise_stds[i] = nonzero_vals.std() + 1e-8

        self.stats['pairwise_mean'] = pairwise_means
        self.stats['pairwise_std'] = pairwise_stds

        return self

    def transform(self, X):
        global_features, object_features = self._parse_objects(X)

        # Normalize global features
        global_norm = (global_features - self.stats['global_mean']) / self.stats['global_std']

        # Normalize object features
        obj_flat = object_features.reshape(object_features.shape[0], -1)
        obj_norm = (obj_flat - self.stats['object_mean']) / self.stats['object_std']

        # Compute and normalize pairwise features
        pairwise_features = self._compute_pairwise_features(object_features)
        pairwise_norm = pairwise_features.clone()

        for i in range(pairwise_features.shape[1]):
            nonzero_mask = pairwise_features[:, i] != 0
            if nonzero_mask.any():
                pairwise_norm[nonzero_mask, i] = ((pairwise_features[nonzero_mask, i] - 
                                                  self.stats['pairwise_mean'][i]) / 
                                                 self.stats['pairwise_std'][i])

        # Concatenate all features: [batch_size, 2 + 90 + 306] = [batch_size, 398]
        combined = torch.cat([global_norm, obj_norm, pairwise_norm], dim=1)
        return combined

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        if isinstance(sample_object, (tuple, list)):
            input_dim = sample_object[0].shape[-1] 
        else:
            input_dim = sample_object.shape[-1]

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        if isinstance(data, (tuple, list)):
            x = data[0]
        else:
            x = data
        return self.classifier(x).squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.BCELoss()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_auc = 0
    patience_counter = 0
    patience = 10
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        train_correct = 0
        train_total = 0
        train_preds = []
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
            optimizer.step()

            epoch_train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())

        # Validation
        model.eval()
        epoch_val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                if isinstance(batch_x, (tuple, list)):
                    batch_x = batch_x[0]
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                epoch_val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

                val_preds.extend(outputs.detach().cpu().numpy())
                val_targets.extend(batch_y.detach().cpu().numpy())

        # Calculate metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        train_auc = roc_auc_score(train_targets, train_preds)
        val_auc = roc_auc_score(val_targets, val_preds)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
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

    # Load best model weights
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

