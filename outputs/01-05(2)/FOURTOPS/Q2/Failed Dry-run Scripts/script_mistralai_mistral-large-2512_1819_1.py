
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
                        
DATASET = {
    "X_train": "./challenges/FOURTOPS/data/train/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/train/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/train/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/train/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv(DATASET["X_train"], dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv(DATASET["Y_train"], dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv(DATASET["X_val"], dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv(DATASET['Y_val'], dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class FourTopsDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        X2 = pre.transform(X) if pre is not None else X
        if not torch.is_tensor(X2):
            X2 = torch.as_tensor(X2)
        self.X = X2.float()
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        self.y = y.long()
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

# ---------- IMPORTS ----------
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.obj_slice = slice(2, None)  # Skip E_T_miss and phi_Et_miss
        self.per_obj_features = 5
        self.max_objects = 18

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": None,

            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False,
                                "batch_size": 512}
        }

    def fit(self, X, y=None):
        # Extract only kinematic features (E, p_T, eta, phi) for scaling
        kinematic_features = X[:, self.obj_slice].reshape(-1, self.per_obj_features)[:, 1:]
        # Remove zero-padded objects (obj_id == 0)
        mask = (X[:, self.obj_slice].reshape(-1, self.per_obj_features)[:, 0] != 0)
        kinematic_features = kinematic_features[mask]
        self.scaler.fit(kinematic_features)
        return self

    def transform(self, X):
        # Create a copy to avoid modifying the original
        X_transformed = X.clone()

        # Scale kinematic features (E, p_T, eta, phi)
        kinematic_features = X_transformed[:, self.obj_slice].reshape(-1, self.per_obj_features)[:, 1:]
        mask = (X_transformed[:, self.obj_slice].reshape(-1, self.per_obj_features)[:, 0] != 0)
        kinematic_features[mask] = self.scaler.transform(kinematic_features[mask])
        X_transformed[:, self.obj_slice] = kinematic_features.reshape(X_transformed.shape[0], -1)

        # Add pairwise features: invariant mass and delta R
        n_events = X_transformed.shape[0]
        pairwise_features = torch.zeros((n_events, self.max_objects, self.max_objects, 2), device=X_transformed.device)

        for i in range(self.max_objects):
            for j in range(i+1, self.max_objects):
                # Get object indices
                idx_i = 2 + i * self.per_obj_features
                idx_j = 2 + j * self.per_obj_features

                # Skip if either object is padding
                if (X_transformed[:, idx_i] == 0).all() or (X_transformed[:, idx_j] == 0).all():
                    continue

                # Extract features
                E_i = X_transformed[:, idx_i+1]
                px_i = X_transformed[:, idx_i+2] * torch.cos(X_transformed[:, idx_i+4])
                py_i = X_transformed[:, idx_i+2] * torch.sin(X_transformed[:, idx_i+4])
                pz_i = X_transformed[:, idx_i+2] * torch.sinh(X_transformed[:, idx_i+3])

                E_j = X_transformed[:, idx_j+1]
                px_j = X_transformed[:, idx_j+2] * torch.cos(X_transformed[:, idx_j+4])
                py_j = X_transformed[:, idx_j+2] * torch.sin(X_transformed[:, idx_j+4])
                pz_j = X_transformed[:, idx_j+2] * torch.sinh(X_transformed[:, idx_j+3])

                # Invariant mass
                m_ij = torch.sqrt((E_i + E_j)**2 - (px_i + px_j)**2 - (py_i + py_j)**2 - (pz_i + pz_j)**2)
                pairwise_features[:, i, j, 0] = m_ij
                pairwise_features[:, j, i, 0] = m_ij

                # Delta R
                delta_eta = X_transformed[:, idx_i+3] - X_transformed[:, idx_j+3]
                delta_phi = torch.abs(X_transformed[:, idx_i+4] - X_transformed[:, idx_j+4])
                delta_phi = torch.min(delta_phi, 2*torch.pi - delta_phi)
                delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)
                pairwise_features[:, i, j, 1] = delta_R
                pairwise_features[:, j, i, 1] = delta_R

        # Flatten pairwise features and concatenate with original features
        pairwise_flat = pairwise_features.reshape(n_events, -1)
        X_transformed = torch.cat([X_transformed, pairwise_flat], dim=1)

        return X_transformed

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[1]

        # Feature extraction layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Output layer
        self.output = nn.Linear(128, 1)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Batch norm
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, batch_x):
        # Feature extraction
        x = F.relu(self.bn1(self.fc1(batch_x)))  # [B, 512]
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))        # [B, 256]
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))        # [B, 128]
        x = self.dropout(x)

        # Attention mechanism
        attention_weights = self.attention(x)    # [B, 1]
        x = x * attention_weights                # [B, 128]

        # Output
        logits = self.output(x).squeeze(-1)     # [B]
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())

        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_targets, train_preds)
        train_acc = (torch.sigmoid(torch.tensor(train_preds)) > 0.5).float().eq(torch.tensor(train_targets)).float().mean().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y.float())
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_acc = (torch.sigmoid(torch.tensor(val_preds)) > 0.5).float().eq(torch.tensor(val_targets)).float().mean().item()

        # Update learning rate based on validation AUC
        scheduler.step(val_auc)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping based on validation AUC
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print("Early stopping triggered")
                break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_losses, val_losses, train_accs, val_accs

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:20]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre     = make_preprocessor().fit(X_train, Y_train)
    
    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec, require_torch_collate=False)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, (X_train, Y_train), pre, train=True)
    val_ds       = build_dataset(spec, (X_val,   Y_val),   pre, train=False)
    train_loader = build_dataloader(spec, train_ds, is_eval=False)
    val_loader   = build_dataloader(spec, val_ds,   is_eval=True)

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane_fourtops(spec, first_batch)
    view = make_view_by_lane_fourtops(mode, first_batch, device)

    # Build model
    model = make_model(view.batch_x).to(device)

    # Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        try:
            with torch.no_grad():
                mode = detect_and_assert_lane_fourtops(spec, first_batch)
                view = make_view_by_lane_fourtops(mode, first_batch, device)
                out  = trained_model(view.batch_x)
                scores, kind = assert_binary_output(view, out)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e

    if not dryrun:
        # Persist artefacts
        base = base_from_argv0()
        persist_artefacts(base, SCRIPT_DIR, trained_model, pre, spec)

        # Save plots
        plot_train_val(tr_loss, va_loss, f"{base} Loss", os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        plot_train_val(tr_acc, va_acc, f"{base} Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))
        
        # Write JSON Summary
        summary = {
            "epochs": n_epochs      if n_epochs else None,
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

