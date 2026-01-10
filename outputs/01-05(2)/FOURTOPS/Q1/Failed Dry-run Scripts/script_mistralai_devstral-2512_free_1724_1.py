
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
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import copy

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.mask_value = -999.0  # Special value to mark padding

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 1024}
        }

    def fit(self, X, y=None):
        # Replace padding zeros with mask_value for scaling
        X_masked = X.clone()
        X_masked[X_masked == 0] = self.mask_value

        # Scale all features except object identifiers (which are at positions 2,7,12,...)
        # We'll scale everything and then restore the object identifiers
        obj_mask = torch.zeros_like(X_masked, dtype=torch.bool)
        for i in range(18):
            obj_mask[:, 2 + i*5] = True

        # Save object identifiers
        obj_ids = X_masked[obj_mask].reshape(-1, 18)

        # Scale non-identifier features
        non_obj_features = X_masked[~obj_mask].reshape(-1, 88)
        self.scaler.fit(non_obj_features.numpy())

        return self

    def transform(self, X):
        X_masked = X.clone()
        X_masked[X_masked == 0] = self.mask_value

        # Create mask for object identifiers
        obj_mask = torch.zeros_like(X_masked, dtype=torch.bool)
        for i in range(18):
            obj_mask[:, 2 + i*5] = True

        # Save and restore object identifiers
        obj_ids = X_masked[obj_mask].reshape(-1, 18)

        # Scale non-identifier features
        non_obj_features = X_masked[~obj_mask].reshape(-1, 88)
        scaled_features = self.scaler.transform(non_obj_features.numpy())

        # Reconstruct the tensor
        X_scaled = torch.zeros_like(X_masked)
        X_scaled[~obj_mask] = torch.from_numpy(scaled_features).reshape(-1)
        X_scaled[obj_mask] = obj_ids.reshape(-1)

        # Restore original zeros (padding)
        X_scaled[X_scaled == self.mask_value] = 0

        return X_scaled

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        self.num_objects = 18
        self.obj_dim = 5  # obj_id, E, pT, eta, phi

        # Global features (missing ET)
        self.global_dim = 2

        # Object processing
        self.obj_encoder = nn.Sequential(
            nn.Linear(self.obj_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Attention mechanism for objects
        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # Global feature processing
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Combined processing
        self.combined = nn.Sequential(
            nn.Linear(32 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x shape: [B, 92]
        B = batch_x.size(0)

        # Extract global features (first 2 elements)
        global_feat = batch_x[:, :2]  # [B, 2]

        # Extract object features (remaining 90 elements = 18 objects * 5 features)
        obj_feats = batch_x[:, 2:].view(B, self.num_objects, self.obj_dim)  # [B, 18, 5]

        # Process each object
        obj_encoded = self.obj_encoder(obj_feats)  # [B, 18, 32]

        # Attention weights
        attn_weights = F.softmax(self.attention(obj_encoded), dim=1)  # [B, 18, 1]
        obj_aggregated = (obj_encoded * attn_weights).sum(dim=1)  # [B, 32]

        # Process global features
        global_encoded = self.global_encoder(global_feat)  # [B, 32]

        # Combine features
        combined = torch.cat([obj_aggregated, global_encoded], dim=1)  # [B, 64]

        # Final classification
        out = self.combined(combined)  # [B, 1]

        return out.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    best_model = None
    best_auc = 0.0
    patience = 10
    patience_counter = 0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_targets = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += preds.eq(batch_y.byte()).sum().item()
            train_total += batch_x.size(0)

            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))

                val_loss += loss.item() * batch_x.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += preds.eq(batch_y.byte()).sum().item()
                val_total += batch_x.size(0)

                val_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                val_targets.extend(batch_y.detach().cpu().numpy())

        # Calculate metrics
        train_loss = train_loss / train_total
        val_loss = val_loss / val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        train_auc = roc_auc_score(train_targets, train_preds)
        val_auc = roc_auc_score(val_targets, val_preds)

        # Update scheduler
        scheduler.step(val_auc)

        # Store history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Early stopping and model saving
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} - "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

    return best_model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

