
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
from sklearn.preprocessing import StandardScaler
from torch.nn import Linear, ReLU, Dropout, BatchNorm1d, Sequential, Sigmoid
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torchmetrics import AUROC
import torch.nn.functional as F

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.mask = None

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
        # Create mask for non-zero objects (first feature of each object is obj_id)
        obj_mask = (X[:, 2::5] != 0).float()  # Shape: [N, 18]
        self.mask = obj_mask

        # Flatten all object features for scaling (excluding global features)
        obj_features = X[:, 2:].reshape(X.shape[0], -1)  # Shape: [N, 18*5]
        self.scaler.fit(obj_features)
        return self

    def transform(self, X):
        # Apply mask to zero out padded objects
        if self.mask is not None:
            obj_mask = (X[:, 2::5] != 0).float()  # Shape: [N, 18]
            # Expand mask to cover all object features
            obj_mask_expanded = obj_mask.unsqueeze(-1).expand(-1, -1, 5).reshape(X.shape[0], -1)  # Shape: [N, 90]
            # Apply mask to object features
            X[:, 2:] = X[:, 2:] * obj_mask_expanded

        # Scale object features
        obj_features = X[:, 2:].reshape(X.shape[0], -1)  # Shape: [N, 90]
        obj_features = self.scaler.transform(obj_features.numpy()) if isinstance(obj_features, torch.Tensor) else self.scaler.transform(obj_features)
        X[:, 2:] = torch.as_tensor(obj_features, dtype=torch.float32).reshape(X.shape[0], -1)

        return X

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        num_global = 2
        num_objects = 18
        obj_feature_dim = 5

        # Global features processing
        self.global_net = Sequential(
            Linear(num_global, 32),
            ReLU(),
            BatchNorm1d(32),
            Dropout(0.2),
            Linear(32, 16)
        )

        # Object features processing (per object)
        self.obj_net = Sequential(
            Linear(obj_feature_dim, 32),
            ReLU(),
            BatchNorm1d(32),
            Dropout(0.2),
            Linear(32, 16)
        )

        # Attention mechanism for objects
        self.attention = Sequential(
            Linear(16, 16),
            ReLU(),
            Linear(16, 1),
            Sigmoid()
        )

        # Final classifier
        self.classifier = Sequential(
            Linear(16 + 16, 64),
            ReLU(),
            BatchNorm1d(64),
            Dropout(0.3),
            Linear(64, 32),
            ReLU(),
            BatchNorm1d(32),
            Dropout(0.2),
            Linear(32, 1)
        )

    def forward(self, batch_x):
        # batch_x shape: [B, 92]
        # Split into global and object features
        global_feat = batch_x[:, :2]  # [B, 2]
        obj_feat = batch_x[:, 2:].reshape(batch_x.shape[0], 18, 5)  # [B, 18, 5]

        # Process global features
        global_out = self.global_net(global_feat)  # [B, 16]

        # Process object features
        obj_out = self.obj_net(obj_feat)  # [B, 18, 16]

        # Apply attention to objects
        attention_weights = self.attention(obj_out)  # [B, 18, 1]
        weighted_obj = (obj_out * attention_weights).sum(dim=1)  # [B, 16]

        # Combine features
        combined = torch.cat([global_out, weighted_obj], dim=1)  # [B, 32]

        # Final classification
        logits = self.classifier(combined)  # [B, 1]
        return logits.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)
    criterion = BCEWithLogitsLoss()
    auc_metric = AUROC(task="binary").to(device)

    best_val_auc = 0
    best_model = None

    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        epoch_train_auc = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_auc += auc_metric(logits, batch_y.int())
            num_batches += 1

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        epoch_val_auc = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                epoch_val_loss += loss.item()
                epoch_val_auc += auc_metric(logits, batch_y.int())

        # Calculate metrics
        train_loss = epoch_train_loss / num_batches
        val_loss = epoch_val_loss / len(val_loader)
        train_auc = epoch_train_auc / num_batches
        val_auc = epoch_val_auc / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc.item())
        val_aucs.append(val_auc.item())

        # Update scheduler and track best model
        scheduler.step(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model.state_dict()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
    model.load_state_dict(best_model)

    return model, train_losses, val_losses, train_aucs, val_aucs

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

