
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

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
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import numpy as np

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.obj_ids = None
        self.n_objects = 18
        self.global_features = 2
        self.per_object_features = 4  # E, pT, eta, phi (obj_id is categorical)

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
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Extract global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :2]  # [N, 2]
        self.scaler.fit(global_features)

        # Extract per-object features (E, pT, eta, phi) for all objects
        # Reshape to [N * n_objects, per_object_features]
        per_object_features = X[:, 2:].reshape(-1, 5)[:, 1:]  # [N*18, 4]
        self.scaler.partial_fit(per_object_features)

        # Store unique object IDs for embedding
        obj_ids = X[:, 2:].reshape(-1, 5)[:, 0]  # [N*18]
        self.obj_ids = np.unique(obj_ids)
        self.obj_id_to_idx = {oid: idx for idx, oid in enumerate(self.obj_ids)}

        return self

    def transform(self, X):
        # Scale global features
        global_features = X[:, :2]  # [N, 2]
        global_features = self.scaler.transform(global_features)

        # Process per-object features
        per_object_data = X[:, 2:].reshape(-1, 5)  # [N*18, 5]
        obj_ids = per_object_data[:, 0]  # [N*18]
        obj_features = per_object_data[:, 1:]  # [N*18, 4]

        # Scale per-object features
        obj_features = self.scaler.transform(obj_features)

        # Create one-hot encoding for object IDs
        obj_id_indices = np.array([self.obj_id_to_idx.get(oid, -1) for oid in obj_ids])
        obj_id_onehot = np.zeros((obj_id_indices.shape[0], len(self.obj_ids)))
        obj_id_onehot[np.arange(obj_id_indices.shape[0]), obj_id_indices] = 1

        # Combine features: [N*18, 4 (kinematic) + n_obj_ids (one-hot)]
        processed_obj_features = np.hstack([obj_features, obj_id_onehot])

        # Reshape back to [N, 18*features_per_object]
        n_obj_features = 4 + len(self.obj_ids)
        processed_obj_features = processed_obj_features.reshape(-1, 18 * n_obj_features)

        # Combine global and per-object features
        processed_X = np.hstack([global_features, processed_obj_features])

        return torch.from_numpy(processed_X).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[1]

        # Calculate dimensions
        self.global_dim = 2
        self.obj_id_dim = len(np.unique(sample_object[:, 2:].reshape(-1, 5)[:, 0]))
        self.per_object_dim = 4 + self.obj_id_dim
        self.n_objects = 18

        # Global features branch
        self.global_branch = nn.Sequential(
            nn.Linear(self.global_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Per-object features branch (1D CNN)
        self.obj_branch = nn.Sequential(
            nn.Conv1d(self.per_object_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, batch_x):
        # batch_x shape: [batch_size, input_dim]

        # Split global and per-object features
        global_feats = batch_x[:, :self.global_dim]  # [batch_size, 2]
        obj_feats = batch_x[:, self.global_dim:].reshape(-1, self.n_objects, self.per_object_dim)  # [batch_size, 18, per_object_dim]

        # Process global features
        global_out = self.global_branch(global_feats)  # [batch_size, 64]

        # Process per-object features
        obj_feats = obj_feats.permute(0, 2, 1)  # [batch_size, per_object_dim, 18]
        obj_out = self.obj_branch(obj_feats)  # [batch_size, 128, 1]
        obj_out = obj_out.squeeze(-1)  # [batch_size, 128]

        # Combine features
        combined = torch.cat([global_out, obj_out], dim=1)  # [batch_size, 192]

        # Classifier
        logits = self.classifier(combined)  # [batch_size, 1]

        return logits.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model = None
    patience = 5
    patience_counter = 0

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_running_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)

        # Calculate AUC
        auc_score = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, "
              f"Train Acc: {epoch_acc:.4f}, Val Acc: {val_epoch_acc:.4f}, Val AUC: {auc_score:.4f}")

        # Early stopping based on AUC
        scheduler.step(auc_score)

        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_loss, val_loss, train_acc, val_acc

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

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

    # Build model
    first_batch = next(iter(train_loader))
    view        = normalise_batch(first_batch, device=device)
    model       = make_model(view.batch_x).to(device)

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
                view = normalise_batch(first_batch, device=device)
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


