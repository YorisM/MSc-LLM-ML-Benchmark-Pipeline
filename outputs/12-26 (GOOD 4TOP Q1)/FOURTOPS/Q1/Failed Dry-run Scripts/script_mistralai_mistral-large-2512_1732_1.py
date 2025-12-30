
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
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

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

# ---------- IMPORTS ----------
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.obj_ids = [0, 1, 2, 3, 4, 5, 6, 21, 22]  # Common object IDs in particle physics
        self.n_objects = 18
        self.obj_feature_size = 5
        self.global_features = 2

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True if device.type == "cuda" else False,

            "collate": None,

            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Extract global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :2].numpy()
        self.scaler.fit(global_features)
        return self

    def transform(self, X):
        # Convert to numpy for sklearn
        X_np = X.numpy()

        # Scale global features
        global_features = X_np[:, :2]
        global_features_scaled = self.scaler.transform(global_features)

        # Process object features
        obj_features = []
        for i in range(self.n_objects):
            start_idx = self.global_features + i * self.obj_feature_size
            end_idx = start_idx + self.obj_feature_size
            obj_slice = X_np[:, start_idx:end_idx]

            # One-hot encode object ID (first feature in each object)
            obj_id = obj_slice[:, 0].astype(int)
            obj_id_onehot = np.zeros((obj_id.shape[0], len(self.obj_ids)))
            for j, id_val in enumerate(self.obj_ids):
                obj_id_onehot[:, j] = (obj_id == id_val).astype(float)

            # Scale kinematic features (E, p_T, eta, phi)
            kinematic_features = obj_slice[:, 1:]
            kinematic_scaled = self.scaler.fit_transform(kinematic_features) if i == 0 else self.scaler.transform(kinematic_features)

            # Combine one-hot encoded ID with scaled kinematic features
            obj_processed = np.hstack([obj_id_onehot, kinematic_scaled])
            obj_features.append(obj_processed)

        # Stack all processed objects
        obj_features_stacked = np.stack(obj_features, axis=1)  # [n_events, n_objects, n_features_per_obj]

        # Combine global features with object features
        global_features_scaled_expanded = np.expand_dims(global_features_scaled, axis=1)
        global_features_repeated = np.repeat(global_features_scaled_expanded, self.n_objects, axis=1)

        # Final feature array: [n_events, n_objects, n_features_per_obj + global_features]
        processed_features = np.concatenate([obj_features_stacked, global_features_repeated], axis=2)

        # Convert back to torch tensor
        return torch.from_numpy(processed_features).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Determine input feature size from sample
        self.n_objects = sample_object.shape[1]
        self.input_size = sample_object.shape[2]

        # Object-level processing
        self.obj_encoder = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.BatchNorm1d(self.n_objects),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )

        # Global processing
        self.global_processor = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, batch_x):
        # batch_x shape: [batch_size, n_objects, n_features] (92 features processed to 13 per object)

        # Process each object independently
        obj_features = self.obj_encoder(batch_x)  # [batch_size, n_objects, 64]

        # Attention mechanism
        attention_weights = self.attention(obj_features)  # [batch_size, n_objects, 1]
        attended_features = torch.sum(attention_weights * obj_features, dim=1)  # [batch_size, 64]

        # Global processing
        global_features = self.global_processor(attended_features)  # [batch_size, 32]

        # Classification
        logits = self.classifier(global_features)  # [batch_size, 1]

        return logits.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model = None
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.extend(yb.cpu().numpy())

        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_targets, train_preds)
        train_losses.append(train_loss)
        train_aucs.append(train_auc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                outputs = model(xb)
                loss = criterion(outputs, yb.float())

                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                val_targets.extend(yb.cpu().numpy())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        # Update learning rate based on validation AUC
        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping and model checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered")
                break

    # Load best model
    if best_model is not None:
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

