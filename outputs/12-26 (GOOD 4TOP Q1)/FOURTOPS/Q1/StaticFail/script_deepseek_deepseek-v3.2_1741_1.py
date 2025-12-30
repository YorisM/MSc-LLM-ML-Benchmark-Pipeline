
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

# -------------------------- START OF LLM BLOCK ------------------------------
# ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import math

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# No custom dataset needed - using FourTopsDataset from harness

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_scaler = StandardScaler()
        self.kinematic_scaler = StandardScaler()
        self.phi_scaler = StandardScaler()

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
        # Reshape data to extract features properly
        X_np = X.numpy()

        # Extract global features
        global_features = X_np[:, :2]  # (n_samples, 2)

        # Extract kinematic features (E, pT, eta, phi) for all objects
        kinematic_features = []
        phi_features = []

        # Loop through 18 objects
        for obj_idx in range(18):
            start_idx = 2 + obj_idx * 5
            # Object features: [obj_id, E, pT, eta, phi]
            obj_slice = X_np[:, start_idx:start_idx+5]

            # Get mask for real objects (obj_id != 0)
            mask = obj_slice[:, 0] != 0

            # Extract kinematic features for real objects
            kinematic_features.append(obj_slice[mask, 1:4])  # E, pT, eta
            phi_features.append(obj_slice[mask, 4:5])  # phi

        # Concatenate all kinematic features
        if kinematic_features:
            kinematic_features = np.vstack(kinematic_features)
            phi_features = np.vstack(phi_features)
        else:
            kinematic_features = np.zeros((0, 3))
            phi_features = np.zeros((0, 1))

        # Fit scalers
        self.global_scaler.fit(global_features)
        self.kinematic_scaler.fit(kinematic_features)
        self.phi_scaler.fit(phi_features)

        return self

    def transform(self, X):
        X_np = X.numpy()
        X_transformed = X_np.copy()

        # Normalize global features
        X_transformed[:, :2] = self.global_scaler.transform(X_np[:, :2])

        # Normalize kinematic features for all objects
        for obj_idx in range(18):
            start_idx = 2 + obj_idx * 5
            obj_slice = X_np[:, start_idx:start_idx+5]

            # Create mask for real objects
            mask = obj_slice[:, 0] != 0

            if mask.any():
                # Normalize E, pT, eta
                kinematic = obj_slice[mask, 1:4]
                kinematic_norm = self.kinematic_scaler.transform(kinematic)
                X_transformed[mask, start_idx+1:start_idx+4] = kinematic_norm

                # Normalize phi
                phi = obj_slice[mask, 4:5]
                phi_norm = self.phi_scaler.transform(phi)
                X_transformed[mask, start_idx+4:start_idx+5] = phi_norm

        return torch.from_numpy(X_transformed).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Input shape: [batch_size, 92]
        # Global features: 2
        # Per object: 5 features, max 18 objects

        # Process global features separately
        self.global_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Process object features
        self.object_encoder = nn.Sequential(
            nn.Linear(5, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64)
        )

        # Attention mechanism for objects
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Combine global and object features
        self.combined_net = nn.Sequential(
            nn.Linear(128, 256),  # 64 (global) + 64 (objects) = 128
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x shape: [batch_size, 92]
        batch_size = batch_x.shape[0]

        # Extract global features
        global_features = batch_x[:, :2]  # [batch_size, 2]
        global_encoded = self.global_net(global_features)  # [batch_size, 64]

        # Process objects
        object_features_list = []

        for i in range(18):
            start_idx = 2 + i * 5
            obj_features = batch_x[:, start_idx:start_idx+5]  # [batch_size, 5]

            # Create mask for real objects (obj_id != 0)
            mask = obj_features[:, 0] != 0

            # Encode object features
            obj_encoded = self.object_encoder(obj_features)  # [batch_size, 64]

            # Apply attention weights
            attention_weights = self.attention(obj_encoded)  # [batch_size, 1]
            attention_weights = torch.sigmoid(attention_weights)

            # Apply mask: zero out padded objects
            attention_weights = attention_weights * mask.float().unsqueeze(1)

            # Store weighted object features
            object_features_list.append(obj_encoded * attention_weights)

        # Stack all object features
        object_features = torch.stack(object_features_list, dim=1)  # [batch_size, 18, 64]

        # Aggregate object features (sum over objects)
        object_aggregated = torch.sum(object_features, dim=1)  # [batch_size, 64]

        # Normalize by number of real objects
        real_objects_mask = batch_x[:, 2::5] != 0  # [batch_size, 18]
        num_real_objects = torch.sum(real_objects_mask.float(), dim=1, keepdim=True)  # [batch_size, 1]
        num_real_objects = torch.clamp(num_real_objects, min=1.0)
        object_aggregated = object_aggregated / num_real_objects  # [batch_size, 64]

        # Combine global and object features
        combined = torch.cat([global_encoded, object_aggregated], dim=1)  # [batch_size, 128]

        # Final classification
        output = self.combined_net(combined)  # [batch_size, 1]

        return output.squeeze(-1)  # [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            outputs = model(xb)

            # Convert labels to float for BCEWithLogitsLoss
            yb_float = yb.float()

            loss = criterion(outputs, yb_float)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * xb.shape[0]

            # Calculate accuracy
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                outputs = model(xb)
                yb_float = yb.float()
                loss = criterion(outputs, yb_float)

                val_loss += loss.item() * xb.shape[0]

                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Record metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Update scheduler
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            # Restore best model
            model.load_state_dict(best_model_state)
            break

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return model, train_losses, val_losses, train_accs, val_accs

# ---------------------------  END OF LLM-CODE BLOCK  ---------------------------

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

