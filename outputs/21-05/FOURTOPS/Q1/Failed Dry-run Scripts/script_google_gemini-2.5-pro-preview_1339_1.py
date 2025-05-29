
import os, sys, pickle, torch, gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

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
    X_train = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train),
            torch.from_numpy(Y_train),
            torch.from_numpy(X_val),
            torch.from_numpy(Y_val))

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))
                        
# ----------------  START OF LLM BLOCK  ----------------

import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import math

# 0. ---------- IMPORTS ----------
# (Additional imports if any, already included above)

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.num_features = None

    def fit(self, X, y=None): # X shape: (N, 92)
        # Extract global features (MET, MET_phi)
        met = X[:, 0]      # (N,)
        met_phi = X[:, 1]  # (N,)

        # Preprocess global features
        # Use relu to ensure non-negativity before log, though MET magnitude should be positive.
        processed_met_log = torch.log1p(torch.relu(met))         # (N,)
        processed_met_phi_cos = torch.cos(met_phi)               # (N,)
        processed_met_phi_sin = torch.sin(met_phi)               # (N,)

        # Extract object features. Original shape (N, 90) -> (N, 18, 5)
        # Each object: obj_id, E, pT, eta, phi
        X_objects = X[:, 2:].reshape(X.shape[0], 18, 5) # (N, 18, 5)

        obj_ids = X_objects[:, :, 0]       # (N, 18)
        obj_E = X_objects[:, :, 1]         # (N, 18)
        obj_pT = X_objects[:, :, 2]        # (N, 18)
        obj_eta = X_objects[:, :, 3]       # (N, 18)
        obj_phi = X_objects[:, :, 4]       # (N, 18)

        # Preprocess object features
        # For padded objects (E, pT, eta, phi often 0), these transformations are mostly well-behaved:
        # log1p(relu(0)) = log1p(0) = 0.
        # cos(0)=1, sin(0)=0.
        # eta=0.
        # obj_id for padded objects might be 0.
        processed_obj_E_log = torch.log1p(torch.relu(obj_E))    # (N, 18)
        processed_obj_pT_log = torch.log1p(torch.relu(obj_pT))   # (N, 18)
        # Eta is kept as is. It can be negative and is often symmetric around 0.
        # Phi is cyclical, convert to Cartesian coordinates.
        processed_obj_phi_cos = torch.cos(obj_phi)              # (N, 18)
        processed_obj_phi_sin = torch.sin(obj_phi)              # (N, 18)

        # Stack engineered object features: (obj_id, E_log, pT_log, eta, phi_cos, phi_sin)
        # These are 6 features per object.
        per_object_engineered_features = torch.stack([
            obj_ids,                     # (N, 18)
            processed_obj_E_log,         # (N, 18)
            processed_obj_pT_log,        # (N, 18)
            obj_eta,                     # (N, 18)
            processed_obj_phi_cos,       # (N, 18)
            processed_obj_phi_sin        # (N, 18)
        ], dim=2) # Resulting shape: (N, 18, 6)

        # Flatten object features from (N, 18, 6) to (N, 18*6) = (N, 108)
        flattened_object_features = per_object_engineered_features.reshape(X.shape[0], -1)

        # Concatenate all engineered features
        # Global features: 3 (log_MET, cos(MET_phi), sin(MET_phi))
        # Object features: 108 (18 objects * 6 features/object)
        # Total features: 3 + 108 = 111
        final_features = torch.cat([
            processed_met_log.unsqueeze(1),       # (N, 1)
            processed_met_phi_cos.unsqueeze(1),   # (N, 1)
            processed_met_phi_sin.unsqueeze(1),   # (N, 1)
            flattened_object_features             # (N, 108)
        ], dim=1) # Shape: (N, 111)

        # Calculate mean and std for scaling
        self.mean = torch.mean(final_features, dim=0)
        self.std = torch.std(final_features, dim=0)
        # Prevent division by zero for features with zero variance (e.g. constant features)
        self.std[self.std == 0] = 1.0 
        
        self.num_features = final_features.shape[1]
        return self

    def transform(self, X): # X shape: (N, 92)
        if self.mean is None or self.std is None:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        # Repeat transformations as in fit()
        met = X[:, 0]
        met_phi = X[:, 1]
        processed_met_log = torch.log1p(torch.relu(met))
        processed_met_phi_cos = torch.cos(met_phi)
        processed_met_phi_sin = torch.sin(met_phi)

        X_objects = X[:, 2:].reshape(X.shape[0], 18, 5)
        obj_ids = X_objects[:, :, 0]
        obj_E = X_objects[:, :, 1]
        obj_pT = X_objects[:, :, 2]
        obj_eta = X_objects[:, :, 3]
        obj_phi = X_objects[:, :, 4]

        processed_obj_E_log = torch.log1p(torch.relu(obj_E))
        processed_obj_pT_log = torch.log1p(torch.relu(obj_pT))
        processed_obj_phi_cos = torch.cos(obj_phi)
        processed_obj_phi_sin = torch.sin(obj_phi)

        per_object_engineered_features = torch.stack([
            obj_ids, processed_obj_E_log, processed_obj_pT_log, 
            obj_eta, processed_obj_phi_cos, processed_obj_phi_sin
        ], dim=2)
        flattened_object_features = per_object_engineered_features.reshape(X.shape[0], -1)

        final_features = torch.cat([
            processed_met_log.unsqueeze(1),
            processed_met_phi_cos.unsqueeze(1),
            processed_met_phi_sin.unsqueeze(1),
            flattened_object_features
        ], dim=1)

        # Apply scaling using stored mean and std
        scaled_features = (final_features - self.mean) / self.std
        return scaled_features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
def make_model(input_dim: int):
    # A relatively small MLP with BatchNorm and Dropout for regularization.
    # Model size is well below 50MB.
    # Layers: Input -> 128 (BN, ReLU, Dropout) -> 64 (BN, ReLU, Dropout) -> 1 (Output)
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1)  # Single output for BCEWithLogitsLoss
    )
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20 # Chosen based on typical convergence for similar tasks and time constraints.
            # With batch_size=1024, this should be well within 2h.

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cpu") # Constraints imply CPU-only environment
    model.to(device)

    criterion = nn.BCEWithLogitsLoss() # numerically stable log-sum-exp for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Scheduler to reduce learning rate if validation loss plateaus.
    # 'verbose' argument is not supported/allowed.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs) # Shape: (batch_size, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            # Accuracy: convert logits to probabilities, then to 0/1 predictions
            preds = torch.sigmoid(outputs) > 0.5 
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        
        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = correct_train / total_train
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad(): # Disable gradient calculation for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)
        
        # This print statement is for local testing, would be removed or managed by calling script in production.
        # print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
        #       f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.1e}")

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    pre = make_preprocessor()
    pre.fit(X_train, Y_train)
    X_train = pre.transform(X_train)
    X_val = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    model = make_model(input_dim=X_train.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy = torch.zeros(8, X_train.shape[1])      # 8 fake events
        try:
            _ = trained_model(pre.transform(toy))
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return  # no files in dry-run

    # 4. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

