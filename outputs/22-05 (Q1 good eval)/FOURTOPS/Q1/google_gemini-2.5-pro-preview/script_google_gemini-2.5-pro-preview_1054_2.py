
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
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.means = None
        self.stds = None
        self.epsilon = 1e-7  # Small constant to prevent division by zero
        self.feature_indices_to_standardize = []
        self._fitted = False

    def fit(self, X, y=None):
        # X shape: (N, 92)
        # Index 0: E_T_miss
        # Index 1: phi_Et_miss (do not standardize, convert to cos/sin)
        # Indices 2-6: object 1 (obj_id, E, p_T, eta, phi)
        # ... up to 18 objects

        self.feature_indices_to_standardize.append(0) # E_T_miss

        num_objects = 18
        features_per_object = 5
        for i in range(num_objects):
            base_idx = 2 + i * features_per_object
            self.feature_indices_to_standardize.append(base_idx + 0) # obj_id
            self.feature_indices_to_standardize.append(base_idx + 1) # E
            self.feature_indices_to_standardize.append(base_idx + 2) # p_T
            self.feature_indices_to_standardize.append(base_idx + 3) # eta
            # obj_phi (base_idx + 4) is not standardized, converted to cos/sin
        
        # Ensure indices are unique and sorted, though current construction guarantees this
        # self.feature_indices_to_standardize = sorted(list(set(self.feature_indices_to_standardize)))

        selected_features = X[:, self.feature_indices_to_standardize]
        self.means = torch.mean(selected_features, dim=0)
        self.stds = torch.std(selected_features, dim=0)
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        # X shape: (batch_size, 92)
        transformed_parts = []

        # Process E_T_miss (index 0)
        e_t_miss_std = (X[:, 0] - self.means[0]) / (self.stds[0] + self.epsilon)
        transformed_parts.append(e_t_miss_std.unsqueeze(1)) # Shape: (batch_size, 1)

        # Process phi_Et_miss (index 1)
        phi_et_miss = X[:, 1]
        transformed_parts.append(torch.cos(phi_et_miss).unsqueeze(1)) # Shape: (batch_size, 1)
        transformed_parts.append(torch.sin(phi_et_miss).unsqueeze(1)) # Shape: (batch_size, 1)

        # Standardized feature index tracker
        std_idx_counter = 1 # starts from 1 because self.means[0] was for E_T_miss

        num_objects = 18
        features_per_object = 5
        for i in range(num_objects):
            obj_base_idx = 2 + i * features_per_object
            
            # obj_id (offset 0)
            obj_id_std = (X[:, obj_base_idx + 0] - self.means[std_idx_counter]) / (self.stds[std_idx_counter] + self.epsilon)
            transformed_parts.append(obj_id_std.unsqueeze(1))
            std_idx_counter += 1

            # E (offset 1)
            E_std = (X[:, obj_base_idx + 1] - self.means[std_idx_counter]) / (self.stds[std_idx_counter] + self.epsilon)
            transformed_parts.append(E_std.unsqueeze(1))
            std_idx_counter += 1

            # p_T (offset 2)
            p_T_std = (X[:, obj_base_idx + 2] - self.means[std_idx_counter]) / (self.stds[std_idx_counter] + self.epsilon)
            transformed_parts.append(p_T_std.unsqueeze(1))
            std_idx_counter += 1

            # eta (offset 3)
            eta_std = (X[:, obj_base_idx + 3] - self.means[std_idx_counter]) / (self.stds[std_idx_counter] + self.epsilon)
            transformed_parts.append(eta_std.unsqueeze(1))
            std_idx_counter += 1

            # phi (offset 4)
            obj_phi = X[:, obj_base_idx + 4]
            transformed_parts.append(torch.cos(obj_phi).unsqueeze(1))
            transformed_parts.append(torch.sin(obj_phi).unsqueeze(1))
            
        # Concatenate all parts
        # Output tensor shape: (batch_size, 1 + 2 + 18 * (1+1+1+1+2)) = (batch_size, 3 + 18*6) = (batch_size, 3+108) = (batch_size, 111)
        return torch.cat(transformed_parts, dim=1)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
def make_model(input_dim: int):
    # input_dim expected to be 111 from the preprocessor
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        
        nn.Linear(64, 1) # Output raw logits for BCEWithLogitsLoss
    )
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss() # Numerically stable
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, min_lr=1e-6)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs) # Shape: (batch_size, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            # Accuracy calculation
            # For BCEWithLogitsLoss, output > 0 maps to probability > 0.5
            preds = (outputs > 0.0).float()
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)
        
        epoch_train_loss = running_loss / total_predictions
        epoch_train_acc = correct_predictions / total_predictions
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val_predictions = 0
        total_val_predictions = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
                preds = (outputs > 0.0).float()
                correct_val_predictions += (preds == labels).sum().item()
                total_val_predictions += labels.size(0)

        epoch_val_loss = running_val_loss / total_val_predictions
        epoch_val_acc = correct_val_predictions / total_val_predictions
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Step the scheduler based on validation loss
        scheduler.step(epoch_val_loss)
        
        # Optional: Print epoch stats (comment out for submission if too verbose)
        # print(f"Epoch {epoch+1}/{epochs} -> "
        #       f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
        #       f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} | "
        #       f"LR: {optimizer.param_groups[0]['lr']:.1e}")

    return model, train_losses, val_losses, train_accuracies, val_accuracies

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

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

