
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
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader # DataLoader only for type hints

# 0. ---------- IMPORTS ----------
# (Imports are already covered above)

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.means_ = None
        self.stds_ = None
        self.epsilon_ = 1e-7
        # Define constants for feature indices
        self.met_idx = 0
        self.met_phi_idx = 1
        self.obj_feature_offset = 2
        self.num_kinematic_features_per_obj = 5 # obj_id, E, p_T, eta, phi
        self.num_objects = 18

        # Calculate the number of output features after transformation
        # E_T_miss (1) + (cos(phi_Et_miss), sin(phi_Et_miss)) (2) = 3 global features
        # For each object: obj_id (1), E (1), p_T (1), eta (1), (cos(phi), sin(phi)) (2) = 6 features per object
        self.num_output_features = 1 + 2 + self.num_objects * (1 + 1 + 1 + 1 + 2) # 3 + 18 * 6 = 3 + 108 = 111

    def _feature_engineer(self, X: torch.Tensor) -> torch.Tensor:
        # X shape: (N, 92)
        N = X.shape[0]
        X_transformed = torch.zeros(N, self.num_output_features, device=X.device, dtype=X.dtype)

        # Global features
        X_transformed[:, 0] = X[:, self.met_idx]  # E_T_miss
        
        met_phi = X[:, self.met_phi_idx]
        X_transformed[:, 1] = torch.cos(met_phi)
        X_transformed[:, 2] = torch.sin(met_phi)

        # Per-object features
        current_out_idx = 3
        for i in range(self.num_objects):
            obj_start_idx_raw = self.obj_feature_offset + i * self.num_kinematic_features_per_obj
            
            # obj_id, E, p_T, eta
            X_transformed[:, current_out_idx : current_out_idx + 4] = X[:, obj_start_idx_raw : obj_start_idx_raw + 4]
            
            # phi transformation
            obj_phi = X[:, obj_start_idx_raw + 4]
            X_transformed[:, current_out_idx + 4] = torch.cos(obj_phi)
            X_transformed[:, current_out_idx + 5] = torch.sin(obj_phi)
            
            current_out_idx += 6
        # Output X_transformed shape: (N, 111)
        return X_transformed

    def fit(self, X: torch.Tensor, y: torch.Tensor = None) -> 'MyPreprocessor':
        X_transformed = self._feature_engineer(X)
        self.means_ = torch.mean(X_transformed, dim=0, keepdim=True)
        self.stds_ = torch.std(X_transformed, dim=0, keepdim=True)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        
        X_transformed = self._feature_engineer(X)
        # Move stats to the device of X if they are not already there
        # This is good practice, though for this problem, device is likely CPU consistently.
        if self.means_.device != X.device:
            self.means_ = self.means_.to(X.device)
            self.stds_ = self.stds_.to(X.device)
            
X_scaled = (X_transformed - self.means_) / (self.stds_ + self.epsilon_)
        # Output X_scaled shape: (N, 111)
        return X_scaled

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
def make_model(input_dim: int) -> nn.Module:
    # A simple Multi-Layer Perceptron (MLP)
    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)  # Output one logit for binary classification
    )
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 15 # Number of training epochs

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                epochs: int) -> tuple[nn.Module, list[float], list[float], list[float], list[float]]:
    
    device = torch.device("cpu") # Assuming CPU as per constraints
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss() # Numerically stable loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizer with a common learning rate

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            
            outputs = model(inputs) # outputs are logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
            # Calculate training accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        
        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
        
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Verbose output (optional, can be removed if not desired by evaluation environment)
        # print(f"Epoch {epoch+1}/{epochs} - "
        #       f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
        #       f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

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

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

