
import os, sys, json, pickle, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}

EPOCHS = 10
                        
def load_data():
    to_np = lambda path: pd.read_csv(path).values
    X_train = to_np(DATASET["X_train"])
    Y_train = to_np(DATASET["Y_train"]).ravel()
    X_val   = to_np(DATASET["X_val"])
    Y_val   = to_np(DATASET["Y_val"]).ravel()
    return X_train, Y_train, X_val, Y_val

def make_loaders(X_train, Y_train, X_val, Y_val, batch=1024):
    from torch.utils.data import TensorDataset, DataLoader
    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train))
    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val))
    return (DataLoader(train, batch_size=batch, shuffle=True),
            DataLoader(val, batch_size=batch))
                        
# ----------------  START OF LLM BLOCK  ----------------
# Imports: torch, numpy, pandas, matplotlib, sklearn are already available
# via the wrapper.  Only import extra std-lib modules or torch.nn sub-modules
# you actually use.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import math

class MyPreprocessor:
    #    Must implement:
    #   - fit(X: np.ndarray | torch.Tensor, y: np.ndarray | None) -> self
    #   - transform(X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor

    def __init__(self):
        self.scalers = {}
        self.epsilon = 1e-8
        self.obj_types = set()  # To store unique object types
        self.valid_indices = None
        self.num_features = None

    def fit(self, X, y=None):
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        # Find indices where there's actual data (non-zero object types)
        object_indices = [i for i in range(2, X.shape[1], 5) if np.sum(np.abs(X[:, i])) > 0]
        self.valid_indices = [0, 1]  # Always include ET_miss and phi_ET_miss
        
        for obj_idx in object_indices:
            # Include the object type and its 4-vector components
            self.valid_indices.extend([obj_idx, obj_idx+1, obj_idx+2, obj_idx+3, obj_idx+4])
        
        # Sort indices to maintain order
        self.valid_indices.sort()
        
        # Record unique object types for one-hot encoding
        for obj_idx in object_indices:
            unique_objs = np.unique(X[:, obj_idx])
            self.obj_types.update([int(obj) for obj in unique_objs if obj > 0])
        
        # Create feature scalers for each physical quantity type
        # ET_miss and phi_ET_miss
        self.scalers['ET_miss'] = StandardScaler().fit(X[:, 0].reshape(-1, 1))
        self.scalers['phi_ET_miss'] = StandardScaler().fit(X[:, 1].reshape(-1, 1))
        
        # Energy, pT, eta, phi for all objects
        E_indices = [i+1 for i in object_indices]
        pT_indices = [i+2 for i in object_indices]
        eta_indices = [i+3 for i in object_indices]
        phi_indices = [i+4 for i in object_indices]
        
        if E_indices:
            self.scalers['E'] = StandardScaler().fit(X[:, E_indices].reshape(-1, 1))
            self.scalers['pT'] = StandardScaler().fit(X[:, pT_indices].reshape(-1, 1))
            self.scalers['eta'] = StandardScaler().fit(X[:, eta_indices].reshape(-1, 1))
            self.scalers['phi'] = StandardScaler().fit(X[:, phi_indices].reshape(-1, 1))
        
        # Calculate derived features statistics
        # We'll create these in transform but need scalers
        jet_counts = self.count_objects_by_type(X)
        HT_values = self.calculate_HT(X)
        MET_values = X[:, 0]  # ET_miss
        
        self.scalers['jet_counts'] = {obj_type: StandardScaler().fit(counts.reshape(-1, 1)) 
                              for obj_type, counts in jet_counts.items()}
        self.scalers['HT'] = StandardScaler().fit(HT_values.reshape(-1, 1))
        self.scalers['MET/HT'] = StandardScaler().fit((MET_values / (HT_values + self.epsilon)).reshape(-1, 1))
        
        # Calculate how many features we'll have after transformation
        n_object_types = len(self.obj_types)
        n_jet_counts = len(self.obj_types)
        n_base_features = 2  # ET_miss and phi_ET_miss
        n_object_features = sum(1 for i in range(2, X.shape[1], 5) if np.sum(np.abs(X[:, i])) > 0) * 4  # E, pT, eta, phi
        n_derived_features = 2  # HT and MET/HT
        
        self.num_features = n_base_features + n_object_features + n_derived_features + n_jet_counts
        
        return self

    def count_objects_by_type(self, X):
        # Count objects of each type per event
        object_counts = defaultdict(lambda: np.zeros(X.shape[0]))
        
        for obj_idx in range(2, X.shape[1], 5):
            obj_types = X[:, obj_idx]
            for obj_type in self.obj_types:
                object_counts[obj_type] += (obj_types == obj_type).astype(int)
        
        return object_counts

    def calculate_HT(self, X):
        # HT is the scalar sum of all jet pT
        HT = np.zeros(X.shape[0])
        
        for i in range(4, X.shape[1], 5):  # pT indices (obj_idx + 2)
            # Only sum for non-zero entries
            mask = X[:, i-2] > 0  # Check if object type > 0
            HT += np.where(mask, X[:, i], 0)
        
        return HT

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        
        # Initialize output array with the calculated number of features
        output = np.zeros((X.shape[0], self.num_features))
        
        # Apply scaling to ET_miss and phi_ET_miss
        output[:, 0] = self.scalers['ET_miss'].transform(X[:, 0].reshape(-1, 1)).flatten()
        output[:, 1] = self.scalers['phi_ET_miss'].transform(X[:, 1].reshape(-1, 1)).flatten()
        
        # Track current feature index
        feature_idx = 2
        
        # Process each object's E, pT, eta, phi
        for obj_idx in range(2, X.shape[1], 5):
            if obj_idx in self.valid_indices:
                # Object exists, process its 4-vector
                mask = X[:, obj_idx] > 0  # Check if object type > 0
                
                # Energy
                scaled_E = np.zeros(X.shape[0])
                scaled_E[mask] = self.scalers['E'].transform(X[mask, obj_idx+1].reshape(-1, 1)).flatten()
                output[:, feature_idx] = scaled_E
                feature_idx += 1
                
                # pT
                scaled_pT = np.zeros(X.shape[0])
                scaled_pT[mask] = self.scalers['pT'].transform(X[mask, obj_idx+2].reshape(-1, 1)).flatten()
                output[:, feature_idx] = scaled_pT
                feature_idx += 1
                
                # eta
                scaled_eta = np.zeros(X.shape[0])
                scaled_eta[mask] = self.scalers['eta'].transform(X[mask, obj_idx+3].reshape(-1, 1)).flatten()
                output[:, feature_idx] = scaled_eta
                feature_idx += 1
                
                # phi
                scaled_phi = np.zeros(X.shape[0])
                scaled_phi[mask] = self.scalers['phi'].transform(X[mask, obj_idx+4].reshape(-1, 1)).flatten()
                output[:, feature_idx] = scaled_phi
                feature_idx += 1
        
        # Add object counts
        jet_counts = self.count_objects_by_type(X)
        for obj_type in sorted(self.obj_types):
            output[:, feature_idx] = self.scalers['jet_counts'][obj_type].transform(jet_counts[obj_type].reshape(-1, 1)).flatten()
            feature_idx += 1
        
        # Add HT and MET/HT features
        HT_values = self.calculate_HT(X)
        MET_values = X[:, 0]
        
        output[:, feature_idx] = self.scalers['HT'].transform(HT_values.reshape(-1, 1)).flatten()
        feature_idx += 1
        
        output[:, feature_idx] = self.scalers['MET/HT'].transform(
            (MET_values / (HT_values + self.epsilon)).reshape(-1, 1)
        ).flatten()
        
        return torch.tensor(output, dtype=torch.float32)

def make_preprocessor():
    return MyPreprocessor()

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.lin1(x)))
        out = self.bn2(self.lin2(out))
        out += residual
        out = F.relu(out)
        return out

class FourTopClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # Size of hidden layers
        h1, h2, h3, h4 = 256, 256, 128, 64
        
        # Input projection
        self.input_layer = nn.Linear(input_dim, h1)
        self.bn_input = nn.BatchNorm1d(h1)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(h1)
        self.res_block2 = ResidualBlock(h1)
        
        # Deeper layers
        self.hidden1 = nn.Linear(h1, h2)
        self.bn1 = nn.BatchNorm1d(h2)
        self.hidden2 = nn.Linear(h2, h3)
        self.bn2 = nn.BatchNorm1d(h3)
        self.hidden3 = nn.Linear(h3, h4)
        self.bn3 = nn.BatchNorm1d(h4)
        
        # Output layer
        self.output = nn.Linear(h4, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Input projection
        x = F.relu(self.bn_input(self.input_layer(x)))
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.dropout(x)
        x = self.res_block2(x)
        
        # Deeper layers
        x = F.relu(self.bn1(self.hidden1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.hidden2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.hidden3(x)))
        
        # Output layer
        x = self.output(x).squeeze(1)
        return x

def make_model(input_dim: int):
    return FourTopClassifier(input_dim)

EPOCHS = 30  # Define training epochs

class AUCLoss(nn.Module):
    def __init__(self):
        super(AUCLoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        # First, use sigmoid to get probabilities for binary
        y_pred = torch.sigmoid(y_pred)
        
        # Sort predictions and get corresponding class labels
        sorted_indices = torch.argsort(y_pred, descending=True)
        sorted_labels = y_true[sorted_indices]
        
        # Calculate true positives and false positives
        P = torch.sum(y_true == 1).float()
        N = torch.sum(y_true == 0).float()
        
        # Edge case check
        if P == 0 or N == 0:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        
        # Calculate TPR and FPR at each threshold
        tp_cumsum = torch.cumsum(sorted_labels, dim=0)
        fp_cumsum = torch.cumsum((1 - sorted_labels), dim=0)
        
        # Normalize to get rates
        tpr = tp_cumsum / P
        fpr = fp_cumsum / N
        
        # Calculate AUC using trapezoidal rule
        # Width of trapezoids
        width = fpr[1:] - fpr[:-1]
        
        # Height is average of adjacent TPRs
        height = (tpr[1:] + tpr[:-1]) / 2.0
        
        # AUC is sum of trapezoid areas
        auc = torch.sum(width * height)
        
        # We want to maximize AUC, so return negative for minimization
        return -auc

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=False
    )
    
    # Binary cross entropy loss
    criterion = nn.BCEWithLogitsLoss()
    
    # Track metrics
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
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
    
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
    X_tr, y_tr, X_va, y_va = load_data()
    pre = make_preprocessor();  pre.fit(X_tr, y_tr)
    X_tr = pre.transform(X_tr); X_va = pre.transform(X_va)
    tr_loader, va_loader = make_loaders(X_tr, y_tr, X_va, y_va)

    # 2. Build model
    model = make_model(input_dim=X_tr.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    trained, tr_loss, va_loss, tr_acc, va_acc = train_model(
        model, tr_loader, va_loader, epochs=n_epochs
    )

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy = torch.zeros(8, X_tr.shape[1])      # 8 fake events
        try:
            _ = trained(pre.transform(toy))
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return  # no files in dry-run

    # 4. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
    torch.save(trained.state_dict(), f"{base}_state.pt")
    torch.jit.script(trained).save(f"{base}_scripted.pt")
    torch.jit.script(pre).save(f"{base}_preproc.pt")
    with open(f"{base}_pre.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

