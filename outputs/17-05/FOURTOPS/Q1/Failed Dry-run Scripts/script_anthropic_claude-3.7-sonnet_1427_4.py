
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

class MyPreprocessor:
    def __init__(self):
        self.obj_indices = np.arange(2, 92, 5)
        self.kinematic_indices = None
        self.scale_factors = None
        self.mean_values = None
        self.nonzero_mask = None

    def fit(self, X, y=None):
        # Convert to numpy if it's a tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        # Identify non-zero object indices (first value in each object)
        self.nonzero_mask = (X[:, self.obj_indices] != 0)
        
        # Extract kinematic features (energy, pt, eta, phi for each object)
        self.kinematic_indices = []
        for i in range(18):
            start_idx = 2 + i * 5
            # Skip object identifier, take E, pT, eta, phi
            self.kinematic_indices.extend([start_idx + j for j in range(1, 5)])
        
        # Calculate scale factors for numerical stability
        nonzero_values = X[:, self.kinematic_indices]
        nonzero_values = nonzero_values[nonzero_values != 0]
        self.mean_values = np.mean(X[:, self.kinematic_indices], axis=0)
        self.scale_factors = np.std(X[:, self.kinematic_indices], axis=0)
        self.scale_factors[self.scale_factors == 0] = 1.0
        
        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
            
        # Create a copy to avoid modifying the original data
        X_transformed = X.copy()
        
        # Normalize missing ET and missing ET phi
        X_transformed[:, 0] = (X_transformed[:, 0] - np.mean(X_transformed[:, 0])) / np.std(X_transformed[:, 0])
        
        # Normalize kinematic features
        X_transformed[:, self.kinematic_indices] = (X_transformed[:, self.kinematic_indices] - self.mean_values) / self.scale_factors
        
        # Extract object presence flags (binary features indicating if an object exists)
        obj_exists = (X[:, self.obj_indices] != 0).astype(np.float32)
        
        # Count objects of each type
        obj_counts = {}
        for i in range(18):
            obj_idx = 2 + i * 5
            obj_type = X[:, obj_idx]
            for obj in np.unique(obj_type):
                if obj != 0:  # Skip padding
                    key = f"obj_{int(obj)}_count"
                    if key not in obj_counts:
                        obj_counts[key] = np.zeros(X.shape[0], dtype=np.float32)
                    obj_counts[key] += (obj_type == obj).astype(np.float32)
        
        # Create derived features
        # Total energy, transverse momentum, etc.
        total_energy = np.zeros(X.shape[0], dtype=np.float32)
        total_pt = np.zeros(X.shape[0], dtype=np.float32)
        
        for i in range(18):
            obj_idx = 2 + i * 5
            energy_idx = obj_idx + 1
            pt_idx = obj_idx + 2
            
            # Only add values for non-zero objects
            mask = X[:, obj_idx] != 0
            total_energy[mask] += X[mask, energy_idx]
            total_pt[mask] += X[mask, pt_idx]
        
        # Normalize these derived features
        total_energy = (total_energy - np.mean(total_energy)) / np.std(total_energy)
        total_pt = (total_pt - np.mean(total_pt)) / np.std(total_pt)
        
        # Combine all features
        features = [X_transformed]  # Original features (normalized)
        
        # Add object existence features
        features.append(obj_exists)
        
        # Add object count features
        for key, value in obj_counts.items():
            features.append(value.reshape(-1, 1))
        
        # Add derived features
        features.append(total_energy.reshape(-1, 1))
        features.append(total_pt.reshape(-1, 1))
        
        # Concatenate all features
        result = np.concatenate([feat if feat.ndim > 1 else feat.reshape(-1, 1) for feat in features], axis=1)
        
        return torch.tensor(result, dtype=torch.float32)

def make_preprocessor():
    return MyPreprocessor()

class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x = x + residual
        x = F.relu(x)
        return x

class PhysicsClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Input projection layer
        self.input_layer = nn.Linear(input_dim, 256)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(256, 512)
        self.res_block2 = ResidualBlock(256, 512)
        self.res_block3 = ResidualBlock(256, 512)
        
        # Output layers with dropout for regularization
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.input_bn(x)
        x = F.relu(self.input_layer(x))
        
        # Apply residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Output layers
        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x).squeeze(1)
        
        return x

def make_model(input_dim: int):
    return PhysicsClassifier(input_dim)

class ParticleDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

EPOCHS = 30

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Binary cross entropy with logits loss (combines sigmoid and BCELoss)
    criterion = nn.BCEWithLogitsLoss()
    
    # Use AdamW optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler to reduce learning rate over time
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=False)
    
    # Storage for metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_scores = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            predicted = torch.sigmoid(outputs) > 0.5
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Store predictions for AUC calculation
            y_true.extend(targets.cpu().numpy())
            y_scores.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        epoch_auc = roc_auc_score(y_true, y_scores)
        
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_scores = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy
                predicted = torch.sigmoid(outputs) > 0.5
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Store predictions for AUC calculation
                y_true.extend(targets.cpu().numpy())
                y_scores.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct / total
        epoch_auc = roc_auc_score(y_true, y_scores)
        
        val_loss.append(epoch_loss)
        val_acc.append(epoch_acc)
        
        # Update learning rate based on AUC
        scheduler.step(epoch_auc)
    
    return model, train_loss, val_loss, train_acc, val_acc
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

