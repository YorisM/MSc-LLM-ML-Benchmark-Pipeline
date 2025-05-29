
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

# 0. ---------- IMPORTS ----------
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.object_scalers = {}
        self.et_miss_scaler = StandardScaler()
        self.n_features = None
        self.epsilon = 1e-8
        
    def fit(self, X, y=None):
        # Get missing ET and phi statistics
        et_miss_data = X[:, 0:2].numpy()
        self.et_miss_scaler.fit(et_miss_data)
        
        # Process each object type
        max_objects = 18
        for obj_idx in range(max_objects):
            # Extract features for this object type across all events
            start_idx = 2 + obj_idx * 5
            end_idx = start_idx + 5
            
            # Get object features: obj_id, E, pT, eta, phi
            obj_features = X[:, start_idx:end_idx].clone()
            
            # Create mask for valid objects (non-zero obj_id)
            valid_mask = obj_features[:, 0] != 0
            
            if valid_mask.sum() > 0:
                # Extract only valid objects for scaling
                valid_features = obj_features[valid_mask][:, 1:].numpy()  # Skip obj_id
                
                # Create and fit scaler for this object type
                scaler = StandardScaler()
                scaler.fit(valid_features)
                self.object_scalers[obj_idx] = scaler
        
        return self
    
    def transform(self, X):
        batch_size = X.shape[0]
        processed_features = []
        
        # Process missing ET
        et_miss_data = X[:, 0:2].numpy()
        scaled_et_miss = torch.tensor(self.et_miss_scaler.transform(et_miss_data), dtype=torch.float32)
        processed_features.append(scaled_et_miss)  # Shape: [batch_size, 2]
        
        # Track object counts for each event
        event_obj_counts = torch.zeros(batch_size, dtype=torch.float32).reshape(-1, 1)
        
        # Process each object type and compute physical features
        max_objects = 18
        for obj_idx in range(max_objects):
            start_idx = 2 + obj_idx * 5
            
            # Extract object features: obj_id, E, pT, eta, phi
            obj_id = X[:, start_idx].reshape(-1, 1)  # Shape: [batch_size, 1]
            obj_features = X[:, start_idx+1:start_idx+5]  # Shape: [batch_size, 4] - E, pT, eta, phi
            
            # Create mask for valid objects (non-zero obj_id)
            valid_mask = obj_id.squeeze() != 0
            
            # Create feature tensor for this object (will be filled conditionally)
            obj_processed = torch.zeros((batch_size, 6), dtype=torch.float32)
            
            if valid_mask.sum() > 0 and obj_idx in self.object_scalers:
                # Count valid objects per event
                event_obj_counts[valid_mask] += 1
                
                # Get indices of valid events
                valid_indices = torch.nonzero(valid_mask).squeeze()
                
                # Extract only valid objects for scaling
                valid_features = obj_features[valid_mask].numpy()
                
                # Scale features
                scaler = self.object_scalers[obj_idx]
                scaled_features = torch.tensor(
                    scaler.transform(valid_features),
                    dtype=torch.float32
                )
                
                # Split the scaled features
                E = scaled_features[:, 0].reshape(-1, 1)  # Energy
                pT = scaled_features[:, 1].reshape(-1, 1)  # Transverse momentum
                eta = scaled_features[:, 2].reshape(-1, 1)  # Pseudorapidity
                phi = scaled_features[:, 3].reshape(-1, 1)  # Azimuthal angle
                
                # Calculate transverse energy
                Et = pT * torch.cosh(eta)
                
                # Calculate px, py, pz components
                px = pT * torch.cos(phi)
                py = pT * torch.sin(phi)
                pz = pT * torch.sinh(eta)
                
                # Create object tensor with [E, pT, eta, phi, Et, obj_id/type] for valid objects
                valid_obj_processed = torch.cat([E, pT, eta, phi, Et, obj_id[valid_mask]], dim=1)
                
                # Place processed features at correct indices
                for i, idx in enumerate(valid_indices):
                    obj_processed[idx] = valid_obj_processed[i]
            
            # Append object features to the list
            processed_features.append(obj_processed)  # Shape per append: [batch_size, 6]
        
        # Add event object count feature
        processed_features.append(event_obj_counts)  # Shape: [batch_size, 1]
        
        # Concatenate all features
        result = torch.cat(processed_features, dim=1)  # Shape: [batch_size, 2 + 18*6 + 1] = [batch_size, 111]
        
        # Replace NaN values with zeros
        result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
        
        self.n_features = result.shape[1]
        return result
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
def make_model(input_dim: int):
    class FourTopClassifier(nn.Module):
        def __init__(self, input_dim):
            super(FourTopClassifier, self).__init__()
            self.dropout_rate = 0.2
            
            # First layer block
            self.layer1 = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(self.dropout_rate)
            )
            
            # Second layer block
            self.layer2 = nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(self.dropout_rate)
            )
            
            # Third layer block
            self.layer3 = nn.Sequential(
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                nn.Dropout(self.dropout_rate)
            )
            
            # Output layer
            self.output = nn.Linear(64, 1)
            
            # Weight initialization
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    # He initialization
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.output(x)
            return x.squeeze(1)
    
    model = FourTopClassifier(input_dim)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    
    # Initialize tracking variables
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    best_auc = 0.0
    best_state = None
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update tracking variables
            train_loss += loss.item() * inputs.size(0)
            train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch statistics
        train_loss = train_loss / len(train_loader.dataset)
        train_loss_history.append(train_loss)
        
        # Calculate accuracy and AUC
        train_preds_binary = (np.array(train_preds) > 0.5).astype(int)
        train_targets = np.array(train_targets)
        train_acc = np.mean(train_preds_binary == train_targets)
        train_acc_history.append(train_acc)
        train_auc = roc_auc_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update tracking variables
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch statistics
        val_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(val_loss)
        
        # Calculate accuracy and AUC
        val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
        val_targets = np.array(val_targets)
        val_acc = np.mean(val_preds_binary == val_targets)
        val_acc_history.append(val_acc)
        val_auc = roc_auc_score(val_targets, val_preds)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
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

