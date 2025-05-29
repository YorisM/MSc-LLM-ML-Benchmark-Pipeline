
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
EPOCHS = 10   # <LLM: may overwrite this constant>
                        
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
# via the wrapper. Only import extra std-lib modules or torch.nn sub-modules
# you actually use.
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

class EventPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.n_features = None
        
    def fit(self, X, y=None):
        # Extract relevant features from the data
        processed_data = self._extract_features(X)
        # Fit scaler on the processed data
        self.scaler.fit(processed_data)
        self.n_features = processed_data.shape[1]
        return self
        
    def transform(self, X):
        # Extract features and normalize them
        processed_data = self._extract_features(X)
        normalized_data = self.scaler.transform(processed_data)
        return torch.tensor(normalized_data, dtype=torch.float32)
    
    def _extract_features(self, X):
        # Convert to numpy if it's a tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        
        batch_size = X.shape[0]
        
        # First two columns are E_T_miss and phi_{E_t}_miss
        missing_et = X[:, 0].reshape(-1, 1)  # Missing ET magnitude
        missing_phi = X[:, 1].reshape(-1, 1)  # Missing ET phi
        
        # Extract object features - each object has 5 values (obj_id, E, pT, eta, phi)
        # The data is padded with zeros for events with fewer objects
        n_objects_max = (X.shape[1] - 2) // 5  # Maximum number of objects
        
        # Initialize feature arrays
        features = []
        
        # Always include missing ET information
        features.append(missing_et)
        features.append(missing_phi)
        
        # Calculate global features
        total_energy = np.zeros((batch_size, 1))
        total_pt = np.zeros((batch_size, 1))
        
        # Lists to collect objects by type for further processing
        jets = []
        leptons = []
        b_jets = []
        
        # Process each object in the event
        for i in range(n_objects_max):
            # Get the starting index for this object
            start_idx = 2 + i * 5
            
            # Check if this object exists (non-zero object ID)
            obj_id = X[:, start_idx]
            obj_mask = obj_id != 0
            
            if not np.any(obj_mask):
                continue  # Skip processing if all entries are padding
                
            # Extract object properties
            E = X[:, start_idx + 1].reshape(-1, 1)  # Energy
            pt = X[:, start_idx + 2].reshape(-1, 1)  # Transverse momentum
            eta = X[:, start_idx + 3].reshape(-1, 1)  # Pseudorapidity
            phi = X[:, start_idx + 4].reshape(-1, 1)  # Azimuthal angle
            
            # Update global features
            total_energy += np.where(obj_mask.reshape(-1, 1), E, 0)
            total_pt += np.where(obj_mask.reshape(-1, 1), pt, 0)
            
            # Store object by type based on obj_id
            # In this context, we'll assume:  
            # Jets: obj_id = 1, Leptons: obj_id = 2, b-jets: obj_id = 3
            # You would need to adapt this based on the actual object encoding in your data
            obj_data = np.column_stack([E, pt, eta, phi])
            
            # Create masked version where non-existent objects are set to 0
            masked_data = np.where(obj_mask.reshape(-1, 1), obj_data, 0)
            
            # Categorize objects based on id
            # This is just an example - adjust according to actual object IDs in your data
            jets.append(masked_data * (obj_id.reshape(-1, 1) == 1))
            leptons.append(masked_data * (obj_id.reshape(-1, 1) == 2))
            b_jets.append(masked_data * (obj_id.reshape(-1, 1) == 3))
            
        # Add global features
        features.append(total_energy)
        features.append(total_pt)
        
        # Process object collections
        # For each collection, compute summary statistics
        for collection in [jets, leptons, b_jets]:
            if collection:
                # Stack objects in the collection
                stacked = np.stack(collection, axis=1)  # (batch_size, n_obj, 4)
                
                # Count non-zero objects
                counts = np.sum(np.sum(stacked, axis=2) != 0, axis=1, keepdims=True)
                features.append(counts)
                
                # Sum properties
                sum_props = np.sum(stacked, axis=1)  # Sum across objects
                features.append(sum_props)
                
                # Calculate mean of non-zero elements
                masked_stacked = np.where(stacked != 0, stacked, np.nan)
                mean_props = np.nanmean(masked_stacked, axis=1)
                mean_props = np.nan_to_num(mean_props)  # Replace NaNs with 0
                features.append(mean_props)
                
                # Get properties of leading (highest pT) object
                pt_idx = 1  # Index of pT in the object properties
                leading_idx = np.argmax(stacked[:, :, pt_idx], axis=1)
                batch_indices = np.arange(batch_size)
                leading_obj = stacked[batch_indices, leading_idx]
                # For events with no objects, replace with zeros
                leading_obj = np.where(counts.reshape(-1, 1) > 0, leading_obj, 0)
                features.append(leading_obj)
                
                # Angular separation features (if multiple objects exist)
                if counts.max() >= 2:
                    # Delta Eta between leading objects
                    if stacked.shape[1] >= 2:
                        deta = np.abs(stacked[:, 0, 2] - stacked[:, 1, 2]).reshape(-1, 1)
                        features.append(deta)
                    
                    # Delta Phi between leading objects
                    if stacked.shape[1] >= 2:
                        dphi = np.abs(stacked[:, 0, 3] - stacked[:, 1, 3]).reshape(-1, 1)
                        # Normalize phi difference to [-π, π]
                        dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
                        features.append(dphi)
                        
                    # Delta R = sqrt(deta^2 + dphi^2) between leading objects
                    if stacked.shape[1] >= 2:
                        dr = np.sqrt(deta**2 + dphi**2)
                        features.append(dr)
        
        # Calculate HT (scalar sum of jet pT)
        if jets:
            jet_pt = np.stack([j[:, 1] for j in jets], axis=1)  # Extract pT column
            HT = np.sum(jet_pt, axis=1, keepdims=True)
            features.append(HT)
        
        # Calculate missing ET significance (missing ET / sqrt(HT))
        if jets:
            met_significance = missing_et / np.sqrt(np.maximum(HT, 1e-10))
            features.append(met_significance)
        
        # Concatenate all features
        all_features = np.hstack(features)
        
        # Remove any NaN or infinite values
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return all_features

def make_preprocessor():
    return EventPreprocessor()

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class PhysicsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(PhysicsClassifier, self).__init__()
        
        # Define network architecture
        self.model = nn.Sequential(
            # Initial layer to expand dimensions
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Residual blocks
            ResidualBlock(256),
            nn.Dropout(0.2),
            ResidualBlock(256),
            nn.Dropout(0.2),
            
            # Reduction layers
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Final output layer
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.model(x).squeeze(-1)

def make_model(input_dim):
    return PhysicsClassifier(input_dim)

epochs = 30

def train_model(model, train_loader, val_loader, epochs):
    # Initialize lists to track metrics
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    best_val_auc = 0
    best_model_state = None
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * inputs.size(0)
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
            train_pred_classes = (torch.sigmoid(outputs) > 0.5).int()
            train_correct += (train_pred_classes == targets).sum().item()
            train_total += targets.size(0)
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_auc = roc_auc_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets.float())
                
                # Track metrics
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                val_pred_classes = (torch.sigmoid(outputs) > 0.5).int()
                val_correct += (val_pred_classes == targets).sum().item()
                val_total += targets.size(0)
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_targets, val_preds)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Store metrics history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history
# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(Y_train, Y_val, name, out):
    plt.figure(); plt.plot(Y_train, label=f'Train {name}')
    plt.plot(Y_val, label=f'Validation {name}')
    plt.legend(); plt.title(name); plt.xlabel('epoch')
    plt.savefig(out); plt.close()        

def _run(dryrun=False):
    X_train, Y_train, X_val, Y_val = load_data()
    pre = make_preprocessor()
    pre.fit(X_train, Y_train)
    X_train = pre.transform(X_train);  X_val = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    model = make_model(input_dim=X_train.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    hist     = train_model(model, train_loader, val_loader, epochs=n_epochs)

    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        torch.save(model.state_dict(), f"{base}_state.pt")
        with open(f"{base}_pre.pkl", "wb") as f: pickle.dump(pre, f)
        _plot(hist['loss'], hist['val_loss'], 'Loss',     f"{base}_loss.png")
        _plot(hist['acc'],  hist['val_acc'],  'Accuracy', f"{base}_acc.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)
