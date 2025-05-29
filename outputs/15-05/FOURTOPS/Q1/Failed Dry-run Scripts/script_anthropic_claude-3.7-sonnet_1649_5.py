
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
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class EventPreprocessor(object):
    def __init__(self):
        self.scalers = {}
        self.feature_indices = None
        self.n_features = None
    
    def fit(self, X, y=None):
        # Identify the structure of the data: ET_miss, phi_ET_miss, and then objects
        # Each object has 5 values: obj_id, E, pT, eta, phi
        # We'll extract meaningful features for each event
        
        # Extract ET_miss and phi_ET_miss (first two columns)
        self.scalers['et_miss'] = StandardScaler().fit(X[:, 0].reshape(-1, 1))
        # No need to scale phi as it's already bounded between -π and π
        
        # Process the object features (starting from index 2)
        # Get non-zero object indices
        mask = X[:, 2::5] != 0  # Check obj_id columns for non-zero values
        
        # Features we'll extract: count of each object type, sum of E, pT, mean values, etc.
        # Energy, pT, eta, phi are at regular intervals
        self.scalers['energy'] = StandardScaler().fit(X[:, 3::5][mask].reshape(-1, 1))
        self.scalers['pt'] = StandardScaler().fit(X[:, 4::5][mask].reshape(-1, 1))
        # Eta and phi don't need scaling as they have natural bounds
        
        return self
    
    def transform(self, X):
        # Initialize array to hold our transformed features
        batch_size = X.shape[0]
        transformed = []
        
        # Process ET_miss and phi_ET_miss
        et_miss = self.scalers['et_miss'].transform(X[:, 0].reshape(-1, 1)).flatten()
        phi_et_miss = X[:, 1].flatten()  # No scaling for phi
        
        transformed.append(et_miss)
        transformed.append(np.cos(phi_et_miss))
        transformed.append(np.sin(phi_et_miss))
        
        # Process objects (starting from index 2)
        # We'll create features for each object type and global features
        
        # Count the number of objects by type
        max_objects = (X.shape[1] - 2) // 5  # Max number of objects possible
        obj_counts = {}
        
        # Extract object type counts and features
        for i in range(max_objects):
            obj_id_col = 2 + i * 5
            obj_ids = X[:, obj_id_col].astype(int)
            
            # Process each unique object type
            for obj_type in np.unique(obj_ids):
                if obj_type == 0:  # Skip padding
                    continue
                    
                # Create mask for this object type
                mask = obj_ids == obj_type
                count = np.sum(mask, axis=0)
                
                # Add count feature
                if obj_type not in obj_counts:
                    obj_counts[obj_type] = np.zeros(batch_size)
                obj_counts[obj_type] += mask.astype(int)
                
                # Get object properties where this object type exists
                energy_col = obj_id_col + 1
                pt_col = obj_id_col + 2
                eta_col = obj_id_col + 3
                phi_col = obj_id_col + 4
                
                # For each event that has this object type
                event_mask = np.where(mask)[0]
                if len(event_mask) > 0:
                    # Extract and scale features
                    energies = X[mask, energy_col].reshape(-1, 1)
                    pts = X[mask, pt_col].reshape(-1, 1)
                    etas = X[mask, eta_col]
                    phis = X[mask, phi_col]
                    
                    # Scale numerical features
                    if len(energies) > 0:
                        energies = self.scalers['energy'].transform(energies).flatten()
                        pts = self.scalers['pt'].transform(pts).flatten()
                    
                    # Create aggregate features by object type and event
                    for event_idx in np.unique(event_mask):
                        event_obj_mask = event_mask == event_idx
                        if np.sum(event_obj_mask) > 0:
                            # Extract values for this specific event and object type
                            event_energies = energies[event_obj_mask]
                            event_pts = pts[event_obj_mask]
                            event_etas = etas[event_obj_mask]
                            event_phis = phis[event_obj_mask]
                            
                            # Calculate aggregated features
                            if len(event_energies) > 0:
                                # Sum of energy and pT
                                sum_energy = np.sum(event_energies)
                                sum_pt = np.sum(event_pts)
                                
                                # Mean values
                                mean_energy = np.mean(event_energies)
                                mean_pt = np.mean(event_pts)
                                mean_eta = np.mean(event_etas)
                                
                                # Spread/variance
                                std_energy = np.std(event_energies) if len(event_energies) > 1 else 0
                                std_pt = np.std(event_pts) if len(event_pts) > 1 else 0
                                std_eta = np.std(event_etas) if len(event_etas) > 1 else 0
                                
                                # Store these features
                                if f"sum_energy_{obj_type}" not in locals():
                                    locals()[f"sum_energy_{obj_type}"] = np.zeros(batch_size)
                                    locals()[f"sum_pt_{obj_type}"] = np.zeros(batch_size)
                                    locals()[f"mean_energy_{obj_type}"] = np.zeros(batch_size)
                                    locals()[f"mean_pt_{obj_type}"] = np.zeros(batch_size)
                                    locals()[f"mean_eta_{obj_type}"] = np.zeros(batch_size)
                                    locals()[f"std_energy_{obj_type}"] = np.zeros(batch_size)
                                    locals()[f"std_pt_{obj_type}"] = np.zeros(batch_size)
                                    locals()[f"std_eta_{obj_type}"] = np.zeros(batch_size)
                                
                                locals()[f"sum_energy_{obj_type}"][event_idx] = sum_energy
                                locals()[f"sum_pt_{obj_type}"][event_idx] = sum_pt
                                locals()[f"mean_energy_{obj_type}"][event_idx] = mean_energy
                                locals()[f"mean_pt_{obj_type}"][event_idx] = mean_pt
                                locals()[f"mean_eta_{obj_type}"][event_idx] = mean_eta
                                locals()[f"std_energy_{obj_type}"][event_idx] = std_energy
                                locals()[f"std_pt_{obj_type}"][event_idx] = std_pt
                                locals()[f"std_eta_{obj_type}"][event_idx] = std_eta
        
        # Add all object counts
        for obj_type, counts in obj_counts.items():
            transformed.append(counts)
        
        # Add all the aggregated features
        for var_name, var_value in locals().items():
            if isinstance(var_value, np.ndarray) and var_value.shape == (batch_size,):
                if var_name.startswith(("sum_", "mean_", "std_")):
                    transformed.append(var_value)
        
        # Combine all features
        result = np.column_stack(transformed)
        
        # Store number of features for model creation
        self.n_features = result.shape[1]
        
        return torch.tensor(result, dtype=torch.float32)

def make_preprocessor():
    return EventPreprocessor()

class FourTopClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FourTopClassifier, self).__init__()
        
        # Deep neural network with residual connections
        self.layer1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.layer4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.layer5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.4)
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        # First block with residual connection
        identity = self.layer1(x)
        x = F.relu(self.bn1(identity))
        x = self.layer2(x)
        x = F.relu(self.bn2(x + identity))
        x = self.dropout(x)
        
        # Second block
        identity = self.layer3(x)
        x = F.relu(self.bn3(identity))
        x = self.layer4(x)
        x = F.relu(self.bn4(x + identity))
        x = self.dropout(x)
        
        # Final layers
        x = F.relu(self.bn5(self.layer5(x)))
        x = self.output(x)
        
        return torch.sigmoid(x).squeeze()

def make_model(input_dim: int):
    return FourTopClassifier(input_dim)

class FourTopDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

epochs = 30

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Use binary cross entropy loss for binary classification
    criterion = nn.BCELoss()
    
    # Use Adam optimizer with learning rate scheduling
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, min_lr=1e-5)
    
    # Initialize lists to track metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Store predictions for AUC calculation
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
        
        # Calculate training metrics
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_auc = roc_auc_score(all_labels, all_preds)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                val_all_preds.extend(outputs.detach().cpu().numpy())
                val_all_labels.extend(labels.detach().cpu().numpy())
                
        # Calculate validation metrics
        val_epoch_loss = val_running_loss / val_total
        val_epoch_acc = val_correct / val_total
        val_epoch_auc = roc_auc_score(val_all_labels, val_all_preds)
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)
        
        # Update learning rate based on validation loss
        scheduler.step(val_epoch_loss)
        
        # Store best model based on validation AUC
        if val_epoch_auc > best_val_auc:
            best_val_auc = val_epoch_auc
            best_model_state = model.state_dict().copy()
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Train AUC: {epoch_auc:.4f} - '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, Val AUC: {val_epoch_auc:.4f}')
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_loss, val_loss, train_acc, val_acc
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
