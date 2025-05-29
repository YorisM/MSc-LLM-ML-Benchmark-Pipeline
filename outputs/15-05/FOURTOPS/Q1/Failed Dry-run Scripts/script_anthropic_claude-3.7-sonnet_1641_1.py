
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import torch.optim as optim

class EventPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.n_features = None

    def fit(self, X, y=None):
        # Extract and reshape features
        X_processed = self._process_features(X)
        
        # Fit scaler on the processed features
        self.scaler.fit(X_processed)
        return self

    def transform(self, X):
        # Process features and apply scaling
        X_processed = self._process_features(X)
        X_scaled = self.scaler.transform(X_processed)
        return torch.tensor(X_scaled, dtype=torch.float32)
    
    def _process_features(self, X):
        # Convert to numpy if it's a tensor
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        batch_size = X.shape[0]
        
        # Extract E_T_miss and phi_{E_t}_miss (first two columns)
        E_T_miss = X[:, 0].reshape(-1, 1)
        phi_E_T_miss = X[:, 1].reshape(-1, 1)
        
        # Rest of the columns contain object data
        object_data = X[:, 2:]
        
        # Reshape to make it easier to process object by object
        # Each object has 5 values: obj_id, E, p_T, eta, phi
        n_objects = (object_data.shape[1]) // 5
        object_data_reshaped = object_data.reshape(batch_size, n_objects, 5)
        
        # Extract physics-relevant features
        features_list = []
        
        # Add missing energy features
        features_list.append(E_T_miss)  # Missing transverse energy
        features_list.append(phi_E_T_miss)  # Azimuthal angle of missing energy
        
        # Calculate basic event-level features
        # Sum of transverse momentum across all objects
        sum_pt = np.sum(object_data_reshaped[:, :, 2], axis=1, keepdims=True)
        features_list.append(sum_pt)
        
        # Count different object types (obj_id values)
        # Assuming obj_id is the first value in each object
        for obj_type in range(1, 7):  # Assuming 6 potential object types
            obj_count = np.sum(object_data_reshaped[:, :, 0] == obj_type, axis=1, keepdims=True)
            features_list.append(obj_count)
        
        # Statistics for each object type
        for obj_type in range(1, 7):
            # Create mask for this object type
            mask = (object_data_reshaped[:, :, 0] == obj_type)
            
            # For each event, get max, min, mean of E, p_T, eta, phi for this object type
            for feature_idx, feature_name in enumerate(['E', 'p_T', 'eta', 'phi'], 1):
                # Get values for this feature and object type
                values = object_data_reshaped[:, :, feature_idx]
                
                # For events with no objects of this type, use 0 as placeholder
                masked_values = np.where(mask, values, np.nan)
                
                # Calculate statistics (ignoring NaN values)
                max_val = np.nanmax(masked_values, axis=1, keepdims=True)
                min_val = np.nanmin(masked_values, axis=1, keepdims=True)
                mean_val = np.nanmean(masked_values, axis=1, keepdims=True)
                sum_val = np.nansum(masked_values, axis=1, keepdims=True)
                
                # Replace NaN with 0
                max_val = np.nan_to_num(max_val)
                min_val = np.nan_to_num(min_val)
                mean_val = np.nan_to_num(mean_val)
                sum_val = np.nan_to_num(sum_val)
                
                features_list.extend([max_val, min_val, mean_val, sum_val])
        
        # Pairwise angular separations (ΔR) between the 5 leading objects
        # ΔR = sqrt((Δeta)² + (Δphi)²)
        leading_n = min(5, n_objects)
        for i in range(leading_n):
            for j in range(i+1, leading_n):
                delta_eta = object_data_reshaped[:, i, 3] - object_data_reshaped[:, j, 3]
                delta_phi = np.abs(object_data_reshaped[:, i, 4] - object_data_reshaped[:, j, 4])
                # Ensure delta_phi is in the range [0, pi]
                delta_phi = np.minimum(delta_phi, 2*np.pi - delta_phi)
                delta_r = np.sqrt(delta_eta**2 + delta_phi**2).reshape(-1, 1)
                features_list.append(delta_r)
        
        # Combine all features
        X_processed = np.hstack(features_list)
        
        # Store the number of features
        self.n_features = X_processed.shape[1]
        
        return X_processed

def make_preprocessor():
    return EventPreprocessor()

class PPCollisionClassifier(nn.Module):
    def __init__(self, input_dim):
        super(PPCollisionClassifier, self).__init__()
        
        # Architecture definition
        self.bn_input = nn.BatchNorm1d(input_dim)
        
        # First block
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        # Second block
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        # Third block
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        
        # Output layer
        self.fc_out = nn.Linear(64, 1)
        
    def forward(self, x):
        # Input normalization
        x = self.bn_input(x)
        
        # First block
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout1(x)
        
        # Second block
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout2(x)
        
        # Third block
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout3(x)
        
        # Output layer - using sigmoid in the loss function for numerical stability
        x = self.fc_out(x).squeeze(1)
        
        return x

def make_model(input_dim):
    return PPCollisionClassifier(input_dim)

epochs = 30

def train_model(model, train_loader, val_loader, epochs):
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    
    # Initialize tracking variables
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_auc = 0.0
    best_model_state = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Calculate loss
            loss = criterion(outputs, y_batch.float())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item() * X_batch.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).int()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            # Store predictions and targets for AUC calculation
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
        
        # Calculate training metrics
        epoch_loss /= total
        epoch_acc = correct / total
        epoch_auc = roc_auc_score(all_targets, all_preds)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        # Validation loop
        model.eval()
        val_epoch_loss = 0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                outputs = model(X_batch)
                
                # Calculate loss
                loss = criterion(outputs, y_batch.float())
                
                # Track metrics
                val_epoch_loss += loss.item() * X_batch.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).int()
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
                
                # Store predictions and targets for AUC calculation
                val_all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                val_all_targets.extend(y_batch.cpu().numpy())
                
        # Calculate validation metrics
        val_epoch_loss /= val_total
        val_epoch_acc = val_correct / val_total
        val_epoch_auc = roc_auc_score(val_all_targets, val_all_preds)
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)
        
        # Update learning rate based on validation loss
        scheduler.step(val_epoch_loss)
        
        # Save best model based on validation AUC
        if val_epoch_auc > best_val_auc:
            best_val_auc = val_epoch_auc
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Train AUC: {epoch_auc:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, Val AUC: {val_epoch_auc:.4f}")
    
    # Load best model state
    if best_model_state:
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
