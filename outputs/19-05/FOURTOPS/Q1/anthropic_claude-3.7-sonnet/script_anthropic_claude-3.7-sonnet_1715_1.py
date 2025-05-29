
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
                       
def load_data():
    X_train_df = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv')
    Y_train_df = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv')
    X_val_df   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv')
    Y_val_df   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv')

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
    X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
    Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()
    return X_train, Y_train, X_val, Y_val

def make_loaders(X_train, Y_train, X_val, Y_val, batch=1024):
    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train))
    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val))
    return (DataLoader(train, batch_size=batch, shuffle=True),
            DataLoader(val, batch_size=batch))
                        
# ----------------  START OF LLM BLOCK  ----------------
# Imports
import os, sys, json, pickle, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class MyPreprocessor:
    def __init__(self):
        self.features_scaler = StandardScaler()
        self.object_indices = {}
        self.max_objects = 18
        self.obj_feature_size = 5
        self.global_feature_size = 2
        self.feature_stats = {}
        self.obj_count_hist = None
        self.valid_object_mask = None
        
    def fit(self, X: torch.Tensor, y=None) -> 'MyPreprocessor':
        # Convert tensor to numpy if needed
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X
        
        # Count valid objects per event
        obj_counts = []
        for i in range(X_np.shape[0]):
            count = 0
            for obj_idx in range(self.max_objects):
                start_idx = self.global_feature_size + obj_idx * self.obj_feature_size
                # Check if the object ID field is not zero (valid object)
                if X_np[i, start_idx] != 0:
                    count += 1
            obj_counts.append(count)
            
        self.obj_count_hist = np.bincount(obj_counts, minlength=self.max_objects+1)
            
        # Create valid object mask
        self.valid_object_mask = np.zeros((X_np.shape[0], self.max_objects), dtype=bool)
        for i in range(X_np.shape[0]):
            for obj_idx in range(self.max_objects):
                start_idx = self.global_feature_size + obj_idx * self.obj_feature_size
                if X_np[i, start_idx] != 0:  # If object exists
                    self.valid_object_mask[i, obj_idx] = True
        
        # Extract features for preprocessing
        # We'll standardize continuous features
        features_to_scale = []
        
        # Global features (missing ET and phi)
        global_features = X_np[:, :self.global_feature_size]
        
        # Object features (excluding object ID, which is categorical)
        obj_features = []
        for obj_idx in range(self.max_objects):
            start_idx = self.global_feature_size + obj_idx * self.obj_feature_size
            # Extract E, p_T, eta, phi but not obj_id
            obj_energy = X_np[:, start_idx + 1:start_idx + self.obj_feature_size]
            mask = self.valid_object_mask[:, obj_idx].reshape(-1, 1)
            # Replace invalid object features with NaN to exclude them from scaling
            masked_features = np.where(mask, obj_features, np.nan)
            obj_features.append(obj_energy)
        
        # Combine global features with all valid object features
        all_features = np.hstack([global_features] + obj_features)
        
        # Compute statistics for non-NaN values (valid objects only)
        self.feature_stats['mean'] = np.nanmean(all_features, axis=0)
        self.feature_stats['std'] = np.nanstd(all_features, axis=0)
        self.feature_stats['min'] = np.nanmin(all_features, axis=0)
        self.feature_stats['max'] = np.nanmax(all_features, axis=0)
        
        # Get object types (categorical IDs)
        object_types = set()
        for i in range(X_np.shape[0]):
            for obj_idx in range(self.max_objects):
                start_idx = self.global_feature_size + obj_idx * self.obj_feature_size
                obj_id = X_np[i, start_idx]
                if obj_id != 0:  # If not padding
                    object_types.add(int(obj_id))
        
        # Create a mapping for object types
        self.object_indices = {int(obj_id): idx for idx, obj_id in enumerate(sorted(object_types))}
             
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        # Convert tensor to numpy if needed
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X
        
        batch_size = X_np.shape[0]
        
        # Extract global features (missing ET and phi)
        et_miss = X_np[:, 0].reshape(-1, 1)  # Missing ET magnitude
        phi_miss = X_np[:, 1].reshape(-1, 1)  # Missing ET phi angle
        
        # Normalize missing ET
        et_miss_normalized = (et_miss - self.feature_stats['mean'][0]) / (self.feature_stats['std'][0] + 1e-8)
        
        # Sine and cosine of phi for circular features
        sin_phi_miss = np.sin(phi_miss)
        cos_phi_miss = np.cos(phi_miss)
        
        # Prepare container for processed objects
        n_obj_types = len(self.object_indices)
        
        # Count objects of each type per event
        obj_type_counts = np.zeros((batch_size, n_obj_types))
        
        # Extract features from objects
        # For each potential object slot
        all_E = []
        all_pT = []
        all_eta = []
        all_sin_phi = []
        all_cos_phi = []
        all_obj_type_onehot = []

        # For global features summed over objects
        sum_E = np.zeros((batch_size, 1))
        sum_pT = np.zeros((batch_size, 1))
        sum_weighted_eta = np.zeros((batch_size, 1))
        sum_weighted_phi_sin = np.zeros((batch_size, 1))
        sum_weighted_phi_cos = np.zeros((batch_size, 1))
        
        # Track valid objects for later masking
        valid_mask = np.zeros((batch_size, self.max_objects), dtype=bool)
        
        for obj_idx in range(self.max_objects):
            start_idx = self.global_feature_size + obj_idx * self.obj_feature_size
            
            # Get object ID, energy, pT, eta, phi
            obj_id = X_np[:, start_idx].reshape(-1, 1)
            E = X_np[:, start_idx + 1].reshape(-1, 1)
            pT = X_np[:, start_idx + 2].reshape(-1, 1)
            eta = X_np[:, start_idx + 3].reshape(-1, 1)
            phi = X_np[:, start_idx + 4].reshape(-1, 1)
            
            # Check if object is valid (non-zero ID)
            is_valid = obj_id != 0
            valid_mask[:, obj_idx] = is_valid.flatten()
            
            # One-hot encode object type
            obj_type_onehot = np.zeros((batch_size, n_obj_types))
            for i in range(batch_size):
                if is_valid[i]:
                    obj_type = int(obj_id[i][0])
                    if obj_type in self.object_indices:
                        idx = self.object_indices[obj_type]
                        obj_type_onehot[i, idx] = 1
                        obj_type_counts[i, idx] += 1
            
            # Normalize features
            E_normalized = np.where(is_valid, (E - self.feature_stats['mean'][2]) / (self.feature_stats['std'][2] + 1e-8), 0)
            pT_normalized = np.where(is_valid, (pT - self.feature_stats['mean'][3]) / (self.feature_stats['std'][3] + 1e-8), 0)
            eta_normalized = np.where(is_valid, (eta - self.feature_stats['mean'][4]) / (self.feature_stats['std'][4] + 1e-8), 0)
            
            # Circular features for phi
            sin_phi = np.where(is_valid, np.sin(phi), 0)
            cos_phi = np.where(is_valid, np.cos(phi), 0)
            
            # Add to the arrays
            all_E.append(E_normalized)
            all_pT.append(pT_normalized)
            all_eta.append(eta_normalized)
            all_sin_phi.append(sin_phi)
            all_cos_phi.append(cos_phi)
            all_obj_type_onehot.append(obj_type_onehot)
            
            # Update global sums
            sum_E += np.where(is_valid, E, 0)
            sum_pT += np.where(is_valid, pT, 0)
            sum_weighted_eta += np.where(is_valid, pT * eta, 0)  # pT-weighted eta
            sum_weighted_phi_sin += np.where(is_valid, pT * sin_phi, 0)  # pT-weighted sin(phi)
            sum_weighted_phi_cos += np.where(is_valid, pT * cos_phi, 0)  # pT-weighted cos(phi)
        
        # Calculate average eta and phi (pT weighted)
        avg_eta = np.where(sum_pT > 0, sum_weighted_eta / (sum_pT + 1e-8), 0)
        avg_phi_x = np.where(sum_pT > 0, sum_weighted_phi_sin / (sum_pT + 1e-8), 0)
        avg_phi_y = np.where(sum_pT > 0, sum_weighted_phi_cos / (sum_pT + 1e-8), 0)
        
        # Count total valid objects per event
        obj_count = np.sum(valid_mask, axis=1, keepdims=True)
        
        # Compute total object type distribution
        obj_type_fractions = obj_type_counts / (obj_count + 1e-8)  # Avoid division by zero
        
        # Group all features
        global_features = [
            et_miss_normalized,
            sin_phi_miss,
            cos_phi_miss,
            sum_E,
            sum_pT,
            avg_eta,
            avg_phi_x,
            avg_phi_y,
            obj_count,
            obj_type_fractions
        ]
        
        # Flatten object features by concatenating instead of stacking
        # This gives fixed-width features regardless of objects
        flattened_E = np.hstack([E for E in all_E])  # Shape: (batch_size, max_objects)
        flattened_pT = np.hstack([pT for pT in all_pT])
        flattened_eta = np.hstack([eta for eta in all_eta])
        flattened_sin_phi = np.hstack([sin_phi for sin_phi in all_sin_phi])
        flattened_cos_phi = np.hstack([cos_phi for cos_phi in all_cos_phi])
        
        # Stack all object type one-hot encodings
        stacked_obj_types = np.hstack(all_obj_type_onehot)  # Shape: (batch_size, max_objects*n_obj_types)
        
        # Concatenate all features into a single tensor
        global_features_flat = np.hstack(global_features)
        object_features_flat = np.hstack([
            flattened_E,
            flattened_pT,
            flattened_eta,
            flattened_sin_phi,
            flattened_cos_phi,
            stacked_obj_types
        ])
        
        # Combine global and object features
        processed_features = np.hstack([
            global_features_flat,
            object_features_flat
        ])
        
        # Return as tensor
        return torch.tensor(processed_features, dtype=torch.float32)


def make_preprocessor():
    return MyPreprocessor()


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        x = x + identity  # Skip connection
        x = F.relu(x)
        return x


class ParticleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 256, 128], dropout_rate=0.3):
        super().__init__()
        
        self.bn_input = nn.BatchNorm1d(input_dim)
        
        # Initial layer
        layers = [nn.Linear(input_dim, hidden_dims[0]), 
                 nn.BatchNorm1d(hidden_dims[0]), 
                 nn.ReLU(), 
                 nn.Dropout(dropout_rate)]
                 
        # Add residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(2):  # Add 2 residual blocks
            self.res_blocks.append(ResidualBlock(hidden_dims[0], hidden_dims[0] // 2))
        
        # Add more layers with decreasing sizes
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate * (1 - i / len(hidden_dims)))
            ])
            
        # Output layer
        layers.extend([nn.Linear(hidden_dims[-1], 1)])
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.bn_input(x)
        
        # Process through first layers until residual blocks
        x = self.layers[0:4](x)
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
            
        # Process through remaining layers
        x = self.layers[4:](x)
        
        return x.squeeze()


def make_model(input_dim: int):
    model = ParticleClassifier(input_dim=input_dim, hidden_dims=[512, 256, 128])
    return model


EPOCHS = 20


def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    # Track metrics
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    train_auc, val_auc = [], []
    
    best_val_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            predicted = torch.sigmoid(outputs) > 0.5
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            # Store probabilities and labels for AUC
            all_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_auc = roc_auc_score(all_labels, all_probs)
        
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        train_auc.append(epoch_auc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device).float()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                
                # Calculate accuracy
                predicted = torch.sigmoid(outputs) > 0.5
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                
                # Store probabilities and labels for AUC
                all_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
            
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        epoch_auc = roc_auc_score(all_labels, all_probs)
        
        val_loss.append(epoch_loss)
        val_acc.append(epoch_acc)
        val_auc.append(epoch_auc)
        
        # Update learning rate based on validation AUC
        scheduler.step(epoch_auc)
        
        # Save best model
        if epoch_auc > best_val_auc:
            best_val_auc = epoch_auc
            best_model_state = model.state_dict().copy()
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.2f}%, Train AUC: {train_auc[-1]:.4f} - '
              f'Val Loss: {val_loss[-1]:.4f}, Val Acc: {val_acc[-1]:.2f}%, Val AUC: {val_auc[-1]:.4f}')
    
    # Load best model for inference
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
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
    with open(f"{base}_model.pkl", "wb") as f: pickle.dump(trained, f)
    with open(f"{base}_preproc.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

