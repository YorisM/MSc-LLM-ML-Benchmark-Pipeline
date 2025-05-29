
import os, sys, pickle, torch, gc
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
import math

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        self.input_dim = 0
        
    def fit(self, X, y=None):
        # Extract the non-zero parts of the data
        # Identify valid objects (non-zero) and create features
        n_samples = X.shape[0]
        
        # Process missing ET and phi
        et_miss = X[:, 0].reshape(-1, 1)  # [n_samples, 1]
        phi_et_miss = X[:, 1].reshape(-1, 1)  # [n_samples, 1]
        
        # Create scalers for ET_miss and phi_ET_miss
        self.scalers['et_miss'] = StandardScaler()
        self.scalers['et_miss'].fit(et_miss.numpy())
        
        # phi is circular, no need to scale
        
        # Process all objects
        # We have 18 objects max, each with 5 features
        max_objects = 18
        
        # Initialize scalers for each feature type
        self.scalers['obj_type'] = StandardScaler()
        self.scalers['energy'] = StandardScaler()
        self.scalers['pt'] = StandardScaler()
        self.scalers['eta'] = StandardScaler()
        # phi is circular, no scaling
        
        # Collect all valid objects for fitting scalers
        obj_types = []
        energies = []
        pts = []
        etas = []
        
        for i in range(max_objects):
            obj_start_idx = 2 + i*5
            
            # Get all instances of this object type
            obj_type = X[:, obj_start_idx].reshape(-1, 1)  # [n_samples, 1]
            energy = X[:, obj_start_idx + 1].reshape(-1, 1)  # [n_samples, 1]
            pt = X[:, obj_start_idx + 2].reshape(-1, 1)  # [n_samples, 1]
            eta = X[:, obj_start_idx + 3].reshape(-1, 1)  # [n_samples, 1]
            
            # Add valid objects (non-zero) to our lists
            valid_mask = obj_type != 0
            if valid_mask.sum() > 0:
                obj_types.append(obj_type[valid_mask])
                energies.append(energy[valid_mask])
                pts.append(pt[valid_mask])
                etas.append(eta[valid_mask])
        
        # Concatenate and fit scalers
        if obj_types:
            self.scalers['obj_type'].fit(np.vstack(obj_types))
            self.scalers['energy'].fit(np.vstack(energies))
            self.scalers['pt'].fit(np.vstack(pts))
            self.scalers['eta'].fit(np.vstack(etas))
        
        # Create feature names for easier debugging
        self.feature_names = [
            'et_miss_scaled', 'sin_phi_et_miss', 'cos_phi_et_miss',
        ]
        
        # Features for each object type
        for i in range(max_objects):
            self.feature_names.extend([
                f'obj_{i}_type_scaled', 
                f'obj_{i}_energy_scaled', 
                f'obj_{i}_pt_scaled', 
                f'obj_{i}_eta_scaled',
                f'obj_{i}_sin_phi', 
                f'obj_{i}_cos_phi',
                # Derived features
                f'obj_{i}_e_pt_ratio',
                f'obj_{i}_valid'
            ])
        
        # Add some global statistical features
        self.feature_names.extend([
            'num_objects',
            'sum_pt',
            'sum_energy',
            'mean_eta',
            'mean_phi',
        ])
        
        self.input_dim = len(self.feature_names)
        return self

    def transform(self, X):
        n_samples = X.shape[0]
        max_objects = 18
        
        # Initialize output tensor
        transformed = torch.zeros((n_samples, len(self.feature_names)), dtype=torch.float32)
        
        # Scale ET_miss and transform phi to sin/cos
        et_miss = X[:, 0].reshape(-1, 1)  # [n_samples, 1]
        phi_et_miss = X[:, 1]  # [n_samples]
        
        # Scale ET_miss
        transformed[:, 0] = torch.tensor(self.scalers['et_miss'].transform(et_miss.numpy()).reshape(-1), dtype=torch.float32)
        
        # Transform phi to sin/cos
        transformed[:, 1] = torch.sin(phi_et_miss)
        transformed[:, 2] = torch.cos(phi_et_miss)
        
        # Track global features
        num_objects = torch.zeros(n_samples, dtype=torch.float32)
        sum_pt = torch.zeros(n_samples, dtype=torch.float32)
        sum_energy = torch.zeros(n_samples, dtype=torch.float32)
        eta_values = []
        phi_values = []
        
        # Process all objects
        for i in range(max_objects):
            obj_start_idx = 2 + i*5
            feat_start_idx = 3 + i*8
            
            # Extract object features
            obj_type = X[:, obj_start_idx].reshape(-1, 1)  # [n_samples, 1]
            energy = X[:, obj_start_idx + 1].reshape(-1, 1)  # [n_samples, 1]
            pt = X[:, obj_start_idx + 2].reshape(-1, 1)  # [n_samples, 1]
            eta = X[:, obj_start_idx + 3]  # [n_samples]
            phi = X[:, obj_start_idx + 4]  # [n_samples]
            
            # Create validity mask - nonzero object type indicates valid object
            valid_mask = obj_type.squeeze() != 0
            
            # Set validity feature
            transformed[:, feat_start_idx + 7] = valid_mask.float()
            
            # Update counter of valid objects
            num_objects += valid_mask.float()
            
            # Apply scalers
            obj_type_scaled = torch.zeros(n_samples, dtype=torch.float32)
            energy_scaled = torch.zeros(n_samples, dtype=torch.float32)
            pt_scaled = torch.zeros(n_samples, dtype=torch.float32)
            eta_scaled = torch.zeros(n_samples, dtype=torch.float32)
            
            if valid_mask.sum() > 0:
                # Scale only valid objects
                obj_type_valid = obj_type[valid_mask].numpy()
                energy_valid = energy[valid_mask].numpy()
                pt_valid = pt[valid_mask].numpy()
                eta_valid = eta[valid_mask].reshape(-1, 1).numpy()
                
                # Apply scaling
                obj_type_scaled_valid = self.scalers['obj_type'].transform(obj_type_valid).squeeze()
                energy_scaled_valid = self.scalers['energy'].transform(energy_valid).squeeze()
                pt_scaled_valid = self.scalers['pt'].transform(pt_valid).squeeze()
                eta_scaled_valid = self.scalers['eta'].transform(eta_valid).squeeze()
                
                # Place back into full tensor
                obj_type_scaled[valid_mask] = torch.tensor(obj_type_scaled_valid, dtype=torch.float32)
                energy_scaled[valid_mask] = torch.tensor(energy_scaled_valid, dtype=torch.float32)
                pt_scaled[valid_mask] = torch.tensor(pt_scaled_valid, dtype=torch.float32)
                eta_scaled[valid_mask] = torch.tensor(eta_scaled_valid, dtype=torch.float32)
                
                # Update global sums
                sum_pt += torch.tensor(pt_valid).sum().float()
                sum_energy += torch.tensor(energy_valid).sum().float()
                
                # Store eta and phi for valid objects for calculating means
                eta_values.extend([eta[valid_mask]])
                phi_values.extend([phi[valid_mask]])
            
            # Add the scaled features
            transformed[:, feat_start_idx] = obj_type_scaled
            transformed[:, feat_start_idx + 1] = energy_scaled
            transformed[:, feat_start_idx + 2] = pt_scaled
            transformed[:, feat_start_idx + 3] = eta_scaled
            
            # Add sin and cos of phi
            transformed[:, feat_start_idx + 4] = torch.sin(phi) * valid_mask.float()
            transformed[:, feat_start_idx + 5] = torch.cos(phi) * valid_mask.float()
            
            # Calculate E/pt ratio (useful for identifying object types)
            e_pt_ratio = torch.zeros(n_samples, dtype=torch.float32)
            valid_pt = (pt.squeeze() > 0) & valid_mask
            if valid_pt.sum() > 0:
                e_pt_ratio[valid_pt] = energy[valid_pt].squeeze() / pt[valid_pt].squeeze()
            transformed[:, feat_start_idx + 6] = e_pt_ratio
        
        # Add global statistical features at the end
        global_feat_start_idx = 3 + max_objects * 8
        transformed[:, global_feat_start_idx] = num_objects
        transformed[:, global_feat_start_idx + 1] = sum_pt
        transformed[:, global_feat_start_idx + 2] = sum_energy
        
        # Calculate mean eta and phi if we have objects
        if eta_values:
            all_etas = torch.cat(eta_values)
            all_phis = torch.cat(phi_values)
            
            # Handle empty cases by object
            mean_eta = torch.zeros(n_samples, dtype=torch.float32)
            mean_phi = torch.zeros(n_samples, dtype=torch.float32)
            
            # Calculate by event (placeholder - actual mean would require more complex grouping)
            # As an approximation, we'll use the global means when objects exist
            mean_eta_val = all_etas.mean().item()
            mean_phi_val = all_phis.mean().item()
            
            # Set means only for events with at least one object
            has_objects = num_objects > 0
            mean_eta[has_objects] = mean_eta_val
            mean_phi[has_objects] = mean_phi_val
            
            transformed[:, global_feat_start_idx + 3] = mean_eta
            transformed[:, global_feat_start_idx + 4] = mean_phi
        
        return transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)

class PhysicsInspiredNet(nn.Module):
    def __init__(self, input_dim):
        super(PhysicsInspiredNet, self).__init__()
        
        # Define network dimensions
        hidden_dim1 = 256
        hidden_dim2 = 128
        hidden_dim3 = 64
        
        # Input layer with batch normalization
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Residual blocks for better gradient flow
        self.res_block1 = ResidualBlock(hidden_dim1)
        
        # Intermediate layers with batch normalization
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.res_block2 = ResidualBlock(hidden_dim2)
        
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim3, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.hidden_layer1(x)
        x = self.res_block2(x)
        x = self.hidden_layer2(x)
        x = self.output_layer(x)
        return self.sigmoid(x).squeeze()

def make_model(input_dim: int):
    return PhysicsInspiredNet(input_dim)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5)
    
    # Initialize history lists
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    best_auc = 0.0
    best_model_state = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_preds = []
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Store targets and predictions for AUC calculation
            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.detach().cpu().numpy())
        
        # Calculate epoch statistics
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Calculate AUC
        all_targets_np = np.concatenate(all_targets)
        all_preds_np = np.concatenate(all_preds)
        train_auc = roc_auc_score(all_targets_np, all_preds_np)
        
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_targets = []
        val_preds = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).float()
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Track statistics
                val_running_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                # Store targets and predictions for AUC calculation
                val_targets.append(targets.cpu().numpy())
                val_preds.append(outputs.cpu().numpy())
        
        # Calculate epoch validation statistics
        val_epoch_loss = val_running_loss / val_total
        val_epoch_acc = val_correct / val_total
        
        # Calculate validation AUC
        val_targets_np = np.concatenate(val_targets)
        val_preds_np = np.concatenate(val_preds)
        val_auc = roc_auc_score(val_targets_np, val_preds_np)
        
        val_loss_history.append(val_epoch_loss)
        val_acc_history.append(val_epoch_acc)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save the best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Train AUC: {train_auc:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, Val AUC: {val_auc:.4f}')
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
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
    torch.save(trained_model.state_dict(), f"{base}_state.pt")
    with open(f"{base}_model.pkl", "wb") as f: pickle.dump(trained_model, f)
    with open(f"{base}_preproc.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

