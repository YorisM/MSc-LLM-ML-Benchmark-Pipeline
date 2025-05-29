
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
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
import math

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.n_objects = 18  # Max number of objects in the dataset
        self.object_features = 4  # E, pT, eta, phi
        self.global_features = 2  # E_T_miss, phi_{E_t}_miss
        
        # Define which features to normalize
        self.normalize_features = ['E', 'pT']
        
    def fit(self, X, y=None):
        # Extract features for normalization
        E_T_miss = X[:, 0].reshape(-1, 1)  # Missing ET magnitude
        
        # Create scalers for global features
        self.scalers['E_T_miss'] = StandardScaler()
        self.scalers['E_T_miss'].fit(E_T_miss)
        
        # For each object type, create separate scalers for E and pT
        for i in range(self.n_objects):
            # Base index for this object (obj_id, E, pT, eta, phi)
            base_idx = 2 + i * 5
            
            # Get object ID to check if valid data (non-zero)
            obj_ids = X[:, base_idx].reshape(-1, 1)
            valid_mask = obj_ids != 0
            
            # Create scalers for E and pT if there's valid data
            if np.any(valid_mask):
                # Extract energy values for valid objects
                E_values = X[valid_mask.squeeze(), base_idx + 1].reshape(-1, 1)
                self.scalers[f'E_{i}'] = StandardScaler()
                self.scalers[f'E_{i}'].fit(E_values)
                
                # Extract pT values for valid objects
                pT_values = X[valid_mask.squeeze(), base_idx + 2].reshape(-1, 1)
                self.scalers[f'pT_{i}'] = StandardScaler()
                self.scalers[f'pT_{i}'].fit(pT_values)
        
        return self
    
    def transform(self, X):
        # Create a copy to avoid modifying the original data
        X_transformed = X.clone()
        
        # Normalize E_T_miss
        E_T_miss = X[:, 0].reshape(-1, 1)
        if 'E_T_miss' in self.scalers:
            E_T_miss_norm = torch.tensor(
                self.scalers['E_T_miss'].transform(E_T_miss.numpy()),
                dtype=torch.float32
            )
            X_transformed[:, 0] = E_T_miss_norm.squeeze()
        
        # Process each object
        for i in range(self.n_objects):
            base_idx = 2 + i * 5
            
            # Only process if we have valid objects (non-zero obj_id)
            obj_ids = X[:, base_idx]
            valid_indices = obj_ids != 0
            
            # Normalize E values
            if valid_indices.any() and f'E_{i}' in self.scalers:
                E_idx = base_idx + 1
                E_values = X[valid_indices, E_idx].reshape(-1, 1).numpy()
                E_norm = self.scalers[f'E_{i}'].transform(E_values)
                X_transformed[valid_indices, E_idx] = torch.tensor(
                    E_norm.squeeze(), dtype=torch.float32
                )
            
            # Normalize pT values
            if valid_indices.any() and f'pT_{i}' in self.scalers:
                pT_idx = base_idx + 2
                pT_values = X[valid_indices, pT_idx].reshape(-1, 1).numpy()
                pT_norm = self.scalers[f'pT_{i}'].transform(pT_values)
                X_transformed[valid_indices, pT_idx] = torch.tensor(
                    pT_norm.squeeze(), dtype=torch.float32
                )
        
        # Create engineered features
        batch_size = X.shape[0]
        
        # Calculate derived physical features
        features_list = [X_transformed]
        
        # Sort objects by pT within each event
        sorted_features = self._sort_objects_by_pt(X_transformed)
        features_list.append(sorted_features)
        
        # Extract high-level features
        high_level_features = self._extract_high_level_features(X_transformed)
        features_list.append(high_level_features)
        
        # Calculate object multiplicity features
        multiplicity_features = self._calculate_multiplicity(X_transformed)
        features_list.append(multiplicity_features)
        
        # Calculate delta R between top objects
        delta_r_features = self._calculate_delta_r(X_transformed)
        features_list.append(delta_r_features)
        
        # Combine all features
        result = torch.cat(features_list, dim=1)
        return result
    
    def _sort_objects_by_pt(self, X):
        batch_size = X.shape[0]
        
        # Create a new tensor to store the sorted objects
        sorted_X = torch.zeros_like(X)
        sorted_X[:, :2] = X[:, :2]  # Copy the global features
        
        # For each sample in the batch
        for b in range(batch_size):
            # Extract all objects and their properties
            objects = []
            for i in range(self.n_objects):
                base_idx = 2 + i * 5
                obj_id = X[b, base_idx].item()
                if obj_id != 0:  # Check if this is a real object
                    # Extract object properties: obj_id, E, pT, eta, phi
                    features = X[b, base_idx:base_idx+5].clone()
                    objects.append((features, features[2].item()))  # (features, pT)
            
            # Sort objects by pT (descending)
            objects.sort(key=lambda x: x[1], reverse=True)
            
            # Place sorted objects back into tensor
            for i, (obj_features, _) in enumerate(objects):
                if i < self.n_objects:  # Ensure we don't exceed max objects
                    base_idx = 2 + i * 5
                    sorted_X[b, base_idx:base_idx+5] = obj_features
        
        return sorted_X
    
    def _extract_high_level_features(self, X):
        batch_size = X.shape[0]
        features = torch.zeros((batch_size, 10), dtype=torch.float32)
        
        for b in range(batch_size):
            # Sum of pT for all objects
            pt_sum = 0
            # ΔR between the two highest pT objects
            delta_r_leading = 0
            # Number of objects with pT > 50 GeV
            high_pt_count = 0
            
            # Leading pT and second-leading pT
            pt_leading = 0
            pt_second = 0
            
            # Leading eta and phi
            eta_leading = 0
            phi_leading = 0
            
            # Missing ET and its phi
            missing_et = X[b, 0].item()
            missing_phi = X[b, 1].item()
            
            leading_obj = None
            second_obj = None
            
            for i in range(self.n_objects):
                base_idx = 2 + i * 5
                obj_id = X[b, base_idx].item()
                
                if obj_id != 0:  # Valid object
                    pt = X[b, base_idx + 2].item()
                    pt_sum += pt
                    
                    if pt > 50000:  # 50 GeV threshold (pt is in MeV)
                        high_pt_count += 1
                    
                    if pt > pt_leading:
                        pt_second = pt_leading
                        second_obj = leading_obj
                        
                        pt_leading = pt
                        eta_leading = X[b, base_idx + 3].item()
                        phi_leading = X[b, base_idx + 4].item()
                        leading_obj = (eta_leading, phi_leading)
                    elif pt > pt_second:
                        pt_second = pt
                        phi2 = X[b, base_idx + 4].item()
                        eta2 = X[b, base_idx + 3].item()
                        second_obj = (eta2, phi2)
            
            # Calculate ΔR between leading objects if both exist
            if leading_obj and second_obj:
                eta1, phi1 = leading_obj
                eta2, phi2 = second_obj
                delta_eta = eta1 - eta2
                delta_phi = self._delta_phi(phi1, phi2)
                delta_r_leading = math.sqrt(delta_eta**2 + delta_phi**2)
            
            # Store features
            features[b, 0] = pt_sum / 1000  # Convert to GeV
            features[b, 1] = delta_r_leading
            features[b, 2] = high_pt_count
            features[b, 3] = pt_leading / 1000 if pt_leading > 0 else 0  # Convert to GeV
            features[b, 4] = pt_second / 1000 if pt_second > 0 else 0  # Convert to GeV
            features[b, 5] = eta_leading
            features[b, 6] = phi_leading
            features[b, 7] = missing_et / 1000  # Convert to GeV
            features[b, 8] = missing_phi
            features[b, 9] = pt_sum / (missing_et + 1)  # Ratio of visible to invisible energy
            
        return features
    
    def _delta_phi(self, phi1, phi2):
        # Calculate the difference between two azimuthal angles
        delta = abs(phi1 - phi2)
        return min(delta, 2 * math.pi - delta)
    
    def _calculate_multiplicity(self, X):
        batch_size = X.shape[0]
        
        # Track counts of different object types
        features = torch.zeros((batch_size, 5), dtype=torch.float32)
        
        for b in range(batch_size):
            # Count objects by type
            obj_count = 0
            b_jets = 0
            lep_electrons = 0
            lep_muons = 0
            
            for i in range(self.n_objects):
                base_idx = 2 + i * 5
                obj_id = X[b, base_idx].item()
                
                if obj_id != 0:
                    obj_count += 1
                    
                    # Different particle types based on obj_id
                    # Assuming: obj_id values might indicate particle types
                    # The actual mapping would depend on the dataset specifics
                    if obj_id == 5:  # Assuming 5 is for b-jets (this is a guess)
                        b_jets += 1
                    elif obj_id == 11:  # Assuming 11 is for electrons (this is a guess)
                        lep_electrons += 1
                    elif obj_id == 13:  # Assuming 13 is for muons (this is a guess)
                        lep_muons += 1
            
            # Store the counts
            features[b, 0] = obj_count
            features[b, 1] = b_jets
            features[b, 2] = lep_electrons
            features[b, 3] = lep_muons
            features[b, 4] = lep_electrons + lep_muons  # Total leptons
            
        return features
    
    def _calculate_delta_r(self, X):
        batch_size = X.shape[0]
        
        # We'll calculate ΔR between pairs of the top 4 objects (if they exist)
        max_pairs = 6  # C(4,2) = 6 possible pairs among top 4 objects
        features = torch.zeros((batch_size, max_pairs), dtype=torch.float32)
        
        for b in range(batch_size):
            # Extract top 4 objects
            top_objects = []
            
            for i in range(4):  # Consider only top 4 objects
                base_idx = 2 + i * 5
                obj_id = X[b, base_idx].item()
                
                if obj_id != 0:  # Valid object
                    eta = X[b, base_idx + 3].item()
                    phi = X[b, base_idx + 4].item()
                    top_objects.append((eta, phi))
            
            # Calculate ΔR between all pairs
            pair_idx = 0
            for i in range(len(top_objects)):
                for j in range(i+1, len(top_objects)):
                    if pair_idx < max_pairs:
                        eta1, phi1 = top_objects[i]
                        eta2, phi2 = top_objects[j]
                        
                        delta_eta = eta1 - eta2
                        delta_phi = self._delta_phi(phi1, phi2)
                        delta_r = math.sqrt(delta_eta**2 + delta_phi**2)
                        
                        features[b, pair_idx] = delta_r
                        pair_idx += 1
            
        return features
        
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out

class PhysicsAwareNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # Network architecture parameters
        hidden_dim = 256
        n_res_blocks = 3
        
        # Input layer
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        
        # Residual blocks for deep representation
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim // 2) for _ in range(n_res_blocks)]
        )
        
        # Output layers with dropout for regularization
        self.dropout = nn.Dropout(0.3)
        self.fc_output = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.bn_input(self.fc_input(x)))
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Final classification
        x = self.dropout(x)
        x = self.fc_output(x)
        return x.squeeze()

def make_model(input_dim):
    model = PhysicsAwareNet(input_dim)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Binary classification with class imbalance - use weighted BCE loss
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam optimizer with weight decay for regularization
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler to reduce LR when validation metrics plateau
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5)
    
    # Tracking metrics
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    best_auc = 0
    best_model_state = None
    
    print(f"Training on {device}")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Convert targets to float for BCE loss
            targets_float = targets.float()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets_float)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * inputs.size(0)
            train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_targets.extend(targets.cpu().numpy())
            
            # Track accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += targets.size(0)
            train_correct += (predicted == targets_float).sum().item()
        
        train_loss /= len(train_loader.dataset)
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
                inputs, targets = inputs.to(device), targets.to(device)
                targets_float = targets.float()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets_float)
                
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
                # Track accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted == targets_float).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_targets, val_preds)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model based on validation AUC
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Store history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
    
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

