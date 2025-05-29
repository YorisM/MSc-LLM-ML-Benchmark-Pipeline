
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
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # Store scalers for numerical features
        self.scaler = StandardScaler()
        # Store the means and std for each feature type
        self.E_scaler = StandardScaler()
        self.pT_scaler = StandardScaler()
        self.eta_scaler = StandardScaler()
        self.phi_scaler = StandardScaler()
        self.Et_miss_scaler = StandardScaler()
        self.phi_Et_miss_scaler = StandardScaler()
        
        # Mask to identify valid objects (non-zero)
        self.valid_mask = None
        
    def fit(self, X, y=None):
        # Extract feature columns
        # First two columns are special: missing-ET magnitude and azimuth
        Et_miss = X[:, 0].reshape(-1, 1)
        phi_Et_miss = X[:, 1].reshape(-1, 1)
        
        # Fit scalers for these special features
        self.Et_miss_scaler.fit(Et_miss)
        self.phi_Et_miss_scaler.fit(phi_Et_miss)
        
        # Process object data (18 objects, each with 5 features)
        # Create arrays to store extracted features
        E_values = []
        pT_values = []
        eta_values = []
        phi_values = []
        
        # Extract features for each object
        for i in range(18):
            start_idx = 2 + i * 5  # Starting index for this object
            
            # Check if object exists (obj_id != 0)
            obj_mask = X[:, start_idx] != 0
            
            # Extract features for valid objects
            if torch.any(obj_mask):
                E_values.append(X[obj_mask, start_idx + 1].numpy().reshape(-1, 1))
                pT_values.append(X[obj_mask, start_idx + 2].numpy().reshape(-1, 1))
                eta_values.append(X[obj_mask, start_idx + 3].numpy().reshape(-1, 1))
                phi_values.append(X[obj_mask, start_idx + 4].numpy().reshape(-1, 1))
        
        # Concatenate values for fitting scalers
        if E_values:
            E_concat = np.vstack(E_values)
            pT_concat = np.vstack(pT_values)
            eta_concat = np.vstack(eta_values)
            phi_concat = np.vstack(phi_values)
            
            # Fit scalers for each feature type
            self.E_scaler.fit(E_concat)
            self.pT_scaler.fit(pT_concat)
            self.eta_scaler.fit(eta_concat)
            self.phi_scaler.fit(phi_concat)
        
        return self

    def transform(self, X):
        # Initialize array for transformed features
        batch_size = X.shape[0]
        
        # Extract and scale the missing ET features
        Et_miss = X[:, 0].reshape(-1, 1)
        phi_Et_miss = X[:, 1].reshape(-1, 1)
        
        Et_miss_scaled = torch.from_numpy(self.Et_miss_scaler.transform(Et_miss)).float()
        phi_Et_miss_scaled = torch.from_numpy(self.phi_Et_miss_scaler.transform(phi_Et_miss)).float()
        
        # Create arrays to store features for each object category
        features = []
        
        # Add the scaled missing ET features
        features.append(Et_miss_scaled)
        features.append(phi_Et_miss_scaled)
        
        # Object counts per event
        obj_counts = torch.zeros(batch_size, 1)
        
        # Process each object
        for i in range(18):
            start_idx = 2 + i * 5
            
            # Identify valid objects (where obj_id != 0)
            obj_mask = X[:, start_idx] != 0
            
            # Update object count
            obj_counts += obj_mask.float().unsqueeze(1)
            
            # Extract object type as one-hot encoding
            obj_type = X[:, start_idx].clone()
            
            # Replace zeros with a placeholder value for invalid objects
            obj_type[~obj_mask] = -1
            
            # One-hot encode object types (for valid objects)
            # Using values 0-6 for common object types in HEP
            obj_type_one_hot = torch.zeros(batch_size, 7)
            for j in range(7):
                obj_type_one_hot[:, j] = (obj_type == j).float()
            
            # Only extract features for valid objects
            E = X[:, start_idx + 1].clone().reshape(-1, 1)
            pT = X[:, start_idx + 2].clone().reshape(-1, 1)
            eta = X[:, start_idx + 3].clone().reshape(-1, 1)
            phi = X[:, start_idx + 4].clone().reshape(-1, 1)
            
            # Scale valid features, leave invalid as zero
            if torch.any(obj_mask):
                # Create masks for scaling valid data
                E_mask = E.clone()
                pT_mask = pT.clone()
                eta_mask = eta.clone()
                phi_mask = phi.clone()
                
                # Zero out invalid objects
                E_mask[~obj_mask] = 0
                pT_mask[~obj_mask] = 0
                eta_mask[~obj_mask] = 0
                phi_mask[~obj_mask] = 0
                
                # Apply scaling only to valid objects
                valid_E = E[obj_mask].numpy().reshape(-1, 1)
                valid_pT = pT[obj_mask].numpy().reshape(-1, 1)
                valid_eta = eta[obj_mask].numpy().reshape(-1, 1)
                valid_phi = phi[obj_mask].numpy().reshape(-1, 1)
                
                scaled_E = self.E_scaler.transform(valid_E)
                scaled_pT = self.pT_scaler.transform(valid_pT)
                scaled_eta = self.eta_scaler.transform(valid_eta)
                scaled_phi = self.phi_scaler.transform(valid_phi)
                
                # Put scaled values back in tensors
                E_scaled = torch.zeros_like(E)
                pT_scaled = torch.zeros_like(pT)
                eta_scaled = torch.zeros_like(eta)
                phi_scaled = torch.zeros_like(phi)
                
                E_scaled[obj_mask] = torch.from_numpy(scaled_E).float().view(-1)
                pT_scaled[obj_mask] = torch.from_numpy(scaled_pT).float().view(-1)
                eta_scaled[obj_mask] = torch.from_numpy(scaled_eta).float().view(-1)
                phi_scaled[obj_mask] = torch.from_numpy(scaled_phi).float().view(-1)
                
                # Apply validity mask to entire vectors
                E_scaled = E_scaled * obj_mask.float().unsqueeze(1)
                pT_scaled = pT_scaled * obj_mask.float().unsqueeze(1)
                eta_scaled = eta_scaled * obj_mask.float().unsqueeze(1)
                phi_scaled = phi_scaled * obj_mask.float().unsqueeze(1)
                
                # Create a presence flag for this object (1 if present, 0 if not)
                presence = obj_mask.float().unsqueeze(1)
                
                # Append all features for this object
                features.append(obj_type_one_hot)
                features.append(E_scaled)
                features.append(pT_scaled)
                features.append(eta_scaled)
                features.append(phi_scaled)
                features.append(presence)
            else:
                # If no valid objects of this type in the batch, add zeros
                features.append(torch.zeros(batch_size, 7))
                features.append(torch.zeros(batch_size, 1))
                features.append(torch.zeros(batch_size, 1))
                features.append(torch.zeros(batch_size, 1))
                features.append(torch.zeros(batch_size, 1))
                features.append(torch.zeros(batch_size, 1))
        
        # Add object count as a feature
        features.append(obj_counts)
        
        # Calculate histogram features of pt and E distributions
        pt_bins = torch.linspace(0, 1e6, 10)
        E_bins = torch.linspace(0, 2e6, 10)
        
        pt_hist = torch.zeros(batch_size, len(pt_bins)-1)
        E_hist = torch.zeros(batch_size, len(E_bins)-1)
        
        for i in range(18):
            start_idx = 2 + i * 5
            obj_mask = X[:, start_idx] != 0
            
            if torch.any(obj_mask):
                pT = X[:, start_idx + 2]
                E = X[:, start_idx + 1]
                
                # For each valid object, update the histograms
                for j in range(batch_size):
                    if obj_mask[j]:
                        # Find bin for pT
                        for k in range(len(pt_bins)-1):
                            if pt_bins[k] <= pT[j] < pt_bins[k+1]:
                                pt_hist[j, k] += 1
                                break
                        
                        # Find bin for E
                        for k in range(len(E_bins)-1):
                            if E_bins[k] <= E[j] < E_bins[k+1]:
                                E_hist[j, k] += 1
                                break
        
        # Normalize histograms
        obj_counts_clipped = torch.clamp(obj_counts, min=1)  # Avoid division by zero
        pt_hist = pt_hist / obj_counts_clipped
        E_hist = E_hist / obj_counts_clipped
        
        features.append(pt_hist)
        features.append(E_hist)
        
        # Calculate some physics-inspired features
        
        # Total energy in the event
        total_E = torch.zeros(batch_size, 1)
        # Total transverse momentum
        total_pT = torch.zeros(batch_size, 1)
        
        for i in range(18):
            start_idx = 2 + i * 5
            obj_mask = X[:, start_idx] != 0
            
            if torch.any(obj_mask):
                total_E += X[:, start_idx + 1].unsqueeze(1) * obj_mask.float().unsqueeze(1)
                total_pT += X[:, start_idx + 2].unsqueeze(1) * obj_mask.float().unsqueeze(1)
        
        # Add these derived features
        features.append(total_E)
        features.append(total_pT)
        
        # Combine all features
        X_transformed = torch.cat(features, dim=1)
        
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(in_features)
        self.fc2 = nn.Linear(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x += residual
        x = F.relu(x)
        return x

class DeepParticleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.3):
        super(DeepParticleNet, self).__init__()
        
        # Initial layer to match dimensions
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)
        self.res_block3 = ResidualBlock(hidden_dim)
        
        # Prediction layers with dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_pred1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_pred1 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc_pred2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn_pred2 = nn.BatchNorm1d(hidden_dim // 4)
        self.fc_output = nn.Linear(hidden_dim // 4, 1)
    
    def forward(self, x):
        # Initial processing
        x = F.relu(self.bn_input(self.fc_input(x)))
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Prediction head
        x = self.dropout(x)
        x = F.relu(self.bn_pred1(self.fc_pred1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_pred2(self.fc_pred2(x)))
        x = self.dropout(x)
        x = self.fc_output(x)
        
        return torch.sigmoid(x).squeeze(1)

def make_model(input_dim):
    model = DeepParticleNet(input_dim)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    # Initialize lists to store metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_auc = 0
    best_model_state = None
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Store predictions and targets for AUC calculation
            all_preds.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
        # Calculate AUC for training set
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        train_auc = roc_auc_score(all_targets, all_preds)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets.float())
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Store predictions and targets for AUC calculation
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Calculate AUC for validation set
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        val_auc = roc_auc_score(all_targets, all_preds)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        val_loss.append(epoch_loss)
        val_acc.append(epoch_acc)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, '
              f'Train Acc: {train_acc[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}, '
              f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')
    
    # Load best model weights
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

