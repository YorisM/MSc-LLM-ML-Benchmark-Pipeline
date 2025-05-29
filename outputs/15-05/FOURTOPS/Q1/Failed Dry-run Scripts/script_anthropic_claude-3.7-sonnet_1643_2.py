
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
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

class FeaturePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_mask = None
        
    def fit(self, X, y=None):
        # Extract meaningful features and remove padding
        # First identify which columns are likely padding (all zeros or very low variance)
        variance = np.var(X, axis=0)
        self.feature_mask = variance > 1e-6  # Keep features with some variance
        
        # Fit scaler on non-padding columns
        self.scaler.fit(X[:, self.feature_mask])
        return self
    
    def transform(self, X):
        # Apply feature mask and scale
        X_selected = X[:, self.feature_mask]
        X_scaled = self.scaler.transform(X_selected)
        
        # Create physics-inspired features
        features_list = [X_scaled]  # Start with basic scaled features
        
        # Extract missing energy (first two columns)
        et_miss = X[:, 0:1]  # Missing transverse energy magnitude
        phi_et_miss = X[:, 1:2]  # Missing transverse energy phi
        
        # Calculate additional features from the objects
        # We look at the data format: E, pT, eta, phi features for each object
        num_objects = (X.shape[1] - 2) // 5  # -2 for ET_miss and phi_ET_miss, 5 for obj_id, E, pT, eta, phi
        
        # Calculate sums of energies and momenta
        total_energy = np.zeros((X.shape[0], 1))
        total_pt = np.zeros((X.shape[0], 1))
        
        for i in range(num_objects):
            obj_start = 2 + i * 5  # Skip ET_miss and phi_ET_miss
            
            # Check if this is a valid object (non-zero)
            obj_mask = X[:, obj_start] != 0
            
            # Energy is at obj_start + 1
            energy = X[:, obj_start + 1:obj_start + 2]
            # pT is at obj_start + 2
            pt = X[:, obj_start + 2:obj_start + 3]
            # eta is at obj_start + 3
            eta = X[:, obj_start + 3:obj_start + 4]
            # phi is at obj_start + 4
            phi = X[:, obj_start + 4:obj_start + 5]
            
            # Sum energy and pT when object exists
            total_energy += np.where(obj_mask.reshape(-1, 1), energy, 0)
            total_pt += np.where(obj_mask.reshape(-1, 1), pt, 0)
            
            # Add individual object features if informative
            if np.mean(obj_mask) > 0.1:  # Only add if object appears in >10% of events
                masked_energy = np.where(obj_mask.reshape(-1, 1), energy, 0)
                masked_pt = np.where(obj_mask.reshape(-1, 1), pt, 0)
                masked_eta = np.where(obj_mask.reshape(-1, 1), eta, 0)
                masked_phi = np.where(obj_mask.reshape(-1, 1), phi, 0)
                
                features_list.append(masked_energy)
                features_list.append(masked_pt)
                features_list.append(masked_eta)
                features_list.append(masked_phi)
        
        # Add global event features
        features_list.append(total_energy)  # Total energy in event
        features_list.append(total_pt)     # Total transverse momentum
        features_list.append(et_miss)      # Missing transverse energy
        
        # Count number of objects with pT > 50000 MeV = 50 GeV (typical for top quark decay products)
        high_pt_count = np.zeros((X.shape[0], 1))
        for i in range(num_objects):
            obj_start = 2 + i * 5
            pt_idx = obj_start + 2
            high_pt_count += (X[:, pt_idx] > 50000).astype(float).reshape(-1, 1)
        features_list.append(high_pt_count)
        
        # Calculate some ratios and differences that could be informative
        features_list.append(et_miss / (total_pt + 1e-6))  # Ratio of missing ET to total pT
        
        # Combine all features
        result = np.hstack(features_list)
        return torch.tensor(result, dtype=torch.float32)

def make_preprocessor():
    return FeaturePreprocessor()

class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)
        
    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += identity
        out = F.leaky_relu(out)
        return out

class AttentionModule(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)
        self.scale = np.sqrt(in_features)
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Self-attention
        attention = torch.matmul(q, k.transpose(0, 1)) / self.scale
        attention = F.softmax(attention, dim=1)
        out = torch.matmul(attention, v)
        
        return out + x  # Residual connection

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # Determine network size based on input dimension
        hidden_dim = max(256, input_dim * 2)
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU()
        )
        
        # Deep network with residual connections
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim // 2) for _ in range(3)
        ])
        
        # Attention mechanism
        self.attention = AttentionModule(hidden_dim)
        
        # Output layers with gradual dimension reduction
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, x):
        # Input processing
        x = self.input_layer(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Apply attention
        # We only apply attention to the batch dimension
        # x = self.attention(x)
        
        # Output layers
        x = self.output_layers(x)
        
        # No sigmoid here - we'll use BCEWithLogitsLoss
        return x.squeeze(1)

def make_model(input_dim):
    return BinaryClassifier(input_dim)

epochs = 30

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Binary cross-entropy loss with logits (includes sigmoid)
    criterion = nn.BCEWithLogitsLoss()
    
    # AdamW optimizer with learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Track metrics
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    best_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_predictions = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            # Metrics
            train_loss += loss.item() * inputs.size(0)
            predicted = torch.sigmoid(outputs) >= 0.5
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
            
            # Store for AUC calculation
            train_predictions.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Calculate training metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_auc = roc_auc_score(train_targets, train_predictions)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Metrics
                val_loss += loss.item() * inputs.size(0)
                predicted = torch.sigmoid(outputs) >= 0.5
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
                
                # Store for AUC calculation
                val_predictions.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_targets, val_predictions)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)
    
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
