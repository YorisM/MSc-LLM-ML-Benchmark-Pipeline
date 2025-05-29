
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
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

class PhysicsPreProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scalers = {}
        self.n_features = None
        
    def fit(self, X, y=None):
        # Separate the missing energy features (first two columns)
        et_miss = X[:, 0:2]
        
        # Create a scaler for the missing energy features
        self.scalers['et_miss'] = StandardScaler()
        self.scalers['et_miss'].fit(et_miss)
        
        # Process the object features
        remaining = X[:, 2:].reshape(X.shape[0], -1, 5)  # Reshape into (batch, n_objects, 5)
        
        # Create a mask for non-zero entries (valid objects)
        mask = (remaining.sum(axis=2) != 0)
        
        # Extract valid object data
        valid_objects = remaining[mask]
        
        # Fit scaler on valid object features, ignoring the object identifier
        self.scalers['objects'] = StandardScaler()
        self.scalers['objects'].fit(valid_objects[:, 1:])  # Exclude object identifier
        
        # Calculate derived features for the fit
        self._calculate_feature_stats(X)
        
        return self
    
    def _calculate_feature_stats(self, X):
        # Calculate statistics for feature engineering
        remaining = X[:, 2:].reshape(X.shape[0], -1, 5)  # Reshape into (batch, n_objects, 5)
        
        # Count objects per event
        mask = (remaining.sum(axis=2) != 0)
        self.max_objects = mask.sum(axis=1).max()
        self.mean_objects = mask.sum(axis=1).mean()
        self.std_objects = mask.sum(axis=1).std()
        
    def transform(self, X):
        # Initialize the resulting features tensor
        batch_size = X.shape[0]
        
        # Extract and scale the missing energy features
        et_miss = X[:, 0:2]
        et_miss_scaled = self.scalers['et_miss'].transform(et_miss)
        
        # Process the object features
        remaining = X[:, 2:].reshape(batch_size, -1, 5)  # Reshape into (batch, n_objects, 5)
        n_objects = remaining.shape[1]
        
        # Create a mask for non-zero entries (valid objects)
        mask = (remaining.sum(axis=2) != 0)
        
        # Initialize arrays for engineered features
        n_jets = np.zeros(batch_size)
        n_leptons = np.zeros(batch_size)
        n_photons = np.zeros(batch_size)
        sum_pt = np.zeros(batch_size)
        mean_pt = np.zeros(batch_size)
        std_pt = np.zeros(batch_size)
        max_pt = np.zeros(batch_size)
        min_pt = np.ones(batch_size) * 1e9
        sum_energy = np.zeros(batch_size)
        mean_eta = np.zeros(batch_size)
        std_eta = np.zeros(batch_size)
        max_eta_abs = np.zeros(batch_size)
        
        # Initialize arrays for object types
        jet_pts = [[] for _ in range(batch_size)]
        lepton_pts = [[] for _ in range(batch_size)]
        photon_pts = [[] for _ in range(batch_size)]
        
        # Initialize arrays for angular differences
        delta_phi_max = np.zeros(batch_size)
        delta_eta_max = np.zeros(batch_size)
        delta_R_max = np.zeros(batch_size)
        
        # Process each event
        for i in range(batch_size):
            valid_indices = np.where(mask[i])[0]
            
            if len(valid_indices) == 0:
                continue
                
            valid_objects = remaining[i, valid_indices]
            obj_types = valid_objects[:, 0]
            energies = valid_objects[:, 1]
            pts = valid_objects[:, 2]
            etas = valid_objects[:, 3]
            phis = valid_objects[:, 4]
            
            # Count object types (assuming object type encoded in obj_id)
            n_jets[i] = np.sum(obj_types == 0)  # Assuming 0 encodes jets
            n_leptons[i] = np.sum((obj_types == 1) | (obj_types == 2))  # Assuming 1,2 encode leptons
            n_photons[i] = np.sum(obj_types == 3)  # Assuming 3 encodes photons
            
            # Collect pts by object type
            jet_pts[i] = pts[obj_types == 0].tolist()
            lepton_pts[i] = pts[(obj_types == 1) | (obj_types == 2)].tolist()
            photon_pts[i] = pts[obj_types == 3].tolist()
            
            # Calculate pt statistics
            sum_pt[i] = np.sum(pts)
            mean_pt[i] = np.mean(pts)
            std_pt[i] = np.std(pts) if len(pts) > 1 else 0
            max_pt[i] = np.max(pts) if len(pts) > 0 else 0
            min_pt[i] = np.min(pts) if len(pts) > 0 else 0
            
            # Energy sum
            sum_energy[i] = np.sum(energies)
            
            # Eta statistics
            mean_eta[i] = np.mean(etas)
            std_eta[i] = np.std(etas) if len(etas) > 1 else 0
            max_eta_abs[i] = np.max(np.abs(etas)) if len(etas) > 0 else 0
            
            # Calculate maximum angular differences
            if len(valid_indices) > 1:
                # Calculate all pairwise differences
                phi_diff = np.array([np.abs(np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi) 
                                    for i, phi1 in enumerate(phis[:-1]) for phi2 in phis[i+1:]])
                eta_diff = np.array([np.abs(eta1 - eta2) 
                                    for i, eta1 in enumerate(etas[:-1]) for eta2 in etas[i+1:]])
                R_diff = np.sqrt(phi_diff**2 + eta_diff**2)
                
                delta_phi_max[i] = np.max(phi_diff) if len(phi_diff) > 0 else 0
                delta_eta_max[i] = np.max(eta_diff) if len(eta_diff) > 0 else 0
                delta_R_max[i] = np.max(R_diff) if len(R_diff) > 0 else 0
        
        # Create the features array
        features = np.column_stack([
            et_miss_scaled,
            n_jets, n_leptons, n_photons,
            sum_pt, mean_pt, std_pt, max_pt, min_pt,
            sum_energy, 
            mean_eta, std_eta, max_eta_abs,
            delta_phi_max, delta_eta_max, delta_R_max
        ])
        
        # Add top N pt values for each object type
        top_n = 3
        
        # Add top jet pts
        for j in range(top_n):
            jet_pt_j = np.array([pts[j] if j < len(pts) else 0 for pts in jet_pts])
            features = np.column_stack([features, jet_pt_j])
            
        # Add top lepton pts
        for j in range(top_n):
            lepton_pt_j = np.array([pts[j] if j < len(pts) else 0 for pts in lepton_pts])
            features = np.column_stack([features, lepton_pt_j])
            
        # Add top photon pts
        for j in range(top_n):
            photon_pt_j = np.array([pts[j] if j < len(pts) else 0 for pts in photon_pts])
            features = np.column_stack([features, photon_pt_j])
        
        # Calculate missing energy to sum pt ratio
        et_miss_ratio = et_miss[:, 0] / (sum_pt + 1e-8)
        features = np.column_stack([features, et_miss_ratio])
        
        # Store the number of features
        self.n_features = features.shape[1]
        
        return features

def make_preprocessor():
    return PhysicsPreProcessor()

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.1)
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.leaky_relu(out, negative_slope=0.1)
        return out

class FourTopClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FourTopClassifier, self).__init__()
        # Initial projection to higher dimension
        self.fc_in = nn.Linear(input_dim, 256)
        self.bn_in = nn.BatchNorm1d(256)
        
        # Residual blocks
        self.res1 = ResidualBlock(256, 512)
        self.res2 = ResidualBlock(256, 512)
        self.res3 = ResidualBlock(256, 512)
        
        # Output layers
        self.fc_out1 = nn.Linear(256, 128)
        self.bn_out1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc_out2 = nn.Linear(128, 64)
        self.bn_out2 = nn.BatchNorm1d(64)
        self.fc_out3 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Initial projection
        x = F.leaky_relu(self.bn_in(self.fc_in(x)), negative_slope=0.1)
        
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # Output layers
        x = F.leaky_relu(self.bn_out1(self.fc_out1(x)), negative_slope=0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn_out2(self.fc_out2(x)), negative_slope=0.1)
        x = self.fc_out3(x)
        
        return x.squeeze()

def make_model(input_dim):
    return FourTopClassifier(input_dim)

epochs = 30

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Binary cross entropy loss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Use Adam optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Initialize tracking variables
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
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            predicted = torch.sigmoid(outputs) > 0.5
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Save predictions and targets for AUC calculation
            train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_auc = roc_auc_score(train_targets, train_preds)
        
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                predicted = torch.sigmoid(outputs) > 0.5
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Save predictions and targets for AUC calculation
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct / total
        val_auc = roc_auc_score(val_targets, val_preds)
        
        val_loss.append(epoch_loss)
        val_acc.append(epoch_acc)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save the best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, Train AUC: {train_auc:.4f}, '
              f'Val Loss: {val_loss[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}, Val AUC: {val_auc:.4f}')
    
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
