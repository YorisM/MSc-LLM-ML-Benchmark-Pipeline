
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

EPOCHS = 10
                        
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
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
import math

class MyPreprocessor:
    def __init__(self):
        self.n_objects = 18
        self.object_features = 5
        self.valid_obj_mask = None
        self.is_fitted = False
        self.feature_means = None
        self.feature_stds = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, X, y=None):
        # Extract features from raw data
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        # Identify valid objects (non-zero object IDs)
        object_ids = X[:, 2:self.n_objects*self.object_features:self.object_features]
        self.valid_obj_mask = object_ids != 0
        
        # Extract E_T_miss and phi_E_t_miss
        et_miss = X[:, 0:2]
        
        # Extract object features (E, pT, eta, phi)
        features_list = []
        
        # Iterate through objects
        for i in range(self.n_objects):
            start_idx = 2 + i * self.object_features
            # Skip object ID and extract E, pT, eta, phi
            obj_features = X[:, [start_idx+1, start_idx+2, start_idx+3, start_idx+4]]
            mask = self.valid_obj_mask[:, i].unsqueeze(1).expand_as(obj_features)
            # Zero out invalid objects
            features_list.append(obj_features * mask)
        
        # Concatenate all features
        all_features = torch.cat([et_miss] + features_list, dim=1)
        
        # Compute statistics for normalization
        self.feature_means = torch.mean(all_features, dim=0)
        self.feature_stds = torch.std(all_features, dim=0)
        self.feature_stds[self.feature_stds < 1e-5] = 1.0  # Avoid division by zero
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        # Extract E_T_miss and phi_E_t_miss
        et_miss = X[:, 0:2]
        
        # Identify valid objects
        object_ids = X[:, 2:self.n_objects*self.object_features:self.object_features]
        valid_obj_mask = object_ids != 0
        
        # Extract object features and add derived features
        physics_features = []
        valid_objects_count = torch.sum(valid_obj_mask, dim=1, keepdim=True)
        physics_features.append(valid_objects_count)
        
        # Process E_T_miss and phi_E_t_miss
        physics_features.append(et_miss)
        
        # Create features per object type
        jet_mask = (object_ids >= 1) & (object_ids <= 6)
        electron_mask = object_ids == 11
        muon_mask = object_ids == 13
        
        # Count objects by type
        jet_count = torch.sum(jet_mask, dim=1, keepdim=True)
        electron_count = torch.sum(electron_mask, dim=1, keepdim=True)
        muon_count = torch.sum(muon_mask, dim=1, keepdim=True)
        physics_features.extend([jet_count, electron_count, muon_count])
        
        # Process each object's features
        all_energies = []
        all_pts = []
        all_etas = []
        all_phis = []
        
        for i in range(self.n_objects):
            start_idx = 2 + i * self.object_features
            obj_id = X[:, start_idx]
            obj_e = X[:, start_idx+1]
            obj_pt = X[:, start_idx+2]
            obj_eta = X[:, start_idx+3]
            obj_phi = X[:, start_idx+4]
            
            # Only consider valid objects
            mask = valid_obj_mask[:, i]
            all_energies.append(obj_e.unsqueeze(1) * mask.unsqueeze(1))
            all_pts.append(obj_pt.unsqueeze(1) * mask.unsqueeze(1))
            all_etas.append(obj_eta.unsqueeze(1) * mask.unsqueeze(1))
            all_phis.append(obj_phi.unsqueeze(1) * mask.unsqueeze(1))
        
        # Create physics-motivated features
        # Total energy and pT
        total_energy = torch.sum(torch.stack(all_energies, dim=1), dim=1)
        total_pt = torch.sum(torch.stack(all_pts, dim=1), dim=1)
        physics_features.extend([total_energy, total_pt])
        
        # Sort pT in descending order for the top 5 objects
        sorted_pt, _ = torch.sort(torch.cat(all_pts, dim=1), dim=1, descending=True)
        top_5_pt = sorted_pt[:, :5]
        physics_features.append(top_5_pt)
        
        # Calculate pT ratios for the top 3 objects
        if top_5_pt.size(1) >= 3:
            pt_ratio_1_2 = top_5_pt[:, 0:1] / (top_5_pt[:, 1:2] + 1e-8)
            pt_ratio_1_3 = top_5_pt[:, 0:1] / (top_5_pt[:, 2:3] + 1e-8)
            pt_ratio_2_3 = top_5_pt[:, 1:2] / (top_5_pt[:, 2:3] + 1e-8)
            physics_features.extend([pt_ratio_1_2, pt_ratio_1_3, pt_ratio_2_3])
        
        # Get eta and phi for top pT objects
        top_obj_indices = torch.zeros(X.size(0), 5, dtype=torch.long)
        for i in range(X.size(0)):
            pts = torch.tensor([all_pts[j][i] for j in range(len(all_pts))])
            _, indices = torch.sort(pts, descending=True)
            top_obj_indices[i, :min(5, len(indices))] = indices[:min(5, len(indices))]
        
        top_etas = []
        top_phis = []
        for i in range(min(5, self.n_objects)):
            batch_indices = torch.arange(X.size(0))
            obj_indices = top_obj_indices[:, i]
            top_etas.append(torch.stack([all_etas[obj_indices[j]][j] for j in batch_indices], dim=0))
            top_phis.append(torch.stack([all_phis[obj_indices[j]][j] for j in batch_indices], dim=0))
        
        physics_features.extend(top_etas)
        physics_features.extend(top_phis)
        
        # Calculate delta R between top objects
        for i in range(min(4, len(top_etas))):
            for j in range(i+1, min(5, len(top_etas))):
                delta_eta = top_etas[i] - top_etas[j]
                delta_phi = torch.abs(top_phis[i] - top_phis[j])
                delta_phi = torch.minimum(delta_phi, 2*math.pi - delta_phi)
                delta_r = torch.sqrt(delta_eta**2 + delta_phi**2)
                physics_features.append(delta_r)
        
        # Calculate HT (scalar sum of pT)
        ht = torch.sum(torch.cat(all_pts, dim=1), dim=1, keepdim=True)
        physics_features.append(ht)
        
        # Concatenate all features
        transformed_features = torch.cat([feat for feat in physics_features], dim=1)
        
        # Normalize features
        transformed_features = (transformed_features - self.feature_means[:transformed_features.size(1)]) / \
                               self.feature_stds[:transformed_features.size(1)]
        
        # Replace NaNs with zeros
        transformed_features = torch.nan_to_num(transformed_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return transformed_features

def make_preprocessor():
    return MyPreprocessor()

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        return F.leaky_relu(out)

class TTTTClassifier(nn.Module):
    def __init__(self, input_dim):
        super(TTTTClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(256, 512)
        self.res_block2 = ResidualBlock(256, 512)
        self.res_block3 = ResidualBlock(256, 512)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x.squeeze(1)

def make_model(input_dim):
    return TTTTClassifier(input_dim)

EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use binary cross-entropy loss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Initialize optimizer with learning rate schedule
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=False)
    
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets_float = targets.float()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets_float)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            predicted = torch.sigmoid(outputs) >= 0.5
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Store predictions and targets for AUC calculation
            train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_auc = roc_auc_score(train_targets, train_preds)
        
        train_loss.append(epoch_loss)
        train_acc.append(epoch_auc)  # Using AUC instead of accuracy
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets_float = targets.float()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets_float)
                
                running_loss += loss.item() * inputs.size(0)
                predicted = torch.sigmoid(outputs) >= 0.5
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_auc = roc_auc_score(val_targets, val_preds)
        
        val_loss.append(epoch_loss)
        val_acc.append(epoch_auc)  # Using AUC instead of accuracy
        
        # Update learning rate scheduler
        scheduler.step(epoch_loss)
    
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
    torch.jit.script(trained).save(f"{base}_scripted.pt")
    torch.jit.script(pre).save(f"{base}_preproc.pt")
    with open(f"{base}_pre.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

