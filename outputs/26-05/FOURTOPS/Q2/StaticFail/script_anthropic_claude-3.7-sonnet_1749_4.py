
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
import math
from sklearn.preprocessing import StandardScaler
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.n_objects = 18
        self.n_features_per_object = 5
        self.mask_value = 0.0  # Zero padding value
        self.et_miss_scaler = StandardScaler()
        self.phi_miss_scaler = StandardScaler()
        self.object_scalers = {
            'obj': StandardScaler(),
            'E': StandardScaler(),
            'pT': StandardScaler(),
            'eta': StandardScaler(),
            'phi': StandardScaler()
        }
        
    def fit(self, X, y=None):
        # Extract missing ET and its phi
        et_miss = X[:, 0].reshape(-1, 1)  # [N, 1]
        phi_miss = X[:, 1].reshape(-1, 1)  # [N, 1]
        
        # Fit scalers for missing ET and phi
        self.et_miss_scaler.fit(et_miss.numpy())
        self.phi_miss_scaler.fit(phi_miss.numpy())
        
        # Reshape to extract all objects
        # Each object has 5 features: obj_id, E, pT, eta, phi
        # Extract non-padding values for each feature type
        for feature_idx in range(5):
            values = []
            for obj_idx in range(self.n_objects):
                start_idx = 2 + obj_idx * self.n_features_per_object
                feature_pos = start_idx + feature_idx
                obj_values = X[:, feature_pos].reshape(-1, 1)  # [N, 1]
                
                # Only include non-zero values (assuming 0 is padding)
                mask = obj_values != self.mask_value
                filtered_values = obj_values[mask]
                if len(filtered_values) > 0:
                    values.append(filtered_values)
            
            if values:
                # Concatenate all non-zero values for this feature
                all_values = torch.cat(values).numpy()
                if feature_idx == 0:  # obj_id
                    self.object_scalers['obj'].fit(all_values.reshape(-1, 1))
                elif feature_idx == 1:  # E
                    self.object_scalers['E'].fit(all_values.reshape(-1, 1))
                elif feature_idx == 2:  # pT
                    self.object_scalers['pT'].fit(all_values.reshape(-1, 1))
                elif feature_idx == 3:  # eta
                    self.object_scalers['eta'].fit(all_values.reshape(-1, 1))
                elif feature_idx == 4:  # phi
                    self.object_scalers['phi'].fit(all_values.reshape(-1, 1))
        
        return self
    
    def transform(self, X):
        batch_size = X.shape[0]
        
        # Scale missing ET and phi
        et_miss = X[:, 0].reshape(-1, 1)  # [N, 1]
        phi_miss = X[:, 1].reshape(-1, 1)  # [N, 1]
        
        et_miss_scaled = torch.tensor(self.et_miss_scaler.transform(et_miss.numpy()), dtype=torch.float32)
        phi_miss_scaled = torch.tensor(self.phi_miss_scaler.transform(phi_miss.numpy()), dtype=torch.float32)
        
        # Create tensor to hold object features
        # Format: [batch_size, n_objects, 7] where 7 = (px, py, pz, E, obj_id, eta, phi)
        object_features = torch.zeros((batch_size, self.n_objects, 7), dtype=torch.float32)
        
        # Create mask for valid objects (1 for real objects, 0 for padding)
        object_mask = torch.zeros((batch_size, self.n_objects), dtype=torch.float32)
        
        # Process each object
        for obj_idx in range(self.n_objects):
            start_idx = 2 + obj_idx * self.n_features_per_object
            
            # Extract features for this object
            obj_id = X[:, start_idx].reshape(-1, 1)  # [N, 1]
            E = X[:, start_idx + 1].reshape(-1, 1)  # [N, 1]
            pT = X[:, start_idx + 2].reshape(-1, 1)  # [N, 1]
            eta = X[:, start_idx + 3].reshape(-1, 1)  # [N, 1]
            phi = X[:, start_idx + 4].reshape(-1, 1)  # [N, 1]
            
            # Create mask: 1 where at least one feature is non-zero
            valid_mask = ((obj_id != self.mask_value) | 
                          (E != self.mask_value) | 
                          (pT != self.mask_value) | 
                          (eta != self.mask_value) | 
                          (phi != self.mask_value)).float().squeeze(-1)
            
            # Set mask for this object
            object_mask[:, obj_idx] = valid_mask
            
            # Scale features (only where mask is 1)
            obj_id_scaled = torch.zeros_like(obj_id)
            E_scaled = torch.zeros_like(E)
            pT_scaled = torch.zeros_like(pT)
            eta_scaled = torch.zeros_like(eta)
            phi_scaled = torch.zeros_like(phi)
            
            # Only scale non-zero values
            valid_indices = (valid_mask == 1).squeeze()
            if valid_indices.any():
                obj_id_scaled[valid_indices] = torch.tensor(
                    self.object_scalers['obj'].transform(obj_id[valid_indices].numpy()), dtype=torch.float32
                )
                E_scaled[valid_indices] = torch.tensor(
                    self.object_scalers['E'].transform(E[valid_indices].numpy()), dtype=torch.float32
                )
                pT_scaled[valid_indices] = torch.tensor(
                    self.object_scalers['pT'].transform(pT[valid_indices].numpy()), dtype=torch.float32
                )
                eta_scaled[valid_indices] = torch.tensor(
                    self.object_scalers['eta'].transform(eta[valid_indices].numpy()), dtype=torch.float32
                )
                phi_scaled[valid_indices] = torch.tensor(
                    self.object_scalers['phi'].transform(phi[valid_indices].numpy()), dtype=torch.float32
                )
            
            # Calculate Cartesian components (px, py, pz) from pT, eta, phi
            px = pT * torch.cos(phi)  # [N, 1]
            py = pT * torch.sin(phi)  # [N, 1]
            pz = pT * torch.sinh(eta)  # [N, 1]
            
            # Store features in object_features tensor
            object_features[:, obj_idx, 0] = px.squeeze()  # px
            object_features[:, obj_idx, 1] = py.squeeze()  # py
            object_features[:, obj_idx, 2] = pz.squeeze()  # pz
            object_features[:, obj_idx, 3] = E.squeeze()   # E
            object_features[:, obj_idx, 4] = obj_id_scaled.squeeze()  # scaled obj_id
            object_features[:, obj_idx, 5] = eta_scaled.squeeze()     # scaled eta
            object_features[:, obj_idx, 6] = phi_scaled.squeeze()     # scaled phi
        
        # Calculate missing transverse momentum components
        px_miss = et_miss * torch.cos(phi_miss)  # [N, 1]
        py_miss = et_miss * torch.sin(phi_miss)  # [N, 1]
        
        # Create the final feature tensor
        # We'll combine: object features, object mask, and missing ET components
        # Format: [batch_size, n_objects, 7], [batch_size, n_objects], [batch_size, 2]
        
        # Flatten object features for the model input
        flat_objects = object_features.reshape(batch_size, -1)  # [N, n_objects * 7]
        flat_mask = object_mask.reshape(batch_size, -1)  # [N, n_objects]
        miss_features = torch.cat([px_miss, py_miss], dim=1)  # [N, 2]
        
        # Combine all features
        combined_features = torch.cat([flat_objects, flat_mask, miss_features], dim=1)
        
        return combined_features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class LorentzLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LorentzLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        # Special Lorentz-aware forward pass
        # For 4-vector components, we apply the Minkowski metric
        # Minkowski metric: g = diag(1, -1, -1, -1)
        # For 4-vectors (E, px, py, pz), we want to preserve E^2 - p^2 invariance
        
        return F.linear(x, self.weight, self.bias)

class LorentzEquivariantLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(LorentzEquivariantLayer, self).__init__()
        # Message passing networks for object interactions
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, mask=None):
        # x has shape [batch_size, n_objects, hidden_dim]
        batch_size, n_objects, hidden_dim = x.shape
        
        # Initialize messages
        messages = torch.zeros_like(x)
        
        # For each object, compute messages from all other objects
        for i in range(n_objects):
            # Extract features for current object
            x_i = x[:, i:i+1].expand(-1, n_objects, -1)  # [batch_size, n_objects, hidden_dim]
            
            # Concatenate with features of all objects
            x_ij = torch.cat([x_i, x], dim=2)  # [batch_size, n_objects, 2*hidden_dim]
            
            # Compute messages
            m_ij = self.message_net(x_ij)  # [batch_size, n_objects, hidden_dim]
            
            # Zero out messages from padding objects if mask is provided
            if mask is not None:
                m_ij = m_ij * mask.unsqueeze(-1)
            
            # Aggregate messages for object i
            m_i = torch.sum(m_ij, dim=1, keepdim=True)  # [batch_size, 1, hidden_dim]
            
            # Store message for object i
            messages[:, i:i+1] = m_i
        
        # Update object features with messages
        x_concat = torch.cat([x, messages], dim=2)  # [batch_size, n_objects, 2*hidden_dim]
        x_updated = self.update_net(x_concat) + x  # Residual connection
        
        return x_updated

class LorentzNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_objects=18):
        super(LorentzNetwork, self).__init__()
        
        self.n_objects = n_objects
        self.object_feature_dim = 7  # px, py, pz, E, obj_id, eta, phi
        
        # Embedding for objects
        self.object_embedding = nn.Sequential(
            nn.Linear(self.object_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Embedding for missing ET
        self.miss_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim),  # px_miss, py_miss
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Lorentz-equivariant message passing layers
        self.eq_layers = nn.ModuleList([
            LorentzEquivariantLayer(hidden_dim) for _ in range(3)
        ])
        
        # Final MLP for classification
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * (n_objects + 1), hidden_dim),  # +1 for missing ET
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract object features, mask, and missing ET components
        flat_objects = x[:, :self.n_objects * self.object_feature_dim]
        flat_mask = x[:, self.n_objects * self.object_feature_dim:self.n_objects * self.object_feature_dim + self.n_objects]
        miss_features = x[:, -2:]
        
        # Reshape object features and mask
        objects = flat_objects.reshape(batch_size, self.n_objects, self.object_feature_dim)
        mask = flat_mask.reshape(batch_size, self.n_objects)  # [batch_size, n_objects]
        
        # Embed objects and missing ET
        obj_embedded = self.object_embedding(objects)  # [batch_size, n_objects, hidden_dim]
        miss_embedded = self.miss_embedding(miss_features)  # [batch_size, hidden_dim]
        
        # Apply Lorentz-equivariant layers with masking
        x = obj_embedded
        for layer in self.eq_layers:
            x = layer(x, mask.unsqueeze(-1))
        
        # Mask out padding objects
        x = x * mask.unsqueeze(-1)
        
        # Flatten object embeddings
        x_flat = x.reshape(batch_size, -1)  # [batch_size, n_objects * hidden_dim]
        
        # Concatenate with missing ET features
        x_with_miss = torch.cat([x_flat, miss_embedded], dim=1)  # [batch_size, (n_objects+1) * hidden_dim]
        
        # Final classification
        logits = self.mlp(x_with_miss).squeeze(-1)
        
        return logits

def make_model(input_dim: int):
    # For 18 objects with 7 features each + 18 mask values + 2 missing ET features
    n_objects = 18
    hidden_dim = 128
    return LorentzNetwork(input_dim, hidden_dim, n_objects)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Binary cross-entropy loss
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5)
    
    # Initialize tracking variables
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Convert target to float for BCE loss
            target_float = target.float()
            
            loss = criterion(output, target_float)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # Calculate accuracy
            predicted = (torch.sigmoid(output) > 0.5).float()
            total_train += target.size(0)
            correct_train += (predicted == target_float).sum().item()
        
        # Calculate average training metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = correct_train / total_train
        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                target_float = target.float()
                
                loss = criterion(output, target_float)
                epoch_val_loss += loss.item()
                
                # Store outputs and targets for AUC calculation
                all_outputs.append(output.cpu())
                all_targets.append(target.cpu())
                
                # Calculate accuracy
                predicted = (torch.sigmoid(output) > 0.5).float()
                total_val += target.size(0)
                correct_val += (predicted == target_float).sum().item()
        
        # Calculate average validation metrics
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_acc = correct_val / total_val
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)
        
        # Update learning rate based on validation performance
        # Convert outputs and targets to numpy arrays
        all_outputs = torch.cat(all_outputs).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        # Use validation accuracy for scheduler
        scheduler.step(avg_val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Train Acc: {avg_train_acc:.4f}, Val Acc: {avg_val_acc:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
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

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

