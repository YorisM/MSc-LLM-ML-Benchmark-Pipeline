
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
import math
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.num_objects = 18
        self.features_per_object = 5
        self.static_features = 2  # E_T_miss and phi_Et_miss
        self.valid_object_mask = None
        self.mean_values = None
        self.std_values = None
        
    def fit(self, X, y=None):
        # Reshape the data to extract object information
        batch_size = X.shape[0]
        
        # First, extract the missing ET features
        et_miss_features = X[:, :self.static_features].clone()
        
        # Extract object features
        object_data = []
        
        # For each object position
        for obj_idx in range(self.num_objects):
            start_idx = self.static_features + obj_idx * self.features_per_object
            end_idx = start_idx + self.features_per_object
            
            # Extract this object's data
            obj_data = X[:, start_idx:end_idx].clone()
            
            # Check if object exists (non-zero)
            # Use the first value (object identifier) to determine if object exists
            valid_mask = obj_data[:, 0] != 0
            
            object_data.append(obj_data)
        
        # Combine all valid objects into a large array for standardization
        all_valid_objects = torch.cat(object_data, dim=0)
        
        # Calculate valid object mask based on non-zero object IDs
        self.valid_object_mask = []
        for obj_idx in range(self.num_objects):
            start_idx = self.static_features + obj_idx * self.features_per_object
            obj_id_idx = start_idx
            self.valid_object_mask.append((X[:, obj_id_idx] != 0).float())
        
        # Store the original shape to reconstruct later
        kinematic_features = torch.cat([obj[:, 1:] for obj in object_data], dim=0)
        
        # Standardize kinematic features (E, pT, eta, phi)
        self.mean_values = torch.mean(kinematic_features, dim=0)
        self.std_values = torch.std(kinematic_features, dim=0)
        
        # Also standardize ET_miss
        self.et_miss_mean = torch.mean(et_miss_features[:, 0])
        self.et_miss_std = torch.std(et_miss_features[:, 0])
        
        return self

    def transform(self, X):
        batch_size = X.shape[0]
        
        # Create list to store our engineered features
        all_features = []
        
        # 1. Extract and standardize missing ET
        et_miss = X[:, 0].clone()
        et_miss_standardized = (et_miss - self.et_miss_mean) / self.et_miss_std
        all_features.append(et_miss_standardized.unsqueeze(1))  # [batch_size, 1]
        
        # 2. Extract phi of missing ET - no need to standardize angles
        phi_et_miss = X[:, 1].clone().unsqueeze(1)  # [batch_size, 1]
        all_features.append(phi_et_miss)  # [batch_size, 1]
        
        # Extract object features and calculate global event features
        total_pt = torch.zeros(batch_size, dtype=torch.float32)
        total_energy = torch.zeros(batch_size, dtype=torch.float32)
        jet_count = torch.zeros(batch_size, dtype=torch.float32)
        lepton_count = torch.zeros(batch_size, dtype=torch.float32)
        b_jet_count = torch.zeros(batch_size, dtype=torch.float32)
        leading_jet_pt = torch.zeros(batch_size, dtype=torch.float32)
        subleading_jet_pt = torch.zeros(batch_size, dtype=torch.float32)
        
        # Arrays to store all jet and lepton features for later sorting
        all_jets = []
        all_leptons = []
        
        # Process each object
        for obj_idx in range(self.num_objects):
            start_idx = self.static_features + obj_idx * self.features_per_object
            end_idx = start_idx + self.features_per_object
            
            # Extract object data
            obj_data = X[:, start_idx:end_idx].clone()
            
            # Check if object exists using the object ID
            obj_id = obj_data[:, 0]
            obj_mask = (obj_id != 0).float().unsqueeze(1)  # [batch_size, 1]
            
            # Extract kinematic properties
            obj_E = obj_data[:, 1]    # Energy
            obj_pt = obj_data[:, 2]   # Transverse momentum
            obj_eta = obj_data[:, 3]  # Pseudo-rapidity
            obj_phi = obj_data[:, 4]  # Azimuthal angle
            
            # Standardize kinematic features
            obj_E_std = (obj_E - self.mean_values[0]) / self.std_values[0]
            obj_pt_std = (obj_pt - self.mean_values[1]) / self.std_values[1]
            # For eta and phi (angular variables), we keep them as they are
            
            # Include standardized features for each object
            all_features.append(obj_E_std.unsqueeze(1) * obj_mask)  # [batch_size, 1]
            all_features.append(obj_pt_std.unsqueeze(1) * obj_mask)  # [batch_size, 1]
            all_features.append(obj_eta.unsqueeze(1) * obj_mask)  # [batch_size, 1]
            all_features.append(obj_phi.unsqueeze(1) * obj_mask)  # [batch_size, 1]
            
            # Count object types and accumulate properties
            # Object IDs: Jets (1-4), Leptons (5-7)
            is_jet = ((obj_id >= 1) & (obj_id <= 4)).float()
            is_lepton = ((obj_id >= 5) & (obj_id <= 7)).float()
            is_b_jet = (obj_id == 4).float()  # assuming ID 4 corresponds to b-jets
            
            # Update counts
            jet_count += is_jet
            lepton_count += is_lepton
            b_jet_count += is_b_jet
            
            # Update totals for valid objects
            total_energy += obj_E * (is_jet + is_lepton)
            total_pt += obj_pt * (is_jet + is_lepton)
            
            # Store jet and lepton data for sorting
            if obj_idx == 0:
                # Initialize tensors
                all_jets = torch.zeros((batch_size, self.num_objects), dtype=torch.float32)
                all_leptons = torch.zeros((batch_size, self.num_objects), dtype=torch.float32)
            
            # Store PT values for later determination of leading objects
            all_jets[:, obj_idx] = obj_pt * is_jet
            all_leptons[:, obj_idx] = obj_pt * is_lepton
            
        # Find leading and subleading jet PT
        sorted_jets, _ = torch.sort(all_jets, dim=1, descending=True)
        leading_jet_pt = sorted_jets[:, 0]
        subleading_jet_pt = sorted_jets[:, 1]
        
        # Find leading and subleading lepton PT
        sorted_leptons, _ = torch.sort(all_leptons, dim=1, descending=True)
        leading_lepton_pt = sorted_leptons[:, 0]
        subleading_lepton_pt = sorted_leptons[:, 1]
        
        # Add global features
        all_features.append(total_energy.unsqueeze(1))  # [batch_size, 1]
        all_features.append(total_pt.unsqueeze(1))  # [batch_size, 1]
        all_features.append(jet_count.unsqueeze(1))  # [batch_size, 1]
        all_features.append(lepton_count.unsqueeze(1))  # [batch_size, 1]
        all_features.append(b_jet_count.unsqueeze(1))  # [batch_size, 1]
        all_features.append(leading_jet_pt.unsqueeze(1))  # [batch_size, 1]
        all_features.append(subleading_jet_pt.unsqueeze(1))  # [batch_size, 1]
        all_features.append(leading_lepton_pt.unsqueeze(1))  # [batch_size, 1]
        all_features.append(subleading_lepton_pt.unsqueeze(1))  # [batch_size, 1]
        
        # Combine all features
        result = torch.cat(all_features, dim=1)  # [batch_size, num_features]
        
        return result

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
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(in_features, hidden_dim)
        self.key = nn.Linear(in_features, hidden_dim)
        self.value = nn.Linear(in_features, hidden_dim)
        self.scale = math.sqrt(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, in_features)
        
    def forward(self, x):
        # x shape: [batch_size, features]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for self-attention
        q = q.unsqueeze(1)  # [batch, 1, hidden_dim]
        k = k.unsqueeze(1)  # [batch, 1, hidden_dim]
        v = v.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch, 1, 1]
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [batch, 1, hidden_dim]
        
        # Project back to original dimension
        out = self.out_proj(attn_output.squeeze(1))  # [batch, features]
        return out + x  # Residual connection

class EventClassifier(nn.Module):
    def __init__(self, input_dim):
        super(EventClassifier, self).__init__()
        
        # Initial dimension reduction
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            AttentionBlock(256),
            nn.Dropout(0.3)
        )
        
        # Further dimension reduction
        self.hidden_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output layer
        self.output_layer = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x.squeeze(-1)  # Return logits for BCE with logits loss

def make_model(input_dim):
    model = EventClassifier(input_dim)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 15

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5)
    
    # Tracking metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_auc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        train_pred_list = []
        train_labels_list = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Convert labels to float for BCE loss
            labels_float = labels.float()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_float)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Record metrics
            epoch_train_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predictions == labels_float).sum().item()
            total_train += labels.size(0)
            
            # Save predictions and labels for AUC calculation
            train_pred_list.append(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels_list.append(labels.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_train_loss /= total_train
        epoch_train_acc = correct_train / total_train
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_pred_list = []
        val_labels_list = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels_float = labels.float()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels_float)
                
                epoch_val_loss += loss.item() * inputs.size(0)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (predictions == labels_float).sum().item()
                total_val += labels.size(0)
                
                val_pred_list.append(torch.sigmoid(outputs).cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())
        
        # Calculate validation metrics
        epoch_val_loss /= total_val
        epoch_val_acc = correct_val / total_val
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        # Calculate AUC for validation
        val_preds = np.concatenate(val_pred_list)
        val_labels = np.concatenate(val_labels_list)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Store best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
        
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, AUC: {val_auc:.4f}')
    
    # Load the best model before returning
    model.load_state_dict(best_state)
    
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

