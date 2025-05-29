
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
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.num_objects = 18  # Maximum number of objects in an event
        self.obj_feature_size = 5  # Size of each object's feature vector
        self.leading_objects = 6  # Number of leading objects to focus on
        self.global_stats = {}

    def fit(self, X, y=None):
        # Extract basic statistics for scaling
        # First two features: missing-ET magnitude and azimuth
        self.scalers['missing_et'] = StandardScaler()
        missing_et_features = X[:, 0:2]
        self.scalers['missing_et'].fit(missing_et_features)
        
        # Extract and organize object features
        object_data = defaultdict(list)
        
        # Process each object in the dataset
        for i in range(self.num_objects):
            start_idx = 2 + i * self.obj_feature_size
            end_idx = start_idx + self.obj_feature_size
            
            if end_idx <= X.shape[1]:
                # Get object type and features
                obj_type = X[:, start_idx].cpu().numpy()
                obj_features = X[:, start_idx+1:end_idx].cpu().numpy()
                
                # Find valid objects (non-zero)
                valid_mask = obj_type != 0
                
                if np.any(valid_mask):
                    # Store valid objects by type
                    unique_types = np.unique(obj_type[valid_mask])
                    for t in unique_types:
                        if t != 0:  # Skip padding
                            type_mask = obj_type == t
                            object_data[t].append(obj_features[type_mask])
        
        # Fit scalers for each object type
        for obj_type, features_list in object_data.items():
            if features_list:
                combined_features = np.vstack(features_list)
                self.scalers[f'object_{int(obj_type)}'] = StandardScaler()
                self.scalers[f'object_{int(obj_type)}'].fit(combined_features)
        
        # Calculate global statistics
        self.global_stats['obj_counts'] = self._count_objects(X)
        
        return self

    def _count_objects(self, X):
        # Count objects per event
        counts = {}
        for i in range(self.num_objects):
            start_idx = 2 + i * self.obj_feature_size
            if start_idx < X.shape[1]:
                obj_type = X[:, start_idx].cpu().numpy()
                valid = obj_type != 0
                for t in np.unique(obj_type[valid]):
                    if t != 0:
                        if t not in counts:
                            counts[t] = 0
                        counts[t] += np.sum(obj_type == t)
        return counts

    def transform(self, X):
        # Create a list to hold all our features
        all_features = []
        
        # 1. Global event features - scaled missing ET
        missing_et = X[:, 0:2].clone()
        missing_et_scaled = torch.tensor(
            self.scalers['missing_et'].transform(missing_et.cpu().numpy()),
            dtype=torch.float32
        )
        all_features.append(missing_et_scaled)  # shape: (batch_size, 2)
        
        # 2. Process objects by type
        # For each object type, collect the top N objects by pT
        objects_by_type = {}
        
        # Track all object types in the event
        batch_size = X.shape[0]
        object_presence = torch.zeros((batch_size, 10), dtype=torch.float32)  # Assuming up to 10 object types
        
        for i in range(self.num_objects):
            start_idx = 2 + i * self.obj_feature_size
            end_idx = start_idx + self.obj_feature_size
            
            if end_idx <= X.shape[1]:
                obj_type = X[:, start_idx].long()
                obj_features = X[:, start_idx+1:end_idx]
                
                # Get valid objects (non-zero type)
                for t in range(1, 11):  # assuming object types from 1 to 10
                    type_mask = obj_type == t
                    if torch.any(type_mask):
                        # Mark this object type as present
                        object_presence[type_mask.nonzero().squeeze(1), t-1] = 1.0
                        
                        # Store object features by type for later processing
                        if t not in objects_by_type:
                            objects_by_type[t] = []
                        
                        # Get batch indices where this object type occurs
                        batch_indices = type_mask.nonzero().squeeze(1)
                        
                        for idx in batch_indices:
                            if len(objects_by_type[t]) <= idx:
                                # Initialize empty list for this batch item
                                while len(objects_by_type[t]) <= idx:
                                    objects_by_type[t].append([])
                            
                            # Add object features for this batch item
                            objects_by_type[t][idx].append(obj_features[idx].tolist())
        
        # Add object presence feature
        all_features.append(object_presence)  # shape: (batch_size, num_object_types)
        
        # 3. Process top N objects by pT for each type
        for obj_type, batches in objects_by_type.items():
            # For each batch item
            type_features = []
            
            for idx, obj_list in enumerate(batches):
                if obj_list:  # If we have objects of this type in this event
                    # Convert to tensor for easier handling
                    objects = torch.tensor(obj_list)
                    
                    # Sort by pT (index 1 in the features)
                    sorted_indices = torch.argsort(objects[:, 1], descending=True)
                    sorted_objects = objects[sorted_indices]
                    
                    # Take top N objects
                    top_n = min(self.leading_objects, len(sorted_objects))
                    top_objects = sorted_objects[:top_n]
                    
                    # Pad if needed
                    if top_n < self.leading_objects:
                        padding = torch.zeros((self.leading_objects - top_n, 4))
                        top_objects = torch.cat([top_objects, padding], dim=0)
                    
                    # Scale using the appropriate scaler
                    if f'object_{obj_type}' in self.scalers:
                        scaled_objs = self.scalers[f'object_{obj_type}'].transform(top_objects.cpu().numpy())
                        type_features.append(torch.tensor(scaled_objs.flatten()))
                    else:
                        type_features.append(top_objects.flatten())
                else:
                    # No objects of this type, add zeros
                    type_features.append(torch.zeros(self.leading_objects * 4))
            
            if type_features:
                type_tensor = torch.stack(type_features)  # shape: (batch_size, leading_objects * 4)
                all_features.append(type_tensor)
        
        # 4. Calculate additional derived features
        
        # Count objects of each type per event
        object_counts = torch.zeros((batch_size, 10), dtype=torch.float32)  # Assuming up to 10 object types
        
        for i in range(self.num_objects):
            start_idx = 2 + i * self.obj_feature_size
            if start_idx < X.shape[1]:
                obj_type = X[:, start_idx].long()
                for t in range(1, 11):  # assuming object types from 1 to 10
                    object_counts[:, t-1] += (obj_type == t).float()
        
        all_features.append(object_counts)  # shape: (batch_size, num_object_types)
        
        # Combine all features
        combined = torch.cat(all_features, dim=1)  # shape: (batch_size, total_features)
        
        return combined

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
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out

class PhysicsClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=3, dropout_rate=0.3):
        super(PhysicsClassifier, self).__init__()
        
        # Initial projection layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # Output layers with decreasing dimensions
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        for block in self.res_blocks:
            x = block(x)
            x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x).squeeze(-1)
        
        return x

def make_model(input_dim: int):
    model = PhysicsClassifier(input_dim)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Binary cross-entropy loss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler to reduce LR as training progresses
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, threshold=0.001
    )
    
    # Track metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            pred = torch.sigmoid(outputs) >= 0.5
            correct += (pred == targets.bool()).sum().item()
            total += targets.size(0)
        
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        # Validation step
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets_list = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track loss and accuracy
                val_running_loss += loss.item() * inputs.size(0)
                pred = torch.sigmoid(outputs) >= 0.5
                val_correct += (pred == targets.bool()).sum().item()
                val_total += targets.size(0)
                
                # Store predictions and targets for AUC calculation
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())
        
        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        # Calculate validation AUC using sklearn
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_targets_list, val_preds)
        
        # Update scheduler
        scheduler.step(val_auc)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, '
              f'Val AUC: {val_auc:.4f}')
    
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
    torch.save(trained_model.state_dict(), f"{base}_state.pt")
    with open(f"{base}_model.pkl", "wb") as f: pickle.dump(trained_model, f)
    with open(f"{base}_preproc.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

