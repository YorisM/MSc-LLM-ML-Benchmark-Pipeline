
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_means = None
        self.feature_stds = None
        self.mask = None
    
    def extract_features(self, X):
        # Convert to numpy if it's a torch tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()
            
        batch_size = X.shape[0]
        features = []
        
        # First, extract the missing ET features (indices 0 and 1)
        missing_et_mag = X[:, 0].reshape(-1, 1)  # E_T_miss
        missing_et_phi = X[:, 1].reshape(-1, 1)  # phi_Et_miss
        
        # Log transform the missing ET magnitude (which is typically skewed)
        missing_et_mag = np.log1p(np.abs(missing_et_mag))
        
        features.append(missing_et_mag)
        features.append(missing_et_phi)
        
        # Calculate total energy, total pT, and object counts
        total_energy = np.zeros((batch_size, 1))
        total_pt = np.zeros((batch_size, 1))
        
        # Object type counters
        object_counts = np.zeros((batch_size, 11))
        
        # Variables to collect all object properties for later statistics
        all_energies = []
        all_pts = []
        all_etas = []
        all_phis = []
        
        # Process each object (18 objects * 5 features per object)
        for i in range(18):
            obj_start_idx = 2 + i * 5
            obj_type = X[:, obj_start_idx]
            
            # Only process valid objects (non-zero object types)
            valid_mask = obj_type != 0
            
            # Count each object type (1-11)
            for obj_val in range(1, 12):
                object_counts[:, obj_val-1] += (obj_type == obj_val).astype(int)
            
            # Extract object features
            energy = X[:, obj_start_idx + 1]  # E
            pt = X[:, obj_start_idx + 2]      # pT
            eta = X[:, obj_start_idx + 3]     # eta
            phi = X[:, obj_start_idx + 4]     # phi
            
            # Accumulate to totals where valid
            total_energy += np.where(valid_mask.reshape(-1, 1), energy.reshape(-1, 1), 0)
            total_pt += np.where(valid_mask.reshape(-1, 1), pt.reshape(-1, 1), 0)
            
            # Save valid features for statistics
            all_energies.append(energy[valid_mask])
            all_pts.append(pt[valid_mask])
            all_etas.append(eta[valid_mask])
            all_phis.append(phi[valid_mask])
            
            # Create features per object (only for valid objects)
            for j in range(batch_size):
                if obj_type[j] != 0:
                    # Log transform energy and pT (which are typically skewed)
                    features.append(np.log1p(energy[j]).reshape(1, 1))
                    features.append(np.log1p(pt[j]).reshape(1, 1))
                    features.append(eta[j].reshape(1, 1))
                    features.append(phi[j].reshape(1, 1))
                    features.append(obj_type[j].reshape(1, 1))  # Object type as a feature
        
        # Add global event features
        features.append(total_energy)
        features.append(total_pt)
        features.append(np.log1p(total_energy))
        features.append(np.log1p(total_pt))
        
        # Add object counts
        features.append(object_counts)
        
        # Flatten all arrays into columns and concatenate
        return np.hstack([f.reshape(batch_size, -1) if f.size > 0 else np.zeros((batch_size, 0)) for f in features])
    
    def fit(self, X, y=None):
        # Extract engineered features
        features = self.extract_features(X)
        
        # Save the mask of non-constant features
        # (to avoid potential division by zero in StandardScaler)
        self.mask = np.std(features, axis=0) > 1e-10
        
        # Fit scaler on non-constant features
        self.scaler.fit(features[:, self.mask])
        
        return self
    
    def transform(self, X):
        # Extract engineered features
        features = self.extract_features(X)
        
        # Transform non-constant features
        if self.mask is not None and np.any(self.mask):
            features_scaled = features.copy()
            features_scaled[:, self.mask] = self.scaler.transform(features[:, self.mask])
            return torch.FloatTensor(features_scaled)
        
        return torch.FloatTensor(features)

class AttentionPooling(nn.Module):
    def __init__(self, in_features):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, 1)
        )
    
    def forward(self, x):
        # Input: batch_size x num_objects x features
        attention_weights = F.softmax(self.attention(x).squeeze(-1), dim=1)
        # Apply attention weights
        weighted_sum = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
        return weighted_sum

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features)
        )
    
    def forward(self, x):
        return F.relu(x + self.layers(x))

class ParticleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(ParticleClassifier, self).__init__()
        
        # Main network layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(3)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # Initial layer
        x = F.relu(self.norm1(self.input_layer(x)))
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
            
        # Output
        logits = self.output_layers(x).squeeze(-1)
        return logits

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim):
    model = ParticleClassifier(input_dim)
    return model

EPOCHS = 15

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Track metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            epoch_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        train_auc = roc_auc_score(all_targets, all_preds)
        
        train_loss.append(avg_loss)
        train_acc.append(accuracy)
        
        # Validation phase
        model.eval()
        epoch_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track metrics
                epoch_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = epoch_loss / len(val_loader)
        accuracy = correct / total
        val_auc = roc_auc_score(all_targets, all_preds)
        
        val_loss.append(avg_loss)
        val_acc.append(accuracy)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
    
    # Load the best model
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
    torch.jit.script(trained).save(f"{base}_scripted.pt")
    torch.jit.script(pre).save(f"{base}_preproc.pt")
    with open(f"{base}_pre.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

