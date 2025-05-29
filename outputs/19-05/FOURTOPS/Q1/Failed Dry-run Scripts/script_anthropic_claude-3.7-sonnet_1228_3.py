
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
    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train))
    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val))
    return (DataLoader(train, batch_size=batch, shuffle=True),
            DataLoader(val, batch_size=batch))
                        
# ----------------  START OF LLM BLOCK  ----------------
# Imports
import os, sys, json, pickle, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from collections import defaultdict

class MyPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.feature_maps = {}
        self.n_objects = 18  # Maximum number of objects
        self.features_per_object = 5  # Number of features per object
        self.global_features = 2  # E_T_miss and phi_Et_miss
        
        # Statistics to store during fit
        self.obj_counts = None
        self.obj_types = None
        self.mean_e = None
        self.std_e = None
        self.mean_pt = None
        self.std_pt = None
        self.has_fit = False

    def fit(self, X, y=None):
        # Convert to numpy if tensor
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        # Count object types and their frequencies
        obj_counts = defaultdict(int)
        n_samples = X.shape[0]
        
        # Extract object types from data
        for i in range(self.n_objects):
            obj_idx = 2 + i * self.features_per_object
            if obj_idx < X.shape[1]:
                obj_types = X[:, obj_idx]
                unique_types, counts = np.unique(obj_types[obj_types != 0], return_counts=True)
                for obj_type, count in zip(unique_types, counts):
                    obj_counts[obj_type] += count
        
        # Sort object types by frequency (most common first)
        sorted_obj_types = sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)
        self.obj_types = [int(obj_type) for obj_type, _ in sorted_obj_types]
        self.obj_counts = obj_counts
        
        # Create feature extractors and scalers
        # Extract energy and pT for normalization
        energies = []
        pts = []
        for i in range(self.n_objects):
            base_idx = 2 + i * self.features_per_object
            if base_idx + 1 < X.shape[1]:  # Energy is at +1, pT at +2
                e_vals = X[:, base_idx + 1]
                pt_vals = X[:, base_idx + 2]
                valid_mask = (e_vals != 0) & ~np.isnan(e_vals)
                energies.extend(e_vals[valid_mask])
                pts.extend(pt_vals[valid_mask])
        
        # Store statistics for normalization
        self.mean_e = np.mean(energies)
        self.std_e = np.std(energies)
        self.mean_pt = np.mean(pts)
        self.std_pt = np.std(pts)
        
        # Create scalers for global features
        self.scalers['et_miss'] = StandardScaler().fit(X[:, 0].reshape(-1, 1))
        
        # We don't scale angular variables (phi) as they're cyclic
        
        self.has_fit = True
        return self

    def transform(self, X):
        if not self.has_fit:
            raise ValueError("Preprocessor must be fit before transform")
            
        # Convert to numpy if tensor
        is_tensor = isinstance(X, torch.Tensor)
        if is_tensor:
            X = X.cpu().numpy()
        
        # Get dimensions
        n_samples = X.shape[0]
        
        # Extract features
        # 1. Global features (missing ET magnitude and phi)
        et_miss = self.scalers['et_miss'].transform(X[:, 0].reshape(-1, 1)).flatten()
        phi_et_miss = X[:, 1]
        
        # Convert phi to sin/cos components
        sin_phi_et = np.sin(phi_et_miss)
        cos_phi_et = np.cos(phi_et_miss)
        
        # Initialize processed features arrays
        # We'll use structured features by object type
        feature_list = [
            et_miss.reshape(-1, 1),
            sin_phi_et.reshape(-1, 1),
            cos_phi_et.reshape(-1, 1)
        ]
        
        # Prepare dictionaries to hold object features by type
        obj_features_by_type = {obj_type: [] for obj_type in self.obj_types}
        obj_counts_by_type = {obj_type: np.zeros(n_samples) for obj_type in self.obj_types}
        
        # Process each object position
        for i in range(self.n_objects):
            base_idx = 2 + i * self.features_per_object
            if base_idx + 4 >= X.shape[1]:  # Ensure we have all features
                continue
                
            obj_ids = X[:, base_idx]
            energies = X[:, base_idx + 1]
            pts = X[:, base_idx + 2]
            etas = X[:, base_idx + 3]
            phis = X[:, base_idx + 4]
            
            # Process each object type
            for obj_type in self.obj_types:
                # Find samples where this position contains this object type
                mask = (obj_ids == obj_type)
                if not np.any(mask):
                    continue
                    
                # Count objects of this type per sample
                obj_counts_by_type[obj_type] += mask
                
                # Process features for this object type at this position
                for sample_idx in np.where(mask)[0]:
                    # Normalize E and pT
                    norm_e = (energies[sample_idx] - self.mean_e) / self.std_e
                    norm_pt = (pts[sample_idx] - self.mean_pt) / self.std_pt
                    eta = etas[sample_idx]
                    phi = phis[sample_idx]
                    
                    # Convert phi to sin/cos
                    sin_phi = np.sin(phi)
                    cos_phi = np.cos(phi)
                    
                    # Add to object features
                    obj_features_by_type[obj_type].append((sample_idx, norm_e, norm_pt, eta, sin_phi, cos_phi))
        
        # Convert object features to arrays and compute statistics
        for obj_type in self.obj_types:
            if obj_features_by_type[obj_type]:
                # Extract sample indices and features
                features_data = obj_features_by_type[obj_type]
                sample_indices = [f[0] for f in features_data]
                feature_values = np.array([f[1:] for f in features_data])  # E, pT, eta, sin(phi), cos(phi)
                
                # Initialize arrays for sum and mean features
                sum_features = np.zeros((n_samples, 5))  # E, pT, eta, sin(phi), cos(phi)
                
                # Populate sum features
                for idx, (sample_idx, e, pt, eta, sin_phi, cos_phi) in enumerate(features_data):
                    sum_features[sample_idx] += [e, pt, eta, sin_phi, cos_phi]
                
                # Compute mean features (avoiding division by zero)
                mean_features = np.zeros((n_samples, 5))  # E, pT, eta, sin(phi), cos(phi)
                nonzero_mask = obj_counts_by_type[obj_type] > 0
                for j in range(5):
                    mean_features[nonzero_mask, j] = sum_features[nonzero_mask, j] / obj_counts_by_type[obj_type][nonzero_mask]
                
                # Add statistical features to our feature list
                feature_list.append(sum_features)  # Sum of features
                feature_list.append(mean_features)  # Mean of features
                feature_list.append(obj_counts_by_type[obj_type].reshape(-1, 1))  # Count of objects
        
        # Concatenate all features
        features = np.hstack(feature_list)
        
        # Return as tensor if input was tensor
        if is_tensor:
            features = torch.from_numpy(features).float()
        
        return features

def make_preprocessor():
    return MyPreprocessor()

# Define a residual block for our deep network
class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity
        out = self.relu(out)
        return out

# Define the neural network model
class DeepParticleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # Architecture parameters
        self.hidden_dim = 256
        self.n_res_blocks = 3
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.hidden_dim, self.hidden_dim // 2) 
            for _ in range(self.n_res_blocks)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
            
        x = self.output_layers(x)
        return x.squeeze(-1)

def make_model(input_dim):
    model = DeepParticleClassifier(input_dim)
    return model

EPOCHS = 25

def train_model(model, train_loader, val_loader, epochs):
    # Setup training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function for binary classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=False
    )
    
    # Initialize metrics trackers
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch training metrics
        train_epoch_loss = running_loss / len(train_loader.dataset)
        train_epoch_auc = roc_auc_score(train_targets, train_preds)
        train_epoch_acc = accuracy_score(train_targets, np.round(train_preds))
        
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets.float())
                
                # Track metrics
                val_running_loss += loss.item() * inputs.size(0)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch validation metrics
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_auc = roc_auc_score(val_targets, val_preds)
        val_epoch_acc = accuracy_score(val_targets, np.round(val_preds))
        
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_epoch_auc)
        
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
    with open(f"{base}_model.pkl", "wb") as f: pickle.dump(trained, f)
    with open(f"{base}_preproc.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

