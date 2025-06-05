
import os, sys, pickle, torch, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
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

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        if isinstance(self.x, (tuple, list)):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])      

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = PairDataset(X_train, Y_train)
    val_ds   = PairDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))
                        
# ----------------  START OF LLM BLOCK  ----------------

# 0. ---------- IMPORTS ----------
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # Constants for data structure
        self.n_objects_max = 18  # Maximum number of objects in dataset
        self.features_per_object = 5  # Features per object (ID, E, pT, eta, phi)

        # Statistics for normalization
        self.means = {}
        self.stds = {}

        # Track object types found in the data
        self.object_types = set()

    def fit(self, X, y=None):
        # Extract valid energy, pT, and eta values for normalization
        energies = []
        pts = []
        etas = []

        # Missing ET statistics
        et_miss = X[:, 0][X[:, 0] != 0]

        # Collect object types present in the data
        for i in range(self.n_objects_max):
            base_idx = 2 + i * self.features_per_object
            if base_idx < X.shape[1]:
                # Get unique object IDs
                obj_ids = X[:, base_idx].unique()
                for obj_id in obj_ids:
                    if obj_id != 0:  # Skip padding (zero values)
                        self.object_types.add(int(obj_id.item()))

        # Extract values from each object for normalization stats
        for i in range(self.n_objects_max):
            base_idx = 2 + i * self.features_per_object
            if base_idx + 4 < X.shape[1]:
                # Object exists if ID is non-zero
                valid_mask = X[:, base_idx] != 0

                energy_idx = base_idx + 1
                pt_idx = base_idx + 2
                eta_idx = base_idx + 3

                energies.append(X[valid_mask, energy_idx])
                pts.append(X[valid_mask, pt_idx])
                etas.append(X[valid_mask, eta_idx])

        # Compute statistics for normalization
        if energies:
            energy_tensor = torch.cat(energies)
            self.means['energy'] = torch.mean(energy_tensor)
            self.stds['energy'] = torch.std(energy_tensor) + 1e-6  # Add epsilon to prevent division by zero

        if pts:
            pt_tensor = torch.cat(pts)
            self.means['pt'] = torch.mean(pt_tensor)
            self.stds['pt'] = torch.std(pt_tensor) + 1e-6

        if etas:
            eta_tensor = torch.cat(etas)
            self.means['eta'] = torch.mean(eta_tensor)
            self.stds['eta'] = torch.std(eta_tensor) + 1e-6

        if len(et_miss) > 0:
            self.means['et_miss'] = torch.mean(et_miss)
            self.stds['et_miss'] = torch.std(et_miss) + 1e-6

        return self

    def transform(self, X):
        batch_size = X.shape[0]

        # 1. Count objects by type
        object_counts = self._count_objects_by_type(X)

        # 2. Extract global event features
        global_features = self._extract_global_features(X)

        # 3. Extract object-specific features
        object_features = self._extract_object_features(X)

        # 4. Extract pair-wise features (correlations between objects)
        pair_features = self._extract_pair_features(X)

        # Combine all features
        all_features = torch.cat([
            object_counts, 
            global_features, 
            object_features, 
            pair_features
        ], dim=1)

        return all_features

    def _count_objects_by_type(self, X):
        """Count objects of each type in each event"""
        batch_size = X.shape[0]

        # Initialize counts tensor
        n_types = max(self.object_types) + 1 if self.object_types else 10
        counts = torch.zeros((batch_size, n_types), device=X.device)

        # Count objects by type
        for i in range(self.n_objects_max):
            base_idx = 2 + i * self.features_per_object
            if base_idx < X.shape[1]:
                obj_ids = X[:, base_idx].long()
                for b in range(batch_size):
                    obj_id = obj_ids[b].item()
                    if obj_id > 0 and obj_id < n_types:
                        counts[b, obj_id] += 1

        return counts

    def _extract_global_features(self, X):
        """Extract global event features"""
        batch_size = X.shape[0]

        # Initialize features tensor
        features = torch.zeros((batch_size, 10), device=X.device)

        # Missing ET and phi
        features[:, 0] = X[:, 0] / self.stds.get('et_miss', 1.0)  # Normalized Missing ET
        features[:, 1] = X[:, 1]  # Phi (angle) doesn't need normalization

        # Initialize variables for global sums
        total_pt = torch.zeros(batch_size, device=X.device)
        total_energy = torch.zeros(batch_size, device=X.device)
        total_px = torch.zeros(batch_size, device=X.device)
        total_py = torch.zeros(batch_size, device=X.device)
        total_pz = torch.zeros(batch_size, device=X.device)
        ht = torch.zeros(batch_size, device=X.device)  # Scalar sum of pT (important HEP variable)
        num_objects = torch.zeros(batch_size, device=X.device)  # Count of objects

        # Calculate global sums
        for i in range(self.n_objects_max):
            base_idx = 2 + i * self.features_per_object
            if base_idx + 4 < X.shape[1]:
                # Object exists if ID is non-zero
                valid_mask = X[:, base_idx] != 0
                num_objects += valid_mask.float()

                energy_idx = base_idx + 1
                pt_idx = base_idx + 2
                eta_idx = base_idx + 3
                phi_idx = base_idx + 4

                # Process valid objects
                for b in range(batch_size):
                    if valid_mask[b]:
                        # Extract values
                        energy = X[b, energy_idx]
                        pt = X[b, pt_idx]
                        eta = X[b, eta_idx]
                        phi = X[b, phi_idx]

                        # Calculate momentum components
                        px = pt * torch.cos(phi)
                        py = pt * torch.sin(phi)
                        pz = pt * torch.sinh(eta)  # pz = pT * sinh(eta)

                        # Add to totals
                        total_energy[b] += energy
                        total_pt[b] += pt
                        total_px[b] += px
                        total_py[b] += py
                        total_pz[b] += pz
                        ht[b] += pt  # HT is scalar sum of pT

        # Store global features
        features[:, 2] = total_pt / self.stds.get('pt', 1.0)
        features[:, 3] = total_energy / self.stds.get('energy', 1.0)
        features[:, 4] = ht / self.stds.get('pt', 1.0)
        features[:, 5] = num_objects  # Number of objects is important

        # Missing ET ratio to HT (sensitive to neutrinos from top decays)
        valid_mask = ht != 0
        features[valid_mask, 6] = X[valid_mask, 0] / ht[valid_mask]

        # Invariant mass of the event
        p_squared = total_px**2 + total_py**2 + total_pz**2
        mass_squared = total_energy**2 - p_squared
        # Handle numerical issues
        mass_squared = torch.clamp(mass_squared, min=0)
        invariant_mass = torch.sqrt(mass_squared)
        features[:, 7] = invariant_mass / self.stds.get('energy', 1.0)

        # Transverse mass with missing ET
        phi_met = X[:, 1]
        phi_visible = torch.atan2(total_py, total_px)
        # Calculate phi difference considering the circular nature of phi
        phi_diff = torch.abs(phi_visible - phi_met)
        phi_diff = torch.min(phi_diff, 2 * torch.pi - phi_diff)

        mt_squared = 2 * total_pt * X[:, 0] * (1 - torch.cos(phi_diff))
        mt_squared = torch.clamp(mt_squared, min=0)
        transverse_mass = torch.sqrt(mt_squared)
        features[:, 8] = transverse_mass / self.stds.get('pt', 1.0)

        # Centrality: ratio of total pt to total energy
        valid_mask = total_energy != 0
        features[valid_mask, 9] = total_pt[valid_mask] / total_energy[valid_mask]

        return features

    def _extract_object_features(self, X):
        """Extract features for individual objects"""
        batch_size = X.shape[0]

        # We'll extract statistics for each object type
        n_types = max(self.object_types) + 1 if self.object_types else 10

        # For each type: mean pT, max pT, mean energy, max energy, mean eta
        features_per_type = 5
        features = torch.zeros((batch_size, n_types * features_per_type), device=X.device)

        # Collect values by type
        for type_id in range(1, n_types):
            # Arrays to collect values for this type
            type_pts = [[] for _ in range(batch_size)]
            type_energies = [[] for _ in range(batch_size)]
            type_etas = [[] for _ in range(batch_size)]

            # Scan all objects
            for i in range(self.n_objects_max):
                base_idx = 2 + i * self.features_per_object
                if base_idx + 4 < X.shape[1]:
                    for b in range(batch_size):
                        obj_id = X[b, base_idx].item()
                        if obj_id == type_id:
                            energy = X[b, base_idx + 1].item()
                            pt = X[b, base_idx + 2].item()
                            eta = X[b, base_idx + 3].item()
                            type_pts[b].append(pt)
                            type_energies[b].append(energy)
                            type_etas[b].append(eta)

            # Compute statistics for each batch item
            for b in range(batch_size):
                if type_pts[b]:
                    # Convert to tensor for easier operations
                    pts = torch.tensor(type_pts[b], device=X.device)
                    energies = torch.tensor(type_energies[b], device=X.device)
                    etas = torch.tensor(type_etas[b], device=X.device)

                    # Calculate statistics
                    mean_pt = torch.mean(pts)
                    max_pt = torch.max(pts)
                    mean_energy = torch.mean(energies)
                    max_energy = torch.max(energies)
                    mean_eta = torch.mean(torch.abs(etas))  # Use absolute eta for forward/backward symmetry

                    # Normalize
                    mean_pt = mean_pt / self.stds.get('pt', 1.0)
                    max_pt = max_pt / self.stds.get('pt', 1.0)
                    mean_energy = mean_energy / self.stds.get('energy', 1.0)
                    max_energy = max_energy / self.stds.get('energy', 1.0)

                    # Store in features
                    feature_start = (type_id - 1) * features_per_type
                    features[b, feature_start] = mean_pt
                    features[b, feature_start + 1] = max_pt
                    features[b, feature_start + 2] = mean_energy
                    features[b, feature_start + 3] = max_energy
                    features[b, feature_start + 4] = mean_eta

        return features

    def _extract_pair_features(self, X):
        """Extract features from pairs of objects (important for resonance identification)"""
        batch_size = X.shape[0]

        # Features: min/max delta R, min/max invariant mass, min/max delta phi
        features = torch.zeros((batch_size, 6), device=X.device)

        # Initialize with extreme values
        min_dr = torch.full((batch_size,), float('inf'), device=X.device)
        max_dr = torch.zeros(batch_size, device=X.device)
        min_mass = torch.full((batch_size,), float('inf'), device=X.device)
        max_mass = torch.zeros(batch_size, device=X.device)
        min_dphi = torch.full((batch_size,), float('inf'), device=X.device)
        max_dphi = torch.zeros(batch_size, device=X.device)

        # Loop through all pairs of objects
        for i in range(self.n_objects_max):
            base_i = 2 + i * self.features_per_object
            if base_i + 4 >= X.shape[1]:
                continue

            for j in range(i + 1, self.n_objects_max):
                base_j = 2 + j * self.features_per_object
                if base_j + 4 >= X.shape[1]:
                    continue

                # Process each batch item
                for b in range(batch_size):
                    # Check if both objects exist
                    if X[b, base_i] != 0 and X[b, base_j] != 0:
                        # Extract kinematic values
                        e1 = X[b, base_i + 1]
                        pt1 = X[b, base_i + 2]
                        eta1 = X[b, base_i + 3]
                        phi1 = X[b, base_i + 4]

                        e2 = X[b, base_j + 1]
                        pt2 = X[b, base_j + 2]
                        eta2 = X[b, base_j + 3]
                        phi2 = X[b, base_j + 4]

                        # Calculate delta R = sqrt(delta_eta^2 + delta_phi^2)
                        deta = eta1 - eta2
                        dphi = torch.abs(phi1 - phi2)
                        # Handle phi wrap-around
                        dphi = torch.min(dphi, 2 * torch.pi - dphi)
                        dr = torch.sqrt(deta**2 + dphi**2)

                        # Update min/max delta R
                        min_dr[b] = min(min_dr[b], dr)
                        max_dr[b] = max(max_dr[b], dr)

                        # Update min/max delta phi
                        min_dphi[b] = min(min_dphi[b], dphi)
                        max_dphi[b] = max(max_dphi[b], dphi)

                        # Calculate 4-momentum components
                        px1 = pt1 * torch.cos(phi1)
                        py1 = pt1 * torch.sin(phi1)
                        pz1 = pt1 * torch.sinh(eta1)

                        px2 = pt2 * torch.cos(phi2)
                        py2 = pt2 * torch.sin(phi2)
                        pz2 = pt2 * torch.sinh(eta2)

                        # Calculate invariant mass
                        # m^2 = (E1+E2)^2 - (px1+px2)^2 - (py1+py2)^2 - (pz1+pz2)^2
                        m_squared = (e1 + e2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2

                        # Handle numerical issues
                        if m_squared > 0:
                            mass = torch.sqrt(m_squared)

                            # Update min/max mass
                            min_mass[b] = min(min_mass[b], mass)
                            max_mass[b] = max(max_mass[b], mass)

        # Handle cases where no valid pairs were found
        min_dr[min_dr == float('inf')] = 0
        min_mass[min_mass == float('inf')] = 0
        min_dphi[min_dphi == float('inf')] = 0

        # Store features
        features[:, 0] = min_dr
        features[:, 1] = max_dr
        features[:, 2] = min_mass / self.stds.get('energy', 1.0)
        features[:, 3] = max_mass / self.stds.get('energy', 1.0)
        features[:, 4] = min_dphi
        features[:, 5] = max_dphi

        return features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
def make_model(input_shape, *, use_mask=False):
    # Get input dimension (features)
    input_dim = input_shape[0]

    # Create a model with residual connections for better gradient flow
    class ResidualBlock(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear1 = nn.Linear(in_features, out_features)
            self.bn1 = nn.BatchNorm1d(out_features)
            self.linear2 = nn.Linear(out_features, out_features)
            self.bn2 = nn.BatchNorm1d(out_features)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

            # Projection shortcut if dimensions don't match
            self.projection = None
            if in_features != out_features:
                self.projection = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features)
                )

        def forward(self, x):
            identity = x

            # First block
            out = self.linear1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.dropout(out)

            # Second block
            out = self.linear2(out)
            out = self.bn2(out)

            # Shortcut connection
            if self.projection is not None:
                identity = self.projection(x)

            # Add residual connection
            out += identity
            out = self.relu(out)

            return out

    class PhysicsClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()

            # Initial feature extraction
            self.initial = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

            # Residual blocks for deeper representation learning
            self.res1 = ResidualBlock(256, 256)
            self.res2 = ResidualBlock(256, 128)
            self.res3 = ResidualBlock(128, 64)

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            # If we get a tuple (data, mask), only use the data
            if isinstance(x, tuple):
                x, _ = x

            x = self.initial(x)
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            output = self.classifier(x)
            return output.squeeze()

    # Instantiate the model
    model = PhysicsClassifier(input_dim)

    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 100

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Use focal loss for binary classification (better handles class imbalance)
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            # Binary cross entropy
            bce_loss = nn.BCELoss(reduction='none')(inputs, targets)

            # Apply focal scaling
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

            return focal_loss.mean()

    criterion = FocalLoss()

    # Use AdamW optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Learning rate scheduler with warmup and cosine annealing
    def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs  # Linear warmup
            else:
                # Cosine decay after warmup
                return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = get_lr_scheduler(optimizer, warmup_epochs=5, total_epochs=epochs)

    # Initialize tracking variables
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_auc = 0
    best_model_state = None
    patience = 15  # Early stopping patience
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            if isinstance(data, tuple):
                data = (data[0].to(device), data[1].to(device))
            else:
                data = data.to(device)
            target = target.float().to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)

            # Calculate loss
            loss = criterion(output, target)

            # Backward pass and optimize
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Track statistics
            epoch_train_loss += loss.item()
            predicted = (output > 0.5).float()
            epoch_train_correct += (predicted == target).sum().item()
            epoch_train_total += target.size(0)

        # Calculate epoch statistics
        epoch_train_loss /= len(train_loader)
        epoch_train_acc = epoch_train_correct / epoch_train_total

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        epoch_val_correct = 0
        epoch_val_total = 0

        val_outputs = []
        val_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                if isinstance(data, tuple):
                    data = (data[0].to(device), data[1].to(device))
                else:
                    data = data.to(device)
                target = target.float().to(device)

                # Forward pass
                output = model(data)

                # Calculate loss
                loss = criterion(output, target)

                # Track statistics
                epoch_val_loss += loss.item()
                predicted = (output > 0.5).float()
                epoch_val_correct += (predicted == target).sum().item()
                epoch_val_total += target.size(0)

                # Store outputs and targets for AUC calculation
                val_outputs.extend(output.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        # Calculate epoch statistics
        epoch_val_loss /= len(val_loader)
        epoch_val_acc = epoch_val_correct / epoch_val_total

        # Calculate AUC (our target metric)
        val_auc = roc_auc_score(val_targets, val_outputs)

        # Update learning rate
        scheduler.step()

        # Store statistics
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        train_acc.append(epoch_train_acc)
        val_acc.append(epoch_val_acc)

        # Early stopping based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break  # Stop training if no improvement for 'patience' epochs

    # Load best model
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
    pre = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train) # may be Tensor or Tuple
    X_val   = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    if isinstance(X_train, torch.Tensor):               # single-tensor case
        temp_ref    = X_train
        input_shape = temp_ref.shape[1:]                # e.g. (F,)
        use_mask    = False
    else:                                               # tuple => (data, mask)
        temp_ref    = X_train
        input_shape = temp_ref[0].shape[1:]             # e.g. (L, F)
        use_mask    = True                              
    model = make_model(input_shape, use_mask=use_mask)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy_data = torch.zeros(8, *input_shape, dtype=torch.float32)
        if use_mask:
            toy_mask = torch.zeros(8, input_shape[0], dtype=torch.bool)
            toy_batch = (toy_data, toy_mask)
        else:
            toy_batch = toy_data

        toy_transformed = pre.transform(toy_batch)
        try:
            _ = trained_model(*toy_transformed) if isinstance(toy_transformed, (tuple, list)) \
                else trained_model(toy_transformed)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 6. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
            "train_loss": tr_loss   if tr_loss else None,
            "val_loss":   va_loss   if va_loss else None,
            "train_acc":  tr_acc    if tr_acc else None,
            "val_acc":    va_acc    if va_acc else None,
        }
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

