
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
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.nn import functional as F
import math
import pickle

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.feature_indices = None
        self.n_objects = 18  # Max number of objects
        self.object_features = 5  # Features per object (obj_id, E, pT, eta, phi)

    def fit(self, X, y=None):
        # Extract features and fit scalers
        # First two features are missing-ET magnitude and azimuth
        self.scalers['ET_miss'] = StandardScaler()
        ET_miss_values = X[:, 0].reshape(-1, 1)
        self.scalers['ET_miss'].fit(ET_miss_values)
        
        # We don't scale azimuthal angles as they're cyclic
        
        # Process each object type separately
        valid_objects = []
        E_values = []
        pT_values = []
        eta_values = []
        
        # Extract object features
        for i in range(self.n_objects):
            idx_start = 2 + i * self.object_features
            obj_ids = X[:, idx_start].cpu().numpy()
            
            # Only consider positions with valid object IDs (non-zero)
            valid_mask = (obj_ids != 0)
            valid_objects.append(valid_mask)
            
            # Energy
            E_vals = X[:, idx_start + 1].reshape(-1, 1).cpu().numpy()
            E_values.append(E_vals[valid_mask])
            
            # Transverse momentum
            pT_vals = X[:, idx_start + 2].reshape(-1, 1).cpu().numpy()
            pT_values.append(pT_vals[valid_mask])
            
            # Pseudorapidity
            eta_vals = X[:, idx_start + 3].reshape(-1, 1).cpu().numpy()
            eta_values.append(eta_vals[valid_mask])
        
        # Fit scalers for each feature type
        if len(np.concatenate(E_values)) > 0:
            self.scalers['E'] = StandardScaler()
            self.scalers['E'].fit(np.concatenate(E_values))
        
        if len(np.concatenate(pT_values)) > 0:
            self.scalers['pT'] = StandardScaler()
            self.scalers['pT'].fit(np.concatenate(pT_values))
        
        if len(np.concatenate(eta_values)) > 0:
            self.scalers['eta'] = StandardScaler()
            self.scalers['eta'].fit(np.concatenate(eta_values))
        
        return self

    def transform(self, X):
        batch_size = X.shape[0]
        device = X.device
        
        # Scale missing ET
        scaled_ET_miss = torch.tensor(
            self.scalers['ET_miss'].transform(X[:, 0].cpu().numpy().reshape(-1, 1)),
            dtype=torch.float32, device=device
        ).squeeze()
        
        # Keep phi values as they are (cyclic coordinate)
        phi_ET_miss = X[:, 1]
        
        # Create sin and cos features for the phi values to handle cyclicity
        sin_phi_ET_miss = torch.sin(phi_ET_miss)
        cos_phi_ET_miss = torch.cos(phi_ET_miss)
        
        # Process each object
        object_features = []
        
        for i in range(self.n_objects):
            idx_start = 2 + i * self.object_features
            
            # Object ID
            obj_id = X[:, idx_start]
            
            # Object mask (1 if object exists, 0 if padding)
            obj_mask = (obj_id != 0).float()
            
            # Energy (scaled)
            E = X[:, idx_start + 1]
            E_scaled = torch.tensor(
                self.scalers['E'].transform(E.cpu().numpy().reshape(-1, 1)),
                dtype=torch.float32, device=device
            ).squeeze() * obj_mask
            
            # Transverse momentum (scaled)
            pT = X[:, idx_start + 2]
            pT_scaled = torch.tensor(
                self.scalers['pT'].transform(pT.cpu().numpy().reshape(-1, 1)),
                dtype=torch.float32, device=device
            ).squeeze() * obj_mask
            
            # Pseudorapidity (scaled)
            eta = X[:, idx_start + 3]
            eta_scaled = torch.tensor(
                self.scalers['eta'].transform(eta.cpu().numpy().reshape(-1, 1)),
                dtype=torch.float32, device=device
            ).squeeze() * obj_mask
            
            # Azimuthal angle kept as is, but add sin/cos features
            phi = X[:, idx_start + 4]
            sin_phi = torch.sin(phi) * obj_mask
            cos_phi = torch.cos(phi) * obj_mask
            
            # Four-momentum components (px, py, pz, E)
            px = pT * torch.cos(phi) * obj_mask
            py = pT * torch.sin(phi) * obj_mask
            pz = pT * torch.sinh(eta) * obj_mask  # pz = pT * sinh(η)
            
            # Mass invariant
            # m^2 = E^2 - p^2 = E^2 - (px^2 + py^2 + pz^2)
            p_squared = px**2 + py**2 + pz**2
            m_squared = E**2 - p_squared
            # Clip to avoid negative values due to numerical errors
            m_squared = torch.clamp(m_squared, min=0)
            mass = torch.sqrt(m_squared) * obj_mask
            
            # Collect features for this object
            obj_features = torch.stack([
                obj_mask, E_scaled, pT_scaled, eta_scaled, 
                sin_phi, cos_phi, px, py, pz, mass
            ], dim=1)
            
            object_features.append(obj_features)
        
        # Stack all object features
        all_obj_features = torch.cat(object_features, dim=1)  # [batch_size, 18*10]
        
        # Combine with global features
        global_features = torch.stack([scaled_ET_miss, sin_phi_ET_miss, cos_phi_ET_miss], dim=1)
        
        # Final feature tensor
        features = torch.cat([global_features, all_obj_features], dim=1)
        
        return features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class LorentzInvariantLayer(nn.Module):
    def __init__(self, n_objects):
        super().__init__()
        self.n_objects = n_objects
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape to get individual objects
        # Each object has 10 features, with the first being the mask
        # and the last 4 being px, py, pz, E
        obj_features = x[:, 3:].view(batch_size, self.n_objects, 10)
        
        # Extract object masks and four-momenta
        masks = obj_features[:, :, 0].view(batch_size, self.n_objects, 1)
        px = obj_features[:, :, 6].view(batch_size, self.n_objects, 1)
        py = obj_features[:, :, 7].view(batch_size, self.n_objects, 1)
        pz = obj_features[:, :, 8].view(batch_size, self.n_objects, 1)
        E = obj_features[:, :, 1].view(batch_size, self.n_objects, 1)  # Using E_scaled
        mass = obj_features[:, :, 9].view(batch_size, self.n_objects, 1)
        
        # Dot product between all pairs of four-momenta
        # This creates Lorentz-invariant scalar products
        lorentz_invariants = []
        
        for i in range(self.n_objects):
            for j in range(i+1, self.n_objects):
                # Calculate Minkowski dot product: Ei*Ej - pxi*pxj - pyi*pyj - pzi*pzj
                dot_product = E[:, i] * E[:, j] - px[:, i] * px[:, j] - py[:, i] * py[:, j] - pz[:, i] * pz[:, j]
                valid_pair = masks[:, i] * masks[:, j]  # Both objects must exist
                lorentz_invariants.append(dot_product * valid_pair)
                
                # Optional: Add normalized invariants
                if torch.any(mass[:, i] > 0) and torch.any(mass[:, j] > 0):
                    normalized = dot_product / (mass[:, i] * mass[:, j] + 1e-8)  # Avoid div by zero
                    lorentz_invariants.append(normalized * valid_pair)
        
        # Stack all invariants
        invariant_features = torch.stack(lorentz_invariants, dim=1)
        
        # Combine with original features
        return torch.cat([x, invariant_features], dim=1)

class EquivariantMessagePassing(nn.Module):
    def __init__(self, n_objects, hidden_dim):
        super().__init__()
        self.n_objects = n_objects
        self.hidden_dim = hidden_dim
        
        # Message networks
        self.message_net = nn.Sequential(
            nn.Linear(20, hidden_dim),  # 10 features per object * 2 objects in pair
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(10 + hidden_dim, hidden_dim),  # Original features + aggregated messages
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)  # Output same dimensionality as input
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract object features
        obj_features = x[:, 3:].view(batch_size, self.n_objects, 10)
        masks = obj_features[:, :, 0].view(batch_size, self.n_objects, 1)
        
        # Initialize messages
        messages = torch.zeros(batch_size, self.n_objects, self.hidden_dim, device=x.device)
        
        # Compute messages between all pairs of objects
        for i in range(self.n_objects):
            for j in range(self.n_objects):
                if i != j:  # Don't send messages to self
                    # Combine features from both objects
                    pair_features = torch.cat([obj_features[:, i], obj_features[:, j]], dim=1)
                    
                    # Compute message
                    message = self.message_net(pair_features)
                    
                    # Only consider valid objects
                    valid_message = message * masks[:, i].squeeze() * masks[:, j].squeeze().unsqueeze(1)
                    
                    # Accumulate message
                    messages[:, i] += valid_message
        
        # Update each object's features
        updated_features = []
        for i in range(self.n_objects):
            # Combine original features with received messages
            combined = torch.cat([obj_features[:, i], messages[:, i]], dim=1)
            
            # Update features
            new_features = self.update_net(combined) * masks[:, i]
            updated_features.append(new_features)
        
        # Stack all updated object features
        updated_obj_features = torch.cat(updated_features, dim=1)  # [batch_size, 18*10]
        
        # Combine with global features (keep them unchanged)
        global_features = x[:, :3]
        
        return torch.cat([global_features, updated_obj_features], dim=1)

class FourTopClassifier(nn.Module):
    def __init__(self, input_dim, n_objects=18, hidden_dim=128):
        super().__init__()
        self.n_objects = n_objects
        
        # Lorentz invariant layer
        self.lorentz_layer = LorentzInvariantLayer(n_objects)
        
        # Calculate output dimension after Lorentz layer
        # We add n_objects * (n_objects - 1) / 2 invariants, potentially doubled for normalized ones
        n_invariants = int(n_objects * (n_objects - 1))
        lorentz_output_dim = input_dim + n_invariants
        
        # Message passing layers
        self.message_passing1 = EquivariantMessagePassing(n_objects, hidden_dim)
        self.message_passing2 = EquivariantMessagePassing(n_objects, hidden_dim)
        
        # MLP for final classification
        self.classifier = nn.Sequential(
            nn.Linear(lorentz_output_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # Apply Lorentz-invariant feature extraction
        x = self.lorentz_layer(x)
        
        # Apply equivariant message passing
        x = self.message_passing1(x)
        x = self.message_passing2(x)
        
        # Final classification
        logits = self.classifier(x).squeeze(1)
        
        return logits

def make_model(input_dim: int):
    # Calculate input dimension after preprocessing: 
    # 3 global features + 18 objects * 10 features per object
    model = FourTopClassifier(input_dim=input_dim, n_objects=18, hidden_dim=128)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_auc = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        all_targets = []
        all_preds = []
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.detach().cpu().numpy())
        
        # Calculate training metrics
        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = correct / total
        train_loss.append(avg_train_loss)
        train_acc.append(train_accuracy)
        
        # Validation
        model.eval()
        val_epoch_loss = 0
        val_correct = 0
        val_total = 0
        val_targets = []
        val_preds = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets.float())
                
                # Statistics
                val_epoch_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                
                val_targets.extend(targets.cpu().numpy())
                val_preds.extend(probs.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_epoch_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_loss.append(avg_val_loss)
        val_acc.append(val_accuracy)
        
        # Calculate AUC (sklearn isn't available)
        val_targets = np.array(val_targets)
        val_preds = np.array(val_preds)
        
        # Compute ROC AUC manually
        def calculate_auc(y_true, y_scores):
            # Sort predictions and corresponding true labels
            sorted_indices = np.argsort(y_scores)[::-1]
            y_true_sorted = y_true[sorted_indices]
            
            # Calculate TPR and FPR at different thresholds
            n_pos = np.sum(y_true == 1)
            n_neg = len(y_true) - n_pos
            
            if n_pos == 0 or n_neg == 0:
                return 0.5  # Unable to calculate AUC
            
            # Calculate cumulative TP and FP
            tp_cumsum = np.cumsum(y_true_sorted)
            fp_cumsum = np.cumsum(1 - y_true_sorted)
            
            # Calculate TPR and FPR
            tpr = tp_cumsum / n_pos
            fpr = fp_cumsum / n_neg
            
            # Add (0,0) and (1,1) points
            tpr = np.concatenate(([0], tpr, [1]))
            fpr = np.concatenate(([0], fpr, [1]))
            
            # Calculate AUC using trapezoidal rule
            auc = np.trapz(tpr, fpr)
            return auc
        
        val_auc = calculate_auc(val_targets, val_preds)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}')
        
        # Update learning rate based on AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
    
    # Restore best model
    if 'best_model_state' in locals():
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

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

