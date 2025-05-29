
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
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from typing import List, Tuple, Dict
import math

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # Define parameters for normalization
        self.scalers = {}
        self.object_mask = None
        self.valid_objects = None
        
        # Store the maximum number of objects and features per object
        self.max_objects = 18
        self.features_per_object = 5
        
        # Store global feature indices
        self.et_miss_idx = 0
        self.phi_et_miss_idx = 1
        
        # Store derived features statistics
        self.derived_features_scaler = StandardScaler()

    def fit(self, X, y=None):
        # Extract valid object mask (non-zero object identifiers)
        object_indices = torch.arange(2, X.shape[1], self.features_per_object)
        self.object_mask = (X[:, object_indices] != 0).float()  # [N, max_objects]
        
        # Extract various feature groups for normalization
        # Global features: missing ET and phi
        global_features = X[:, :2]
        self.scalers['global'] = StandardScaler().fit(global_features.numpy())
        
        # Extract object features for each type
        valid_objects_list = []
        
        for i in range(self.max_objects):
            # Offsets for each object's features
            start_idx = 2 + i * self.features_per_object
            
            # Get object mask for this position
            mask = self.object_mask[:, i] == 1
            
            if mask.sum() > 0:
                # Extract features for valid objects
                object_features = X[mask, start_idx:start_idx+self.features_per_object]
                
                # We'll store scalers for each feature type across all objects
                for j in range(1, self.features_per_object):  # Skip object ID
                    feature_key = f'obj_feature_{j}'
                    if feature_key not in self.scalers:
                        self.scalers[feature_key] = StandardScaler()
                    
                    feature_values = object_features[:, j].reshape(-1, 1)
                    self.scalers[feature_key].partial_fit(feature_values.numpy())
        
        # Build derived features and fit scaler
        sample_derived = self._build_derived_features(X[:1000])  # Use subset for fitting
        self.derived_features_scaler.fit(sample_derived.numpy())
        
        return self

    def _build_derived_features(self, X):
        batch_size = X.shape[0]
        
        # Initialize tensor to hold derived features
        derived_features = torch.zeros((batch_size, 15), dtype=torch.float32)
        
        # Missing ET features
        et_miss = X[:, self.et_miss_idx]
        phi_et_miss = X[:, self.phi_et_miss_idx]
        
        # Count valid objects per event
        object_indices = torch.arange(2, X.shape[1], self.features_per_object)
        valid_objects = (X[:, object_indices] != 0).sum(dim=1).float()
        derived_features[:, 0] = valid_objects
        
        # Sum of energies, pTs, and other global quantities
        total_energy = torch.zeros(batch_size)
        total_pt = torch.zeros(batch_size)
        pt_weighted_eta = torch.zeros(batch_size)
        pt_weighted_phi = torch.zeros(batch_size)
        
        # Sphericity tensor components
        Sxx = torch.zeros(batch_size)
        Syy = torch.zeros(batch_size)
        Szz = torch.zeros(batch_size)
        Sxy = torch.zeros(batch_size)
        Sxz = torch.zeros(batch_size)
        Syz = torch.zeros(batch_size)
        
        # Variables for tracking high-pt objects
        max_pt = torch.zeros(batch_size)
        second_max_pt = torch.zeros(batch_size)
        
        for i in range(self.max_objects):
            start_idx = 2 + i * self.features_per_object
            obj_id = X[:, start_idx]
            
            # Process only valid objects
            valid_mask = (obj_id != 0)
            
            if valid_mask.sum() > 0:
                E = X[valid_mask, start_idx + 1]  # Energy
                pt = X[valid_mask, start_idx + 2]  # Transverse momentum
                eta = X[valid_mask, start_idx + 3]  # Pseudorapidity
                phi = X[valid_mask, start_idx + 4]  # Azimuthal angle
                
                # Update totals for valid objects
                total_energy[valid_mask] += E
                total_pt[valid_mask] += pt
                
                # Update pt-weighted eta and phi
                pt_weighted_eta[valid_mask] += pt * eta
                pt_weighted_phi[valid_mask] += pt * phi
                
                # Update max and second max pt
                update_max = pt > max_pt[valid_mask]
                second_max_pt[valid_mask[update_max]] = max_pt[valid_mask[update_max]]
                max_pt[valid_mask[update_max]] = pt[update_max]
                
                update_second = (~update_max) & (pt > second_max_pt[valid_mask])
                second_max_pt[valid_mask[update_second]] = pt[update_second]
                
                # Calculate Cartesian momentum components for sphericity tensor
                px = pt * torch.cos(phi)
                py = pt * torch.sin(phi)
                pz = pt * torch.sinh(eta)  # pz = pt * sinh(eta)
                p_mag_squared = px*px + py*py + pz*pz
                
                # Update sphericity tensor components
                Sxx[valid_mask] += px * px / p_mag_squared
                Syy[valid_mask] += py * py / p_mag_squared
                Szz[valid_mask] += pz * pz / p_mag_squared
                Sxy[valid_mask] += px * py / p_mag_squared
                Sxz[valid_mask] += px * pz / p_mag_squared
                Syz[valid_mask] += py * pz / p_mag_squared
        
        # Normalize pt-weighted eta and phi
        mask = total_pt > 0
        pt_weighted_eta[mask] /= total_pt[mask]
        pt_weighted_phi[mask] /= total_pt[mask]
        
        # Store derived features
        derived_features[:, 1] = total_energy
        derived_features[:, 2] = total_pt
        derived_features[:, 3] = pt_weighted_eta
        derived_features[:, 4] = pt_weighted_phi
        derived_features[:, 5] = max_pt
        derived_features[:, 6] = second_max_pt
        derived_features[:, 7] = Sxx
        derived_features[:, 8] = Syy
        derived_features[:, 9] = Szz
        derived_features[:, 10] = Sxy
        derived_features[:, 11] = Sxz
        derived_features[:, 12] = Syz
        
        # Calculate missing ET features relative to the event
        derived_features[:, 13] = et_miss / (total_pt + 1e-8)  # ET_miss / sum_pt
        
        # Calculate delta phi between weighted event phi and missing ET phi
        delta_phi = torch.abs(phi_et_miss - pt_weighted_phi)
        delta_phi = torch.min(delta_phi, 2*np.pi - delta_phi)  # Make sure delta_phi is in [0, pi]
        derived_features[:, 14] = delta_phi
        
        return derived_features

    def transform(self, X):
        batch_size = X.shape[0]
        
        # Normalize global features
        global_features = X[:, :2].clone()
        global_features = torch.tensor(
            self.scalers['global'].transform(global_features.numpy()), 
            dtype=torch.float32
        )  # [N, 2]
        
        # Extract and normalize object features
        # We'll create a tensor with structure [batch, max_objects, features]
        normalized_objects = torch.zeros(
            (batch_size, self.max_objects, self.features_per_object-1), 
            dtype=torch.float32
        )  # [N, max_objects, 4] - excluding object ID
        
        # Create object mask
        object_indices = torch.arange(2, X.shape[1], self.features_per_object)
        object_mask = (X[:, object_indices] != 0).float()  # [N, max_objects]
        
        # Process each object
        for i in range(self.max_objects):
            start_idx = 2 + i * self.features_per_object
            
            # Skip object ID and process 4 features: E, pT, eta, phi
            for j in range(1, self.features_per_object):
                feature_idx = start_idx + j
                feature_key = f'obj_feature_{j}'
                
                # Extract feature and reshape
                feature = X[:, feature_idx].clone().reshape(-1, 1)
                
                # Normalize feature
                normalized_feature = torch.tensor(
                    self.scalers[feature_key].transform(feature.numpy()), 
                    dtype=torch.float32
                ).reshape(-1)
                
                # Store normalized feature
                normalized_objects[:, i, j-1] = normalized_feature
        
        # Build derived features
        derived_features = self._build_derived_features(X)  # [N, derived_features]
        
        # Normalize derived features
        derived_features = torch.tensor(
            self.derived_features_scaler.transform(derived_features.numpy()),
            dtype=torch.float32
        )
        
        # Return flattened object tensor for MLP approach
        flattened_objects = normalized_objects.reshape(batch_size, -1)  # [N, max_objects * 4]
        
        # Concatenate features
        result = torch.cat([
            global_features,           # [N, 2] - ET_miss, phi_ET_miss
            flattened_objects,         # [N, max_objects * 4] - normalized object features
            object_mask,               # [N, max_objects] - valid object mask
            derived_features           # [N, derived_features] - derived global features
        ], dim=1)  # [N, 2 + max_objects*4 + max_objects + derived_features]
        
        return result

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class LorentzInvariantLayer(nn.Module):
    def __init__(self, max_objects=18):
        super().__init__()
        self.max_objects = max_objects
        
    def forward(self, x, mask):
        # x shape: [batch_size, max_objects, 4] where 4 represents (E, px, py, pz)
        # mask shape: [batch_size, max_objects]
        
        batch_size = x.shape[0]
        
        # Apply mask to ensure only valid objects are considered
        masked_x = x * mask.unsqueeze(-1)  # [batch_size, max_objects, 4]
        
        # Calculate pairwise Lorentz dot products
        # For 4-vectors a and b, dot product is: a·b = a_0*b_0 - a_1*b_1 - a_2*b_2 - a_3*b_3
        products = torch.zeros(batch_size, self.max_objects, self.max_objects, dtype=x.dtype, device=x.device)
        
        for i in range(self.max_objects):
            for j in range(self.max_objects):
                # Compute Minkowski inner product
                # E_i * E_j - px_i * px_j - py_i * py_j - pz_i * pz_j
                products[:, i, j] = masked_x[:, i, 0] * masked_x[:, j, 0] \
                                    - masked_x[:, i, 1] * masked_x[:, j, 1] \
                                    - masked_x[:, i, 2] * masked_x[:, j, 2] \
                                    - masked_x[:, i, 3] * masked_x[:, j, 3]
                
                # Apply mask to ensure invalid pairs are 0
                products[:, i, j] *= mask[:, i] * mask[:, j]
        
        # Flatten the products tensor
        flat_products = products.view(batch_size, -1)  # [batch_size, max_objects²]
        
        return flat_products

class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim, message_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Message computation network
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, message_dim),
            nn.LayerNorm(message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
        
        # Update network
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + message_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, node_features, mask):
        # node_features shape: [batch_size, num_nodes, hidden_dim]
        # mask shape: [batch_size, num_nodes]
        
        batch_size, num_nodes, hidden_dim = node_features.shape
        device = node_features.device
        
        # Create mask for valid edges (both nodes must be valid)
        edge_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # [batch_size, num_nodes, num_nodes]
        
        # Compute messages for all pairs of nodes
        messages = torch.zeros(batch_size, num_nodes, num_nodes, self.message_mlp[-1].out_features, device=device)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Skip self-loops
                    # Concatenate features from both nodes
                    node_pair = torch.cat([node_features[:, i, :], node_features[:, j, :]], dim=1)
                    # Compute message
                    message = self.message_mlp(node_pair)
                    messages[:, i, j, :] = message * edge_mask[:, i, j].unsqueeze(-1)
        
        # Aggregate messages per node
        aggregated_messages = torch.sum(messages, dim=2)  # [batch_size, num_nodes, message_dim]
        
        # Update node features
        updated_features = torch.cat([node_features, aggregated_messages], dim=2)
        updated_features = self.update_mlp(updated_features)
        
        # Apply original mask to ensure only valid nodes are updated
        result = node_features + updated_features * mask.unsqueeze(-1)
        
        return result

class LorentzEquivariantNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_message_passing=2):
        super().__init__()
        
        # Parameters for preprocessing
        self.max_objects = 18
        self.features_per_object = 4  # E, pT, eta, phi
        
        # Define dimensions for various feature groups
        self.global_dim = 2  # ET_miss, phi_ET_miss
        self.derived_dim = 15  # Various derived features
        
        # Initial embeddings for different object types
        self.object_embedding = nn.Linear(self.features_per_object, hidden_dim)
        self.global_embedding = nn.Linear(self.global_dim, hidden_dim)
        self.derived_embedding = nn.Linear(self.derived_dim, hidden_dim)
        
        # Lorentz-invariant layer
        self.lorentz_layer = LorentzInvariantLayer(max_objects=self.max_objects)
        self.lorentz_projection = nn.Linear(self.max_objects * self.max_objects, hidden_dim)
        
        # Message passing layers for learning interactions
        self.message_passing_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) for _ in range(num_message_passing)
        ])
        
        # Final MLP for classification
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def _convert_to_cartesian(self, x):
        # Convert (E, pT, eta, phi) to (E, px, py, pz)
        batch_size = x.shape[0]
        cartesian = torch.zeros(batch_size, self.max_objects, 4, device=x.device)
        
        # Extract components
        E = x[:, :, 0]
        pt = x[:, :, 1]
        eta = x[:, :, 2]
        phi = x[:, :, 3]
        
        # Convert to Cartesian coordinates
        cartesian[:, :, 0] = E  # Energy
        cartesian[:, :, 1] = pt * torch.cos(phi)  # px
        cartesian[:, :, 2] = pt * torch.sin(phi)  # py
        cartesian[:, :, 3] = pt * torch.sinh(eta)  # pz
        
        return cartesian
            
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Unpack different feature groups
        # Given shape: [batch, 2 + max_objects*4 + max_objects + derived_dim]
        global_features = x[:, :self.global_dim]  # [batch, 2]
        
        # Extract object features and reshape to [batch, max_objects, 4]
        start_idx = self.global_dim
        end_idx = start_idx + self.max_objects * self.features_per_object
        object_features = x[:, start_idx:end_idx].reshape(batch_size, self.max_objects, self.features_per_object)
        
        # Extract object mask
        start_idx = end_idx
        end_idx = start_idx + self.max_objects
        object_mask = x[:, start_idx:end_idx]  # [batch, max_objects]
        
        # Extract derived features
        derived_features = x[:, end_idx:]  # [batch, derived_dim]
        
        # Apply embeddings
        object_embeds = self.object_embedding(object_features)  # [batch, max_objects, hidden_dim]
        global_embeds = self.global_embedding(global_features)  # [batch, hidden_dim]
        derived_embeds = self.derived_embedding(derived_features)  # [batch, hidden_dim]
        
        # Convert to Cartesian coordinates for Lorentz operations
        cartesian_coords = self._convert_to_cartesian(object_features)
        
        # Apply Lorentz-invariant layer
        lorentz_products = self.lorentz_layer(cartesian_coords, object_mask)  # [batch, max_objects²]
        lorentz_embeds = self.lorentz_projection(lorentz_products)  # [batch, hidden_dim]
        
        # Apply message passing to learn interactions between particles
        message_features = object_embeds
        for layer in self.message_passing_layers:
            message_features = layer(message_features, object_mask)
        
        # Global pooling over objects with masking
        pooled_objects = torch.sum(message_features * object_mask.unsqueeze(-1), dim=1)  # [batch, hidden_dim]
        
        # Combine all features
        combined_features = torch.cat([
            pooled_objects,
            global_embeds,
            lorentz_embeds
        ], dim=1)  # [batch, hidden_dim*3]
        
        # Final classification MLP
        logits = self.final_mlp(combined_features).squeeze(-1)
        
        return logits

def make_model(input_dim: int):
    return LorentzEquivariantNetwork(input_dim)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Track metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    best_val_auc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float()
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            epoch_train_loss += loss.item() * X_batch.size(0)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
        
        epoch_train_loss /= total
        epoch_train_acc = correct / total
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).float()
                
                # Forward pass
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                # Track metrics
                epoch_val_loss += loss.item() * X_batch.size(0)
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
                
                all_probs.append(probs.cpu())
                all_labels.append(y_batch.cpu())
        
        epoch_val_loss /= total
        epoch_val_acc = correct / total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        # Calculate AUC for scheduler
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        
        # Convert to numpy for scikit-learn
        probs_np = all_probs.numpy()
        labels_np = all_labels.numpy()
        
        # Calculate AUC using sklearn
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(labels_np, probs_np)
        
        # Update scheduler based on validation AUC
        scheduler.step(val_auc)
        
        # Track best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pt')
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val AUC: {val_auc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
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

