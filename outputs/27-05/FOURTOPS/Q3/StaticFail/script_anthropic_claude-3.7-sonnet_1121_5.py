
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
from torch.nn import functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.object_scaler = StandardScaler()
        self.et_miss_scaler = StandardScaler()
        self.n_objects = 18
        self.obj_features = 5  # obj_id, E, pT, eta, phi
        self.valid_object_mask = None
        self.mean_valid_objects = None
    
    def fit(self, X, y=None):
        # Extract features
        # First two features are ET_miss and phi_ET_miss
        et_miss_features = X[:, :2].numpy()  # [N, 2]
        self.et_miss_scaler.fit(et_miss_features)
        
        # Extract all object features and mask invalid objects (zero-padded)
        all_object_features = []
        valid_objects = []
        
        for i in range(self.n_objects):
            start_idx = 2 + i * self.obj_features
            end_idx = start_idx + self.obj_features
            obj_features = X[:, start_idx:end_idx]
            
            # Check if object is valid (non-zero)
            obj_mask = obj_features[:, 0] != 0  # Check if object ID is non-zero
            valid_objects.append(obj_mask)
            
            # Only include valid objects for scaling
            valid_obj_features = obj_features[obj_mask]
            if len(valid_obj_features) > 0:
                all_object_features.append(valid_obj_features)
        
        # Stack all valid object features for scaling
        if all_object_features:
            all_objects_stacked = torch.cat(all_object_features, dim=0).numpy()
            self.object_scaler.fit(all_objects_stacked)
        
        # Store valid object mask statistics
        self.valid_object_mask = torch.stack(valid_objects, dim=1)  # [N, n_objects]
        self.mean_valid_objects = self.valid_object_mask.float().mean(dim=0)
        
        return self
    
    def transform(self, X):
        batch_size = X.shape[0]
        
        # Transform ET_miss features
        et_miss_transformed = torch.tensor(
            self.et_miss_scaler.transform(X[:, :2].numpy()), 
            dtype=torch.float32
        )  # [N, 2]
        
        # Initialize transformed object features tensor
        all_transformed_objects = []
        valid_object_masks = []
        
        # Calculate additional physics-informed features
        for i in range(self.n_objects):
            start_idx = 2 + i * self.obj_features
            end_idx = start_idx + self.obj_features
            obj_features = X[:, start_idx:end_idx].clone()  # [N, 5]
            
            # Check if object is valid (non-zero)
            obj_mask = obj_features[:, 0] != 0  # Check object ID
            valid_object_masks.append(obj_mask)
            
            # Apply scaler only to valid objects
            valid_indices = torch.where(obj_mask)[0]
            if len(valid_indices) > 0:
                valid_obj = obj_features[valid_indices]
                obj_features[valid_indices] = torch.tensor(
                    self.object_scaler.transform(valid_obj.numpy()),
                    dtype=torch.float32
                )
            
            # Add physics-informed features (for valid objects only)
            if len(valid_indices) > 0:
                valid_obj_raw = X[valid_indices, start_idx:end_idx]
                # Extract raw values for physics calculations
                obj_id = valid_obj_raw[:, 0]  # Object type ID
                E = valid_obj_raw[:, 1]       # Energy
                pT = valid_obj_raw[:, 2]      # Transverse momentum
                eta = valid_obj_raw[:, 3]     # Pseudorapidity
                phi = valid_obj_raw[:, 4]     # Azimuthal angle
                
                # Calculate mass (m^2 = E^2 - |p|^2)
                # |p|^2 = pT^2 * cosh(eta)^2
                p_squared = pT**2 * torch.cosh(eta)**2
                m_squared = E**2 - p_squared
                # Set negative m_squared to zero (numerical precision issues)
                mass = torch.sqrt(torch.clamp(m_squared, min=0.0))
                
                # Calculate transverse energy
                ET = E * torch.sin(torch.atan(torch.exp(-eta)) * 2)
                
                # Create augmented features tensor for valid objects
                aug_features = torch.stack([mass, ET], dim=1)  # [valid_N, 2]
                
                # Initialize augmented features for all objects in batch
                batch_aug_features = torch.zeros((batch_size, 2), dtype=torch.float32)
                # Fill in values for valid objects
                batch_aug_features[valid_indices] = aug_features
                
                # Append augmented features to object features
                obj_features = torch.cat([obj_features, batch_aug_features], dim=1)  # [N, 7]
            else:
                # For invalid objects, add zeros for augmented features
                zeros = torch.zeros((batch_size, 2), dtype=torch.float32)
                obj_features = torch.cat([obj_features, zeros], dim=1)  # [N, 7]
            
            all_transformed_objects.append(obj_features)
        
        # Stack all transformed objects
        objects_transformed = torch.stack(all_transformed_objects, dim=1)  # [N, n_objects, 7]
        
        # Stack object validity masks
        valid_mask = torch.stack(valid_object_masks, dim=1).float()  # [N, n_objects]
        
        # Calculate pairwise ΔR and invariant mass between objects for all valid combinations
        n_pairs = (self.n_objects * (self.n_objects - 1)) // 2
        pair_dR = torch.zeros((batch_size, n_pairs), dtype=torch.float32)
        pair_mass = torch.zeros((batch_size, n_pairs), dtype=torch.float32)
        
        pair_idx = 0
        for i in range(self.n_objects):
            for j in range(i+1, self.n_objects):
                # Extract raw values from original tensor
                obj_i_start = 2 + i * self.obj_features
                obj_j_start = 2 + j * self.obj_features
                
                # Create a mask where both objects are valid
                both_valid = valid_object_masks[i] & valid_object_masks[j]
                valid_indices = torch.where(both_valid)[0]
                
                if len(valid_indices) > 0:
                    # Extract raw values for valid pairs
                    obj_i_raw = X[valid_indices, obj_i_start:obj_i_start+self.obj_features]
                    obj_j_raw = X[valid_indices, obj_j_start:obj_j_start+self.obj_features]
                    
                    # Extract individual components
                    E_i = obj_i_raw[:, 1]
                    pT_i = obj_i_raw[:, 2]
                    eta_i = obj_i_raw[:, 3]
                    phi_i = obj_i_raw[:, 4]
                    
                    E_j = obj_j_raw[:, 1]
                    pT_j = obj_j_raw[:, 2]
                    eta_j = obj_j_raw[:, 3]
                    phi_j = obj_j_raw[:, 4]
                    
                    # Calculate ΔR = sqrt(Δη^2 + Δφ^2)
                    # Handle phi differences properly (circular coordinate)
                    delta_phi = torch.abs(phi_i - phi_j)
                    delta_phi = torch.min(delta_phi, 2*math.pi - delta_phi)
                    delta_eta = eta_i - eta_j
                    delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)
                    
                    # Calculate invariant mass
                    p_i = pT_i * torch.cosh(eta_i)
                    p_j = pT_j * torch.cosh(eta_j)
                    
                    # px_i = pT_i * torch.cos(phi_i)
                    # py_i = pT_i * torch.sin(phi_i)
                    # pz_i = pT_i * torch.sinh(eta_i)
                    
                    # px_j = pT_j * torch.cos(phi_j)
                    # py_j = pT_j * torch.sin(phi_j)
                    # pz_j = pT_j * torch.sinh(eta_j)
                    
                    # Calculate invariant mass using E^2 - |p|^2 formula
                    # m^2 = (E_i + E_j)^2 - (px_i + px_j)^2 - (py_i + py_j)^2 - (pz_i + pz_j)^2
                    # Simplified using dot product formula
                    # Apply dot product formula for 3-vectors
                    cos_delta_phi = torch.cos(delta_phi)
                    inv_mass_squared = (E_i + E_j)**2 - (pT_i**2 + pT_j**2 + 2*pT_i*pT_j*cos_delta_phi)*torch.cosh(eta_i)*torch.cosh(eta_j)
                    # Handle numerical precision issues
                    inv_mass = torch.sqrt(torch.clamp(inv_mass_squared, min=0.0))
                    
                    # Assign to pair features
                    pair_dR[valid_indices, pair_idx] = delta_R
                    pair_mass[valid_indices, pair_idx] = inv_mass
                
                pair_idx += 1
        
        # Calculate event-level features
        # Sum of pT for all objects
        sum_pT = torch.zeros(batch_size, dtype=torch.float32)
        for i in range(self.n_objects):
            obj_mask = valid_object_masks[i]
            if torch.any(obj_mask):
                pT_values = X[obj_mask, 2 + i*self.obj_features + 2]  # +2 for pT index in obj features
                # Add to the corresponding entries in sum_pT
                sum_pT[obj_mask] += pT_values
        
        # Sphericity tensor components
        spher_tensor = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        
        for i in range(self.n_objects):
            obj_mask = valid_object_masks[i]
            valid_indices = torch.where(obj_mask)[0]
            
            if len(valid_indices) > 0:
                obj_raw = X[valid_indices, 2 + i*self.obj_features:2 + (i+1)*self.obj_features]
                
                pT = obj_raw[:, 2]
                eta = obj_raw[:, 3]
                phi = obj_raw[:, 4]
                
                # Calculate momentum components
                px = pT * torch.cos(phi)
                py = pT * torch.sin(phi)
                pz = pT * torch.sinh(eta)
                p_mag = torch.sqrt(px**2 + py**2 + pz**2)
                
                # Normalize by momentum magnitude
                px_norm = px / p_mag
                py_norm = py / p_mag
                pz_norm = pz / p_mag
                
                # Construct sphericity tensor components
                spher_tensor[valid_indices, 0, 0] += px_norm * px_norm
                spher_tensor[valid_indices, 0, 1] += px_norm * py_norm
                spher_tensor[valid_indices, 0, 2] += px_norm * pz_norm
                spher_tensor[valid_indices, 1, 0] += py_norm * px_norm
                spher_tensor[valid_indices, 1, 1] += py_norm * py_norm
                spher_tensor[valid_indices, 1, 2] += py_norm * pz_norm
                spher_tensor[valid_indices, 2, 0] += pz_norm * px_norm
                spher_tensor[valid_indices, 2, 1] += pz_norm * py_norm
                spher_tensor[valid_indices, 2, 2] += pz_norm * pz_norm
        
        # Get eigenvalues of sphericity tensor
        sphericity_eigenvals = torch.zeros((batch_size, 3), dtype=torch.float32)
        for i in range(batch_size):
            try:
                eigenvals = torch.linalg.eigvalsh(spher_tensor[i])
                sphericity_eigenvals[i] = eigenvals
            except Exception:
                # Handle numerical issues by setting to zeros
                pass
        
        # Sort eigenvalues in ascending order (PyTorch's eigvalsh might already do this)
        sphericity_eigenvals, _ = torch.sort(sphericity_eigenvals, dim=1)
        
        # Calculate sphericity and aplanarity
        sphericity = 1.5 * (sphericity_eigenvals[:, 0] + sphericity_eigenvals[:, 1])
        aplanarity = 1.5 * sphericity_eigenvals[:, 0]
        
        # Count number of valid objects of each type
        obj_counts = torch.zeros((batch_size, 5), dtype=torch.float32)  # Assuming 5 object types
        
        for i in range(self.n_objects):
            obj_mask = valid_object_masks[i]
            valid_indices = torch.where(obj_mask)[0]
            
            if len(valid_indices) > 0:
                obj_ids = X[valid_indices, 2 + i*self.obj_features].to(torch.long) - 1  # 0-indexed
                for j, idx in enumerate(valid_indices):
                    if 0 <= obj_ids[j] < 5:  # Check valid range
                        obj_counts[idx, obj_ids[j]] += 1
        
        # Calculate total valid objects per event
        total_objects = torch.sum(valid_mask, dim=1, keepdim=True)  # [N, 1]
        
        # Combine all features
        # 1. ET_miss features [N, 2]
        # 2. Flattened object features [N, n_objects * 7]
        # 3. Valid object mask [N, n_objects]
        # 4. Pairwise ΔR and mass [N, n_pairs * 2]
        # 5. Sum of pT [N, 1]
        # 6. Sphericity and aplanarity [N, 2]
        # 7. Object counts [N, 5]
        # 8. Total objects [N, 1]
        
        # Reshape objects_transformed to [N, n_objects * 7]
        objects_flat = objects_transformed.reshape(batch_size, -1)  # [N, n_objects * 7]
        
        # Combine all features
        combined_features = torch.cat([
            et_miss_transformed,                    # [N, 2]
            objects_flat,                           # [N, n_objects * 7]
            valid_mask,                             # [N, n_objects]
            pair_dR,                                # [N, n_pairs]
            pair_mass,                              # [N, n_pairs]
            sum_pT.unsqueeze(1),                    # [N, 1]
            sphericity.unsqueeze(1),                # [N, 1]
            aplanarity.unsqueeze(1),                # [N, 1]
            obj_counts,                             # [N, 5]
            total_objects                           # [N, 1]
        ], dim=1)
        
        return combined_features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class SlotAttention(nn.Module):
    def __init__(self, dim, num_slots, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        
        # Learnable slots initialization
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, dim))
        
        # Linear projections for queries, keys, values
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        # Slot update function
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_inputs = nn.LayerNorm(dim)

    def forward(self, inputs, mask=None):
        batch_size, num_inputs, _ = inputs.shape
        inputs = self.norm_inputs(inputs)
        
        # Initialize slots from mu and sigma
        mu = self.slots_mu.expand(batch_size, -1, -1)
        sigma = self.slots_log_sigma.exp().expand(batch_size, -1, -1)
        
        slots = mu + torch.randn(mu.shape, device=mu.device) * sigma
        
        # Prepare mask if provided
        if mask is not None:
            # Ensure mask is properly shaped and has correct values
            mask = mask.to(inputs.device)
        
        # Attention iterations
        for _ in range(self.iters):
            slots_prev = slots
            
            # Normalize slots
            slots = self.norm_slots(slots)
            
            # Computing attention
            q = self.to_q(slots)  # (batch, num_slots, dim)
            k = self.to_k(inputs)  # (batch, num_inputs, dim)
            v = self.to_v(inputs)  # (batch, num_inputs, dim)
            
            # Compute attention weights
            attn_logits = torch.matmul(q, k.transpose(2, 1))  # (batch, num_slots, num_inputs)
            attn_logits = attn_logits * self.scale
            
            # Apply mask to attention logits if provided
            if mask is not None:
                # Expand mask for broadcasting: (batch, num_inputs) -> (batch, 1, num_inputs)
                mask_expanded = mask.unsqueeze(1)
                # Set attention logits to negative infinity for masked inputs
                attn_logits = attn_logits.masked_fill(~mask_expanded, -1e9)
            
            attn = F.softmax(attn_logits, dim=-1)  # (batch, num_slots, num_inputs)
            
            # Weighted mean
            updates = torch.matmul(attn, v)  # (batch, num_slots, dim)
            
            # Update slots with GRU
            slots = slots_prev.reshape(-1, self.dim)
            updates = updates.reshape(-1, self.dim)
            slots = self.gru(updates, slots)  # (batch * num_slots, dim)
            slots = slots.reshape(batch_size, self.num_slots, self.dim)
            
            # Apply MLP
            slots = slots + self.mlp(slots)
        
        return slots, attn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attended, _ = self.attention(x, x, x, key_padding_mask=mask if mask is not None else None)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class PhysicsInformedTopClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_slots=8, num_objects=18, obj_feature_dim=7, 
                 num_transformer_layers=3, num_heads=4, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.num_objects = num_objects
        self.obj_feature_dim = obj_feature_dim
        
        # Process ET_miss features
        self.et_miss_processor = nn.Sequential(
            nn.Linear(2, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Process object features
        self.object_embedder = nn.Sequential(
            nn.Linear(obj_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer encoder layers to process object embeddings
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Slot attention to group particles
        self.slot_attention = SlotAttention(
            dim=hidden_dim,
            num_slots=num_slots,
            iters=3,
            hidden_dim=hidden_dim*2
        )
        
        # Process global features
        self.global_processor = nn.Sequential(
            nn.Linear(input_dim - 2 - num_objects * obj_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Split the input into different parts based on our preprocessing
        et_miss = x[:, :2]  # [N, 2]
        
        objects_flat = x[:, 2:2 + self.num_objects * self.obj_feature_dim]
        objects = objects_flat.reshape(batch_size, self.num_objects, self.obj_feature_dim)  # [N, num_objects, 7]
        
        obj_mask_start = 2 + self.num_objects * self.obj_feature_dim
        obj_mask = x[:, obj_mask_start:obj_mask_start + self.num_objects]  # [N, num_objects]
        
        # Global features are all remaining features
        global_features = x[:, obj_mask_start + self.num_objects:]
        
        # Process ET_miss
        et_miss_embedding = self.et_miss_processor(et_miss)  # [N, hidden_dim]
        
        # Process objects
        object_embeddings = self.object_embedder(objects)  # [N, num_objects, hidden_dim]
        
        # Apply transformer layers with masking
        # Convert mask to boolean where 1.0 means valid (not masked)
        transformer_mask = (obj_mask < 0.5)  # [N, num_objects], True means masked/invalid
        
        for layer in self.transformer_layers:
            object_embeddings = layer(object_embeddings, transformer_mask)
        
        # Apply slot attention with masking
        # Slot attention needs mask where True means valid
        slot_mask = (obj_mask > 0.5)  # [N, num_objects], True means valid
        slots, attention = self.slot_attention(object_embeddings, slot_mask)
        
        # Pool slots with mean for a slot embedding
        slot_embedding = slots.mean(dim=1)  # [N, hidden_dim]
        
        # Process global features
        global_embedding = self.global_processor(global_features)  # [N, hidden_dim]
        
        # Concatenate all embeddings
        combined = torch.cat([et_miss_embedding, slot_embedding, global_embedding], dim=1)  # [N, hidden_dim*3]
        
        # Final classification
        logits = self.classifier(combined).squeeze(-1)  # [N]
        
        return logits

def make_model(input_dim: int):
    model = PhysicsInformedTopClassifier(
        input_dim=input_dim,
        hidden_dim=128,
        num_slots=6,
        num_objects=18,
        obj_feature_dim=7,
        num_transformer_layers=2,
        num_heads=4,
        dropout=0.1
    )
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Binary cross entropy loss
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    
    # Track metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            epoch_train_loss += loss.item() * batch_X.size(0)
            epoch_train_correct += ((outputs > 0.0) == batch_y.bool()).sum().item()
            epoch_train_total += batch_X.size(0)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0
        val_outputs_all = []
        val_labels_all = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.float().to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Track metrics
                epoch_val_loss += loss.item() * batch_X.size(0)
                epoch_val_correct += ((outputs > 0.0) == batch_y.bool()).sum().item()
                epoch_val_total += batch_X.size(0)
                
                # Store outputs and labels for AUC calculation
                val_outputs_all.append(outputs.cpu())
                val_labels_all.append(batch_y.cpu())
        
        # Calculate epoch metrics
        epoch_train_loss /= epoch_train_total
        epoch_train_acc = epoch_train_correct / epoch_train_total
        epoch_val_loss /= epoch_val_total
        epoch_val_acc = epoch_val_correct / epoch_val_total
        
        # Calculate AUC for validation set
        val_outputs_all = torch.cat(val_outputs_all).numpy()
        val_labels_all = torch.cat(val_labels_all).numpy()
        val_auc = roc_auc_score(val_labels_all, torch.sigmoid(torch.tensor(val_outputs_all)).numpy())
        
        # Update scheduler based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Store metrics
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        train_acc.append(epoch_train_acc)
        val_acc.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train loss: {epoch_train_loss:.4f}, Train acc: {epoch_train_acc:.4f} - "
              f"Val loss: {epoch_val_loss:.4f}, Val acc: {epoch_val_acc:.4f}, Val AUC: {val_auc:.4f}")
    
    # Load best model state
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

