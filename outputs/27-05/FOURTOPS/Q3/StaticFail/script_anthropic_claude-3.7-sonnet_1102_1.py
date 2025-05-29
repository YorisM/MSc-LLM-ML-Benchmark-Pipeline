
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
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # Define scalers for numerical features
        self.et_miss_scaler = RobustScaler()
        self.phi_et_miss_scaler = RobustScaler()
        self.energy_scaler = RobustScaler()
        self.pt_scaler = RobustScaler()
        self.eta_scaler = RobustScaler()
        self.phi_scaler = RobustScaler()
        
        # Constants for physics-informed features
        self.max_objects = 18
        self.features_per_object = 5  # obj_id, E, pT, eta, phi

    def fit(self, X, y=None):
        # Extract features from tensor
        et_miss = X[:, 0].reshape(-1, 1)  # E_T_miss
        phi_et_miss = X[:, 1].reshape(-1, 1)  # phi_Et_miss
        
        # Initialize arrays to store extracted features
        energies = []
        pts = []
        etas = []
        phis = []
        
        # Extract object features
        for i in range(self.max_objects):
            start_idx = 2 + i * self.features_per_object
            obj_id = X[:, start_idx].reshape(-1, 1)
            
            # Only consider valid objects (obj_id != 0)
            valid_mask = obj_id != 0
            
            # Extract and collect features for valid objects
            if valid_mask.any():
                energy = X[:, start_idx + 1].reshape(-1, 1)
                pt = X[:, start_idx + 2].reshape(-1, 1)
                eta = X[:, start_idx + 3].reshape(-1, 1)
                phi = X[:, start_idx + 4].reshape(-1, 1)
                
                energies.append(energy[valid_mask])
                pts.append(pt[valid_mask])
                etas.append(eta[valid_mask])
                phis.append(phi[valid_mask])
        
        # Fit scalers on non-zero values
        if et_miss.size > 0:
            self.et_miss_scaler.fit(et_miss[et_miss != 0])
        if phi_et_miss.size > 0:
            self.phi_et_miss_scaler.fit(phi_et_miss[phi_et_miss != 0])
        
        # Concatenate all valid features and fit the corresponding scalers
        if energies:
            all_energies = torch.cat(energies, dim=0)
            all_pts = torch.cat(pts, dim=0)
            all_etas = torch.cat(etas, dim=0)
            all_phis = torch.cat(phis, dim=0)
            
            self.energy_scaler.fit(all_energies.numpy())
            self.pt_scaler.fit(all_pts.numpy())
            self.eta_scaler.fit(all_etas.numpy())
            self.phi_scaler.fit(all_phis.numpy())
            
        return self

    def transform(self, X):
        batch_size = X.shape[0]
        
        # Process missing ET and phi
        et_miss = X[:, 0].clone().reshape(-1, 1)
        phi_et_miss = X[:, 1].clone().reshape(-1, 1)
        
        # Scale non-zero values
        non_zero_et = et_miss != 0
        if non_zero_et.any():
            et_miss[non_zero_et] = torch.tensor(self.et_miss_scaler.transform(et_miss[non_zero_et].numpy())).float()
        
        non_zero_phi_et = phi_et_miss != 0
        if non_zero_phi_et.any():
            phi_et_miss[non_zero_phi_et] = torch.tensor(self.phi_et_miss_scaler.transform(phi_et_miss[non_zero_phi_et].numpy())).float()
        
        # Extract and scale object features
        # We'll create a tensor of shape [batch_size, max_objects, features]
        objects_tensor = torch.zeros(batch_size, self.max_objects, 15)  # 15 = 4 original + 11 derived features
        
        for i in range(self.max_objects):
            start_idx = 2 + i * self.features_per_object
            obj_id = X[:, start_idx].reshape(-1, 1)
            energy = X[:, start_idx + 1].clone().reshape(-1, 1)
            pt = X[:, start_idx + 2].clone().reshape(-1, 1)
            eta = X[:, start_idx + 3].clone().reshape(-1, 1)
            phi = X[:, start_idx + 4].clone().reshape(-1, 1)
            
            # Scale non-zero values
            non_zero_energy = energy != 0
            if non_zero_energy.any():
                energy[non_zero_energy] = torch.tensor(self.energy_scaler.transform(energy[non_zero_energy].numpy())).float()
            
            non_zero_pt = pt != 0
            if non_zero_pt.any():
                pt[non_zero_pt] = torch.tensor(self.pt_scaler.transform(pt[non_zero_pt].numpy())).float()
            
            non_zero_eta = eta != 0
            if non_zero_eta.any():
                eta[non_zero_eta] = torch.tensor(self.eta_scaler.transform(eta[non_zero_eta].numpy())).float()
            
            non_zero_phi = phi != 0
            if non_zero_phi.any():
                phi[non_zero_phi] = torch.tensor(self.phi_scaler.transform(phi[non_zero_phi].numpy())).float()
            
            # Calculate physics-informed features
            mask = (obj_id != 0).float()  # Object validity mask
            
            # Calculate transverse energy (ET)
            et = pt.clone()
            
            # Calculate pz (longitudinal momentum)
            pz = pt * torch.sinh(eta) * mask
            
            # Calculate px, py using pt and phi
            px = pt * torch.cos(phi) * mask
            py = pt * torch.sin(phi) * mask
            
            # E/pT ratio (related to mass)
            e_pt_ratio = (energy / (pt + 1e-6)) * mask
            
            # ET/pT ratio
            et_pt_ratio = (et / (pt + 1e-6)) * mask
            
            # Calculate delta phi relative to missing ET
            delta_phi = (phi - phi_et_miss) * mask
            # Normalize to [-π, π]
            delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))
            
            # Object id one-hot encoding as feature
            # Here we assume obj_id takes values 0-5 (jets, leptons, etc)
            obj_type = obj_id.clone()
            obj_type_onehot = F.one_hot(obj_type.long().clamp(0, 5).squeeze(-1), num_classes=6).float()
            
            # Stack features: [original_features, derived_features]
            org_features = torch.cat([energy, pt, eta, phi], dim=1)
            derived_features = torch.cat([pz, px, py, et, e_pt_ratio, et_pt_ratio, delta_phi], dim=1)
            obj_features = torch.cat([org_features, derived_features, obj_type_onehot], dim=1)
            
            # Store in tensor
            objects_tensor[:, i, :obj_features.shape[1]] = obj_features
        
        # Global event features
        event_features = torch.cat([et_miss, phi_et_miss], dim=1)  # [batch_size, 2]
        
        # Reshape objects tensor to [batch_size, max_objects * features]
        flat_objects = objects_tensor.reshape(batch_size, -1)  # [batch_size, max_objects * 15]
        
        # Combine event features with object features
        final_features = torch.cat([event_features, flat_objects], dim=1)  # [batch_size, 2 + max_objects * 15]
        
        return final_features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        # x shape: [batch_size, seq_len, embed_dim]
        
        # Linear projections
        query = self.query(x).reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.key(x).reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.value(x).reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # query, key, value shape: [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention weights
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores shape: [batch_size, num_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            # Mask shape: [batch_size, seq_len]
            # Expand mask to [batch_size, 1, 1, seq_len]  
            expanded_mask = mask.unsqueeze(1).unsqueeze(1)            
            # Apply mask by setting attention scores to a large negative value
            scores = scores.masked_fill(expanded_mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights shape: [batch_size, num_heads, seq_len, seq_len]
        
        # Apply attention to values
        context = torch.matmul(attn_weights, value)  # [batch_size, num_heads, seq_len, head_dim]
        context = context.permute(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim)
        # context shape: [batch_size, seq_len, embed_dim]
        
        # Output projection
        output = self.proj(context)  # [batch_size, seq_len, embed_dim]
        
        return output

class SlotAttention(nn.Module):
    def __init__(self, num_slots, hidden_dim, num_iterations=3, slot_dim=None):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.slot_dim = hidden_dim if slot_dim is None else slot_dim
        
        # Initialize slots parameters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.slot_dim))
        self.slots_sigma = nn.Parameter(torch.ones(1, 1, self.slot_dim))
        
        # Slot attention specific layers
        self.norm_inputs = nn.LayerNorm(hidden_dim)
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.norm_mlp = nn.LayerNorm(self.slot_dim)
        
        # Linear projections for attention
        self.k_proj = nn.Linear(hidden_dim, self.slot_dim)
        self.q_proj = nn.Linear(self.slot_dim, self.slot_dim)
        self.v_proj = nn.Linear(hidden_dim, self.slot_dim)
        
        # MLP for position-wise feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, self.slot_dim * 2),
            nn.ReLU(),
            nn.Linear(self.slot_dim * 2, self.slot_dim)
        )
        
        # GRU for slot updates
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        
    def forward(self, inputs, mask=None):
        batch_size = inputs.shape[0]
        num_inputs = inputs.shape[1]
        inputs = self.norm_inputs(inputs)  # [batch_size, num_inputs, hidden_dim]
        
        # Initialize slots
        slots = self.slots_mu + self.slots_sigma * torch.randn(
            batch_size, self.num_slots, self.slot_dim, device=inputs.device)
        
        # Multiple rounds of attention
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Compute attention weights
            queries = self.q_proj(slots)  # [batch_size, num_slots, slot_dim]
            keys = self.k_proj(inputs)  # [batch_size, num_inputs, slot_dim]
            values = self.v_proj(inputs)  # [batch_size, num_inputs, slot_dim]
            
            # Compute attention scores and handle masked inputs if needed
            attn_logits = torch.matmul(queries, keys.transpose(-1, -2))  # [batch_size, num_slots, num_inputs]
            attn_logits = attn_logits / math.sqrt(self.slot_dim)
            
            # Apply mask if provided
            if mask is not None:  # mask shape: [batch_size, num_inputs]
                expanded_mask = mask.unsqueeze(1)  # [batch_size, 1, num_inputs]
                attn_logits = attn_logits.masked_fill(expanded_mask == 0, -1e9)
            
            # Normalize attention weights
            attn = F.softmax(attn_logits, dim=-1)  # [batch_size, num_slots, num_inputs]
            
            # Normalize attention weights along the slots dimension
            attn_slot_norm = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)  # [batch_size, num_slots, num_inputs]
            
            # Apply attention to input values
            updates = torch.matmul(attn_slot_norm, values)  # [batch_size, num_slots, slot_dim]
            
            # Update slots with GRU
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            ).reshape(batch_size, self.num_slots, self.slot_dim)
            
            # Apply MLP
            slots = slots + self.mlp(self.norm_mlp(slots))
            
        return slots, attn  # [batch_size, num_slots, slot_dim], [batch_size, num_slots, num_inputs]

class TopQuarkTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_slots=4, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.global_event_dim = 2  # ET_miss and phi_ET_miss
        self.max_objects = 18
        
        # Input embedding for objects
        self.embed_dim = 64
        self.object_embedding = nn.Linear(15, self.embed_dim)  # 15 features per object from preprocessor
        
        # Event feature embedding
        self.event_embedding = nn.Linear(self.global_event_dim, hidden_dim)
        
        # Transformer layers for object processing
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Slot Attention for grouping objects into top quark candidates
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            hidden_dim=self.embed_dim,
            num_iterations=3
        )
        
        # Combine slot representations
        self.slot_mlp = nn.Sequential(
            nn.Linear(num_slots * self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract global event features (ET_miss, phi_ET_miss)
        global_features = x[:, :self.global_event_dim]  # [batch_size, 2]
        
        # Process global features
        global_embedding = self.event_embedding(global_features)  # [batch_size, hidden_dim]
        
        # Extract and reshape object features
        object_features = x[:, self.global_event_dim:].reshape(batch_size, self.max_objects, -1)
        # Now object_features has shape [batch_size, max_objects, 15]
        
        # Create object mask (1 for valid objects, 0 for padding)
        # We use the energy feature (index 0) to determine validity
        obj_mask = (object_features[:, :, 0] != 0).float()  # [batch_size, max_objects]
        
        # Embed each object
        object_embeddings = self.object_embedding(object_features)  # [batch_size, max_objects, embed_dim]
        
        # Apply transformer encoder layers
        x_obj = object_embeddings.transpose(0, 1)  # [max_objects, batch_size, embed_dim]
        for layer in self.transformer_layers:
            x_obj = layer(x_obj)
        x_obj = x_obj.transpose(0, 1)  # [batch_size, max_objects, embed_dim]
        
        # Apply slot attention to group objects
        slots, attention = self.slot_attention(x_obj, obj_mask)  # [batch_size, num_slots, embed_dim]
        
        # Flatten slots for further processing
        flat_slots = slots.reshape(batch_size, -1)  # [batch_size, num_slots * embed_dim]
        slot_features = self.slot_mlp(flat_slots)  # [batch_size, hidden_dim]
        
        # Combine global and slot features
        combined_features = torch.cat([global_embedding, slot_features], dim=1)  # [batch_size, hidden_dim*2]
        
        # Final classification
        output = self.classifier(combined_features).squeeze(-1)  # [batch_size]
        
        return output

def make_model(input_dim: int):
    # For the top quark transformer, we'll use a model that processes both
    # global event features and individual objects with slot attention
    model = TopQuarkTransformer(input_dim=input_dim)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Initialize lists to track metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs >= 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets.float())
                
                # Update metrics
                running_loss += loss.item() * inputs.size(0)
                predicted = (outputs >= 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # Calculate validation metrics
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}')
    
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

