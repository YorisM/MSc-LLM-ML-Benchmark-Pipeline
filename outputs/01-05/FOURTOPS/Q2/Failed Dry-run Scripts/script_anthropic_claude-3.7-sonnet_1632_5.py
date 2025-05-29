# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import math
from typing import List, Tuple, Dict
from torch.nn import functional as F

# ----- FIXED SECTION: Data Loading -----
def load_data():
    X_train_df = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv')
    Y_train_df = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv')
    X_val_df   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv')
    Y_val_df   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv')

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
    X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
    Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()
    return X_train, Y_train, X_val, Y_val

# ----- FREE SECTION: Data Preprocessing -----
class PreprocessModule(torch.nn.Module):
    # TorchScript-compatible module applying pre-fitted transformations.
    def __init__(self, **kwargs):
        super().__init__()
        # Store the normalization statistics
        if "mean" in kwargs and "std" in kwargs:
            self.register_buffer("mean", kwargs["mean"])
            self.register_buffer("std", kwargs["std"])
            self.normalize = True
        else:
            self.normalize = False
            
        # Store the mask for non-zero entries
        if "mask" in kwargs:
            self.register_buffer("mask", kwargs["mask"])
        else:
            self.register_buffer("mask", torch.ones(105, dtype=torch.bool))
            
        # Store object indices for reshaping
        if "obj_indices" in kwargs:
            self.register_buffer("obj_indices", kwargs["obj_indices"])
            self.has_obj_indices = True
        else:
            self.has_obj_indices = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply normalization if available
        if self.normalize:
            x = (x - self.mean) / (self.std + 1e-8)
            
        # Replace NaNs and infinities with zeros
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply mask to keep only relevant features
        if torch.any(~self.mask):
            x = x[:, self.mask]
            
        # Reshape into particle features if object indices are available
        if self.has_obj_indices:
            batch_size = x.shape[0]
            # Extract missing energy features (first 2 columns)
            et_miss = x[:, :2]  # E_T_miss and phi
            
            # Extract particle features and reshape for message passing
            objects = []
            for i in range(len(self.obj_indices) - 1):
                start_idx = self.obj_indices[i]
                end_idx = self.obj_indices[i+1]
                if start_idx < x.shape[1] and end_idx <= x.shape[1] and end_idx - start_idx == 5:
                    obj_features = x[:, start_idx:end_idx]
                    objects.append(obj_features)
            
            if objects:  # Check if there are valid objects
                # Stack objects into a tensor of shape [batch_size, n_objects, 5]
                particles = torch.stack(objects, dim=1)
                
                # Calculate mask for valid particles (non-zero energy)
                valid_mask = particles[:, :, 1] != 0  # Energy feature
                
                # Combine with original tensor
                x = torch.cat([et_miss.unsqueeze(1), particles, valid_mask.unsqueeze(2).float()], dim=2)
            
        return x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=128):
    # Extract object indices
    obj_indices = []
    for i in range(2, X_train.shape[1], 5):  # Start from 2 (after ET_miss and phi)
        if i < X_train.shape[1]:
            obj_indices.append(i)
    if len(obj_indices) > 0:  # Add the end index
        obj_indices.append(min(obj_indices[-1] + 5, X_train.shape[1]))
    
    # Calculate statistics for normalization
    # Use robust statistics for non-zero values
    mask = (X_train != 0).sum(0) > 0
    masked_data = X_train[:, mask]
    mean = masked_data.mean(0)
    std = masked_data.std(0)
    
    # Create preprocessor with the calculated statistics
    preproc = PreprocessModule(
        mean=mean, 
        std=std, 
        mask=mask,
        obj_indices=torch.tensor(obj_indices)
    )

    # Apply preprocessing
    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class ParticleAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ParticleAttention, self).__init__()
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        
    def forward(self, x, mask=None):
        # x: [batch_size, num_particles, features]
        q = self.query_proj(x)  # [batch_size, num_particles, hidden_dim]
        k = self.key_proj(x)    # [batch_size, num_particles, hidden_dim]
        v = self.value_proj(x)  # [batch_size, num_particles, hidden_dim]
        
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [batch_size, num_particles, num_particles]
        
        # Apply mask if provided
        if mask is not None:
            # Create 2D attention mask [batch_size, num_particles, num_particles]
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
            scores = scores.masked_fill(~mask_2d, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_particles, num_particles]
        
        # Apply attention weights to values
        output = torch.bmm(attn_weights, v)  # [batch_size, num_particles, hidden_dim]
        
        return output

class LorentzLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(LorentzLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        
    def forward(self, x):
        # x contains 4-vectors: [batch_size, num_particles, 4+features]
        batch_size, num_particles = x.shape[:2]
        
        # Extract 4-vectors (E, px, py, pz)
        four_vectors = x[:, :, :4]
        
        # Compute Lorentz invariant quantities
        # 1. Invariant mass: m^2 = E^2 - p^2
        E = four_vectors[:, :, 0]  # Energy
        p_mag_squared = torch.sum(four_vectors[:, :, 1:4]**2, dim=-1)  # |p|^2
        m_squared = E**2 - p_mag_squared
        
        # 2. Compute pairwise dot products between 4-vectors (Minkowski metric)
        # For each pair of particles i,j: p_i · p_j = E_i*E_j - px_i*px_j - py_i*py_j - pz_i*pz_j
        dot_products = torch.zeros(batch_size, num_particles, num_particles, device=x.device)
        
        for i in range(num_particles):
            for j in range(num_particles):
                # Minkowski inner product with (-+++) signature
                dot_products[:, i, j] = (four_vectors[:, i, 0] * four_vectors[:, j, 0] - 
                                        torch.sum(four_vectors[:, i, 1:4] * four_vectors[:, j, 1:4], dim=-1))
        
        # 3. Compute transverse momentum and pseudorapidity
        pt = torch.sqrt(four_vectors[:, :, 1]**2 + four_vectors[:, :, 2]**2)  # √(px^2 + py^2)
        eta = torch.zeros_like(pt)
        non_zero_mask = pt > 0
        pz = four_vectors[:, :, 3]
        p = torch.sqrt(p_mag_squared)
        
        # Safe computation of pseudorapidity
        safe_vals = pz[non_zero_mask] / (p[non_zero_mask] + 1e-8)
        safe_vals = torch.clamp(safe_vals, -0.99999, 0.99999)  # Avoid exact ±1
        eta[non_zero_mask] = 0.5 * torch.log((1 + safe_vals) / (1 - safe_vals + 1e-8))
        
        # Compute delta R between all pairs
        phi = torch.atan2(four_vectors[:, :, 2], four_vectors[:, :, 1])  # arctan(py/px)
        delta_r = torch.zeros_like(dot_products)
        
        for i in range(num_particles):
            for j in range(num_particles):
                if i != j:
                    delta_phi = torch.abs(phi[:, i] - phi[:, j])
                    # Ensure delta_phi is between 0 and π
                    delta_phi = torch.where(delta_phi > math.pi, 2*math.pi - delta_phi, delta_phi)
                    delta_eta = torch.abs(eta[:, i] - eta[:, j])
                    delta_r[:, i, j] = torch.sqrt(delta_eta**2 + delta_phi**2)
        
        # Combine all Lorentz invariant features
        invariant_features = torch.cat([
            m_squared.unsqueeze(-1),             # Invariant mass squared
            pt.unsqueeze(-1),                    # Transverse momentum
            eta.unsqueeze(-1),                   # Pseudorapidity
            phi.unsqueeze(-1),                   # Azimuthal angle
            torch.sum(dot_products, dim=-1).unsqueeze(-1),  # Sum of Minkowski dot products
            torch.mean(delta_r, dim=-1).unsqueeze(-1)       # Mean delta R
        ], dim=-1)
        
        # Apply linear transformation
        transformed = F.linear(invariant_features, self.W, self.b)
        transformed = F.relu(transformed)
        
        return transformed

class LorentzEquivariantLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LorentzEquivariantLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Transformations for message passing
        self.msg_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Lorentz-specific layer
        self.lorentz_layer = LorentzLayer(hidden_dim)
        
    def forward(self, x, mask=None):
        batch_size, num_particles, feature_dim = x.shape
        
        # Create messages between all pairs of particles
        src_features = x.unsqueeze(2).expand(-1, -1, num_particles, -1)  # [batch, num_p, num_p, feat]
        dst_features = x.unsqueeze(1).expand(-1, num_particles, -1, -1)  # [batch, num_p, num_p, feat]
        pair_features = torch.cat([src_features, dst_features], dim=-1)  # [batch, num_p, num_p, 2*feat]
        
        # Process pair features through MLP
        flattened_pairs = pair_features.reshape(batch_size * num_particles * num_particles, -1)
        messages = self.msg_mlp(flattened_pairs)
        messages = messages.reshape(batch_size, num_particles, num_particles, self.hidden_dim)
        
        # Apply mask if available
        if mask is not None:
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)  # [batch, num_p, num_p]
            mask_3d = mask_2d.unsqueeze(3).expand(-1, -1, -1, self.hidden_dim)
            messages = messages * mask_3d.float()
        
        # Aggregate messages
        aggregated_messages = messages.sum(dim=2)  # [batch, num_p, hidden_dim]
        
        # Process through Lorentz layer
        lorentz_features = self.lorentz_layer(x)
        
        # Update node features
        combined = torch.cat([x, aggregated_messages], dim=-1)  # [batch, num_p, feat+hidden_dim]
        updates = self.update_mlp(combined)
        new_features = x + updates + lorentz_features
        
        return new_features

class GlobalAttentionPool(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(GlobalAttentionPool, self).__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, mask=None):
        # x: [batch_size, num_nodes, features]
        scores = self.att_mlp(x)  # [batch_size, num_nodes, 1]
        
        # Apply mask if available
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
            
        weights = torch.softmax(scores, dim=1)  # [batch_size, num_nodes, 1]
        
        # Weighted sum
        x_weighted = x * weights
        pooled = x_weighted.sum(dim=1)  # [batch_size, features]
        
        return pooled

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.preprocessing = nn.Linear(input_dim, 128)
        
        # Define dimensions
        self.hidden_dim = 64
        self.num_layers = 3
        
        # Message passing layers with Lorentz equivariance
        self.mp_layers = nn.ModuleList([LorentzEquivariantLayer(128, self.hidden_dim) for _ in range(self.num_layers)])
        
        # Attention mechanism between particles
        self.attention = ParticleAttention(128, self.hidden_dim)
        
        # Global pooling with attention
        self.global_pool = GlobalAttentionPool(128, self.hidden_dim)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # Extract particle information from the input tensor
        batch_size = x.shape[0]
        
        # For flat input: reshape to [batch_size, num_particles, features]
        if len(x.shape) == 2:
            # Assume the first two elements are ET_miss and phi
            et_miss = x[:, :2]  # [batch_size, 2]
            
            # Reshape the rest of the tensor to have particle objects
            particles = x[:, 2:].reshape(batch_size, -1, 5)  # [batch_size, num_particles, 5]
            
            # Create a mask for valid particles (non-zero energy)
            mask = particles[:, :, 1] != 0  # Energy feature
            
            # Combine ET_miss with particle features
            et_miss_expanded = et_miss.unsqueeze(1).expand(-1, particles.shape[1], -1)
            x = torch.cat([particles, et_miss_expanded], dim=2)  # [batch_size, num_particles, 5+2]
        else:
            # Extract mask from the last feature dimension if it exists
            if x.shape[2] > 6:  # ET_miss(2) + particles(4) + mask(1)
                mask = x[:, :, -1].bool()  # [batch_size, num_particles]
                x = x[:, :, :-1]  # Remove the mask from input
            else:
                # Create a default mask where all particles are valid
                mask = torch.ones(batch_size, x.shape[1], dtype=torch.bool, device=x.device)
        
        # Process through preprocessing layer
        x = self.preprocessing(x)  # [batch_size, num_particles, hidden_dim]
        
        # Apply message passing with Lorentz equivariance
        for mp_layer in self.mp_layers:
            x = mp_layer(x, mask)
        
        # Apply attention mechanism
        x = x + self.attention(x, mask)  # Residual connection
        
        # Global pooling with attention
        x = self.global_pool(x, mask)  # [batch_size, hidden_dim]
        
        # Final classification
        x = self.classifier(x)  # [batch_size, 1]
        
        return x.squeeze(-1)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_auc = 0.0
    best_model = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_preds = []
        epoch_targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            epoch_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            epoch_targets.extend(batch_y.cpu().numpy())
        
        train_loss = epoch_loss / len(train_loader.dataset)
        train_acc = accuracy_score(epoch_targets, np.round(epoch_preds))
        train_auc = roc_auc_score(epoch_targets, epoch_preds)
        training_loss.append(train_loss)
        training_acc.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.float())
                
                val_loss += loss.item() * batch_X.size(0)
                val_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, np.round(val_preds))
        val_auc = roc_auc_score(val_targets, val_preds)
        validation_loss.append(val_loss)
        validation_acc.append(val_acc)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
        print(f"Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.4f}, Valid AUC: {val_auc:.4f}")
        print("-" * 50)
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model, training_loss, validation_loss, training_acc, validation_acc

# ----- FIXED SECTION: Plotting and Saving Outputs -----
def plot_and_save(metric_train, metric_val, metric_name, filename):
    plt.figure()
    plt.plot(metric_train, label=f'Training {metric_name}')
    plt.plot(metric_val, label=f'Validation {metric_name}')
    plt.title(f'{metric_name} per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# ----- FIXED SECTION: Main Function -----
def main(dryrun=False):
    # Data Loading
    X_train, Y_train, X_val, Y_val = load_data()

    # Preprocessing
    batch_size = 64 if not dryrun else 128
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=batch_size)

    # Model Initialization
    sample_X, _ = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[-1])

    # Training
    epochs = 1 if dryrun else 15

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)

    if not dryrun:
        # determine base name & script directory
        base       = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.makedirs(script_dir, exist_ok=True)

        # save model
        model_path = os.path.join(script_dir, f"{base}_model.pth")
        torch.save(trained_model.state_dict(), model_path)

        # save scripted model
        scripted_path = os.path.join(script_dir, f"{base}_scripted.pt")
        torch.jit.script(trained_model).save(scripted_path)

        # save preprocessor
        scripted_preproc = torch.jit.script(preproc)
        scripted_preproc.save(os.path.join(script_dir, f"{base}_preproc.pt"))

        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, f"Loss - {base}", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, f"Accuracy - {base}", os.path.join(script_dir, f"{base}_accuracy.png"))

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)