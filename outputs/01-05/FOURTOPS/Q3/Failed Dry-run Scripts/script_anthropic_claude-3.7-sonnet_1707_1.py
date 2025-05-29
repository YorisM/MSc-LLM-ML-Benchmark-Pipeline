# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

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
        # Register normalization statistics
        self.register_buffer("mean", kwargs.get("mean", torch.zeros(1)))
        self.register_buffer("std", kwargs.get("std", torch.ones(1)))
        
        # Feature masks to extract different parts of the input
        self.register_buffer("etmiss_mask", kwargs.get("etmiss_mask", torch.zeros(105, dtype=torch.bool)))
        self.register_buffer("particle_mask", kwargs.get("particle_mask", torch.zeros(105, dtype=torch.bool)))
        
        # Constants for physics-inspired features
        self.register_buffer("top_mass", torch.tensor(173000.0))  # Top quark mass in MeV
        self.register_buffer("w_mass", torch.tensor(80420.0))     # W boson mass in MeV
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Extract ETmiss and phi components (first 2 columns)
        etmiss = x[:, :2]
        
        # Reshape particle features to parse them correctly
        # Format: obj_id, E, pT, eta, phi - repeating every 5 columns after the first 2
        particle_data = x[:, 2:]
        particle_features = []
        
        # Process in groups of 5 features (obj_id, E, pT, eta, phi)
        for i in range(0, particle_data.shape[1], 5):
            if i+5 <= particle_data.shape[1]:
                group = particle_data[:, i:i+5]
                # Only include particles with non-zero values (filtering padding)
                valid_mask = torch.sum(group, dim=1) != 0
                
                if valid_mask.any():
                    # Normalize the basic features
                    obj_id = group[:, 0:1]  # Keep object ID as is
                    normalized_features = (group[:, 1:] - self.mean) / (self.std + 1e-8)
                    
                    # Extract physics components
                    E = group[:, 1]   # Energy
                    pT = group[:, 2]  # Transverse momentum
                    eta = group[:, 3] # Pseudorapidity
                    phi = group[:, 4] # Azimuthal angle
                    
                    # Calculate physics-inspired features
                    # Transverse energy
                    ET = pT.clone()  # For hadrons, ET ~= pT
                    
                    # Mass estimation using E and pT
                    px = pT * torch.cos(phi)
                    py = pT * torch.sin(phi)
                    pz = pT * torch.sinh(eta)
                    m2 = E*E - px*px - py*py - pz*pz
                    mass = torch.sqrt(torch.clamp(m2, min=0.0))
                    
                    # Distance from top mass for particle groups
                    mass_diff_top = torch.abs(mass - self.top_mass) / self.top_mass
                    
                    # Distance from W mass for particle groups
                    mass_diff_w = torch.abs(mass - self.w_mass) / self.w_mass
                    
                    # Create augmented feature vector
                    augmented_features = torch.cat([
                        obj_id,                    # Object ID
                        normalized_features,       # Normalized basic features
                        ET.unsqueeze(1),          # Transverse energy
                        mass.unsqueeze(1),        # Estimated mass
                        mass_diff_top.unsqueeze(1), # Distance from top mass
                        mass_diff_w.unsqueeze(1),   # Distance from W mass
                    ], dim=1)
                    
                    particle_features.append(augmented_features)
        
        # Stack all valid particles
        if particle_features:
            particles_tensor = torch.cat(particle_features, dim=1)
            # Reshape to [batch_size, num_particles, features_per_particle]
            num_features = particle_features[0].shape[1]
            max_particles = particles_tensor.shape[1] // num_features
            particles_tensor = particles_tensor.reshape(batch_size, max_particles, num_features)
        else:
            # If no valid particles, create empty tensor with proper dimensions
            num_features = 10  # obj_id + 4 original + 5 derived
            particles_tensor = torch.zeros((batch_size, 1, num_features), device=x.device)
        
        # Normalize ET_miss
        etmiss_normalized = (etmiss - self.mean[:2]) / (self.std[:2] + 1e-8)
        
        # Return dictionary of processed features
        return {
            'etmiss': etmiss_normalized,
            'particles': particles_tensor,
        }

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=64):
    # Calculate statistics for normalization (excluding object IDs and zeros from padding)
    # Extract all non-zero values for energy, pT, eta, and phi
    non_zero_mask = X_train[:, 2:] != 0.0
    
    # Group features into obj_id, E, pT, eta, phi
    feature_values = []
    
    # Skip ETmiss first 2 columns
    for i in range(2, X_train.shape[1], 5):
        # Skip obj_id column
        if i+4 < X_train.shape[1]:  # Check if we have all 4 features
            # Add E, pT, eta, phi to our feature list
            feature_values.append(X_train[:, i+1:i+5].reshape(-1)[non_zero_mask[:, i-1:i+3].reshape(-1)])
    
    # Combine all non-zero feature values
    if feature_values:
        all_features = torch.cat(feature_values)
        mean = all_features.mean()
        std = all_features.std()
    else:
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)
    
    # Create masks for different parts of the data
    etmiss_mask = torch.zeros(X_train.shape[1], dtype=torch.bool)
    etmiss_mask[:2] = True  # First two columns are ETmiss and phi_ETmiss
    
    particle_mask = torch.zeros(X_train.shape[1], dtype=torch.bool)
    particle_mask[2:] = True  # The rest are particle features
    
    # Initialize preprocessor with computed statistics
    preproc = PreprocessModule(
        mean=mean,
        std=std,
        etmiss_mask=etmiss_mask,
        particle_mask=particle_mask
    )

    # Preprocess data
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, preproc

# Slot Attention mechanism
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        
        hidden_dim = max(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, inputs, mask=None):
        b, n, d = inputs.shape
        slots = self.slots_mu.expand(b, self.num_slots, -1) + self.slots_sigma.expand(b, self.num_slots, -1) * torch.randn(b, self.num_slots, d, device=inputs.device)
        inputs = self.norm_input(inputs)
        
        # Multiple rounds of attention
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention
            q = self.to_q(slots)
            k = self.to_k(inputs)
            v = self.to_v(inputs)
            
            # Dot product attention
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            
            # Apply mask if provided (for ignoring padding)
            if mask is not None:
                mask = mask.unsqueeze(1)
                dots.masked_fill_(~mask, -1e9)
                
            # Softmax attention weights
            attn = dots.softmax(dim=2) # [B, num_slots, N]
            attn_weighted_inputs = torch.einsum('bij,bjd->bid', attn, v)
            
            # Update slots
            slots = self.gru(attn_weighted_inputs.reshape(-1, d), slots_prev.reshape(-1, d))
            slots = slots.reshape(b, self.num_slots, d)
            slots = slots + self.mlp(self.norm_mlp(slots))
            
        return slots

# Multi-head self-attention module
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(x.shape[0], -1, self.heads, x.shape[-1] // self.heads).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, seq_len]
            dots = dots.masked_fill(~mask, -1e9)

        attn = self.attend(dots)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(x.shape[0], -1, x.shape[-1])
        return self.to_out(out)

# Feed-forward network for Transformer blocks
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Transformer encoder block
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ff(self.norm2(x))
        return x

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_slots=8, num_transformer_layers=4, num_heads=8):
        super(Classifier, self).__init__()
        
        # Define dimensions
        self.particle_embed_dim = 64   # Dimension for particle embeddings
        self.etmiss_embed_dim = 32     # Dimension for ETmiss embeddings
        self.slot_dim = 128            # Dimension for slot attention
        self.num_slots = num_slots     # Number of slots for slot attention
        
        # Embeddings for particles
        self.particle_embedding = nn.Sequential(
            nn.Linear(10, 128),  # 10 features per particle from preprocessor
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, self.particle_embed_dim)
        )
        
        # Embedding for ETmiss
        self.etmiss_embedding = nn.Sequential(
            nn.Linear(2, self.etmiss_embed_dim),
            nn.LayerNorm(self.etmiss_embed_dim),
            nn.ReLU()
        )
        
        # Slot attention to group particles by physical meaning
        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=self.particle_embed_dim,
            iters=3,
            hidden_dim=hidden_dim
        )
        
        # Transformer layers to process slot representations
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                dim=self.slot_dim,
                heads=num_heads,
                dim_head=self.slot_dim // num_heads,
                mlp_dim=hidden_dim,
                dropout=0.1
            ) for _ in range(num_transformer_layers)
        ])
        
        # Combine ETmiss with slot features
        self.combined_dim = self.slot_dim * self.num_slots + self.etmiss_embed_dim
        
        # Project slots to match required dimensions
        self.project_slots = nn.Linear(self.particle_embed_dim, self.slot_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # Extract features from preprocessed input
        etmiss = x['etmiss']  # [batch_size, 2]
        particles = x['particles']  # [batch_size, num_particles, 10]
        
        batch_size = particles.shape[0]
        
        # Create mask for valid particles (non-zero)
        particle_mask = torch.sum(particles, dim=-1) != 0
        
        # Embed particles
        particle_embeddings = self.particle_embedding(particles)  # [batch_size, num_particles, particle_embed_dim]
        
        # Apply slot attention to group particles
        slots = self.slot_attention(particle_embeddings, mask=particle_mask)  # [batch_size, num_slots, particle_embed_dim]
        
        # Project slots to match transformer dimensions
        slots = self.project_slots(slots)  # [batch_size, num_slots, slot_dim]
        
        # Apply transformer layers to process slot representations
        for transformer_layer in self.transformer_layers:
            slots = transformer_layer(slots)
        
        # Embed ETmiss
        etmiss_embedding = self.etmiss_embedding(etmiss)  # [batch_size, etmiss_embed_dim]
        
        # Flatten slots and concatenate with ETmiss
        slots_flat = slots.reshape(batch_size, -1)  # [batch_size, num_slots * slot_dim]
        combined = torch.cat([slots_flat, etmiss_embedding], dim=1)  # [batch_size, combined_dim]
        
        # Classification
        logits = self.classifier(combined).squeeze(-1)  # [batch_size]
        
        return logits

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Binary cross entropy with logits as loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # For storing metrics
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_val_auc = 0.0
    
    # Preprocessing component
    preprocessor = train_loader.dataset.tensors[0][0].device
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_preds = []
        epoch_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            
            # Process inputs through preprocessor
            processed_inputs = model.preprocess(inputs) if hasattr(model, 'preprocess') else inputs
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(processed_inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item() * inputs.size(0)
            epoch_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            epoch_targets.append(targets.cpu().numpy())
        
        # Compute epoch metrics
        epoch_loss /= len(train_loader.dataset)
        epoch_preds = np.concatenate(epoch_preds)
        epoch_targets = np.concatenate(epoch_targets)
        epoch_acc = accuracy_score(epoch_targets > 0.5, epoch_preds > 0.5)
        epoch_auc = roc_auc_score(epoch_targets, epoch_preds)
        
        training_loss.append(epoch_loss)
        training_acc.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                
                # Process inputs through preprocessor
                processed_inputs = model.preprocess(inputs) if hasattr(model, 'preprocess') else inputs
                
                # Forward pass
                outputs = model(processed_inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Accumulate metrics
                val_loss += loss.item() * inputs.size(0)
                val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        # Compute validation metrics
        val_loss /= len(val_loader.dataset)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_acc = accuracy_score(val_targets > 0.5, val_preds > 0.5)
        val_auc = roc_auc_score(val_targets, val_preds)
        
        validation_loss.append(val_loss)
        validation_acc.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Train AUC: {epoch_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        print("-" * 50)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
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

    # Set batch size based on available memory
    batch_size = 64 if not dryrun else 16

    # Preprocessing
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=batch_size)

    # Model Initialization
    # We'll pass a dummy batch through the preprocessor to get the right dimensions
    sample_X, _ = next(iter(train_loader))
    processed_sample = preproc(sample_X)
    
    # Get the dimensionality of processed input
    # In this case, processed_sample is a dictionary
    particle_dim = processed_sample['particles'].shape[2]  # Number of features per particle
    
    # Create the classifier with physics-informed architecture
    model = Classifier(input_dim=particle_dim)
    
    # Attach preprocessor to model for convenience during training
    model.preprocess = preproc

    # Training
    epochs = 1 if dryrun else 10

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