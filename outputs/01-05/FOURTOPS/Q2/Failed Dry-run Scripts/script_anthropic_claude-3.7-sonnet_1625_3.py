# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
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
        # Register normalization constants
        for name, tensor in kwargs.items():
            self.register_buffer(name, tensor)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get missing energy components (first two columns)
        e_t_miss = x[:, 0:1]  # Missing transverse energy magnitude
        phi_miss = x[:, 1:2]  # Missing transverse energy azimuthal angle
        
        # Calculate missing energy x and y components
        e_t_miss_x = e_t_miss * torch.cos(phi_miss)
        e_t_miss_y = e_t_miss * torch.sin(phi_miss)
        
        # Extract particles' data from the input
        batch_size = x.size(0)
        n_features = (x.size(1) - 2) // 5  # Number of objects (excluding E_T_miss and phi)
        
        # Prepare output with transformed features
        transformed = torch.zeros((batch_size, n_features * 8 + 3), device=x.device)
        
        # Add missing energy information as normalized features
        transformed[:, 0] = (e_t_miss.squeeze() - self.et_miss_mean) / self.et_miss_std
        transformed[:, 1] = e_t_miss_x.squeeze() / self.et_miss_std
        transformed[:, 2] = e_t_miss_y.squeeze() / self.et_miss_std
        
        # Process each particle
        for i in range(n_features):
            base_idx = 2 + i * 5  # Starting index in the original data
            obj_type = x[:, base_idx]  # Object type
            energy = x[:, base_idx + 1]  # Energy
            pt = x[:, base_idx + 2]      # Transverse momentum
            eta = x[:, base_idx + 3]     # Pseudo-rapidity
            phi = x[:, base_idx + 4]     # Azimuthal angle
            
            # Skip if this is padding (zero values)
            mask = (obj_type != 0).float()
            
            # Calculate px, py, pz components
            px = pt * torch.cos(phi)
            py = pt * torch.sin(phi)
            pz = pt * torch.sinh(eta)
            
            # Create a feature mask to indicate presence of particles
            out_idx = 3 + i * 8  # Starting index in the output
            
            # Store normalized features
            transformed[:, out_idx] = mask
            transformed[:, out_idx + 1] = (obj_type * mask) / 10  # Normalize object type
            transformed[:, out_idx + 2] = (energy * mask - self.energy_mean) / self.energy_std
            transformed[:, out_idx + 3] = (pt * mask - self.pt_mean) / self.pt_std
            transformed[:, out_idx + 4] = eta * mask / 3.0  # Normalize eta to approximately [-1, 1]
            transformed[:, out_idx + 5] = torch.cos(phi) * mask  # Encode phi as sin/cos for circular continuity
            transformed[:, out_idx + 6] = torch.sin(phi) * mask
            transformed[:, out_idx + 7] = (pz * mask - self.pz_mean) / self.pz_std
            
        return transformed

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=512):
    # Calculate statistics for normalization from training data
    et_miss = X_train[:, 0]
    et_miss_mean = et_miss[et_miss > 0].mean()
    et_miss_std = et_miss[et_miss > 0].std()
    
    # Process particles
    n_features = (X_train.size(1) - 2) // 5
    all_energy = []
    all_pt = []
    all_pz = []
    
    for i in range(n_features):
        base_idx = 2 + i * 5
        obj_type = X_train[:, base_idx]
        mask = obj_type != 0  # Not padding
        
        energy = X_train[:, base_idx + 1][mask]
        pt = X_train[:, base_idx + 2][mask]
        eta = X_train[:, base_idx + 3][mask]
        
        # Calculate pz
        pz = pt * torch.sinh(eta)
        
        all_energy.append(energy)
        all_pt.append(pt)
        all_pz.append(pz)
    
    # Combine and calculate statistics
    all_energy = torch.cat(all_energy)
    all_pt = torch.cat(all_pt)
    all_pz = torch.cat(all_pz)
    
    energy_mean = all_energy.mean()
    energy_std = all_energy.std()
    pt_mean = all_pt.mean()
    pt_std = all_pt.std()
    pz_mean = all_pz.mean()
    pz_std = all_pz.std()
    
    # Create the preprocessor with statistics
    preproc = PreprocessModule(
        et_miss_mean=et_miss_mean,
        et_miss_std=et_miss_std,
        energy_mean=energy_mean,
        energy_std=energy_std,
        pt_mean=pt_mean,
        pt_std=pt_std,
        pz_mean=pz_mean,
        pz_std=pz_std
    )
    
    # Apply preprocessing
    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)
    
    # Create datasets and dataloaders
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, preproc

# Define Lorentz-invariant operations
class LorentzLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LorentzLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

# Message passing network for particle interactions
class ParticleInteractionNetwork(nn.Module):
    def __init__(self, feature_dim):
        super(ParticleInteractionNetwork, self).__init__()
        # Message function
        self.message = nn.Sequential(
            LorentzLayer(feature_dim * 2, feature_dim * 2),
            nn.LeakyReLU(0.1),
            LorentzLayer(feature_dim * 2, feature_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Update function
        self.update = nn.Sequential(
            LorentzLayer(feature_dim * 2, feature_dim),
            nn.LeakyReLU(0.1),
            LorentzLayer(feature_dim, feature_dim),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, particles, mask):
        batch_size, n_particles, feat_dim = particles.shape
        
        # Compute pairwise interactions (messages)
        messages = torch.zeros(batch_size, n_particles, feat_dim, device=particles.device)
        
        for i in range(n_particles):
            # Create pairs with particle i
            for j in range(n_particles):
                if i != j:
                    sender = particles[:, j]
                    receiver = particles[:, i]
                    
                    # Compute the message
                    pair = torch.cat([sender, receiver], dim=1)  # Concatenate features
                    message = self.message(pair) * mask[:, j].unsqueeze(1) * mask[:, i].unsqueeze(1)
                    
                    # Aggregate messages
                    messages[:, i] += message
        
        # Update each particle
        updated_particles = torch.zeros_like(particles)
        for i in range(n_particles):
            particle = particles[:, i]
            message = messages[:, i]
            
            # Update step
            update_input = torch.cat([particle, message], dim=1)
            updated_particles[:, i] = self.update(update_input) * mask[:, i].unsqueeze(1) + particle * mask[:, i].unsqueeze(1)
        
        return updated_particles

# Lorentz-invariant attention mechanism
class LorentzAttention(nn.Module):
    def __init__(self, feature_dim):
        super(LorentzAttention, self).__init__()
        self.query_proj = LorentzLayer(feature_dim, feature_dim)
        self.key_proj = LorentzLayer(feature_dim, feature_dim)
        self.value_proj = LorentzLayer(feature_dim, feature_dim)
        self.scale = torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32))
        
    def forward(self, particles, mask=None):
        batch_size, n_particles, feat_dim = particles.shape
        
        # Compute query, key, value projections
        q = self.query_proj(particles.reshape(-1, feat_dim)).view(batch_size, n_particles, -1)
        k = self.key_proj(particles.reshape(-1, feat_dim)).view(batch_size, n_particles, -1)
        v = self.value_proj(particles.reshape(-1, feat_dim)).view(batch_size, n_particles, -1)
        
        # Compute attention scores
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale
        
        # Apply mask
        if mask is not None:
            # Create an attention mask from the particle mask
            attn_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            attn = attn.masked_fill(~attn_mask.bool(), float('-inf'))
        
        # Normalize with softmax
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        output = torch.bmm(attn, v)
        
        return output

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        # Get parameters
        self.n_particles = (input_dim - 3) // 8  # First 3 are missing ET features
        self.particle_feature_dim = 8
        self.hidden_dim = 64
        
        # Particle embedding
        self.particle_embedding = nn.Sequential(
            LorentzLayer(self.particle_feature_dim, self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.2)
        )
        
        # Interaction networks for equivariant message passing
        self.interaction1 = ParticleInteractionNetwork(self.hidden_dim)
        self.interaction2 = ParticleInteractionNetwork(self.hidden_dim)
        
        # Lorentz attention for global context
        self.attention = LorentzAttention(self.hidden_dim)
        
        # Missing energy embedding
        self.et_miss_embedding = nn.Sequential(
            LorentzLayer(3, self.hidden_dim),  # 3 missing ET features
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            LorentzLayer(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.Dropout(0.3),
            LorentzLayer(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.2),
            LorentzLayer(self.hidden_dim, 1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract missing energy features
        et_miss_features = x[:, :3]
        et_miss_embedded = self.et_miss_embedding(et_miss_features)
        
        # Extract and embed particles
        particles = []
        masks = []
        
        for i in range(self.n_particles):
            start_idx = 3 + i * 8
            particle_features = x[:, start_idx:start_idx + 8]
            
            # Get mask (presence of particle)
            mask = particle_features[:, 0].bool()  # First feature is the mask
            masks.append(mask)
            
            # Embed each particle separately to maintain Lorentz invariance
            particles.append(particle_features)
        
        # Stack particles into a single tensor
        particles_tensor = torch.stack(particles, dim=1)  # [batch_size, n_particles, feature_dim]
        masks_tensor = torch.stack(masks, dim=1)  # [batch_size, n_particles]
        
        # Embed each particle
        embedded_particles = torch.zeros(batch_size, self.n_particles, self.hidden_dim, device=x.device)
        
        for i in range(self.n_particles):
            valid_indices = masks[i]
            if valid_indices.sum() > 0:
                particle_feats = particles[i][valid_indices]
                embedded = self.particle_embedding(particle_feats)
                embedded_particles[valid_indices, i] = embedded
        
        # Message passing for particle interactions
        particles_after_mp1 = self.interaction1(embedded_particles, masks_tensor)
        particles_after_mp2 = self.interaction2(particles_after_mp1, masks_tensor)
        
        # Apply attention to capture global relationships
        attended_particles = self.attention(particles_after_mp2, masks_tensor)
        
        # Aggregate particle features with attention (weighted sum)
        particle_sum = torch.sum(attended_particles * masks_tensor.unsqueeze(2), dim=1)
        
        # Combine with missing energy
        combined_features = torch.cat([particle_sum, et_miss_embedded], dim=1)
        
        # Final classification
        logits = self.classifier(combined_features)
        
        return torch.sigmoid(logits).squeeze(1)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.BCELoss()
    
    # Metrics tracking
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_auc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_pred = []
        train_true = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record metrics
            train_losses.append(loss.item())
            train_pred.extend(outputs.detach().cpu().numpy())
            train_true.extend(targets.cpu().numpy())
        
        # Calculate training metrics
        epoch_train_loss = sum(train_losses) / len(train_losses)
        train_pred_binary = (np.array(train_pred) > 0.5).astype(int)
        epoch_train_acc = accuracy_score(train_true, train_pred_binary)
        train_auc = roc_auc_score(train_true, train_pred)
        
        training_loss.append(epoch_train_loss)
        training_acc.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_pred = []
        val_true = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Record metrics
                val_losses.append(loss.item())
                val_pred.extend(outputs.detach().cpu().numpy())
                val_true.extend(targets.cpu().numpy())
        
        # Calculate validation metrics
        epoch_val_loss = sum(val_losses) / len(val_losses)
        val_pred_binary = (np.array(val_pred) > 0.5).astype(int)
        epoch_val_acc = accuracy_score(val_true, val_pred_binary)
        val_auc = roc_auc_score(val_true, val_pred)
        
        validation_loss.append(epoch_val_loss)
        validation_acc.append(epoch_val_acc)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save the best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | '
              f'Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f} | '
              f'Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}')
    
    # Load the best model
    if epochs > 1:
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

    # Preprocessing
    batch_size = 512 if not dryrun else 32
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size)

    # Model Initialization
    sample_X, _ = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[1])

    # Training
    epochs = 1 if dryrun else 20

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