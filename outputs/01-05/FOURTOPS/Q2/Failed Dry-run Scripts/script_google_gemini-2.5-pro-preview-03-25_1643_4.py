# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
# <FREE: You may only import python and torch native modules here. NO OTHER MODULES.>
import math
from typing import List, Tuple

# ----- FIXED SECTION: Data Loading -----
# Ensure data files are accessible from the execution path
DATA_DIR = './challenges/FOURTOPS/data/'

def load_data():
    X_train_df = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    Y_train_df = pd.read_csv(os.path.join(DATA_DIR, 'Y_train.csv'))
    X_val_df   = pd.read_csv(os.path.join(DATA_DIR, 'X_val.csv'))
    Y_val_df   = pd.read_csv(os.path.join(DATA_DIR, 'Y_val.csv'))

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
    X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
    Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()
    return X_train, Y_train, X_val, Y_val

# ----- FREE SECTION: Data Preprocessing -----
class PreprocessModule(torch.nn.Module):
    # TorchScript-compatible module applying pre-fitted transformations.
    # All fitted statistics/constants must be registered as buffers.
    # Torch operations ONLY (no numpy, no pandas).
    # Deterministic behavior required (no randomness in forward pass).
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, epsilon: float = 1e-7):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        # Use a small epsilon to prevent division by zero for features with zero std
        self.register_buffer("epsilon", torch.tensor(epsilon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standardize the features
        # Add epsilon to std to avoid division by zero
        x_standardized = (x - self.mean) / (self.std + self.epsilon)
        return x_standardized

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size):
    # Derive standardization statistics from the training set
    mean = torch.mean(X_train, dim=0)
    std = torch.std(X_train, dim=0)

    preproc = PreprocessModule(mean=mean, std=std)

    # Apply preprocessing - although defined, it's usually applied within the training loop or dataloader
    # For simplicity here and compatibility with template, we apply it before creating datasets.
    # Note: In a production setting, preprocessing might be deferred to the model's forward pass
    # or handled by the DataLoader for efficiency.
    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----

# Helper: Minkowski metric tensor
MINKOWSKI_METRIC = torch.tensor([1., -1., -1., -1.])

@torch.jit.script
def to_cartesian(pt: torch.Tensor, eta: torch.Tensor, phi: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    # Ensure pt is non-negative
    pt = torch.relu(pt) # Use relu to ensure pt >= 0
    # Clamp eta to avoid large values in sinh causing overflow/NaN
    eta = torch.clamp(eta, min=-7.0, max=7.0)

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return torch.stack([E, px, py, pz], dim=-1) # Output shape: [..., 4]

@torch.jit.script
def minkowski_dot(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    # v1, v2 shape: [..., 4]
    # Ensure metric is on the same device
    metric = MINKOWSKI_METRIC.to(v1.device)
    return torch.sum(v1 * v2 * metric, dim=-1) # Output shape: [...]

class LorentzLayer(nn.Module):
    """A layer that computes Lorentz invariant features."""
    def __init__(self, num_scalars_in: int, num_scalars_out: int, zero_init: bool = False):
        super().__init__()
        # Simple MLP acting on scalar features derived from Lorentz invariants
        self.mlp = nn.Sequential(
            nn.Linear(num_scalars_in, num_scalars_out),
            nn.ReLU(),
            nn.LayerNorm(num_scalars_out)
        )
        if zero_init:
            # Initialize the last layer biases to zero
            if isinstance(self.mlp[-2], nn.Linear):
                 torch.nn.init.zeros_(self.mlp[-2].bias)

    def forward(self, p4: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Input:
            p4: Particle 4-vectors (Batch, NumParticles, 4)
            mask: Particle mask (Batch, NumParticles)
        Output:
            scalars: Derived scalar features per particle (Batch, NumParticles, NumScalarsOut)
        """
        batch_size, num_particles, _ = p4.shape
        device = p4.device

        # Compute pairwise features (masked)
        p4_i = p4.unsqueeze(2) # (B, N, 1, 4)
        p4_j = p4.unsqueeze(1) # (B, 1, N, 4)

        # Pairwise invariant mass squared m_ij^2 = (p_i + p_j)^2
        p_sum_ij = p4_i + p4_j # (B, N, N, 4)
        m2_ij = minkowski_dot(p_sum_ij, p_sum_ij) # (B, N, N)

        # Pairwise Minkowski dot product p_i . p_j
        dot_ij = minkowski_dot(p4_i, p4_j) # (B, N, N)

        # Create pairwise mask
        mask_ij = mask.unsqueeze(2) * mask.unsqueeze(1) # (B, N, N)
        # Prevent self-interaction contribution in sums if needed, but let's include it for now
        # mask_ij.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Apply mask (set invalid entries to 0 before aggregation)
        m2_ij = m2_ij * mask_ij
        dot_ij = dot_ij * mask_ij

        # Aggregate pairwise features per particle
        # Sum over j for each i
        sum_m2 = torch.sum(m2_ij, dim=2) # (B, N)
        sum_dot = torch.sum(dot_ij, dim=2) # (B, N)
        
        # Calculate particle invariant mass squared (p_i)^2
        m2_i = minkowski_dot(p4, p4) # (B, N)

        # Normalize aggregations by number of valid pairs (handle division by zero)
        num_valid_j = torch.sum(mask, dim=1, keepdim=True).clamp(min=1.0) # (B, 1)
        # Correct normalization: Sum over j for particle i needs num_valid_j for particle i
        num_valid_j_per_i = torch.sum(mask_ij, dim=2).clamp(min=1e-6) # (B, N)

        mean_m2 = sum_m2 / num_valid_j_per_i
        mean_dot = sum_dot / num_valid_j_per_i

        # Combine scalar features for each particle
        # Features: m_i^2, sum(m_ij^2), sum(p_i.p_j), mean(m_ij^2), mean(p_i.p_j), E, |p|
        E_i = p4[:, :, 0]
        p_mag = torch.norm(p4[:, :, 1:4], dim=-1)
        particle_scalars = torch.stack([
            m2_i,
            sum_m2, 
            sum_dot,
            mean_m2,
            mean_dot,
            E_i,
            p_mag
        ], dim=-1) # (B, N, NumScalarFeatures) == (B, N, 7)

        # Apply mask to the input features before MLP
        particle_scalars = particle_scalars * mask.unsqueeze(-1)
        
        # Pass through MLP
        out_scalars = self.mlp(particle_scalars)
         # Apply mask again after MLP 
        out_scalars = out_scalars * mask.unsqueeze(-1)
        
        return out_scalars

class Classifier(nn.Module):
    def __init__(self, input_dim: int = 105, num_particle_features: int = 5, max_particles: int = 20, hidden_dim: int = 64, num_lorentz_layers: int = 2):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.num_particle_features = num_particle_features
        self.max_particles = max_particles
        self.num_global_features = 2 # E_T_miss, phi_E_t_miss
        
        # Check consistency
        expected_dim = self.num_global_features + self.max_particles * self.num_particle_features
        if input_dim < expected_dim:
             print(f"Warning: input_dim {input_dim} is less than expected {expected_dim}. Adjusting max_particles or assuming padding.")
             # Assuming the remainder is padding, recalculate max_particles based on input_dim
             self.max_particles = (input_dim - self.num_global_features) // self.num_particle_features
             print(f"Adjusted max_particles to {self.max_particles}")
        
        current_scalar_dim = 7 # Number of features output by LorentzLayer's feature engineering
        lorentz_layers = []
        for i in range(num_lorentz_layers):
            lorentz_layers.append(LorentzLayer(current_scalar_dim, hidden_dim, zero_init=(i == num_lorentz_layers - 1)))
            current_scalar_dim = hidden_dim # Output dim becomes input for next layer
        self.lorentz_net = nn.Sequential(*lorentz_layers)

        # Final MLP to combine aggregated features
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim + self.num_global_features, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device

        # Separate global features (MET) and particle features
        global_features = x[:, :self.num_global_features] # (B, 2)
        particle_data = x[:, self.num_global_features : self.num_global_features + self.max_particles * self.num_particle_features]
        # Reshape particle data: (Batch, MaxParticles, NumFeaturesPerParticle)
        particles = particle_data.view(batch_size, self.max_particles, self.num_particle_features)

        # Extract kinematic variables (assuming order: Type, E, pT, eta, phi)
        # obj_type = particles[:, :, 0] # Currently unused
        E   = particles[:, :, 1]
        pT  = particles[:, :, 2]
        eta = particles[:, :, 3]
        phi = particles[:, :, 4]

        # Create mask for valid particles (based on pT > 0, assuming padding uses 0)
        mask = (pT > 1e-6).float() # (B, N), use small threshold due to float precision

        # Convert to Cartesian 4-vectors
        p4 = to_cartesian(pT, eta, phi, E) # (B, N, 4)

        # Apply mask to 4-vectors (zero out padded particles)
        p4 = p4 * mask.unsqueeze(-1)

        # Pass through Lorentz layers
        particle_scalars = self.lorentz_net(p4, mask) # (B, N, hidden_dim)

        # Aggregate particle features: Use masked mean pooling
        # Sum masked features and divide by the number of valid particles
        summed_features = torch.sum(particle_scalars * mask.unsqueeze(-1), dim=1) # (B, hidden_dim)
        num_valid_particles = torch.sum(mask, dim=1, keepdim=True).clamp(min=1.0) # (B, 1)
        mean_features = summed_features / num_valid_particles # (B, hidden_dim)

        # Concatenate aggregated particle features with global features
        combined_features = torch.cat([mean_features, global_features], dim=1) # (B, hidden_dim + 2)

        # Final prediction MLP
        logits = self.final_mlp(combined_features) # (B, 1)

        return logits.squeeze(-1) # Output shape: (B)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs, learning_rate=1e-4, weight_decay=1e-5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    # Scheduler reduces LR if validation AUC stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True)

    training_loss_hist = []
    validation_loss_hist = []
    training_acc_hist = []
    validation_acc_hist = []
    validation_auc_hist = [] # Track AUC specifically

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss_epoch = 0.0
        train_correct = 0
        train_total = 0

        for i, (batch_X, batch_Y) in enumerate(train_loader):
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item() * batch_X.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == batch_Y.byte()).sum().item()
            train_total += batch_Y.size(0)
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_train_loss = train_loss_epoch / train_total
        epoch_train_acc = train_correct / train_total
        training_loss_hist.append(epoch_train_loss)
        training_acc_hist.append(epoch_train_acc)

        # Validation phase
        model.eval()
        val_loss_epoch = 0.0
        val_correct = 0
        val_total = 0
        all_val_outputs = []
        all_val_labels = []
        with torch.no_grad():
            for batch_X_val, batch_Y_val in val_loader:
                batch_X_val, batch_Y_val = batch_X_val.to(device), batch_Y_val.to(device).float()
                outputs = model(batch_X_val)
                loss = criterion(outputs, batch_Y_val)

                val_loss_epoch += loss.item() * batch_X_val.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == batch_Y_val.byte()).sum().item()
                val_total += batch_Y_val.size(0)
                all_val_outputs.append(torch.sigmoid(outputs).cpu()) # Store raw probabilities for AUC
                all_val_labels.append(batch_Y_val.cpu())

        epoch_val_loss = val_loss_epoch / val_total
        epoch_val_acc = val_correct / val_total
        validation_loss_hist.append(epoch_val_loss)
        validation_acc_hist.append(epoch_val_acc)

        # Calculate validation AUC
        all_val_outputs_tensor = torch.cat(all_val_outputs)
        all_val_labels_tensor = torch.cat(all_val_labels)
        epoch_val_auc = roc_auc_score(all_val_labels_tensor.numpy(), all_val_outputs_tensor.numpy())
        validation_auc_hist.append(epoch_val_auc)

        print(f"Epoch {epoch+1}/{epochs} Completed:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"  Val Loss:   {epoch_val_loss:.4f}, Val Acc:   {epoch_val_acc:.4f}, Val AUC: {epoch_val_auc:.4f}")
        
        # Scheduler step based on validation AUC
        scheduler.step(epoch_val_auc)

    print("Training finished.")
    return model, training_loss_hist, validation_loss_hist, training_acc_hist, validation_acc_hist, validation_auc_hist

# ----- FIXED SECTION: Plotting and Saving Outputs -----
def plot_and_save(metric_train, metric_val, metric_name, filename):
    if not metric_train or not metric_val: # Handle cases with few epochs
        print(f"Skipping plot for {metric_name} due to insufficient data.")
        return
    plt.figure(figsize=(8, 6))
    plt.plot(metric_train, label=f'Training {metric_name}')
    plt.plot(metric_val, label=f'Validation {metric_name}')
    plt.title(f'{metric_name} per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# ----- FIXED SECTION: Main Function -----
def main(dryrun=False):
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    X_train, Y_train, X_val, Y_val = load_data()
    print(f"Data loaded: X_train shape {X_train.shape}, Y_train shape {Y_train.shape}")

    # Preprocessing & DataLoader
    batch_size = 512 # Adjust based on GPU memory
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size)
    print("Preprocessing applied and DataLoaders created.")

    # Model Initialization
    # Determine input dimension from loader if needed, otherwise use fixed size
    # sample_X, _ = next(iter(train_loader))
    # input_dim = sample_X.shape[1] 
    input_dim = X_train.shape[1] # Use shape from loaded data
    
    # Adjust model hyperparameters here if needed
    model = Classifier(input_dim=input_dim, hidden_dim=128, num_lorentz_layers=3)
    print(f"Model initialized: {model.__class__.__name__}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    # Training
    epochs = 3 if dryrun else 30 # Increase Epochs for potentially better results

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc, validation_auc = train_model(
        model, train_loader, val_loader, epochs=epochs, device=device
    )

    if not dryrun:
        # determine base name & script directory
        script_name = os.path.basename(sys.argv[0])
        # Handle interactive environments where sys.argv[0] might not be a file path
        if script_name == '-f': # Check common marker for interactive environments like Jupyter
             base = 'interactive_lorentz_model'
             script_dir = '.' # Save in current directory
        else:
             base = os.path.splitext(script_name)[0].replace("script_", "")
             script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
             os.makedirs(script_dir, exist_ok=True)
        
        print(f"Saving artifacts with base name: {base}")
        print(f"Saving artifacts in directory: {script_dir}")

        # save model state dict
        model_path = os.path.join(script_dir, f"{base}_model.pth")
        torch.save(trained_model.state_dict(), model_path)
        print(f"Model state_dict saved to {model_path}")

        # save scripted model (ensure model is scriptable)
        try:
            scripted_path = os.path.join(script_dir, f"{base}_scripted.pt")
            # Ensure model is on CPU before scripting if necessary to avoid device mismatch
            # Or ensure scripting handles device placement correctly
            trained_model.to('cpu') # Move model to CPU before scripting
            scripted_model = torch.jit.script(trained_model)
            scripted_model.save(scripted_path)
            print(f"Scripted model saved to {scripted_path}")
        except Exception as e:
             print(f"Could not script model: {e}")

        # save scripted preprocessor
        try:
            scripted_preproc = torch.jit.script(preproc)
            preproc_path = os.path.join(script_dir, f"{base}_preproc.pt")
            scripted_preproc.save(preproc_path)
            print(f"Scripted preprocessor saved to {preproc_path}")
        except Exception as e:
            print(f"Could not script preprocessor: {e}")

        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, f"Loss - {base}", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, f"Accuracy - {base}", os.path.join(script_dir, f"{base}_accuracy.png"))
        # Plot AUC specifically
        plt.figure(figsize=(8, 6))
        plt.plot(validation_auc, label=f'Validation AUC')
        plt.title(f'Validation AUC per Epoch - {base}')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(script_dir, f"{base}_auc.png"))
        plt.close()
        print("Plots saved.")

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    # Simple check for '--dryrun' argument
    dryrun = any(arg == '--dryrun' for arg in sys.argv)
    main(dryrun=dryrun)