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
from typing import Tuple, List

# ----- FIXED SECTION: Data Loading -----
def load_data():
    # Assuming data files are in a subdirectory relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
    data_path = os.path.join(script_dir, 'challenges', 'FOURTOPS', 'data')
    if not os.path.exists(data_path):
        # Fallback if running interactively or from a different working dir
        # This path might need adjustment depending on execution context
        data_path = './challenges/FOURTOPS/data'
        if not os.path.exists(data_path):
             raise FileNotFoundError(f"Data directory not found. Please check the path: {data_path}")

    X_train_df = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    Y_train_df = pd.read_csv(os.path.join(data_path, 'Y_train.csv'))
    X_val_df   = pd.read_csv(os.path.join(data_path, 'X_val.csv'))
    Y_val_df   = pd.read_csv(os.path.join(data_path, 'Y_val.csv'))

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
    X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
    Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()
    return X_train, Y_train, X_val, Y_val

# ----- FREE SECTION: Data Preprocessing -----
class PreprocessModule(torch.nn.Module):
    # TorchScript-compatible module applying pre-fitted transformations.
    def __init__(self, particle_means, particle_stds, met_means, met_stds, max_particles=25, particle_input_dim=4):
        super().__init__()
        # Register buffers for statistics and constants
        self.register_buffer("particle_means", particle_means)
        self.register_buffer("particle_stds", particle_stds)
        self.register_buffer("met_means", met_means)
        self.register_buffer("met_stds", met_stds)
        self.max_particles = max_particles
        self.particle_input_dim = particle_input_dim
        # Based on input_dim=105 -> 2 global + 103 particle features
        # Assuming 103 = 25 particles * 4 features + 3 padding
        self.particle_feature_end_idx = 2 + self.max_particles * self.particle_input_dim

    def _calculate_derived_particle_features(self, p: torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        # p shape: [batch, max_particles, 4] (E, pT, eta, phi)
        # mask shape: [batch, max_particles, 1]
        E = p[..., 0:1]
        pT = p[..., 1:2]
        eta = p[..., 2:3]
        phi = p[..., 3:4]

        # Prevent log(0) or sqrt(<0) issues by masking or adding epsilon
        pT_safe = torch.clamp(pT, min=1e-6)
        E_safe = torch.clamp(E, min=1e-6)

        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)

        # Calculate mass: m^2 = E^2 - p^2 = E^2 - (px^2 + py^2 + pz^2)
        # More stable: m^2 = E^2 - pT^2 * cosh^2(eta)
        m2 = E.pow(2) - pT.pow(2) * torch.cosh(eta).pow(2)
        # Ensure mass is non-negative and applies mask
        mass = torch.sqrt(torch.relu(m2)) * mask

        # New features: E, pT, eta, phi, px, py, pz, mass, logE, logpT
        # Log features might help with large dynamic range
        logE = torch.log(E_safe) * mask
        logpT = torch.log(pT_safe) * mask

        augmented_p = torch.cat([E, pT, eta, phi, px, py, pz, mass, logE, logpT], dim=-1) # [batch, max_particles, 10]
        return augmented_p * mask # ensure padded entries remain zero

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: [batch, 105]
        # Extract MET and particle features based on assumed structure
        met_features = x[:, 0:2] # [batch, 2] (E_T_miss, phi_E_t_miss)
        particles_flat = x[:, 2:self.particle_feature_end_idx] # [batch, 100]

        batch_size = x.shape[0]
        # Reshape particles: [batch, max_particles, 4]
        particles = particles_flat.reshape(batch_size, self.max_particles, self.particle_input_dim)

        # Create mask for valid particles (non-padding)
        # Use E > 0 as indicator (or pT > 0)
        mask = (particles[..., 0] > 1e-6).float().unsqueeze(-1) # [batch, max_particles, 1]

        # Calculate derived particle features
        augmented_particles = self._calculate_derived_particle_features(particles, mask)

        # Standardize features (apply mask during calculation of stats, apply here to all)
        # Ensure stds are not zero; add epsilon
        safe_particle_stds = self.particle_stds + 1e-8
        standardized_particles = (augmented_particles - self.particle_means) / safe_particle_stds
        # Apply mask again AFTER standardization to zero out padded values
        standardized_particles = standardized_particles * mask

        safe_met_stds = self.met_stds + 1e-8
        standardized_met = (met_features - self.met_means) / safe_met_stds

        # Detach mask before returning if it's not needed for gradients later
        mask = mask.squeeze(-1).detach() # [batch, max_particles]

        return standardized_particles, standardized_met, mask

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=256):
    # Calculate standardization statistics only on training data
    max_particles = 25
    particle_input_dim = 4
    particle_feature_end_idx = 2 + max_particles * particle_input_dim

    # Extract initial features for stats calculation
    train_met = X_train[:, 0:2]
    train_particles_flat = X_train[:, 2:particle_feature_end_idx]
    train_particles = train_particles_flat.reshape(-1, max_particles, particle_input_dim)
    train_mask = (train_particles[..., 0] > 1e-6).float().unsqueeze(-1)

    # Need a temporary PreprocessModule instance just to calculate derived features
    temp_preproc = PreprocessModule(torch.zeros(10), torch.ones(10), torch.zeros(2), torch.ones(2)) # Dummy stats
    train_aug_particles = temp_preproc._calculate_derived_particle_features(train_particles, train_mask)
    num_aug_particle_features = train_aug_particles.shape[-1]

    # Calculate means and stds only for valid particles
    num_valid_particles = train_mask.sum()
    masked_particles = train_aug_particles * train_mask
    particle_sums = masked_particles.sum(dim=(0, 1))
    particle_means = particle_sums / num_valid_particles

    particle_sq_diff = ((masked_particles - particle_means) * train_mask).pow(2).sum(dim=(0, 1))
    particle_stds = torch.sqrt(particle_sq_diff / num_valid_particles)

    # Calculate means and stds for MET features
    met_means = train_met.mean(dim=0)
    met_stds = train_met.std(dim=0)

    # Create the actual PreprocessModule with fitted stats
    preproc = PreprocessModule(
        particle_means=particle_means.detach(),
        particle_stds=particle_stds.detach(),
        met_means=met_means.detach(),
        met_stds=met_stds.detach(),
        max_particles=max_particles,
        particle_input_dim=particle_input_dim
    )

    # Preprocess data - Note: Preprocessing happens *inside* the training loop / data loading
    # Here, we just set up the datasets and loaders
    # The preprocessing module will be applied dynamically
    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val,   Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=min(4, os.cpu_count()))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False, pin_memory=True, num_workers=min(4, os.cpu_count()))

    # Return loaders and the *fitted* preprocessor module
    return train_loader, val_loader, preproc, num_aug_particle_features

# ----- FREE SECTION: Binary Classifier Definition -----
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim_slots, dim_input, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.dim_slots = dim_slots
        self.dim_input = dim_input
        self.iters = iters
        self.eps = eps
        self.scale = dim_slots ** -0.5

        # Use fixed random initialization for slots
        self.slots = nn.Parameter(torch.randn(1, num_slots, dim_slots))

        self.to_q = nn.Linear(dim_slots, dim_slots, bias=False)
        self.to_k = nn.Linear(dim_input, dim_slots, bias=False)
        self.to_v = nn.Linear(dim_input, dim_slots, bias=False)

        # Slot update mechanism: Use MLP + Residual
        self.mlp = nn.Sequential(
            nn.Linear(dim_slots, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim_slots)
        )

        self.norm_input = nn.LayerNorm(dim_input)
        self.norm_slots = nn.LayerNorm(dim_slots)
        self.norm_pre_mlp = nn.LayerNorm(dim_slots)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # inputs: [batch, num_inputs, dim_input]
        # mask: [batch, num_inputs]
        b, n_in, d_in = inputs.shape
        n_s, d_s = self.num_slots, self.dim_slots

        slots = self.slots.expand(b, -1, -1)

        inputs_norm = self.norm_input(inputs)
        k = self.to_k(inputs_norm) # [b, n_in, d_s]
        v = self.to_v(inputs_norm) # [b, n_in, d_s]

        mask = mask.unsqueeze(1) # [b, 1, n_in]

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots) # [b, n_s, d_s]

            dots = torch.einsum('bsd,bnd->bsn', q, k) * self.scale # [b, n_s, n_in]

            # Masking before softmax
            dots.masked_fill_(~mask.bool(), -1e9) # Use large negative value

            attn_probs = dots.softmax(dim=-1) # [b, n_s, n_in], softmax over inputs

            # Weighted mean update calculation (competition happens via softmax normalization)
            # Need weights summing to 1 over slots for each input feature -> softmax over slots dim
            attn_compete = dots.softmax(dim=1) + self.eps # [b, n_s, n_in]
            attn_compete = attn_compete / attn_compete.sum(dim=-1, keepdim=True) # Renormalize over inputs

            updates = torch.einsum('bsn,bnd->bsd', attn_compete, v) # [b, n_s, d_s]

            # Update slots using MLP + Residual
            slots = slots + self.mlp(self.norm_pre_mlp(updates))
            # Note: Original SlotAttn often uses GRU, MLP is simpler here

        return slots # [batch, num_slots, dim_slots]

class Classifier(nn.Module):
    def __init__(self, particle_feature_dim=10, met_feature_dim=2, d_model=128, nhead=4, num_encoder_layers=2, num_slots=8, slot_dim=128, mlp_hidden_dim=256, dropout=0.1):
        super(Classifier, self).__init__()

        self.particle_feature_dim = particle_feature_dim
        self.met_feature_dim = met_feature_dim
        self.d_model = d_model

        # Particle Embedding
        self.particle_embed = nn.Linear(particle_feature_dim, d_model)

        # Optional: Transformer Encoder for particles (helps learn relationships before slot attention)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=mlp_hidden_dim, dropout=dropout, batch_first=True, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Slot Attention mechanism
        self.slot_attention = SlotAttention(num_slots=num_slots, dim_slots=slot_dim, dim_input=d_model, iters=3, hidden_dim=mlp_hidden_dim)

        # MET Processing (optional, could just concatenate raw)
        self.met_embed = nn.Linear(met_feature_dim, d_model // 2) # Project MET to smaller dim

        # Final Classifier Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(num_slots * slot_dim + d_model // 2),
            nn.Linear(num_slots * slot_dim + d_model // 2, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 2, 1)
        )

    def forward(self, particles: torch.Tensor, met: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # particles: [batch, max_particles, particle_feature_dim]
        # met: [batch, met_feature_dim]
        # mask: [batch, max_particles] (0 for pad, 1 for valid)

        # 1. Embed Particles
        particle_embeddings = self.particle_embed(particles) # [b, max_p, d_model]

        # Create attention mask for Transformer Encoder (expects True for padded)
        # Transformer mask needs shape [b, max_p] where True indicates padding
        encoder_mask = (mask == 0)

        # 2. Process Particles with Transformer Encoder
        # Requires mask where True indicates positions to be *ignored*
        encoded_particles = self.transformer_encoder(particle_embeddings, src_key_padding_mask=encoder_mask)

        # 3. Apply Slot Attention
        # SlotAttention needs mask where 1 indicates valid tokens
        slots = self.slot_attention(encoded_particles, mask) # [b, num_slots, slot_dim]

        # 4. Process MET Features
        met_embeddings = self.met_embed(met) # [b, d_model/2]

        # 5. Combine Slot representations and MET
        slots_flat = slots.flatten(start_dim=1) # [b, num_slots * slot_dim]
        combined_features = torch.cat([slots_flat, met_embeddings], dim=1)

        # 6. Classification Head
        logits = self.mlp_head(combined_features) # [b, 1]

        return logits

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, preproc, train_loader, val_loader, epochs=10, lr=1e-4, weight_decay=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    preproc.to(device) # Ensure preprocessor is on the right device too

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    # Optional: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3, verbose=True)

    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    validation_auc = []

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0

        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device).float().unsqueeze(1)

            # Apply preprocessing dynamically
            particles, met, mask = preproc(features)

            optimizer.zero_grad()
            outputs = model(particles, met, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * features.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            epoch_train_correct += (preds == labels.bool()).sum().item()
            epoch_train_total += labels.size(0)

            if i % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {i}/{len(train_loader)}, Train Loss: {loss.item():.4f}', end='\r')

        avg_train_loss = epoch_train_loss / epoch_train_total
        avg_train_acc = epoch_train_correct / epoch_train_total
        training_loss.append(avg_train_loss)
        training_acc.append(avg_train_acc)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device).float().unsqueeze(1)
                particles, met, mask = preproc(features)

                outputs = model(particles, met, mask)
                loss = criterion(outputs, labels)

                epoch_val_loss += loss.item() * features.size(0)
                probs = torch.sigmoid(outputs)
                preds = probs > 0.5
                epoch_val_correct += (preds == labels.bool()).sum().item()
                epoch_val_total += labels.size(0)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(probs.cpu().numpy())

        avg_val_loss = epoch_val_loss / epoch_val_total
        avg_val_acc = epoch_val_correct / epoch_val_total
        validation_loss.append(avg_val_loss)
        validation_acc.append(avg_val_acc)

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        epoch_auc = roc_auc_score(all_labels, all_preds)
        validation_auc.append(epoch_auc)

        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}, Val Acc:   {avg_val_acc:.4f}, Val AUC: {epoch_auc:.4f}")

        # Step the scheduler based on validation AUC
        scheduler.step(epoch_auc)

    # Return trained model and metrics
    return {
        "model": model,
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "training_acc": training_acc,
        "validation_acc": validation_acc,
        "validation_auc": validation_auc # Include AUC history
    }

# ----- FIXED SECTION: Plotting and Saving Outputs -----
def plot_and_save(metric_train, metric_val, metric_name, filename):
    plt.figure()
    plt.plot(metric_train, label=f'Training {metric_name}')
    plt.plot(metric_val, label=f'Validation {metric_name}')
    plt.title(f'{metric_name} per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

# ----- FIXED SECTION: Main Function -----
def main(dryrun=False):
    print("Loading data...")
    X_train, Y_train, X_val, Y_val = load_data()
    print(f"Data loaded: X_train shape {X_train.shape}, Y_train shape {Y_train.shape}")

    print("Preprocessing data and setting up loaders...")
    # Set hyperparameters for preprocessing/model here
    batch_size = 256
    train_loader, val_loader, preproc, num_aug_particle_features = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=batch_size)
    print(f"Using {num_aug_particle_features} augmented particle features.")

    # Model Initialization Hyperparameters
    d_model = 128
    nhead = 4
    num_encoder_layers = 2 # Set to 0 to disable transformer encoder
    num_slots = 8 # Number of concepts/groups (e.g., W bosons, tops, ISR)
    slot_dim = 128
    mlp_hidden_dim = 256
    dropout = 0.1

    print("Initializing model...")
    model = Classifier(
        particle_feature_dim=num_aug_particle_features,
        met_feature_dim=2, # E_T_miss, phi_E_t_miss
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_slots=num_slots,
        slot_dim=slot_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout=dropout
    )
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params:,} trainable parameters.")

    # Training Hyperparameters
    epochs = 3 if dryrun else 20 # Reduced epochs for dry run, increased for potentially better convergence
    lr = 3e-4 # Adjusted learning rate
    weight_decay = 1e-5

    print("Starting training...")
    # Train the model
    training_results = train_model(
        model, preproc, train_loader, val_loader, epochs=epochs, lr=lr, weight_decay=weight_decay)

    trained_model = training_results["model"]
    # Retrieve metrics
    training_loss = training_results["training_loss"]
    validation_loss = training_results["validation_loss"]
    training_acc = training_results["training_acc"]
    validation_acc = training_results["validation_acc"]
    validation_auc = training_results["validation_auc"]

    print("Training finished.")
    print(f"Final Validation Accuracy: {validation_acc[-1]:.4f}")
    print(f"Final Validation AUC: {validation_auc[-1]:.4f}")

    if not dryrun:
        print("Saving model and plots...")
        # determine base name & script directory
        try:
            script_path = os.path.abspath(sys.argv[0])
            base = os.path.splitext(os.path.basename(script_path))[0].removeprefix("script_")
            script_dir = os.path.dirname(script_path)
        except: # Fallback for environments where sys.argv[0] is not reliable
            base = "fourtops_transformer_slotattn"
            script_dir = "."

        os.makedirs(script_dir, exist_ok=True)

        # save model state dict
        model_path = os.path.join(script_dir, f"{base}_model.pth")
        torch.save(trained_model.state_dict(), model_path)
        print(f"Saved model state dict: {model_path}")

        # save scripted model (ensure model and preproc are scriptable)
        try:
            # Move model to CPU before scripting for broader compatibility
            trained_model.cpu()
            preproc.cpu()
            # Create a wrapper for scripted model including preprocessing
            class FullModel(nn.Module):
                def __init__(self, preprocessor, classifier):
                    super().__init__()
                    self.preprocessor = preprocessor
                    self.classifier = classifier
                
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    particles, met, mask = self.preprocessor(x)
                    return self.classifier(particles, met, mask)

            full_model_instance = FullModel(preproc, trained_model)
            full_model_instance.eval() # Set to evaluation mode

            scripted_path = os.path.join(script_dir, f"{base}_scripted.pt")
            scripted_model = torch.jit.script(full_model_instance)
            scripted_model.save(scripted_path)
            print(f"Saved scripted model (incl. preproc): {scripted_path}")

            # Also save preprocessor separately if needed
            scripted_preproc = torch.jit.script(preproc)
            scripted_preproc.save(os.path.join(script_dir, f"{base}_preproc.pt"))
            print(f"Saved scripted preprocessor: {os.path.join(script_dir, f'{base}_preproc.pt')}")

        except Exception as e:
            print(f"Could not script model: {e}")
            # Fallback: Save model state dict only (already done)

        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, f"Loss - {base}", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, f"Accuracy - {base}", os.path.join(script_dir, f"{base}_accuracy.png"))
        # Plot AUC
        plt.figure()
        plt.plot(validation_auc, label='Validation AUC')
        plt.title(f'Validation AUC per Epoch - {base}')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(script_dir, f"{base}_auc.png"))
        print(f"Saved plot: {os.path.join(script_dir, f'{base}_auc.png')}")
        plt.close()

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    # Simple argument parsing
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)