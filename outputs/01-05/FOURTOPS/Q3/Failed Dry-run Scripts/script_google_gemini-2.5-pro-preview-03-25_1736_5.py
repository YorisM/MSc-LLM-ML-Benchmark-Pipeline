# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
# <FREE: You may only import python and torch native modules here. NO OTHER MODULES.>
import math

# ----- FIXED SECTION: Data Loading -----
def load_data():
    # Assuming data files are in './challenges/FOURTOPS/data/' relative to script execution
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    data_dir = os.path.join(script_dir, 'challenges', 'FOURTOPS', 'data')

    # Check if directory exists, create dummy data if not (for local testing outside platform)
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} not found. Generating dummy data.")
        os.makedirs(data_dir, exist_ok=True)
        # Create dummy CSV files if they don't exist
        if not os.path.exists(os.path.join(data_dir, 'X_train.csv')):
            dummy_X_train = np.random.rand(100, 105)
            dummy_Y_train = np.random.randint(0, 2, 100)
            dummy_X_val = np.random.rand(20, 105)
            dummy_Y_val = np.random.randint(0, 2, 20)
            pd.DataFrame(dummy_X_train).to_csv(os.path.join(data_dir, 'X_train.csv'), index=False)
            pd.DataFrame(dummy_Y_train).to_csv(os.path.join(data_dir, 'Y_train.csv'), index=False)
            pd.DataFrame(dummy_X_val).to_csv(os.path.join(data_dir, 'X_val.csv'), index=False)
            pd.DataFrame(dummy_Y_val).to_csv(os.path.join(data_dir, 'Y_val.csv'), index=False)

    X_train_df = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    Y_train_df = pd.read_csv(os.path.join(data_dir, 'Y_train.csv'))
    X_val_df   = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
    Y_val_df   = pd.read_csv(os.path.join(data_dir, 'Y_val.csv'))

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
    X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
    Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()
    return X_train, Y_train, X_val, Y_val

# ----- FREE SECTION: Data Preprocessing -----
class PreprocessModule(torch.nn.Module):
    def __init__(self, means, stds, max_particles=25, input_dim=105):
        super().__init__()
        self.max_particles = max_particles
        self.input_dim = input_dim
        # Assuming 4 features per particle (E, pT, eta, phi)
        self.particle_feat_dim = 4
        self.global_feat_dim = 2
        self.augmented_particle_feat_dim = 7 # E, pT, eta, phi, px, py, pz

        # Register means and stds as buffers for TorchScript compatibility
        self.register_buffer("global_means", means[:self.global_feat_dim])
        self.register_buffer("global_stds", stds[:self.global_feat_dim])
        self.register_buffer("particle_means", means[self.global_feat_dim:])
        self.register_buffer("particle_stds", stds[self.global_feat_dim:])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        # Extract global features (MET, MET_phi)
        global_features = x[:, :self.global_feat_dim]

        # Extract particle features block
        # Shape: [batch_size, max_particles * particle_feat_dim]
        particle_features_flat = x[:, self.global_feat_dim : self.global_feat_dim + self.max_particles * self.particle_feat_dim]

        # Reshape to [batch_size, max_particles, particle_feat_dim]
        particle_features = particle_features_flat.view(batch_size, self.max_particles, self.particle_feat_dim)

        # Create mask for valid particles (E > 1e-6 assumed for valid particles)
        # Mask shape: [batch_size, max_particles]
        mask = (particle_features[:, :, 0] > 1e-6).float()

        # Feature Augmentation: Calculate px, py, pz
        E   = particle_features[:, :, 0]
        pT  = particle_features[:, :, 1]
        eta = particle_features[:, :, 2]
        phi = particle_features[:, :, 3]

        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        # Clamp sinh argument to avoid large values for large eta
        pz = pT * torch.sinh(torch.clamp(eta, -5, 5))

        # Combine original and augmented features
        # Shape: [batch_size, max_particles, augmented_particle_feat_dim]
        augmented_particles = torch.stack([E, pT, eta, phi, px, py, pz], dim=-1)

        # Apply mask: set padded particle features to zero (optional, handled by mask later, but good practice)
        augmented_particles = augmented_particles * mask.unsqueeze(-1)

        # Standardization
        # Ensure stds are not zero to avoid division by zero
        global_stds_safe = torch.where(self.global_stds == 0, torch.ones_like(self.global_stds), self.global_stds)
        particle_stds_safe = torch.where(self.particle_stds == 0, torch.ones_like(self.particle_stds), self.particle_stds)

        global_features_std = (global_features - self.global_means) / global_stds_safe

        # Standardize only non-masked particle features
        # Need to broadcast means/stds: [augmented_particle_feat_dim] -> [1, 1, augmented_particle_feat_dim]
        particle_means_b = self.particle_means.unsqueeze(0).unsqueeze(0)
        particle_stds_b = particle_stds_safe.unsqueeze(0).unsqueeze(0)

        augmented_particles_std = ((augmented_particles - particle_means_b) / particle_stds_b) * mask.unsqueeze(-1)

        # Return standardized features and mask
        # Global: [batch, 2], Particles: [batch, max_particles, 7], Mask: [batch, max_particles]
        return global_features_std, augmented_particles_std, mask

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size):
    MAX_PARTICLES = 25 # Based on (105 - 2) / 4 = 25.75
    PARTICLE_FEAT_DIM = 4
    GLOBAL_FEAT_DIM = 2
AUGMENTED_PARTICLE_FEAT_DIM = 7

    # Calculate statistics for standardization (means and stds)
    # Calculate global stats
    global_train_features = X_train[:, :GLOBAL_FEAT_DIM]
    global_means = torch.mean(global_train_features, dim=0)
    global_stds = torch.std(global_train_features, dim=0)

    # Calculate particle stats - carefully handling padding
    particle_features_flat = X_train[:, GLOBAL_FEAT_DIM : GLOBAL_FEAT_DIM + MAX_PARTICLES * PARTICLE_FEAT_DIM]
    particle_features = particle_features_flat.view(-1, MAX_PARTICLES, PARTICLE_FEAT_DIM)
    mask = (particle_features[:, :, 0] > 1e-6).float() # [N_train, MAX_PARTICLES]

    # Augment features to calculate stats on the final feature set
    E   = particle_features[:, :, 0]
    pT  = particle_features[:, :, 1]
    eta = particle_features[:, :, 2]
    phi = particle_features[:, :, 3]
    px = pT * torch.cos(phi)
    py = pT * torch.sin(phi)
    pz = pT * torch.sinh(torch.clamp(eta, -5, 5))
    augmented_particles = torch.stack([E, pT, eta, phi, px, py, pz], dim=-1)

    # Calculate mean and std only for valid particles
    num_valid_particles = mask.sum()
    masked_particles = augmented_particles * mask.unsqueeze(-1)
    particle_sum = masked_particles.sum(dim=(0, 1))
    particle_means = particle_sum / num_valid_particles

    # Calculate variance (std = sqrt(variance))
    particle_var_sum = (((masked_particles - particle_means.unsqueeze(0).unsqueeze(0)) * mask.unsqueeze(-1))**2).sum(dim=(0, 1))
    particle_stds = torch.sqrt(particle_var_sum / num_valid_particles)

    # Combine means and stds
    means = torch.cat([global_means, particle_means])
    stds = torch.cat([global_stds, particle_stds])

    # Instantiate Preprocessing Module with calculated stats
    preproc_module = PreprocessModule(means=means, stds=stds, max_particles=MAX_PARTICLES)

    # Apply preprocessing (primarily for getting dataset structures right)
    # Note: Actual transformation happens inside the training loop / model forward pass implicitly
    # We create TensorDatasets with the *original* data.
    # The PreprocessModule will be applied within the model or just before it.
    # This structure avoids storing large preprocessed datasets.

    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val,   Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # Return loaders and the *fitted* preprocessor module
    return train_loader, val_loader, preproc_module

# ----- FREE SECTION: Slot Attention Module -----
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, dim))
        torch.nn.init.xavier_uniform_(self.slots_log_sigma)

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

        self.norm_input   = nn.LayerNorm(dim)
        self.norm_slots   = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, mask=None, num_slots=None):
        b, n, d = inputs.shape
        k = num_slots if num_slots is not None else self.num_slots

        # Initialize slots
        mu = self.slots_mu.expand(b, k, -1)
        sigma = self.slots_log_sigma.exp().expand(b, k, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=inputs.device)

        inputs = self.norm_input(inputs)
        queries = self.to_q(slots)
        keys    = self.to_k(inputs)
        values  = self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention
            dots = torch.einsum('bid,bjd->bij', queries, keys) * self.scale # [batch, k, n]
            if mask is not None:
                 dots = dots.masked_fill(~mask.unsqueeze(1).bool(), -1e10) # Mask invalid particles

            attn = dots.softmax(dim=1) + self.eps # Attention over slots [batch, k, n]
            attn = attn / attn.sum(dim=-1, keepdim=True) # Weighted mean calculation

            updates = torch.einsum('bij,bjd->bid', attn, values) # [batch, k, d]

            # GRU update
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))
            slots = slots.reshape(b, k, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, preprocessor, d_model=128, num_slots=4, slot_iters=3, num_transformer_layers=2, num_heads=4, dim_feedforward=512, dropout=0.1):
        super(Classifier, self).__init__()
        self.preprocessor = preprocessor

        self.particle_embed = nn.Sequential(
            nn.Linear(self.preprocessor.augmented_particle_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.global_embed = nn.Sequential(
            nn.Linear(self.preprocessor.global_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=d_model,
            iters=slot_iters,
            hidden_dim=d_model * 2
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model * 2), # Input is concatenation of pooled slots and global embedding
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # Preprocess input
        global_feat_std, particle_feat_std, mask = self.preprocessor(x)
        # particle_feat_std shape: [batch, max_particles, 7]
        # global_feat_std shape: [batch, 2]
        # mask shape: [batch, max_particles]

        # Embed features
        particle_embeddings = self.particle_embed(particle_feat_std) # [batch, max_particles, d_model]
        global_embedding = self.global_embed(global_feat_std)    # [batch, d_model]

        # Apply Slot Attention to particle embeddings
        slots = self.slot_attention(particle_embeddings, mask=mask) # [batch, num_slots, d_model]

        # Process slots with Transformer
        transformer_output = self.transformer_encoder(slots) # [batch, num_slots, d_model]

        # Aggregate slot information (e.g., mean pooling)
        pooled_slots = transformer_output.mean(dim=1) # [batch, d_model]

        # Combine with global features
        combined_features = torch.cat([pooled_slots, global_embedding], dim=-1) # [batch, 2 * d_model]

        # Classification
        logits = self.cls_head(combined_features) # [batch, 1]

        return logits.squeeze(-1) # Return logits [batch]

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs, learning_rate=1e-4, weight_decay=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Optional: learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    validation_auc = [] # Track AUC per epoch

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0

        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * batch_X.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            epoch_train_correct += (preds == batch_Y.bool()).sum().item()
            epoch_train_total += batch_Y.size(0)

        avg_train_loss = epoch_train_loss / epoch_train_total
        avg_train_acc = epoch_train_correct / epoch_train_total
        training_loss.append(avg_train_loss)
        training_acc.append(avg_train_acc)

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device).float()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)

                epoch_val_loss += loss.item() * batch_X.size(0)
                probs = torch.sigmoid(outputs)
                preds = probs > 0.5
                epoch_val_correct += (preds == batch_Y.bool()).sum().item()
                epoch_val_total += batch_Y.size(0)

                all_val_labels.extend(batch_Y.cpu().numpy())
                all_val_preds.extend(probs.cpu().numpy())

        avg_val_loss = epoch_val_loss / epoch_val_total
        avg_val_acc = epoch_val_correct / epoch_val_total
        validation_loss.append(avg_val_loss)
        validation_acc.append(avg_val_acc)

        # Calculate AUC for the epoch
        epoch_auc = roc_auc_score(all_val_labels, all_val_preds)
        validation_auc.append(epoch_auc)

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, Val AUC: {epoch_auc:.4f}")

        # Optional: Step the scheduler based on validation loss
        # scheduler.step(avg_val_loss)

    return model, training_loss, validation_loss, training_acc, validation_acc, validation_auc

# ----- FIXED SECTION: Plotting and Saving Outputs -----
def plot_and_save(metric_train, metric_val, metric_name, title, filename):
    plt.figure()
    plt.plot(metric_train, label=f'Training {metric_name}')
    plt.plot(metric_val, label=f'Validation {metric_name}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# ----- FIXED SECTION: Main Function -----
def main(dryrun=False):
    BATCH_SIZE = 512
    EPOCHS = 30 if not dryrun else 1
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-5
    D_MODEL = 128
    NUM_SLOTS = 4       # Hypothesize 4 slots for 4 top quarks
    SLOT_ITERS = 3
    TRANSFORMER_LAYERS = 3
    NUM_HEADS = 4
    DIM_FEEDFORWARD = 256
    DROPOUT = 0.1

    # Data Loading
    X_train, Y_train, X_val, Y_val = load_data()

    # Preprocessing
    train_loader, val_loader, preproc_module = preprocess_data(
        X_train, Y_train, X_val, Y_val, batch_size=BATCH_SIZE)

    # Model Initialization
    model = Classifier(
        preprocessor=preproc_module,
        d_model=D_MODEL,
        num_slots=NUM_SLOTS,
        slot_iters=SLOT_ITERS,
        num_transformer_layers=TRANSFORMER_LAYERS,
        num_heads=NUM_HEADS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
        )

    # Training
    trained_model, training_loss, validation_loss, training_acc, validation_acc, validation_auc = train_model(
        model, train_loader, val_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"Final Validation AUC: {validation_auc[-1]:.4f}")

    if not dryrun:
        # determine base name & script directory
        script_name = os.path.basename(sys.argv[0])
        base = os.path.splitext(script_name)[0].replace("script_", "") # Use actual script name
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if '__file__' in globals() else '.' # Handle interactive use
        output_dir = os.path.join(script_dir, "outputs", base)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Saving outputs to: {output_dir}")

        # save model
        model_path = os.path.join(output_dir, f"{base}_model.pth")
        torch.save(trained_model.state_dict(), model_path)

        # Ensure model is on CPU before scripting
        trained_model.to('cpu')
        trained_model.eval() 

        # save scripted model
        try:
            scripted_path = os.path.join(output_dir, f"{base}_scripted_model.pt")
            scripted_model = torch.jit.script(trained_model)
            scripted_model.save(scripted_path)
        except Exception as e:
            print(f"Could not script model: {e}")

        # save preprocessor
        try:
            scripted_preproc = torch.jit.script(preproc_module)
            scripted_preproc.save(os.path.join(output_dir, f"{base}_preproc.pt"))
        except Exception as e:
            print(f"Could not script preprocessor: {e}")

        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, "Loss", f"Loss vs Epochs - {base}", os.path.join(output_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, "Accuracy", f"Accuracy vs Epochs - {base}", os.path.join(output_dir, f"{base}_accuracy.png"))
        # Plot AUC
        plt.figure()
        plt.plot(validation_auc, label='Validation AUC')
        plt.title(f'Validation AUC per Epoch - {base}')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{base}_auc.png"))
        plt.close()

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)