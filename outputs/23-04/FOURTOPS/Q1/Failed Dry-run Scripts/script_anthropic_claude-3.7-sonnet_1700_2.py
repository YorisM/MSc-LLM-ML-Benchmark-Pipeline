# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

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
    # All fitted statistics/constants must be registered as buffers.
    # Torch operations ONLY (no numpy, no pandas).
    # Deterministic behavior required (no randomness in forward pass).
    def __init__(self, **kwargs):
        super().__init__()
        if "sample_means" in kwargs and "sample_stds" in kwargs:
            self.register_buffer("means", kwargs["sample_means"])
            self.register_buffer("stds", kwargs["sample_stds"])
        if "valid_mask" in kwargs:
            self.register_buffer("valid_mask", kwargs["valid_mask"])
        if "padding_mask" in kwargs:
            self.register_buffer("padding_mask", kwargs["padding_mask"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply normalization only to valid features (non-padding, non-zero-variance)
        if hasattr(self, "valid_mask"):
            # Normalize only valid features
            valid_x = x[:, self.valid_mask]
            normalized = (valid_x - self.means) / (self.stds + 1e-8)
            
            # Prepare output with the same shape as input
            output = torch.zeros_like(x)
            output[:, self.valid_mask] = normalized
            
            # Add physics-aware features
            # Calculate energy sums for different particle types
            n_features = 4  # E, pT, eta, phi for each object
            n_particles = (x.shape[1] - 3) // 5  # Subtract weight, ET_miss, phi_ET_miss and divide by 5 (obj_id + 4 features)
            
            # Process each particle/object group
            energy_sum = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
            pt_sum = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
            
            for i in range(n_particles):
                # Calculate base indices for each particle group
                base_idx = 3 + i * 5  # 3 for weight, ET_miss, phi_ET_miss then 5 columns per particle
                
                # Extract object_id and check if it's non-zero (not padding)
                # obj_id is at base_idx, energy at base_idx+1, pT at base_idx+2
                obj_id = x[:, base_idx]
                obj_mask = obj_id != 0  # Mask for non-padding objects
                
                # Add energy and pT to respective sums if the object is not padding
                particle_energy = x[:, base_idx + 1]
                particle_pt = x[:, base_idx + 2]
                
                # Use broadcasting to apply mask
                energy_sum += torch.where(obj_mask.unsqueeze(1), particle_energy.unsqueeze(1), torch.zeros_like(particle_energy.unsqueeze(1)))
                pt_sum += torch.where(obj_mask.unsqueeze(1), particle_pt.unsqueeze(1), torch.zeros_like(particle_pt.unsqueeze(1)))
            
            # Add derived features to output
            output = torch.cat([output, energy_sum, pt_sum], dim=1)
            
            return output
        return x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=128):
    # Identify valid features (non-zero variance)
    vars = torch.var(X_train, dim=0)
    valid_mask = vars > 1e-8
    
    # Calculate means and stds for valid features
    valid_X_train = X_train[:, valid_mask]
    means = torch.mean(valid_X_train, dim=0)
    stds = torch.std(valid_X_train, dim=0)
    
    # Identify padding features
    # In this dataset, padded values are zeros
    non_zero_counts = torch.sum(X_train != 0, dim=0)
    padding_mask = non_zero_counts > 0.1 * X_train.shape[0]  # Features with at least 10% non-zero values

    preproc = PreprocessModule(
        sample_means=means,
        sample_stds=stds,
        valid_mask=valid_mask,
        padding_mask=padding_mask
    )

    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Linear projections
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.scale
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.fc_out(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.layer1(x)))
        out = self.dropout(out)
        out = self.bn2(self.layer2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        
        # Architecture dimensions
        self.reduced_dim = 256
        hidden1 = 512
        hidden2 = 256
        hidden3 = 128
        dropout_rate = 0.3
        
        # Initial dimensionality reduction
        self.reduce_dim = nn.Sequential(
            nn.Linear(input_dim, self.reduced_dim),
            nn.BatchNorm1d(self.reduced_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks
        self.residual1 = ResidualBlock(self.reduced_dim, hidden1)
        self.residual2 = ResidualBlock(self.reduced_dim, hidden1)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self.reduced_dim, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(hidden3, 1)
        )
        
    def forward(self, x):
        # Input dimensionality reduction
        x = self.reduce_dim(x)
        
        # Apply residual blocks
        x = self.residual1(x)
        x = self.residual2(x)
        
        # Classification head
        logits = self.classifier(x)
        return logits.squeeze()

# ----- FREE SECTION: Training Loop Implementation -----
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize optimizer and loss function focused on AUC optimization
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Use focal loss to handle class imbalance better
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_auc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        y_true_train = []
        y_pred_train = []
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        training_loss.append(epoch_loss)
        
        # Calculate training accuracy and AUC
        y_pred_binary = (np.array(y_pred_train) > 0.5).astype(int)
        train_acc = accuracy_score(y_true_train, y_pred_binary)
        train_auc = roc_auc_score(y_true_train, y_pred_train)
        training_acc.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        y_true_val = []
        y_pred_val = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                
                val_loss += loss.item() * inputs.size(0)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        validation_loss.append(val_epoch_loss)
        
        # Calculate validation accuracy and AUC
        y_val_pred_binary = (np.array(y_pred_val) > 0.5).astype(int)
        val_acc = accuracy_score(y_true_val, y_val_pred_binary)
        val_auc = roc_auc_score(y_true_val, y_pred_val)
        validation_acc.append(val_acc)
        
        # Learning rate scheduler step based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Print statistics
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        print("-" * 50)
    
    # Load best model
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
    batch_size = 64 if not dryrun else 16
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size)

    # Model Initialization
    sample_X, _ = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[1])

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