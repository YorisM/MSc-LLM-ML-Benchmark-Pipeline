# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import math
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    def __init__(self, **kwargs):
        super().__init__()
        # Register normalization stats
        for key, value in kwargs.items():
            self.register_buffer(key, value)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First separate the weights, E_T_miss, phi_{E_t}_miss from the object features
        weights = x[:, 0:1]
        et_miss = x[:, 1:2]
        phi_et_miss = x[:, 2:3]
        
        # Process object features (groups of 5: obj_n, E_n, p_T_n, eta_n, phi_n)
        rest = x[:, 3:]
        batch_size = x.shape[0]
        
        # Reshape to make object extraction easier
        # New shape: [batch_size, num_objects, 5]
        num_objects = (rest.shape[1] // 5)
        reshaped = rest.view(batch_size, num_objects, 5)
        
        # Extract features
        obj_ids = reshaped[:, :, 0]
        energies = reshaped[:, :, 1]
        pt = reshaped[:, :, 2]
        eta = reshaped[:, :, 3]
        phi = reshaped[:, :, 4]
        
        # Create mask for valid objects (non-zero obj_ids)
        valid_mask = (obj_ids != 0).float().unsqueeze(-1)
        
        # Normalize the features
        norm_energies = (energies.unsqueeze(-1) - self.energy_mean) / self.energy_std
        norm_pt = (pt.unsqueeze(-1) - self.pt_mean) / self.pt_std
        norm_eta = eta.unsqueeze(-1) / 5.0  # normalize eta to approx [-1, 1] range
        
        # Process angles (phi is periodic, convert to sin/cos)
        sin_phi = torch.sin(phi).unsqueeze(-1)
        cos_phi = torch.cos(phi).unsqueeze(-1)
        
        # Also normalize E_T_miss
        norm_et_miss = (et_miss - self.et_miss_mean) / self.et_miss_std
        sin_phi_miss = torch.sin(phi_et_miss)
        cos_phi_miss = torch.cos(phi_et_miss)
        
        # One-hot encoding for object IDs (using predefined mapping)
        # Map each object ID to its corresponding index
        obj_id_mapped = torch.zeros_like(obj_ids)
        for i, obj_type in enumerate(self.obj_types):
            obj_id_mapped = torch.where(obj_ids == obj_type, torch.ones_like(obj_ids) * i, obj_id_mapped)
        
        # Create one-hot encoding
        one_hot = F.one_hot(obj_id_mapped.long(), num_classes=len(self.obj_types))
        one_hot = one_hot.float() * valid_mask  # Apply mask to zero-out invalid objects
        
        # Concatenate all features for each object
        object_features = torch.cat([
            one_hot,                  # Object type encoding
            norm_energies,            # Normalized energy
            norm_pt,                 # Normalized transverse momentum
            norm_eta,                # Normalized pseudorapidity
            sin_phi,                 # Sin of azimuthal angle
            cos_phi                  # Cos of azimuthal angle
        ], dim=-1)
        
        # Calculate global event features
        total_energy = torch.sum(energies * valid_mask.squeeze(-1), dim=1, keepdim=True)
        total_pt = torch.sum(pt * valid_mask.squeeze(-1), dim=1, keepdim=True)
        num_objects_present = torch.sum(valid_mask, dim=1)
        
        # Normalize global features
        norm_total_energy = (total_energy - self.total_energy_mean) / self.total_energy_std
        norm_total_pt = (total_pt - self.total_pt_mean) / self.total_pt_std
        norm_num_objects = (num_objects_present - self.num_objects_mean) / self.num_objects_std
        
        # Sort objects by transverse momentum (descending)
        pt_values = pt.clone()
        pt_values[valid_mask.squeeze(-1) == 0] = -float('inf')  # Set padding objects to negative infinity
        _, indices = torch.sort(pt_values, dim=1, descending=True)
        batch_indices = torch.arange(batch_size).view(-1, 1).repeat(1, num_objects)
        sorted_object_features = object_features[batch_indices, indices]
        
        # Flatten the objects (preserving the batch dimension)
        flattened = sorted_object_features.reshape(batch_size, -1)
        
        # Concatenate all features
        # Global event features and object features
        features = torch.cat([
            weights,                  # Event weight
            norm_et_miss,             # Normalized missing transverse energy
            sin_phi_miss,             # Sin of missing ET phi
            cos_phi_miss,             # Cos of missing ET phi
            norm_total_energy,        # Total event energy
            norm_total_pt,            # Total transverse momentum
            norm_num_objects,         # Number of objects in the event
            flattened                 # All object features (sorted by pT)
        ], dim=1)
        
        return features

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=256):
    # Separate into components for analysis
    weights = X_train[:, 0]
    et_miss = X_train[:, 1]
    phi_et_miss = X_train[:, 2]
    rest = X_train[:, 3:]
    
    # Create masks and extract features from the object data
    batch_size, total_features = rest.shape
    num_objects = total_features // 5  # Each object has 5 features
    
    # Reshape to get object features in a structured way
    reshaped = rest.view(batch_size, num_objects, 5)
    obj_ids = reshaped[:, :, 0]
    energies = reshaped[:, :, 1]
    pt = reshaped[:, :, 2]
    eta = reshaped[:, :, 3]
    phi = reshaped[:, :, 4]
    
    # Create mask for valid objects (non-zero obj_ids)
    valid_mask = (obj_ids != 0).float()
    
    # Compute statistics for normalization
    energy_mean = torch.mean(energies[valid_mask == 1])
    energy_std = torch.std(energies[valid_mask == 1])
    pt_mean = torch.mean(pt[valid_mask == 1])
    pt_std = torch.std(pt[valid_mask == 1])
    et_miss_mean = torch.mean(et_miss)
    et_miss_std = torch.std(et_miss)
    
    # Calculate global features
    total_energy = torch.sum(energies * valid_mask, dim=1)
    total_pt = torch.sum(pt * valid_mask, dim=1)
    num_objects_present = torch.sum(valid_mask, dim=1)
    
    total_energy_mean = torch.mean(total_energy)
    total_energy_std = torch.std(total_energy)
    total_pt_mean = torch.mean(total_pt)
    total_pt_std = torch.std(total_pt)
    num_objects_mean = torch.mean(num_objects_present)
    num_objects_std = torch.std(num_objects_present)
    
    # Get unique object IDs (excluding zeros)
    unique_obj_ids = torch.unique(obj_ids[obj_ids > 0])
    unique_obj_ids, _ = torch.sort(unique_obj_ids)  # Sort for consistent mapping
    
    # Create preprocessing module
    preproc = PreprocessModule(
        energy_mean=energy_mean.view(1, 1),
        energy_std=energy_std.view(1, 1),
        pt_mean=pt_mean.view(1, 1),
        pt_std=pt_std.view(1, 1),
        et_miss_mean=et_miss_mean.view(1, 1),
        et_miss_std=et_miss_std.view(1, 1),
        total_energy_mean=total_energy_mean.view(1, 1),
        total_energy_std=total_energy_std.view(1, 1),
        total_pt_mean=total_pt_mean.view(1, 1),
        total_pt_std=total_pt_std.view(1, 1),
        num_objects_mean=num_objects_mean.view(1, 1),
        num_objects_std=num_objects_std.view(1, 1),
        obj_types=unique_obj_ids
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

# ----- FREE SECTION: Binary Classifier Definition -----
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for feature recalibration"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual block with SE attention"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        
        # Global event features
        self.global_features = 7  # weight, et_miss, sin/cos phi_miss, total_energy, total_pt, num_objects
        
        # Object-level features processing
        object_features = input_dim - self.global_features
        
        # Reshape object features for 1D convolutions
        self.object_feature_size = 11  # one-hot (5) + energy + pt + eta + sin_phi + cos_phi
        self.num_objects = object_features // self.object_feature_size
        
        # Conv layers for object features
        self.conv_layers = nn.Sequential(
            ResidualBlock(self.object_feature_size, 64),
            nn.MaxPool1d(2, stride=2),
            ResidualBlock(64, 128),
            nn.MaxPool1d(2, stride=2),
            ResidualBlock(128, 256),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fully connected layers
        self.fc_global = nn.Sequential(
            nn.Linear(self.global_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined features processing
        self.fc_combined = nn.Sequential(
            nn.Linear(64 + 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract global and object features
        global_features = x[:, :self.global_features]
        object_features = x[:, self.global_features:]
        
        # Reshape object features for convolution: [batch, features, objects]
        batch_size = x.shape[0]
        reshaped = object_features.view(batch_size, self.num_objects, self.object_feature_size)
        reshaped = reshaped.permute(0, 2, 1)  # [batch, features, objects]
        
        # Process object features with convolutions
        obj_features = self.conv_layers(reshaped).squeeze(-1)  # [batch, channels]
        
        # Process global features
        global_out = self.fc_global(global_features)
        
        # Combine features
        combined = torch.cat([global_out, obj_features], dim=1)
        output = self.fc_combined(combined)
        
        return output.squeeze(1)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    # Initialize metrics tracking
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_auc = 0.0
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target.float())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Accumulate metrics
            train_loss += loss.item() * data.size(0)
            train_preds.extend(torch.sigmoid(output).cpu().detach().numpy())
            train_targets.extend(target.cpu().numpy())
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs} [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}')
        
        # Normalize metrics
        train_loss /= len(train_loader.dataset)
        train_auc = roc_auc_score(train_targets, train_preds)
        train_acc = accuracy_score(train_targets, [1 if p >= 0.5 else 0 for p in train_preds])
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Calculate loss
                loss = criterion(output, target.float())
                
                # Accumulate metrics
                val_loss += loss.item() * data.size(0)
                val_preds.extend(torch.sigmoid(output).cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        # Normalize metrics
        val_loss /= len(val_loader.dataset)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_acc = accuracy_score(val_targets, [1 if p >= 0.5 else 0 for p in val_preds])
        
        # Update LR scheduler based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Store metrics
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        training_acc.append(train_acc)
        validation_acc.append(val_acc)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f} - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
    
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

    # Preprocessing
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=256)

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