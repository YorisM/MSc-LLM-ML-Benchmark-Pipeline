# ----- FIXED SECTION: Import Libraries -----
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

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
def preprocess_data(X_train, Y_train, X_val, Y_val):
    # Extract features and organize them
    def extract_features(X):
        # First 3 columns: weight, E_T_miss, phi_{E_t}_miss
        basic_features = X[:, :3].clone()
        
        # The rest are objects with their properties (obj_id, E, p_T, eta, phi)
        obj_features = X[:, 3:].clone()
        
        # Reshape to get each object separately
        n_samples = X.shape[0]
        obj_features = obj_features.reshape(n_samples, -1, 5)
        
        # Count non-zero objects (those with non-zero energy)
        obj_counts = torch.sum(obj_features[:, :, 1] > 0, dim=1, keepdim=True)
        
        # Calculate object statistics
        # Sum of energies, transverse momenta, and counts of different object types
        total_energy = torch.sum(obj_features[:, :, 1], dim=1, keepdim=True)
        total_pt = torch.sum(obj_features[:, :, 2], dim=1, keepdim=True)
        
        # Count objects by type (object IDs usually start at 1)
        max_obj_id = int(torch.max(obj_features[:, :, 0]).item())
        obj_type_counts = torch.zeros((n_samples, max_obj_id + 1), dtype=torch.float32)
        
        for i in range(n_samples):
            for j in range(obj_features.shape[1]):
                if obj_features[i, j, 1] > 0:  # If energy > 0
                    obj_id = int(obj_features[i, j, 0].item())
                    if 0 <= obj_id <= max_obj_id:
                        obj_type_counts[i, obj_id] += 1
        
        # Calculate pairwise angular differences and invariant masses for objects
        n_objects = obj_features.shape[1]
        angular_diffs = []
        invariant_masses = []
        
        for i in range(n_objects):
            for j in range(i+1, n_objects):
                # Angular difference in phi
                phi_i = obj_features[:, i, 4]
                phi_j = obj_features[:, j, 4]
                dphi = torch.abs(phi_i - phi_j)
                dphi = torch.min(dphi, 2*np.pi - dphi)
                angular_diffs.append(dphi.unsqueeze(1))
                
                # Pseudorapidity difference
                eta_i = obj_features[:, i, 3]
                eta_j = obj_features[:, j, 3]
                deta = torch.abs(eta_i - eta_j)
                angular_diffs.append(deta.unsqueeze(1))
                
                # Invariant mass calculation (approximate formula)
                E_i = obj_features[:, i, 1]
                E_j = obj_features[:, j, 1]
                pt_i = obj_features[:, i, 2]
                pt_j = obj_features[:, j, 2]
                
                # Only calculate for pairs where both objects have energy
                valid_pair = (E_i > 0) & (E_j > 0)
                m_ij = torch.zeros_like(E_i)
                
                # Use a simplified invariant mass formula
                dR = torch.sqrt(dphi**2 + deta**2)
                m_ij[valid_pair] = torch.sqrt(2 * pt_i[valid_pair] * pt_j[valid_pair] * 
                                              (torch.cosh(deta[valid_pair]) - torch.cos(dphi[valid_pair])))
                invariant_masses.append(m_ij.unsqueeze(1))
        
        # Combine all features
        all_features = [basic_features, obj_counts, total_energy, total_pt, obj_type_counts]
        
        if angular_diffs:
            all_features.extend(angular_diffs)
        
        if invariant_masses:
            all_features.extend(invariant_masses)
        
        # Concatenate all features
        combined_features = torch.cat(all_features, dim=1)
        
        return combined_features
    
    # Apply feature extraction
    X_train_processed = extract_features(X_train)
    X_val_processed = extract_features(X_val)
    
    # Normalize features (excluding weight column)
    weight_train = X_train_processed[:, 0:1].clone()
    weight_val = X_val_processed[:, 0:1].clone()
    
    features_train = X_train_processed[:, 1:].clone()
    features_val = X_val_processed[:, 1:].clone()
    
    # Calculate mean and std on training data only
    mean = torch.mean(features_train, dim=0)
    std = torch.std(features_train, dim=0) + 1e-8  # Avoid division by zero
    
    # Normalize both train and validation sets using training statistics
    features_train = (features_train - mean) / std
    features_val = (features_val - mean) / std
    
    # Replace NaN and inf values
    features_train = torch.nan_to_num(features_train, nan=0.0, posinf=0.0, neginf=0.0)
    features_val = torch.nan_to_num(features_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Reattach weight column
    X_train_processed = torch.cat([weight_train, features_train], dim=1)
    X_val_processed = torch.cat([weight_val, features_val], dim=1)
    
    return X_train_processed, Y_train, X_val_processed, Y_val

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        # Define architecture with residual connections
        self.input_layer = nn.Linear(input_dim, 256)
        
        # Several residual blocks
        self.fc1 = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 128)
        self.bn6 = nn.BatchNorm1d(128)
        
        # Output layer
        self.output_layer = nn.Linear(128, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Extract sample weights (first column)
        weights = x[:, 0:1]
        x = x[:, 1:]  # Remove weights from input features
        
        # Initial layer
        x = F.leaky_relu(self.input_layer(x))
        
        # First residual block
        residual = x
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        x = F.leaky_relu(x + residual)  # Residual connection
        
        # Second residual block
        residual = x
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.bn4(self.fc4(x))
        x = F.leaky_relu(x + residual)  # Residual connection
        
        # Final layers
        x = F.leaky_relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn6(self.fc6(x)))
        
        # Output layer (no activation, will use sigmoid in loss function)
        x = self.output_layer(x)
        
        # Return output and sample weights separately
        return x, weights

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, X_train, Y_train, X_val, Y_val, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    # Use binary cross entropy loss
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # 'none' to allow weighting by sample
    
    # Adam optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    # Arrays to store metrics
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    validation_auc = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_predictions = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, sample_weights = model(inputs)
            outputs = outputs.squeeze()
            
            # Calculate weighted loss
            loss = criterion(outputs, targets.float())
            weighted_loss = (loss * sample_weights.squeeze()).mean()
            
            # Backward pass and optimize
            weighted_loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += weighted_loss.item() * inputs.size(0)
            predicted = torch.sigmoid(outputs) >= 0.5
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
            
            # Save predictions and targets for AUC calculation
            train_predictions.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Compute training metrics for the epoch
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_auc = roc_auc_score(train_targets, train_predictions)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs, sample_weights = model(inputs)
                outputs = outputs.squeeze()
                
                # Calculate weighted loss
                loss = criterion(outputs, targets.float())
                weighted_loss = (loss * sample_weights.squeeze()).mean()
                
                # Calculate metrics
                val_loss += weighted_loss.item() * inputs.size(0)
                predicted = torch.sigmoid(outputs) >= 0.5
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
                
                # Save predictions and targets for AUC calculation
                val_predictions.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Compute validation metrics for the epoch
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_targets, val_predictions)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Store metrics
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        training_acc.append(train_acc)
        validation_acc.append(val_acc)
        validation_auc.append(val_auc)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
    
    # Print best validation AUC
    best_epoch = np.argmax(validation_auc)
    print(f'Best Validation AUC: {validation_auc[best_epoch]:.4f} at Epoch {best_epoch+1}')
    
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
    X_train, Y_train, X_val, Y_val = preprocess_data(X_train, Y_train, X_val, Y_val)

    # Model Initialization
    model = Classifier(input_dim=X_train.shape[1]-1)  # -1 because we separate the weight

    # Training (dryrun limits epochs)
    epochs = 1 if dryrun else 20

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, X_train, Y_train, X_val, Y_val, epochs=epochs)

    # Save Model
    model_filename = sys.argv[0].replace(".py", "") + "_model.pth"
    torch.save(trained_model.state_dict(), model_filename)

    # Plot Metrics
    plot_and_save(training_loss, validation_loss, "Loss", "training_loss.png")
    plot_and_save(training_acc, validation_acc, "Accuracy", "training_accuracy.png")

    print("Training complete. Outputs and model saved successfully.")

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)