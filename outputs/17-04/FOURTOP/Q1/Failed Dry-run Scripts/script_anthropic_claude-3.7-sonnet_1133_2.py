# ----- FREE SECTION: Import Libraries -----
import numpy as np
import pandas as pd
import math
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.nn.utils import weight_norm

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
    # Feature extraction from raw data
    # Identify weight, E_T_miss, phi_miss, then objects
    X_train_processed = feature_engineering(X_train)
    X_val_processed = feature_engineering(X_val)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_processed = torch.tensor(scaler.fit_transform(X_train_processed), dtype=torch.float32)
    X_val_processed = torch.tensor(scaler.transform(X_val_processed), dtype=torch.float32)
    
    # Create dataloaders
    batch_size = 256
    train_dataset = TensorDataset(X_train_processed, Y_train)
    val_dataset = TensorDataset(X_val_processed, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def feature_engineering(X):
    # Convert tensor to numpy for easier manipulation
    X_np = X.numpy()
    
    # Extract first three columns: weight, E_T_miss, phi_miss
    base_features = X_np[:, :3]
    
    # Process object features
    obj_data = X_np[:, 3:]
    
    # Reshape to interpret 4-vectors: each object has 5 values (obj_id, E, p_T, eta, phi)
    n_samples = X_np.shape[0]
    obj_data_reshaped = obj_data.reshape(n_samples, -1, 5)
    
    # Count non-zero objects per sample
    mask = (obj_data_reshaped[:, :, 0] != 0)  # objects have non-zero identifiers
    obj_counts = mask.sum(axis=1, keepdims=True)
    
    # Extract basic kinematic features
    features_list = [base_features, obj_counts]
    
    # Calculate physics-motivated features
    for i in range(min(5, obj_data_reshaped.shape[1])):  # Use first 5 objects
        obj_mask = mask[:, i:i+1]  # Keep dim for broadcasting
        
        # Only process valid objects
        valid_objs = obj_data_reshaped[:, i]
        obj_id = valid_objs[:, 0:1]  # Object identifier
        E = valid_objs[:, 1:2]      # Energy
        pt = valid_objs[:, 2:3]     # Transverse momentum
        eta = valid_objs[:, 3:4]    # Pseudorapidity
        phi = valid_objs[:, 4:5]    # Azimuthal angle
        
        # Set features to 0 for padding
        obj_id = obj_id * obj_mask
        E = E * obj_mask
        pt = pt * obj_mask
        eta = eta * obj_mask
        phi = phi * obj_mask
        
        # Add basic features and derived features
        features_list.extend([obj_id, E, pt, eta, phi, E/pt, pt/E])
    
    # Calculate inter-object features for first few objects
    for i in range(min(3, obj_data_reshaped.shape[1])):
        for j in range(i+1, min(4, obj_data_reshaped.shape[1])):
            # Make sure both objects are valid
            combined_mask = mask[:, i:i+1] & mask[:, j:j+1]
            
            # Get 4-vectors
            obj_i = obj_data_reshaped[:, i]
            obj_j = obj_data_reshaped[:, j]
            
            # Calculate delta phi (handle circular nature of phi)
            phi_i = obj_i[:, 4]
            phi_j = obj_j[:, 4]
            delta_phi = np.abs(phi_i - phi_j)
            delta_phi = np.minimum(delta_phi, 2*np.pi - delta_phi)
            delta_phi = delta_phi.reshape(-1, 1)
            
            # Calculate delta eta
            eta_i = obj_i[:, 3]
            eta_j = obj_j[:, 3]
            delta_eta = np.abs(eta_i - eta_j).reshape(-1, 1)
            
            # Calculate delta R = sqrt(delta_phi^2 + delta_eta^2)
            delta_R = np.sqrt(delta_phi**2 + delta_eta**2)
            
            # Pt ratio
            pt_i = obj_i[:, 2]
            pt_j = obj_j[:, 2]
            pt_ratio = np.minimum(pt_i, pt_j) / np.maximum(pt_i, pt_j)
            pt_ratio = pt_ratio.reshape(-1, 1)
            pt_ratio = np.where(np.isnan(pt_ratio), 0, pt_ratio)  # Handle division by zero
            
            # Apply mask
            delta_phi = delta_phi * combined_mask
            delta_eta = delta_eta * combined_mask
            delta_R = delta_R * combined_mask
            pt_ratio = pt_ratio * combined_mask
            
            features_list.extend([delta_phi, delta_eta, delta_R, pt_ratio])
    
    # Calculate global event features
    valid_pts = obj_data_reshaped[:, :, 2] * mask  # Only consider pt of valid objects
    
    # Sum of pt for all objects
    sum_pt = np.sum(valid_pts, axis=1, keepdims=True)
    
    # Missing ET ratio to sum of pT
    et_miss_ratio = base_features[:, 1:2] / np.maximum(sum_pt, 1e-8)  # Avoid division by zero
    et_miss_ratio = np.where(np.isnan(et_miss_ratio), 0, et_miss_ratio)
    
    features_list.extend([sum_pt, et_miss_ratio])
    
    # Combine all features
    processed_features = np.hstack(features_list)
    
    # Replace NaNs and infs
    processed_features = np.nan_to_num(processed_features)
    
    return processed_features

# ----- FREE SECTION: Binary Classifier Definition -----
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = weight_norm(nn.Linear(input_dim, hidden_dim))
        self.linear2 = weight_norm(nn.Linear(hidden_dim, input_dim))
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        
    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.1)
        out = self.bn2(self.linear2(out))
        out += identity
        return F.leaky_relu(out, negative_slope=0.1)

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        
        # Define network architecture
        self.input_dim = input_dim
        hidden_dim = 256
        
        # Input layer with batch normalization
        self.input_layer = nn.Sequential(
            weight_norm(nn.Linear(input_dim, hidden_dim)),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Residual blocks
        self.res_block1 = ResidualBlock(hidden_dim, hidden_dim//2)
        self.res_block2 = ResidualBlock(hidden_dim, hidden_dim//2)
        self.res_block3 = ResidualBlock(hidden_dim, hidden_dim//2)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Dropout(0.3),
            weight_norm(nn.Linear(hidden_dim, hidden_dim//2)),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        logits = self.output_layer(x).squeeze(-1)
        return logits

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Track metrics
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_val_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == labels.float()).sum().item()
            train_total += labels.size(0)
            
            # Save predictions and labels for AUC calculation
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_auc = roc_auc_score(train_labels, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                
                val_loss += loss.item() * inputs.size(0)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == labels.float()).sum().item()
                val_total += labels.size(0)
                
                # Save predictions and labels for AUC calculation
                val_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_labels, val_preds)
        
        # Update learning rate using AUC score
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Store metrics
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        training_acc.append(train_acc)
        validation_acc.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, '
              f'Val Acc: {val_acc:.4f}, '
              f'Train AUC: {train_auc:.4f}, '
              f'Val AUC: {val_auc:.4f}')
    
    # Load best model
    if best_model_state is not None:
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
    train_loader, val_loader = preprocess_data(X_train, Y_train, X_val, Y_val)

    # Get input dimension from first batch
    for X_batch, _ in train_loader:
        input_dim = X_batch.shape[1]
        break

    # Model Initialization
    model = Classifier(input_dim=input_dim)

    # Training
    epochs = 1 if dryrun else 15

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)

    if not dryrun:
        # Save Model
        model_filename = sys.argv[0].replace(".py", "") + "_model.pth"
        torch.save(trained_model.state_dict(), model_filename)

        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, "Loss", "training_loss.png")
        plot_and_save(training_acc, validation_acc, "Accuracy", "training_accuracy.png")

        print("Full run complete. Outputs and model saved successfully.")
    else:
        print("Dry-run complete. No outputs saved.")

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)