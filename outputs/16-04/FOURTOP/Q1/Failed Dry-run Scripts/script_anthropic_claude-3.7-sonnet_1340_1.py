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
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn.functional as F
from torch.utils.data import Dataset

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
    # Reshape data to extract physics information
    def extract_features(X):
        batch_size = X.shape[0]
        
        # First 3 elements: weight, E_T_miss, phi_{E_t}_miss
        basic_features = X[:, :3]
        
        # Rest of the data is objects with 5 values per object (obj_id, E, p_T, eta, phi)
        object_data = X[:, 3:].reshape(batch_size, -1, 5)
        
        # Create masks for non-zero objects (where object ID is not 0)
        mask = object_data[:, :, 0] != 0
        
        # Extract kinematic features
        energy = object_data[:, :, 1]  # E
        pt = object_data[:, :, 2]      # p_T
        eta = object_data[:, :, 3]     # eta
        phi = object_data[:, :, 4]     # phi
        
        # Identify different physics objects by their ID
        # Assuming IDs might be integers: 1=electron, 2=muon, 3=jet, etc.
        obj_ids = object_data[:, :, 0]
        
        # Features to extract (could be expanded)
        features = []
        
        # Add basic features
        features.append(basic_features)
        
        # Calculate mean, std, max, min, sum for each kinematic variable
        for var in [energy, pt, eta, phi]:
            # Apply mask to only consider non-zero objects
            masked_var = torch.where(mask.unsqueeze(-1), var.unsqueeze(-1), torch.zeros_like(var.unsqueeze(-1)))
            
            # Global statistics across all objects
            features.append(torch.sum(masked_var, dim=1))
            features.append(torch.mean(masked_var, dim=1))
            features.append(torch.std(masked_var, dim=1))
            
            # Get max/min (ignoring zeros)
            max_val = torch.max(torch.where(mask, var, torch.tensor(-float('inf'), dtype=torch.float32)), dim=1)[0].unsqueeze(-1)
            min_val = torch.min(torch.where(mask, var, torch.tensor(float('inf'), dtype=torch.float32)), dim=1)[0].unsqueeze(-1)
            features.append(max_val)
            features.append(min_val)
        
        # Count number of objects
        num_objects = torch.sum(mask.float(), dim=1).unsqueeze(-1)
        features.append(num_objects)
        
        # Unique object type counts
        unique_obj_ids = torch.unique(obj_ids[obj_ids > 0])
        for obj_id in unique_obj_ids:
            obj_count = torch.sum((obj_ids == obj_id).float(), dim=1).unsqueeze(-1)
            features.append(obj_count)
            
            # Statistics per object type
            obj_mask = obj_ids == obj_id
            for var in [energy, pt, eta, phi]:
                # Apply object-specific mask
                masked_obj_var = torch.where(obj_mask.unsqueeze(-1), var.unsqueeze(-1), torch.zeros_like(var.unsqueeze(-1)))
                if torch.sum(obj_mask) > 0:  # Only if we have any of these objects
                    features.append(torch.sum(masked_obj_var, dim=1))
                    
                    # Get mean for this object type (avoid division by zero)
                    obj_count_safe = torch.clamp(obj_count, min=1.0)  # Avoid division by zero
                    obj_mean = torch.sum(masked_obj_var, dim=1) / obj_count_safe
                    features.append(obj_mean)
        
        # Calculate delta R between top N objects by pT
        # Sort objects by pT
        sorted_indices = torch.argsort(pt, dim=1, descending=True)
        n_top = 4  # Consider top 4 objects
        if sorted_indices.shape[1] >= n_top:
            top_indices = sorted_indices[:, :n_top]
            
            # Get eta and phi of top objects
            top_eta = torch.gather(eta, 1, top_indices)
            top_phi = torch.gather(phi, 1, top_indices)
            
            # Calculate delta R between pairs of top objects
            for i in range(n_top):
                for j in range(i+1, n_top):
                    delta_eta = top_eta[:, i] - top_eta[:, j]
                    delta_phi = torch.abs(top_phi[:, i] - top_phi[:, j])
                    # Handle periodicity of phi
                    delta_phi = torch.where(delta_phi > math.pi, 2*math.pi - delta_phi, delta_phi)
                    delta_r = torch.sqrt(delta_eta**2 + delta_phi**2).unsqueeze(-1)
                    features.append(delta_r)
        
        # Concatenate all features
        return torch.cat(features, dim=1)
    
    # Extract features
    X_train_features = extract_features(X_train)
    X_val_features = extract_features(X_val)
    
    # Apply standard scaling
    scaler = StandardScaler()
    X_train_scaled = torch.tensor(scaler.fit_transform(X_train_features), dtype=torch.float32)
    X_val_scaled = torch.tensor(scaler.transform(X_val_features), dtype=torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_scaled, Y_train)
    val_dataset = TensorDataset(X_val_scaled, Y_val)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    return train_loader, val_loader

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        # Define a more complex architecture
        self.dropout_rate = 0.3
        
        # First block
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Second block
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Third block
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Output layer
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        # First block with skip connection
        x1 = F.leaky_relu(self.bn1(self.fc1(x)))
        x1 = F.dropout(x1, p=self.dropout_rate, training=self.training)
        
        # Second block with skip connection
        x2 = F.leaky_relu(self.bn2(self.fc2(x1)))
        x2 = F.dropout(x2, p=self.dropout_rate, training=self.training)
        
        # Third block
        x3 = F.leaky_relu(self.bn3(self.fc3(x2)))
        x3 = F.dropout(x3, p=self.dropout_rate, training=self.training)
        
        # Output layer
        x_out = self.fc_out(x3)
        
        return torch.sigmoid(x_out).squeeze()

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Initialize metrics tracking
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_predictions = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs >= 0.5).float()
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
            
            # Store predictions and targets for AUC calculation
            train_predictions.extend(outputs.cpu().detach().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch metrics for training
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        train_auc = roc_auc_score(train_targets, train_predictions)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update metrics
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs >= 0.5).float()
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
                
                # Store predictions and targets for AUC calculation
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch metrics for validation
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_targets, val_predictions)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Store metrics
        training_loss.append(epoch_train_loss)
        validation_loss.append(epoch_val_loss)
        training_acc.append(epoch_train_acc)
        validation_acc.append(epoch_val_acc)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Train AUC: {train_auc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, '
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

    # Get input dimension from a batch
    for inputs, _ in train_loader:
        input_dim = inputs.shape[1]
        break

    # Model Initialization
    model = Classifier(input_dim=input_dim)

    # Training
    epochs = 1 if dryrun else 20

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

        # Calculate and plot ROC curve and AUC for final model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trained_model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = trained_model(inputs)
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(val_targets, val_predictions)
        auc = roc_auc_score(val_targets, val_predictions)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig('roc_curve.png')

        print(f"Full run complete. Outputs and model saved successfully. Final AUC: {auc:.4f}")
    else:
        print("Dry-run complete. No outputs saved.")

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)