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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
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
def preprocess_data(X_train, Y_train, X_val, Y_val):
    # Replace NaN values with zeros
    X_train = torch.nan_to_num(X_train, nan=0.0)
    X_val = torch.nan_to_num(X_val, nan=0.0)
    
    # Extract the event weights (first column)
    weights_train = X_train[:, 0].clone()
    weights_val = X_val[:, 0].clone()
    
    # Normalize the data (excluding weights)
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train[:, 1:].numpy())
    X_val_np = scaler.transform(X_val[:, 1:].numpy())
    
    X_train_processed = torch.tensor(X_train_np, dtype=torch.float32)
    X_val_processed = torch.tensor(X_val_np, dtype=torch.float32)
    
    # Create feature groups based on physics knowledge
    # Each event is represented as: weight, E_T_miss, phi_{E_t}_miss, obj_1, E_1, p_T1, eta_1, phi_1, ...
    
    # Extract global features (missing energy)
    global_features_train = X_train_processed[:, :2]  # E_T_miss and phi_{E_t}_miss
    global_features_val = X_val_processed[:, :2]
    
    # Extract object features and reshape
    # Each object has 5 features: obj_id, E, p_T, eta, phi
    n_objects = (X_train_processed.shape[1] - 2) // 5
    
    # Reshape to have objects as separate entities
    object_features_train = X_train_processed[:, 2:].reshape(X_train.shape[0], n_objects, 5)
    object_features_val = X_val_processed[:, 2:].reshape(X_val.shape[0], n_objects, 5)
    
    # Create masks to identify valid objects (non-zero entries)
    # Object IDs are the first feature of each object
    object_mask_train = (object_features_train[:, :, 0] != 0)
    object_mask_val = (object_features_val[:, :, 0] != 0)
    
    # Replace object_id with 1 where there's a valid object (to use as a mask in the model)
    object_features_train[:, :, 0] = object_mask_train.float()
    object_features_val[:, :, 0] = object_mask_val.float()
    
    # Calculate derived features for each object
    # 1. Calculate transverse mass: m_T = sqrt(E^2 - p_T^2)
    E_train = object_features_train[:, :, 1]
    pt_train = object_features_train[:, :, 2]
    m_T_train = torch.sqrt(torch.clamp(E_train**2 - pt_train**2, min=0))
    
    E_val = object_features_val[:, :, 1]
    pt_val = object_features_val[:, :, 2]
    m_T_val = torch.sqrt(torch.clamp(E_val**2 - pt_val**2, min=0))
    
    # Add transverse mass as a new feature
    object_features_train = torch.cat([object_features_train, m_T_train.unsqueeze(-1)], dim=2)
    object_features_val = torch.cat([object_features_val, m_T_val.unsqueeze(-1)], dim=2)
    
    # 2. Add rapidity: y = 0.5 * ln((E+pz)/(E-pz)) where pz = pT*sinh(eta)
    eta_train = object_features_train[:, :, 3]
    pz_train = pt_train * torch.sinh(eta_train)
    rapidity_train = 0.5 * torch.log((E_train + pz_train) / torch.clamp(E_train - pz_train, min=1e-10))
    
    eta_val = object_features_val[:, :, 3]
    pz_val = pt_val * torch.sinh(eta_val)
    rapidity_val = 0.5 * torch.log((E_val + pz_val) / torch.clamp(E_val - pz_val, min=1e-10))
    
    # Add rapidity as a new feature
    object_features_train = torch.cat([object_features_train, rapidity_train.unsqueeze(-1)], dim=2)
    object_features_val = torch.cat([object_features_val, rapidity_val.unsqueeze(-1)], dim=2)
    
    # Calculate global event features
    # 1. Total scalar sum of pT (HT)
    HT_train = torch.sum(pt_train * object_mask_train, dim=1, keepdim=True)
    HT_val = torch.sum(pt_val * object_mask_val, dim=1, keepdim=True)
    
    # 2. Number of objects
    n_objects_train = torch.sum(object_mask_train, dim=1, keepdim=True)
    n_objects_val = torch.sum(object_mask_val, dim=1, keepdim=True)
    
    # Append global derived features
    global_features_train = torch.cat([global_features_train, HT_train, n_objects_train], dim=1)
    global_features_val = torch.cat([global_features_val, HT_val, n_objects_val], dim=1)
    
    # Flatten object features for the dense model
    object_features_train_flat = object_features_train.reshape(X_train.shape[0], -1)
    object_features_val_flat = object_features_val.reshape(X_val.shape[0], -1)
    
    # Combine global and object features
    X_train_final = torch.cat([global_features_train, object_features_train_flat], dim=1)
    X_val_final = torch.cat([global_features_val, object_features_val_flat], dim=1)
    
    # Create dataloaders
    batch_size = 256
    train_dataset = TensorDataset(X_train_final, Y_train, weights_train)
    val_dataset = TensorDataset(X_val_final, Y_val, weights_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        
        # Define network architecture
        self.bn_input = nn.BatchNorm1d(input_dim)
        
        # Define hidden layers with residual connections
        hidden_dim1 = 512
        hidden_dim2 = 256
        hidden_dim3 = 128
        hidden_dim4 = 64
        
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.bn3 = nn.BatchNorm1d(hidden_dim3)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.bn4 = nn.BatchNorm1d(hidden_dim4)
        self.dropout4 = nn.Dropout(0.2)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim4, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Apply batch normalization to the input
        x = self.bn_input(x)
        
        # First block
        x1 = F.leaky_relu(self.bn1(self.fc1(x)))
        x1 = self.dropout1(x1)
        
        # Second block
        x2 = F.leaky_relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout2(x2)
        
        # Third block
        x3 = F.leaky_relu(self.bn3(self.fc3(x2)))
        x3 = self.dropout3(x3)
        
        # Fourth block
        x4 = F.leaky_relu(self.bn4(self.fc4(x3)))
        x4 = self.dropout4(x4)
        
        # Output layer
        out = self.fc_out(x4)
        
        return torch.sigmoid(out).squeeze()

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use binary cross entropy loss
    criterion = nn.BCELoss()
    
    # Use Adam optimizer with learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Initialize metrics tracking
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    validation_auc = []
    best_auc = 0.0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for inputs, targets, weights in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float()  # Convert to float for BCE
            weights = weights.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate weighted loss
            loss = criterion(outputs, targets)
            weighted_loss = (loss * weights).mean()
            
            # Backward pass and optimize
            weighted_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += weighted_loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Store predictions and targets for AUC calculation
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_loss /= total
        epoch_acc = correct / total
        epoch_auc = roc_auc_score(all_targets, all_preds)
        training_loss.append(epoch_loss)
        training_acc.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets, weights in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).float()
                weights = weights.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate weighted loss
                loss = criterion(outputs, targets)
                weighted_loss = (loss * weights).mean()
                
                # Update metrics
                val_loss += weighted_loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Store predictions and targets for AUC calculation
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch metrics
        val_loss /= total
        val_acc = correct / total
        val_auc = roc_auc_score(all_targets, all_preds)
        validation_loss.append(val_loss)
        validation_acc.append(val_acc)
        validation_auc.append(val_auc)
        
        # Print epoch metrics
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Train AUC: {epoch_auc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save the best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            print(f'New best model with validation AUC: {best_auc:.4f}')
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Loaded best model with validation AUC: {best_auc:.4f}')
    
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

    # Model Initialization
    # Get input dimension from the first batch
    for batch in train_loader:
        input_dim = batch[0].shape[1]
        break
        
    model = Classifier(input_dim=input_dim)

    # Training
    epochs = 1 if dryrun else 30

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