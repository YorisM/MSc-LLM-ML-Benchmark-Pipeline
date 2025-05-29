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
import os
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
    # Extract the event weights (first column)
    train_weights = X_train[:, 0].clone()
    val_weights = X_val[:, 0].clone()
    
    # Extract missing energy features (2nd and 3rd columns)
    E_T_miss_train = X_train[:, 1].clone()
    phi_ET_miss_train = X_train[:, 2].clone()
    E_T_miss_val = X_val[:, 1].clone()
    phi_ET_miss_val = X_val[:, 2].clone()
    
    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    X_train_np = X_train.numpy()
    X_val_np = X_val.numpy()
    
    # Fit the scaler on training data and transform both datasets
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_val_scaled = scaler.transform(X_val_np)
    
    # Convert back to tensors
    X_train_processed = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_processed = torch.tensor(X_val_scaled, dtype=torch.float32)
    
    # Extract physics objects
    # Reshape the data to extract 4-vectors
    # Starting from index 3, we have objects with 5 values each (obj_id, E, pT, eta, phi)
    object_features = []
    
    for i in range(3, X_train.shape[1], 5):
        if i + 4 < X_train.shape[1]:
            # For each object, calculate some physics-inspired features
            obj_train = X_train[:, i:i+5].clone()
            obj_val = X_val[:, i:i+5].clone()
            
            # Check if this is a valid object (non-zero values)
            valid_objs_train = torch.sum(torch.abs(obj_train[:, 1:]), dim=1) > 1e-6  # Check E, pT, eta, phi
            valid_objs_val = torch.sum(torch.abs(obj_val[:, 1:]), dim=1) > 1e-6
            
            # Create masks for valid objects
            train_mask = valid_objs_train.float().unsqueeze(1)
            val_mask = valid_objs_val.float().unsqueeze(1)
            
            # Apply masks to the objects
            obj_train_masked = obj_train * train_mask
            obj_val_masked = obj_val * val_mask
            
            # Add these as new features
            object_features.append({
                'train': obj_train_masked,
                'val': obj_val_masked
            })
    
    # Count number of valid objects per event
    num_objects_train = torch.zeros(X_train.shape[0], dtype=torch.float32)
    num_objects_val = torch.zeros(X_val.shape[0], dtype=torch.float32)
    
    for obj_dict in object_features:
        num_objects_train += (torch.sum(torch.abs(obj_dict['train'][:, 1:]), dim=1) > 1e-6).float()
        num_objects_val += (torch.sum(torch.abs(obj_dict['val'][:, 1:]), dim=1) > 1e-6).float()
    
    # Add object count as a feature
    num_objects_train = num_objects_train.unsqueeze(1)
    num_objects_val = num_objects_val.unsqueeze(1)
    
    # Create dataset with weights for sampling
    batch_size = 256
    train_dataset = TensorDataset(X_train_processed, Y_train, train_weights)
    val_dataset = TensorDataset(X_val_processed, Y_val, val_weights)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        # Define architecture with attention to physics-based features
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        x = self.fc5(x)
        return torch.sigmoid(x).squeeze()

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    # Use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use Binary Cross Entropy loss
    criterion = nn.BCELoss()  # Binary Cross Entropy loss
    
    # Adam optimizer with learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Tracking metrics
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_val_auc = 0.0
    best_model = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_predictions = []
        train_true_labels = []
        
        for inputs, labels, weights in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            weights = weights.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Apply sample weights to loss
            sample_loss = F.binary_cross_entropy(outputs, labels, reduction='none')
            loss = (sample_loss * weights).mean()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs >= 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Collect predictions for AUC calculation
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_true_labels.extend(labels.detach().cpu().numpy())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for inputs, labels, weights in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                weights = weights.to(device)
                
                outputs = model(inputs)
                
                # Apply sample weights to loss
                sample_loss = F.binary_cross_entropy(outputs, labels, reduction='none')
                loss = (sample_loss * weights).mean()
                
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs >= 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Collect predictions for AUC calculation
                val_predictions.extend(outputs.detach().cpu().numpy())
                val_true_labels.extend(labels.detach().cpu().numpy())
        
        # Calculate epoch metrics
        train_loss = train_loss / train_total
        val_loss = val_loss / val_total
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        
        # Calculate AUC
        train_auc = roc_auc_score(train_true_labels, train_predictions)
        val_auc = roc_auc_score(val_true_labels, val_predictions)
        
        # Store metrics
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        training_acc.append(train_auc)  # Using AUC instead of accuracy
        validation_acc.append(val_auc)  # Using AUC instead of accuracy
        
        # Update scheduler based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model.state_dict().copy()
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Training Loss: {train_loss:.4f}, Training AUC: {train_auc:.4f}')
        print(f'  Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}')
    
    # Load the best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
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
    model = Classifier(input_dim=X_train.shape[1])

    # Training
    epochs = 1 if dryrun else 15

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)

    if not dryrun:
        # Save Model
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        model_filename = os.path.join(script_dir, f"{base}_model.pth")
        os.makedirs(script_dir, exist_ok=True)

        torch.save(trained_model.state_dict(), f"{base}_model.pth")

        scripted = torch.jit.script(trained_model)
        scripted_filename = os.path.join(script_dir, f"{base}_scripted.pt")
        scripted.save(scripted_filename)
        print(f"Saved TorchScript module to {scripted_filename}")

        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, f"Loss - {base}", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, f"AUC - {base}", os.path.join(script_dir, f"{base}_accuracy.png"))

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)