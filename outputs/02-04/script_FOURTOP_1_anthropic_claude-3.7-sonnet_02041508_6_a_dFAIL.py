import pandas as pd
import torch

X_train_df = pd.read_csv('./data/fourtops/X_train.csv')
Y_train_df  = pd.read_csv('./data/fourtops/Y_train.csv')
X_val_df   = pd.read_csv('./data/fourtops/X_val.csv')
Y_val_df   = pd.read_csv('./data/fourtops/Y_val.csv')

X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()

print(f"X_train shape: {X_train.shape}\nY_train shape: {Y_train.shape}\nX_val shape: {X_val.shape}\nY_val shape: {Y_val.shape}")

# LLM Reponse Below
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def preprocess_data(X):
    # Extract features from the data
    n_samples = X.shape[0]
    
    # First 3 columns are: weight, E_T_miss, phi_{E_t}_miss
    weights = X[:, 0].reshape(-1, 1)
    et_miss = X[:, 1].reshape(-1, 1)
    phi_et_miss = X[:, 2].reshape(-1, 1)
    
    # The rest are objects with their properties
    # Each object has 5 values: obj_id, E, p_T, eta, phi
    object_features = X[:, 3:]
    
    # Reshape to get individual objects
    num_objects = (object_features.shape[1]) // 5
    objects = object_features.reshape(n_samples, num_objects, 5)
    
    # Count non-zero objects (assuming obj_id=0 means padding)
    object_counts = torch.sum(objects[:, :, 0] != 0, dim=1).reshape(-1, 1).float()
    
    # Calculate basic statistical features for each property across objects
    # Ignore zeroes (padding)
    energy_stats = calculate_stats(objects[:, :, 1])
    pt_stats = calculate_stats(objects[:, :, 2])
    eta_stats = calculate_stats(objects[:, :, 3])
    phi_stats = calculate_stats(objects[:, :, 4])
    
    # Count specific object types (assuming obj_id indicates type)
    # For example, count objects of type 1, 2, etc.
    obj_type_counts = []
    for i in range(1, 11):  # Assuming 10 possible object types
        count = torch.sum(objects[:, :, 0] == i, dim=1).reshape(-1, 1).float()
        obj_type_counts.append(count)
    obj_type_counts = torch.cat(obj_type_counts, dim=1)
    
    # Calculate pairwise delta R between objects
    delta_r_stats = calculate_delta_r_stats(objects)
    
    # Concatenate all features
    features = torch.cat([weights, et_miss, phi_et_miss, object_counts, 
                          energy_stats, pt_stats, eta_stats, phi_stats,
                          obj_type_counts, delta_r_stats], dim=1)
    
    # Normalize features
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    std[std == 0] = 1  # Avoid division by zero
    normalized_features = (features - mean) / std
    
    return normalized_features


def calculate_stats(property_values):
    # Calculate statistics for non-zero values
    mask = property_values != 0
    
    # Handle empty rows (all zeros)
    row_has_values = torch.sum(mask, dim=1) > 0
    
    # Initialize stats with zeros
    sum_values = torch.zeros(property_values.shape[0], 1, device=property_values.device)
    mean_values = torch.zeros(property_values.shape[0], 1, device=property_values.device)
    max_values = torch.zeros(property_values.shape[0], 1, device=property_values.device)
    min_values = torch.zeros(property_values.shape[0], 1, device=property_values.device)
    std_values = torch.zeros(property_values.shape[0], 1, device=property_values.device)
    
    # Calculate stats for rows with non-zero values
    for i in range(property_values.shape[0]):
        if row_has_values[i]:
            valid_values = property_values[i][mask[i]]
            sum_values[i] = valid_values.sum()
            mean_values[i] = valid_values.mean()
            max_values[i] = valid_values.max()
            min_values[i] = valid_values.min()
            if valid_values.shape[0] > 1:
                std_values[i] = valid_values.std()
    
    # Concatenate all stats
    return torch.cat([sum_values, mean_values, max_values, min_values, std_values], dim=1)


def calculate_delta_r_stats(objects):
    n_samples = objects.shape[0]
    delta_r_mean = torch.zeros(n_samples, 1, device=objects.device)
    delta_r_max = torch.zeros(n_samples, 1, device=objects.device)
    delta_r_min = torch.ones(n_samples, 1, device=objects.device) * float('inf')
    
    for i in range(n_samples):
        # Get non-padding objects
        valid_objects = objects[i, objects[i, :, 0] != 0]
        n_valid = valid_objects.shape[0]
        
        if n_valid > 1:
            # Calculate pairwise delta R
            delta_r_values = []
            for j in range(n_valid):
                for k in range(j + 1, n_valid):
                    eta1, phi1 = valid_objects[j, 3], valid_objects[j, 4]
                    eta2, phi2 = valid_objects[k, 3], valid_objects[k, 4]
                    delta_eta = eta1 - eta2
                    delta_phi = abs(phi1 - phi2)
                    # Handle phi periodicity
                    delta_phi = torch.min(delta_phi, 2 * torch.tensor(np.pi) - delta_phi)
                    delta_r = torch.sqrt(delta_eta**2 + delta_phi**2)
                    delta_r_values.append(delta_r)
            
            if delta_r_values:  # Check if there are any values
                delta_r_tensor = torch.tensor(delta_r_values, device=objects.device)
                delta_r_mean[i] = delta_r_tensor.mean()
                delta_r_max[i] = delta_r_tensor.max()
                delta_r_min[i] = delta_r_tensor.min()
    
    # Replace inf with 0
    delta_r_min[delta_r_min == float('inf')] = 0
    
    # Concatenate all delta R stats
    return torch.cat([delta_r_mean, delta_r_max, delta_r_min], dim=1)


class PhysicsClassifier(nn.Module):
    def __init__(self, input_size):
        super(PhysicsClassifier, self).__init__()
        # Define layers
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 1)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.batch_norm4 = nn.BatchNorm1d(32)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm3(self.layer3(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm4(self.layer4(x)))
        x = self.sigmoid(self.layer5(x))
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    train_losses = []
    val_losses = []
    val_aucs = []
    best_auc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validate the model
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate AUC
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_aucs.append(val_auc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        
        # Save the best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig('training_history.png')
    
    return model, best_auc


def dry_run():
    # Create synthetic data for dry run
    n_samples = 1000
    n_features = 106
    
    X_train = torch.rand(n_samples, n_features)
    Y_train = torch.randint(0, 2, (n_samples,)).float()
    X_val = torch.rand(n_samples // 10, n_features)
    Y_val = torch.randint(0, 2, (n_samples // 10,)).float()
    
    # Processing, model creation, and training as in main function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Preprocess data
    print("Preprocessing data...")
    X_train_processed = preprocess_data(X_train)
    X_val_processed = preprocess_data(X_val)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(X_train_processed, Y_train)
    val_dataset = TensorDataset(X_val_processed, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create and train the model
    input_size = X_train_processed.shape[1]
    model = PhysicsClassifier(input_size).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model with fewer epochs for the dry run
    print("Training model...")
    model, best_auc = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=2)
    
    print(f"Dry run complete. Best AUC: {best_auc:.4f}")


def main():
    # Check if X_train, Y_train, X_val, Y_val exist in the global scope
    try:
        # These variables are assumed to be already loaded
        global X_train, Y_train, X_val, Y_val
    except NameError:
        print("Error: Required data variables (X_train, Y_train, X_val, Y_val) not found.")
        print("This script is designed to work with preloaded data tensors.")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Preprocess data
    print("Preprocessing data...")
    X_train_processed = preprocess_data(X_train)
    X_val_processed = preprocess_data(X_val)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(X_train_processed, Y_train)
    val_dataset = TensorDataset(X_val_processed, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Create and train the model
    input_size = X_train_processed.shape[1]
    model = PhysicsClassifier(input_size).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model
    print("Training model...")
    model, best_auc = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30)
    
    # Final evaluation
    model.eval()
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(batch_y.cpu().numpy())
    
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    final_auc = roc_auc_score(val_targets, val_preds)
    
    print(f"Final AUC score: {final_auc:.6f}")
    
    # Save the model
    torch.save(model.state_dict(), "particle_physics_model.pth")
    print("Model saved as 'particle_physics_model.pth'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun", action="store_true", help="Run the model with synthetic data")
    args = parser.parse_args()
    
    if args.dryrun:
        dry_run()
    else:
        main()