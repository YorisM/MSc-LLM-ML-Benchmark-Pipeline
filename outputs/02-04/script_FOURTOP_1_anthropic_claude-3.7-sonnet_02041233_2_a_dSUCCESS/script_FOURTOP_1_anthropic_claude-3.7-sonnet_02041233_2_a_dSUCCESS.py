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
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a binary classifier for particle physics events')
parser.add_argument('--dryrun', action='store_true', help='Run a quick test with minimal data')
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the neural network architecture
class EventClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(EventClassifier, self).__init__()
        
        # Feature extraction layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()

# Function for feature engineering
def preprocess_data(X):
    # Extract basic components from the data
    weight = X[:, 0].reshape(-1, 1)
    missing_ET = X[:, 1].reshape(-1, 1)
    missing_phi = X[:, 2].reshape(-1, 1)
    
    # Reshape object data (excluding weight, missing_ET, missing_phi)
    object_data = X[:, 3:]
    
    # The rest of the data is in the format [obj_id, E, pT, eta, phi, ...]
    # Reshape to handle each object separately
    n_events = X.shape[0]
    n_objects_max = (X.shape[1] - 3) // 5  # Maximum number of objects
    
    # Initialize arrays to store engineered features
    sum_pt = np.zeros((n_events, 1))
    sum_energy = np.zeros((n_events, 1))
    n_jets = np.zeros((n_events, 1))
    n_leptons = np.zeros((n_events, 1))
    n_b_jets = np.zeros((n_events, 1))
    
    # Process each object
    for i in range(n_objects_max):
        obj_idx = 3 + i * 5
        
        # Skip if no object (zero padding)
        obj_id = X[:, obj_idx]
        valid_objs = (obj_id != 0)
        
        if np.sum(valid_objs) > 0:
            # Object energy, pt, eta, phi
            E = X[:, obj_idx + 1]
            pt = X[:, obj_idx + 2]
            eta = X[:, obj_idx + 3]
            phi = X[:, obj_idx + 4]
            
            # Count objects by type (example: obj_id 1-4 might be jets, 5-6 electrons, etc.)
            # This is an assumption based on common conventions
            n_jets += np.logical_and(valid_objs, (obj_id >= 1) & (obj_id <= 4)).reshape(-1, 1)
            n_leptons += np.logical_and(valid_objs, (obj_id >= 5) & (obj_id <= 6)).reshape(-1, 1)
            n_b_jets += np.logical_and(valid_objs, (obj_id == 4)).reshape(-1, 1)  # Assuming ID 4 is b-jet
            
            # Sum pT and energy for valid objects
            sum_pt[valid_objs] += pt[valid_objs].reshape(-1, 1)
            sum_energy[valid_objs] += E[valid_objs].reshape(-1, 1)
    
    # Calculate HT (scalar sum of jet pT)
    HT = sum_pt
    
    # Calculate MET to HT ratio
    met_ht_ratio = missing_ET / (HT + 1e-8)  # Avoid division by zero
    
    # Combine features
    features = np.concatenate([
        weight, missing_ET, missing_phi,
        sum_pt, sum_energy, HT, met_ht_ratio, 
        n_jets, n_leptons, n_b_jets,
        object_data
    ], axis=1)
    
    return features

# Load and preprocess the data
def load_and_preprocess_data(is_dryrun=False):
    if is_dryrun:
        # Create synthetic data for dry run
        print("Running in dry run mode with synthetic data")
        X_train = torch.randn(1000, 106)
        Y_train = torch.randint(0, 2, (1000,)).float()
        X_val = torch.randn(200, 106)
        Y_val = torch.randint(0, 2, (200,)).float()
    else:
        # In a real scenario, you would load the data here
        # For demonstration, we'll assume X_train, Y_train, X_val, Y_val are already loaded
        # These variables should be defined outside this function in a real implementation
        try:
            # Try to access the pre-defined variables
            X_train
            Y_train
            X_val
            Y_val
        except NameError:
            print("ERROR: Data not found. Running in dry run mode instead.")
            return load_and_preprocess_data(is_dryrun=True)
    
    # Convert to numpy for preprocessing
    X_train_np = X_train.numpy()
    X_val_np = X_val.numpy()
    
    # Apply feature engineering
    X_train_processed = preprocess_data(X_train_np)
    X_val_processed = preprocess_data(X_val_np)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_val_scaled = scaler.transform(X_val_processed)
    
    # Convert back to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    
    return X_train_tensor, Y_train, X_val_tensor, Y_val

# Train the model
def train_model(X_train, Y_train, X_val, Y_val, model_filename, is_dryrun=False):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define model
    input_dim = X_train.shape[1]
    model = EventClassifier(input_dim)
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training parameters
    batch_size = 64
    num_epochs = 5 if is_dryrun else 50
    
    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Validation dataset
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_aucs = []
    best_val_auc = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_Y in val_dataloader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item()
                
                # Store predictions and labels for AUC calculation
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(batch_Y.cpu().numpy())
        
        # Calculate AUC
        val_auc = roc_auc_score(all_labels, all_preds)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_filename)
        
        # Record metrics
        train_losses.append(train_loss / len(dataloader))
        val_losses.append(val_loss / len(val_dataloader))
        val_aucs.append(val_auc)
        
        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(dataloader):.4f}, '
              f'Val Loss: {val_loss/len(val_dataloader):.4f}, Val AUC: {val_auc:.4f}')
    
    # Load the best model
    model.load_state_dict(torch.load(model_filename))
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_Y in val_dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_Y.cpu().numpy())
    
    final_auc = roc_auc_score(all_labels, all_preds)
    print(f'\nFinal validation AUC: {final_auc:.6f}')
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, label='Validation AUC')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Classifier')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    return model, final_auc

# Main function
def main():
    # Get the name of this script without extension
    import sys
    script_name = sys.argv[0].split('.')[0]
    model_filename = f"{script_name}_model.pth"
    
    # Load and preprocess data
    X_train, Y_train, X_val, Y_val = load_and_preprocess_data(is_dryrun=args.dryrun)
    
    # Train the model
    model, final_auc = train_model(X_train, Y_train, X_val, Y_val, model_filename, is_dryrun=args.dryrun)
    
    print(f"\nModel saved as {model_filename}")
    print(f"Final AUC: {final_auc:.6f}")

if __name__ == "__main__":
    main()