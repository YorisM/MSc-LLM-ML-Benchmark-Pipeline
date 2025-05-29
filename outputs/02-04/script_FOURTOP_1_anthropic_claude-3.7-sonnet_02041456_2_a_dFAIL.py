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

# Command line arguments
parser = argparse.ArgumentParser(description='Train a binary classifier for particle physics data')
parser.add_argument('--dryrun', action='store_true', help='Run a quick test with minimal epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
args = parser.parse_args()

# Set parameters based on dryrun flag
if args.dryrun:
    args.epochs = 2
    args.batch_size = 64
    
# Fix random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Helper function to generate synthetic data for dry run
def generate_synthetic_data():
    # Create basic synthetic data for testing
    n_train = 1000
    n_val = 200
    feature_dim = 106
    
    # Generate random data
    X_train = torch.randn(n_train, feature_dim)
    Y_train = torch.randint(0, 2, (n_train,)).float()
    X_val = torch.randn(n_val, feature_dim)
    Y_val = torch.randint(0, 2, (n_val,)).float()
    
    return X_train, Y_train, X_val, Y_val

# Load or generate data based on dryrun flag
if args.dryrun:
    X_train, Y_train, X_val, Y_val = generate_synthetic_data()
    print("Using synthetic data for dry run")
else:
    # Assume data is already loaded as mentioned in the problem
    try:
        # This would be where the data is actually imported in a real scenario
        # For now, we'll generate synthetic data as a fallback
        X_train
    except NameError:
        print("Real data not found, using synthetic data instead")
        X_train, Y_train, X_val, Y_val = generate_synthetic_data()

# Data preprocessing
class EventPreprocessor:
    def __init__(self):
        self.feature_means = None
        self.feature_stds = None
    
    def fit(self, data):
        # Calculate mean and std for each feature, ignoring padded zeros
        # First three columns are weight, ET_miss, phi_ET_miss
        # Rest are object features grouped as (obj_id, E, pT, eta, phi)
        
        # For the first three columns (weight, ET_miss, phi_ET_miss)
        self.weight_mean = data[:, 0].mean().item()
        self.weight_std = data[:, 0].std().item() or 1.0  # Avoid division by zero
        
        self.et_miss_mean = data[:, 1].mean().item()
        self.et_miss_std = data[:, 1].std().item() or 1.0
        
        # phi_ET_miss is an angle - don't standardize
        
        # For the object features
        # Create masks for non-zero values (to ignore padding)
        obj_data = data[:, 3:].reshape(data.shape[0], -1, 5)  # reshape to get objects
        
        # Mask for non-zero objects (where object ID is not zero)
        mask = obj_data[:, :, 0] != 0
        
        # Calculate stats only for real objects
        self.obj_means = []
        self.obj_stds = []
        
        # For each feature in the object (obj_id, E, pT, eta, phi)
        for i in range(5):
            if i == 0:  # obj_id - no normalization needed
                self.obj_means.append(0)
                self.obj_stds.append(1)
            else:
                # Get only non-zero values
                valid_values = obj_data[mask, i]
                if valid_values.numel() > 0:
                    mean = valid_values.mean().item()
                    std = valid_values.std().item() or 1.0
                else:
                    mean, std = 0, 1
                self.obj_means.append(mean)
                self.obj_stds.append(std)
    
    def transform(self, data):
        result = data.clone()
        
        # Normalize weight
        result[:, 0] = (data[:, 0] - self.weight_mean) / self.weight_std
        
        # Normalize ET_miss
        result[:, 1] = (data[:, 1] - self.et_miss_mean) / self.et_miss_std
        
        # phi_ET_miss stays the same (angle)
        
        # Normalize object features
        obj_data = data[:, 3:].reshape(data.shape[0], -1, 5)
        obj_result = result[:, 3:].reshape(result.shape[0], -1, 5)
        
        # Apply normalization to each feature except obj_id
        for i in range(1, 5):
            # Only normalize non-zero values
            mask = obj_data[:, :, 0] != 0
            indices = torch.nonzero(mask, as_tuple=True)
            
            if indices[0].numel() > 0:
                obj_result[indices[0], indices[1], i] = (
                    obj_data[indices[0], indices[1], i] - self.obj_means[i]
                ) / self.obj_stds[i]
        
        return result

# Function to extract meaningful features
def extract_features(data):
    # Extract basic features
    batch_size = data.shape[0]
    
    # Weight, ET_miss, phi_ET_miss are the first 3 columns
    basic_features = data[:, :3]  # shape: [batch_size, 3]
    
    # Reshape object data to [batch_size, num_objects, 5]
    # where 5 = (obj_id, E, pT, eta, phi)
    obj_data = data[:, 3:].reshape(batch_size, -1, 5)
    num_objects = obj_data.shape[1]
    
    # Count number of each type of object per event
    # Create a mask for non-zero objects
    obj_mask = obj_data[:, :, 0] != 0
    num_real_objects = obj_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
    
    # Compute sums, means and standard deviations of kinematic properties for real objects
    # Initialize with zeros
    sum_e = torch.zeros(batch_size, 1, device=data.device)
    sum_pt = torch.zeros(batch_size, 1, device=data.device)
    mean_eta = torch.zeros(batch_size, 1, device=data.device)
    std_eta = torch.zeros(batch_size, 1, device=data.device)
    
    # Only consider real objects (where mask is True)
    for i in range(batch_size):
        real_objs = obj_data[i, obj_mask[i]]
        if real_objs.size(0) > 0:  # if there are any real objects
            sum_e[i, 0] = real_objs[:, 1].sum()  # Sum of energies
            sum_pt[i, 0] = real_objs[:, 2].sum()  # Sum of pT
            
            # Mean and std of eta
            eta_values = real_objs[:, 3]
            mean_eta[i, 0] = eta_values.mean()
            if eta_values.size(0) > 1:  # need at least 2 values for std
                std_eta[i, 0] = eta_values.std()
    
    # Combine all aggregate features
    aggregate_features = torch.cat([num_real_objects, sum_e, sum_pt, mean_eta, std_eta], dim=1)
    
    # Calculate event shape variables
    # Thrust and sphericity-like quantities
    thrust = torch.zeros(batch_size, 1, device=data.device)
    sphericity = torch.zeros(batch_size, 3, device=data.device)  # Eigenvalues of sphericity tensor
    
    for i in range(batch_size):
        real_objs = obj_data[i, obj_mask[i]]
        if real_objs.size(0) > 0:
            # Simplified thrust calculation
            pt_values = real_objs[:, 2]
            phi_values = real_objs[:, 4]
            
            # Calculate thrust using px and py components
            px = pt_values * torch.cos(phi_values)
            py = pt_values * torch.sin(phi_values)
            
            p_mag = torch.sqrt(px.pow(2) + py.pow(2))
            thrust[i, 0] = p_mag.sum() / pt_values.sum() if pt_values.sum() > 0 else 0
            
            # Simplified sphericity calculation using momentum tensor
            if real_objs.size(0) > 2:  # Need at least 3 particles for meaningful sphericity
                # Use pT, eta, phi to approximate 3D momentum
                eta = real_objs[:, 3]
                theta = 2 * torch.atan(torch.exp(-eta))  # Convert eta to theta
                
                # 3D momentum components
                px = pt_values * torch.cos(phi_values)
                py = pt_values * torch.sin(phi_values)
                pz = pt_values / torch.tan(theta)
                
                # Normalize
                p_sum_sq = (px.pow(2) + py.pow(2) + pz.pow(2)).sum()
                
                if p_sum_sq > 0:
                    # Momentum tensor elements
                    tensor = torch.zeros(3, 3, device=data.device)
                    tensor[0, 0] = (px * px).sum() / p_sum_sq
                    tensor[0, 1] = tensor[1, 0] = (px * py).sum() / p_sum_sq
                    tensor[0, 2] = tensor[2, 0] = (px * pz).sum() / p_sum_sq
                    tensor[1, 1] = (py * py).sum() / p_sum_sq
                    tensor[1, 2] = tensor[2, 1] = (py * pz).sum() / p_sum_sq
                    tensor[2, 2] = (pz * pz).sum() / p_sum_sq
                    
                    # Get eigenvalues (sphericity components)
                    try:
                        eigenvalues = torch.linalg.eigvalsh(tensor)
                        sphericity[i] = eigenvalues
                    except:
                        pass  # Fallback if eigenvalue calculation fails
    
    # Combine all features
    all_features = torch.cat([basic_features, aggregate_features, thrust, sphericity], dim=1)
    
    # Include the original object data (flattened) for the model to use
    # Note: for simplicity, we're keeping the original data structure too
    # A more advanced model could use a better structure to handle variable-length data
    
    return all_features

# Define the neural network model
class PhysicsNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(PhysicsNet, self).__init__()
        
        # Feature extraction block
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification block
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output.squeeze()

# Process data for training
preprocessor = EventPreprocessor()
preprocessor.fit(X_train)
X_train_norm = preprocessor.transform(X_train)
X_val_norm = preprocessor.transform(X_val)

# Extract features
print("Extracting features...")
X_train_features = extract_features(X_train_norm)
X_val_features = extract_features(X_val_norm)
print(f"Feature extraction complete. Feature dimension: {X_train_features.shape[1]}")

# Create dataloaders
train_dataset = TensorDataset(X_train_features, Y_train)
val_dataset = TensorDataset(X_val_features, Y_val)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = PhysicsNet(X_train_features.shape[1], hidden_dim=args.hidden_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

# Training loop
def train():
    train_losses = []
    val_losses = []
    val_aucs = []
    best_auc = 0.0
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_auc = roc_auc_score(all_labels, all_preds)
        val_aucs.append(val_auc)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "physics_model.pth")
            print(f"Model saved with AUC: {best_auc:.4f}")
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, label='Val AUC')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guessing')
    plt.legend()
    plt.title('AUC Curve')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    return best_auc

# Evaluate the best model
def evaluate():
    # Load the best model
    best_model = PhysicsNet(X_train_features.shape[1], hidden_dim=args.hidden_dim)
    best_model.load_state_dict(torch.load("physics_model.pth"))
    best_model.eval()
    
    # Evaluate on validation set
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = best_model(batch_X)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    final_auc = roc_auc_score(all_labels, all_preds)
    print(f"\nFinal evaluation AUC: {final_auc:.4f}")
    
    return final_auc

# Run the training process
best_auc = train()

# Evaluate final model
final_auc = evaluate()

print(f"\nTraining completed. Best validation AUC: {best_auc:.4f}")
print(f"Final AUC score: {final_auc:.4f}")