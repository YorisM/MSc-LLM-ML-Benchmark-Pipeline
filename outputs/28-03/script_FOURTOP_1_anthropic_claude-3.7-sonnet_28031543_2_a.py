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

# LLM Reponse Belowimport torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to preprocess the data
def preprocess_data(X):
    # Extract features
    batch_size = X.shape[0]
    
    # Extract weight, missing ET and phi
    weight = X[:, 0].reshape(-1, 1)
    et_miss = X[:, 1].reshape(-1, 1)
    phi_miss = X[:, 2].reshape(-1, 1)
    
    # The rest of the data contains object information (groups of 5 values)
    obj_data = X[:, 3:]
    
    # Reshape to extract object types and their properties
    reshaped_data = obj_data.reshape(batch_size, -1, 5)
    
    # Count number of each object type
    # Extract object types (first column of reshaped data)
    obj_types = reshaped_data[:, :, 0]
    
    # Define object type ranges (assuming object types are encoded as integers)
    # Count occurrences of each object type
    max_obj_type = int(torch.max(obj_types).item()) + 1
    obj_counts = torch.zeros((batch_size, max_obj_type), device=X.device)
    
    for i in range(max_obj_type):
        obj_counts[:, i] = torch.sum((obj_types == i).float(), dim=1)
    
    # Extract kinematic features for objects
    # Energy, PT, eta, phi statistics for each object type
    kinematic_features = []
    
    for i in range(max_obj_type):
        # Create mask for this object type
        mask = (obj_types == i).unsqueeze(-1).expand(-1, -1, 4)
        
        # Extract kinematic properties (last 4 columns of reshaped data) for this object type
        obj_kinematics = reshaped_data[:, :, 1:5]
        masked_kinematics = obj_kinematics * mask.float()
        
        # Calculate statistics (sum, mean where count > 0, max)
        # Sum
        sum_kinematics = torch.sum(masked_kinematics, dim=1)
        
        # Mean (handling division by zero)
        counts = torch.sum(mask[:, :, 0].float(), dim=1, keepdim=True)
        counts_nonzero = torch.max(counts, torch.ones_like(counts))
        mean_kinematics = sum_kinematics / counts_nonzero
        
        # Max (setting to 0 where no objects of this type)
        max_kinematics = torch.max(masked_kinematics, dim=1)[0] * (counts > 0).float()
        
        # Concatenate statistics
        kinematic_features.append(sum_kinematics)
        kinematic_features.append(mean_kinematics)
        kinematic_features.append(max_kinematics)
    
    # Additional global features
    # Calculate total energy and total PT
    total_energy = torch.sum(reshaped_data[:, :, 1], dim=1, keepdim=True)
    total_pt = torch.sum(reshaped_data[:, :, 2], dim=1, keepdim=True)
    
    # Calculate number of non-zero objects
    num_objects = torch.sum((torch.sum(reshaped_data[:, :, 1:], dim=2) > 0).float(), dim=1, keepdim=True)
    
    # Concatenate all features
    all_features = [weight, et_miss, phi_miss, total_energy, total_pt, num_objects, obj_counts]
    all_features.extend(kinematic_features)
    
    processed_features = torch.cat(all_features, dim=1)
    
    # Normalize features to improve training stability
    # Calculate mean and std across the batch dimension
    mean = torch.mean(processed_features, dim=0)
    std = torch.std(processed_features, dim=0) + 1e-8  # Add small epsilon to avoid division by zero
    
    # Normalize
    normalized_features = (processed_features - mean) / std
    
    return normalized_features

# Define the neural network model
class PhysicsNet(nn.Module):
    def __init__(self, input_size):
        super(PhysicsNet, self).__init__()
        
        # Define network architecture
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.01)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.01)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.batch_norm3(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.01)
        x = self.dropout(x)
        
        x = self.layer4(x)
        x = self.batch_norm4(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.01)
        
        x = self.layer5(x)
        x = torch.sigmoid(x)
        
        return x.squeeze()

# Preprocess the data
X_train_processed = preprocess_data(X_train)
X_val_processed = preprocess_data(X_val)

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_processed, Y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

val_dataset = TensorDataset(X_val_processed, Y_val)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Initialize model
input_size = X_train_processed.shape[1]
model = PhysicsNet(input_size).to(device)

# Define loss function and optimizer
class_weights = torch.bincount(Y_train.long())
class_weights = class_weights.float() / class_weights.sum()
class_weights = 1.0 / class_weights
class_weights = class_weights / class_weights.sum()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20):
    best_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate AUC
        val_auc = roc_auc_score(all_targets, all_preds)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}')
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save the best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
    
    # Load the best model
    model.load_state_dict(best_model_state)
    return model, best_auc

# Train the model
best_model, best_auc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
print(f'Best validation AUC: {best_auc:.4f}')

# Make final predictions on validation set
model.eval()
val_preds = []

with torch.no_grad():
    for inputs, _ in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        val_preds.extend(outputs.cpu().numpy())

# Calculate final AUC
final_auc = roc_auc_score(Y_val.cpu().numpy(), val_preds)
print(f'Final validation AUC: {final_auc:.4f}')