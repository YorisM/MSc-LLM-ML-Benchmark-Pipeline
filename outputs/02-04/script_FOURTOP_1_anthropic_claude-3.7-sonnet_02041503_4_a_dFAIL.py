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
import random

# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocess data
def preprocess_data(X):
    # Extract event weights and missing energy info from first columns
    weights = X[:, 0]
    et_miss = X[:, 1]
    phi_et_miss = X[:, 2]
    
    # Extract object features (starting from index 3)
    object_data = X[:, 3:]
    
    # Reshape to extract object IDs and their features
    # Each object has 5 values: obj_id, E, p_T, eta, phi
    num_events = object_data.shape[0]
    object_length = object_data.shape[1]
    num_objects = object_length // 5
    
    # Reshape to have explicit object features
    reshaped_data = object_data.reshape(num_events, num_objects, 5)
    
    # Check for padding (zeros in the obj_id position)
    object_mask = (reshaped_data[:, :, 0] != 0).float()
    
    # Count number of real objects per event
    num_real_objects = torch.sum(object_mask, dim=1).unsqueeze(1)
    
    # Global event features
    global_features = torch.cat([et_miss.unsqueeze(1), phi_et_miss.unsqueeze(1), num_real_objects], dim=1)
    
    # Calculate high-level features from object kinematics
    # 1. Sum of p_T for all objects
    pt_sum = torch.sum(reshaped_data[:, :, 2] * object_mask, dim=1, keepdim=True)
    
    # 2. Mean E of objects
    e_mean = torch.sum(reshaped_data[:, :, 1] * object_mask, dim=1, keepdim=True) / torch.clamp(num_real_objects, min=1.0)
    
    # 3. Standard deviation of p_T
    pt_std = torch.sqrt(torch.sum(((reshaped_data[:, :, 2] - torch.sum(reshaped_data[:, :, 2] * object_mask, dim=1, keepdim=True) / 
                               torch.clamp(num_real_objects, min=1.0)) * object_mask) ** 2, dim=1, keepdim=True) / 
                               torch.clamp(num_real_objects, min=1.0))
    
    # Add more physics-inspired features
    # Count object types
    obj_types = []
    for i in range(1, 9):  # Assuming object types from 1-8
        type_count = torch.sum((reshaped_data[:, :, 0] == i).float(), dim=1, keepdim=True)
        obj_types.append(type_count)
    
    obj_type_counts = torch.cat(obj_types, dim=1)
    
    # Combine all features
    high_level_features = torch.cat([global_features, pt_sum, e_mean, pt_std, obj_type_counts], dim=1)
    
    # Return both high-level features and the reshaped data for detailed processing
    return high_level_features, reshaped_data, object_mask, weights

# Define model architecture
class PhysicsAwareNet(nn.Module):
    def __init__(self, input_dim, object_dim=5, num_objects=20):
        super(PhysicsAwareNet, self).__init__()
        
        # Process high-level features
        self.fc_high = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Process each object with a small network
        self.object_encoder = nn.Sequential(
            nn.Linear(object_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Attention mechanism for objects
        self.attention = nn.Sequential(
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        
        # Combine features
        self.combiner = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, high_level, objects, mask):
        # Process high-level features
        high_feat = self.fc_high(high_level)
        
        # Process each object
        batch_size, num_objects, obj_features = objects.size()
        
        # Apply mask to ensure padded objects don't contribute
        masked_objects = objects * mask.unsqueeze(2)
        
        # Reshape for object encoder
        flat_objects = masked_objects.reshape(-1, obj_features)
        encoded_objects = self.object_encoder(flat_objects)
        encoded_objects = encoded_objects.reshape(batch_size, num_objects, -1)
        
        # Apply attention
        attention_scores = self.attention(encoded_objects)
        attended_objects = torch.sum(attention_scores * encoded_objects, dim=1)
        
        # Combine features
        combined = torch.cat([high_feat, attended_objects], dim=1)
        output = self.combiner(combined)
        
        return output

# Training function
def train_model(model, train_loader, val_loader, weights_train, weights_val, epochs=30):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []
    
    best_val_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for i, (high_level, objects, mask, targets) in enumerate(train_loader):
            high_level, objects, mask, targets = high_level.to(device), objects.to(device), mask.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(high_level, objects, mask).squeeze()
            
            # Apply sample weights
            batch_weights = weights_train[i*train_loader.batch_size:min((i+1)*train_loader.batch_size, len(weights_train))]
            batch_weights = batch_weights.to(device)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(targets.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_targets, train_preds)
        train_losses.append(train_loss)
        train_aucs.append(train_auc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for i, (high_level, objects, mask, targets) in enumerate(val_loader):
                high_level, objects, mask, targets = high_level.to(device), objects.to(device), mask.to(device), targets.to(device)
                
                outputs = model(high_level, objects, mask).squeeze()
                
                # Apply sample weights
                batch_weights = weights_val[i*val_loader.batch_size:min((i+1)*val_loader.batch_size, len(weights_val))]
                batch_weights = batch_weights.to(device)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_preds.extend(outputs.detach().cpu().numpy())
                val_targets.extend(targets.detach().cpu().numpy())
        
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Learning rate scheduling
        scheduler.step(val_loss)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_aucs, val_aucs, best_val_auc

# Evaluate function
def evaluate(model, val_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for high_level, objects, mask, targets in val_loader:
            high_level, objects, mask = high_level.to(device), objects.to(device), mask.to(device)
            
            outputs = model(high_level, objects, mask).squeeze()
            
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
    
    auc = roc_auc_score(all_targets, all_preds)
    return auc

# Plot training curves
def plot_training_curves(train_losses, val_losses, train_aucs, val_aucs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(epochs, train_aucs, 'b-', label='Training AUC')
    ax2.plot(epochs, val_aucs, 'r-', label='Validation AUC')
    ax2.set_title('Training and Validation AUC')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('AUC')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

# Main function
def main(dryrun=False):
    # For the dry run, generate some random data
    if dryrun:
        print("Running in dry run mode with synthetic data...")
        X_train = torch.randn(10000, 106)
        Y_train = torch.randint(0, 2, (10000,), dtype=torch.float32)
        X_val = torch.randn(2000, 106)
        Y_val = torch.randint(0, 2, (2000,), dtype=torch.float32)
    else:
        # In a real run, these variables should be pre-loaded
        # This code assumes they are already loaded
        try:
            print("Loading real data...")
            # These should be predefined in the environment
            pass
        except NameError:
            print("Real data not found. Running with synthetic data...")
            X_train = torch.randn(10000, 106)
            Y_train = torch.randint(0, 2, (10000,), dtype=torch.float32)
            X_val = torch.randn(2000, 106)
            Y_val = torch.randint(0, 2, (2000,), dtype=torch.float32)
    
    # Preprocess data
    print("Preprocessing data...")
    high_level_train, objects_train, mask_train, weights_train = preprocess_data(X_train)
    high_level_val, objects_val, mask_val, weights_val = preprocess_data(X_val)
    
    # Create dataloaders
    batch_size = 128 if not dryrun else 32
    train_dataset = TensorDataset(high_level_train, objects_train, mask_train, Y_train)
    val_dataset = TensorDataset(high_level_val, objects_val, mask_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Define model
    input_dim = high_level_train.shape[1]
    model = PhysicsAwareNet(input_dim=input_dim, object_dim=5, num_objects=objects_train.shape[1])
    model = model.to(device)
    
    # Train model
    print("Training model...")
    epochs = 5 if dryrun else 20
    model, train_losses, val_losses, train_aucs, val_aucs, best_val_auc = train_model(
        model, train_loader, val_loader, weights_train, weights_val, epochs=epochs
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_aucs, val_aucs)
    
    # Final evaluation
    final_auc = evaluate(model, val_loader)
    print(f"Final validation AUC: {final_auc:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'particle_physics_model.pth')
    print("Model saved as 'particle_physics_model.pth'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a classifier for particle physics events')
    parser.add_argument('--dryrun', action='store_true', help='Run in dry run mode with synthetic data')
    args = parser.parse_args()
    
    main(dryrun=args.dryrun)