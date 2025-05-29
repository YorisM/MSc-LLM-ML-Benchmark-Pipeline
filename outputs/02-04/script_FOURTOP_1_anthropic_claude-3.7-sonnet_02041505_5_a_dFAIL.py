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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a model for four-top classification')
parser.add_argument('--dryrun', action='store_true', help='Run a quick test with minimal training')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of the model')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
args = parser.parse_args()

# Set parameters based on dryrun flag
if args.dryrun:
    args.batch_size = 16
    args.epochs = 2
    print("Running in dry run mode with minimal epochs")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Assume data is already loaded as mentioned in the problem
# For this example, we'll create dummy data if not available
try:
    # Access the existing tensors mentioned in the problem
    X_train
    Y_train
    X_val
    Y_val
except NameError:
    # Create dummy data for dry run
    print("Creating dummy data for demonstration")
    X_train = torch.randn(10000, 106)
    Y_train = torch.randint(0, 2, (10000,)).float()
    X_val = torch.randn(1000, 106)
    Y_val = torch.randint(0, 2, (1000,)).float()

# Preprocess the data
def preprocess_data(X):
    """
    Preprocess the data: extract features and normalize
    """
    # The first three entries are weight, ET_miss, phi_ET_miss
    event_info = X[:, :3]
    
    # The rest are object properties (obj_id, E, pT, eta, phi) repeating
    obj_data = X[:, 3:]
    
    # Reshape to separate objects and properties
    n_events = X.shape[0]
    n_objs = (X.shape[1] - 3) // 5
    
    # Extract properties for each object
    objs_reshaped = obj_data.view(n_events, n_objs, 5)
    
    # Check if an object exists (non-zero energy)
    obj_exists = (objs_reshaped[:, :, 1] > 0).float()
    
    # Count objects per event
    obj_counts = obj_exists.sum(dim=1, keepdim=True)
    
    # Calculate object-level features (energies, momenta, etc.)
    total_energy = torch.sum(objs_reshaped[:, :, 1] * obj_exists, dim=1, keepdim=True)
    total_pt = torch.sum(objs_reshaped[:, :, 2] * obj_exists, dim=1, keepdim=True)
    
    # Calculate statistics of properties across objects
    # Mean and std of energy, pt, eta, phi for valid objects
    mean_e = torch.sum(objs_reshaped[:, :, 1] * obj_exists, dim=1, keepdim=True) / (obj_counts + 1e-8)
    mean_pt = torch.sum(objs_reshaped[:, :, 2] * obj_exists, dim=1, keepdim=True) / (obj_counts + 1e-8)
    mean_eta = torch.sum(objs_reshaped[:, :, 3] * obj_exists, dim=1, keepdim=True) / (obj_counts + 1e-8)
    mean_phi = torch.sum(objs_reshaped[:, :, 4] * obj_exists, dim=1, keepdim=True) / (obj_counts + 1e-8)
    
    # Standard deviations
    std_e = torch.sqrt(torch.sum(((objs_reshaped[:, :, 1] - mean_e) * obj_exists) ** 2, dim=1, keepdim=True) / (obj_counts + 1e-8))
    std_pt = torch.sqrt(torch.sum(((objs_reshaped[:, :, 2] - mean_pt) * obj_exists) ** 2, dim=1, keepdim=True) / (obj_counts + 1e-8))
    std_eta = torch.sqrt(torch.sum(((objs_reshaped[:, :, 3] - mean_eta) * obj_exists) ** 2, dim=1, keepdim=True) / (obj_counts + 1e-8))
    std_phi = torch.sqrt(torch.sum(((objs_reshaped[:, :, 4] - mean_phi) * obj_exists) ** 2, dim=1, keepdim=True) / (obj_counts + 1e-8))
    
    # Max values
    max_e = torch.max(objs_reshaped[:, :, 1] * obj_exists, dim=1, keepdim=True)[0]
    max_pt = torch.max(objs_reshaped[:, :, 2] * obj_exists, dim=1, keepdim=True)[0]
    
    # Combine engineered features
    engineered_features = torch.cat([
        event_info,
        obj_counts,
        total_energy,
        total_pt,
        mean_e, std_e, max_e,
        mean_pt, std_pt, max_pt,
        mean_eta, std_eta,
        mean_phi, std_phi,
    ], dim=1)
    
    # Flatten the object data
    flattened = objs_reshaped.reshape(n_events, -1)
    
    # Combine all features
    combined = torch.cat([engineered_features, flattened], dim=1)
    
    # Normalize features (excluding weight which is at position 0)
    mean = combined[:, 1:].mean(dim=0)
    std = combined[:, 1:].std(dim=0) + 1e-8
    combined_normalized = combined.clone()
    combined_normalized[:, 1:] = (combined[:, 1:] - mean) / std
    
    return combined_normalized, mean, std

# Process training and validation data
X_train_processed, mean_train, std_train = preprocess_data(X_train)
X_val_processed = X_val.clone()
X_val_processed[:, 1:] = (X_val[:, 1:] - mean_train) / std_train

# Create data loaders
train_dataset = TensorDataset(X_train_processed, Y_train)
val_dataset = TensorDataset(X_val_processed, Y_val)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Define the model
class FourTopClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(FourTopClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return self.sigmoid(x).squeeze()

# Initialize model
input_dim = X_train_processed.shape[1]
model = FourTopClassifier(input_dim, args.hidden_dim, args.dropout).to(device)
print(f"Model initialized with input dimension: {input_dim}")

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Training and evaluation functions
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X)
        
        # Calculate weighted loss
        loss = criterion(outputs, y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate AUC
    auc = roc_auc_score(all_labels, all_preds)
    
    return auc

# Training loop
best_auc = 0.0
train_losses = []
val_aucs = []

print("Starting training...")
for epoch in range(args.epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_auc = evaluate(model, val_loader, device)
    
    train_losses.append(train_loss)
    val_aucs.append(val_auc)
    
    print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")
    
    # Save the best model
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "four_top_classifier_model.pth")
        print(f"Model saved with improved validation AUC: {val_auc:.4f}")

print(f"Training complete. Best validation AUC: {best_auc:.4f}")

# Plot training curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(val_aucs)
plt.title('Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.close()

# Load the best model and evaluate on validation set
model.load_state_dict(torch.load("four_top_classifier_model.pth"))
final_auc = evaluate(model, val_loader, device)
print(f"Final Model AUC on Validation Set: {final_auc:.4f}")