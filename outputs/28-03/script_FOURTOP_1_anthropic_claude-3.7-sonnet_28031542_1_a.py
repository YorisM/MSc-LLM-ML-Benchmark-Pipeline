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
import numpy as np
from sklearn.metrics import roc_auc_score

# Define a custom dataset class for handling the physics data
class PhysicsDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Function to extract meaningful features from the raw data
def extract_features(X):
    batch_size = X.shape[0]
    feature_list = []
    
    # Extract weights and missing energy features (first 3 columns)
    weights = X[:, 0].reshape(-1, 1)
    et_miss = X[:, 1].reshape(-1, 1)
    phi_et_miss = X[:, 2].reshape(-1, 1)
    feature_list.extend([weights, et_miss, phi_et_miss])
    
    # Process the object data (columns 3 onwards)
    obj_features = X[:, 3:]
    obj_features = obj_features.reshape(batch_size, -1, 5)  # Reshape to [batch_size, n_objects, 5]
    
    # Count non-zero objects
    obj_mask = (obj_features[:, :, 0] != 0).float()
    n_objects = torch.sum(obj_mask, dim=1, keepdim=True)
    feature_list.append(n_objects)
    
    # Calculate object type counts
    max_obj_type = 10  # Assuming max 10 different object types
    obj_type_counts = torch.zeros((batch_size, max_obj_type))
    
    for i in range(max_obj_type):
        obj_type_counts[:, i] = torch.sum((obj_features[:, :, 0] == i+1).float(), dim=1)
    
    feature_list.append(obj_type_counts)
    
    # Calculate sum of pT for different object types
    pt_sums = torch.zeros((batch_size, max_obj_type))
    
    for i in range(max_obj_type):
        mask = (obj_features[:, :, 0] == i+1).float().unsqueeze(-1)
        pt_vals = obj_features[:, :, 2] * mask.squeeze(-1)  # pT is at index 2
        pt_sums[:, i] = torch.sum(pt_vals, dim=1)
    
    feature_list.append(pt_sums)
    
    # Calculate HT (scalar sum of pT of all objects)
    ht = torch.sum(obj_features[:, :, 2] * obj_mask, dim=1, keepdim=True)
    feature_list.append(ht)
    
    # Calculate MET/HT ratio
    met_ht_ratio = et_miss / (ht + 1e-8)  # Add small epsilon to avoid division by zero
    feature_list.append(met_ht_ratio)
    
    # Calculate statistics of object properties
    for prop_idx in range(1, 5):  # E, pT, eta, phi
        prop_vals = obj_features[:, :, prop_idx] * obj_mask
        prop_sum = torch.sum(prop_vals, dim=1, keepdim=True)
        prop_mean = prop_sum / (n_objects + 1e-8)
        
        # Calculate squared values for standard deviation
        prop_sq = (prop_vals ** 2) * obj_mask
        prop_sq_sum = torch.sum(prop_sq, dim=1, keepdim=True)
        prop_var = prop_sq_sum / (n_objects + 1e-8) - (prop_mean ** 2)
        prop_std = torch.sqrt(torch.clamp(prop_var, min=1e-8))
        
        feature_list.extend([prop_sum, prop_mean, prop_std])
    
    # Feature engineering: calculate dR between objects
    max_pairs = 5  # Take top 5 pairs with highest pT products
    dR_features = torch.zeros((batch_size, max_pairs))
    pt_products = torch.zeros((batch_size, max_pairs))
    
    for b in range(batch_size):
        valid_objects = int(n_objects[b].item())
        if valid_objects >= 2:
            # Calculate pairwise deltaR for valid objects
            eta_i = obj_features[b, :valid_objects, 3]
            phi_i = obj_features[b, :valid_objects, 4]
            pt_i = obj_features[b, :valid_objects, 2]
            
            pair_indices = []
            for i in range(valid_objects):
                for j in range(i+1, valid_objects):
                    deta = eta_i[i] - eta_i[j]
                    dphi = phi_i[i] - phi_i[j]
                    # Adjust dphi to be in [-pi, pi]
                    dphi = torch.remainder(dphi + torch.tensor(np.pi), torch.tensor(2 * np.pi)) - torch.tensor(np.pi)
                    dr = torch.sqrt(deta**2 + dphi**2)
                    pt_prod = pt_i[i] * pt_i[j]
                    pair_indices.append((i, j, dr, pt_prod))
            
            # Sort by pT product and take top max_pairs
            if pair_indices:
                sorted_pairs = sorted(pair_indices, key=lambda x: x[3], reverse=True)[:max_pairs]
                for p, (i, j, dr, pt_prod) in enumerate(sorted_pairs):
                    if p < max_pairs:
                        dR_features[b, p] = dr
                        pt_products[b, p] = pt_prod
    
    feature_list.extend([dR_features, pt_products])
    
    # Concatenate all features
    return torch.cat([f.float() for f in feature_list], dim=1)

# Neural Network architecture for the classifier
class PhysicsNet(nn.Module):
    def __init__(self, input_dim):
        super(PhysicsNet, self).__init__()
        
        # Define layers with batch normalization and dropout
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        
        self.fc5 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.relu(self.bn4(self.fc4(x)))
        
        x = self.sigmoid(self.fc5(x))
        return x

# Main training function
def train_physics_model(X_train, Y_train, X_val, Y_val):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract features
    print("Extracting features...")
    X_train_features = extract_features(X_train)
    X_val_features = extract_features(X_val)
    
    # Create datasets and dataloaders
    batch_size = 128
    train_dataset = PhysicsDataset(X_train_features, Y_train)
    val_dataset = PhysicsDataset(X_val_features, Y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = X_train_features.shape[1]
    model = PhysicsNet(input_dim).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training loop
    num_epochs = 30
    best_val_auc = 0.0
    best_model_state = None
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device).view(-1, 1)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item() * data.size(0)
                val_preds.extend(output.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(val_loss)
        
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        val_auc = roc_auc_score(val_targets, val_preds)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            print(f'New best model with AUC: {best_val_auc:.4f}')
    
    # Load best model and evaluate final performance
    model.load_state_dict(best_model_state)
    model.eval()
    final_val_preds = []
    final_val_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device).view(-1, 1)
            output = model(data)
            
            final_val_preds.extend(output.cpu().numpy())
            final_val_targets.extend(target.cpu().numpy())
    
    final_val_preds = np.array(final_val_preds).flatten()
    final_val_targets = np.array(final_val_targets).flatten()
    final_val_auc = roc_auc_score(final_val_targets, final_val_preds)
    
    print(f'Final validation AUC: {final_val_auc:.4f}')
    return model, final_val_auc, final_val_preds

# Execute training
if __name__ == "__main__":
    model, auc, predictions = train_physics_model(X_train, Y_train, X_val, Y_val)
    print(f"Final model AUC: {auc}")