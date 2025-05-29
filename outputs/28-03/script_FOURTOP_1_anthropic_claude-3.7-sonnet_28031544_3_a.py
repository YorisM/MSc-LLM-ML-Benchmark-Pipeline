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

# LLM Reponse Belowimport numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Define constants
BATCH_SIZE = 256
EPOCHS = 15
LEARNING_RATE = 0.001

# Process the data
def preprocess_data(X):
    # Extract features
    weight = X[:, 0].unsqueeze(1)  # event weight
    et_miss = X[:, 1].unsqueeze(1)  # missing transverse energy
    phi_et_miss = X[:, 2].unsqueeze(1)  # phi of missing transverse energy
    
    # The rest of the features are object properties
    # Each object has 5 features: obj_id, E, p_T, eta, phi
    # We have (106-3)/5 = 20.6 max objects, but let's be safe and use 20
    
    # Reshape the data to extract object properties
    n_samples = X.shape[0]
    objects_features = []
    
    for i in range(3, X.shape[1], 5):
        if i+4 < X.shape[1]:  # Make sure we have a complete object
            obj_id = X[:, i].unsqueeze(1)  # object identifier
            E = X[:, i+1].unsqueeze(1)  # energy
            p_T = X[:, i+2].unsqueeze(1)  # transverse momentum
            eta = X[:, i+3].unsqueeze(1)  # pseudo-rapidity
            phi = X[:, i+4].unsqueeze(1)  # azimuthal angle
            
            # Filter out zero-padded entries
            mask = (obj_id != 0).float()
            
            # Combine features for this object
            obj_features = torch.cat([obj_id, E*mask, p_T*mask, eta*mask, phi*mask], dim=1)
            objects_features.append(obj_features)
    
    # Combine all object features
    all_objects = torch.cat(objects_features, dim=1)
    
    # Combine with event-level features
    processed_data = torch.cat([weight, et_miss, phi_et_miss, all_objects], dim=1)
    
    # Add engineered features
    # 1. Log of ET_miss (important for missing energy distributions)
    log_et_miss = torch.log1p(et_miss)
    
    # 2. Count the number of non-zero objects
    obj_counts = sum([(X[:, i] != 0).float().unsqueeze(1) for i in range(3, X.shape[1], 5)])
    
    # Add engineered features to processed_data
    processed_data = torch.cat([processed_data, log_et_miss, obj_counts], dim=1)
    
    return processed_data

# Define a PyTorch model
class TTTTClassifier(nn.Module):
    def __init__(self, input_size):
        super(TTTTClassifier, self).__init__()
        
        # Use a more complex architecture for better performance
        self.layer1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.layer2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.layer4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.layer5 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn4(self.layer4(x)))
        
        x = self.sigmoid(self.layer5(x))
        return x

# Preprocess the data
X_train_processed = preprocess_data(X_train)
X_val_processed = preprocess_data(X_val)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_processed, Y_train.unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the model
input_size = X_train_processed.shape[1]
model = TTTTClassifier(input_size)

# Define loss function and optimizer
class_weights = torch.tensor([(Y_train == 0).sum() / (Y_train == 1).sum()])
bce_loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Training loop
train_losses = []
val_aucs = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        
        # Calculate loss
        loss = bce_loss(outputs, batch_y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_processed)
        val_preds = val_outputs.squeeze().numpy()
        val_auc = roc_auc_score(Y_val.numpy(), val_preds)
        val_aucs.append(val_auc)
    
    # Update learning rate
    scheduler.step(avg_loss)
    
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Validation AUC: {val_auc:.4f}')

# Plot training progress
plt.figure(figsize=(10, 5))
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
plt.show()

# Final evaluation
model.eval()
with torch.no_grad():
    final_val_outputs = model(X_val_processed)
    final_val_preds = final_val_outputs.squeeze().numpy()
    final_val_auc = roc_auc_score(Y_val.numpy(), final_val_preds)
    print(f'Final Validation AUC: {final_val_auc:.4f}')