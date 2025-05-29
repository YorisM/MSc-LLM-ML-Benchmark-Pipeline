import pandas as pd
import torch

X_train_df = pd.read_csv('./data/fourtops/X_train.csv')
Y_train_df = pd.read_csv('./data/fourtops/Y_train.csv')
X_val_df   = pd.read_csv('./data/fourtops/X_val.csv')
Y_val_df   = pd.read_csv('./data/fourtops/Y_val.csv')

X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
Y_train = torch.tensor(Y_train_df.values, dtype=torch.long)
X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long)


#LLM Response below

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Data preprocessing function
def preprocess_data(X):
    # Remove zero-padded entries (masking)
    mask = X[:, 3:] != 0
    X_nonzero = X[:, 3:][mask].reshape(X.shape[0], -1, 5)
    
    # Compute basic physics-driven features
    E_sum = X_nonzero[:, :, 1].sum(dim=1, keepdim=True)  # Sum of energy per event
    pT_sum = X_nonzero[:, :, 2].sum(dim=1, keepdim=True) # Sum of transverse momentum
    eta_mean = X_nonzero[:, :, 3].mean(dim=1, keepdim=True) # Mean pseudorapidity
    
    # Stack the features back, normalizing values
    E_sum /= 1e6
    pT_sum /= 1e6
    X_features = torch.cat([X[:, :3], E_sum, pT_sum, eta_mean], dim=1)
    
    return X_features

# Preprocess dataset
X_train_proc = preprocess_data(X_train)
X_val_proc = preprocess_data(X_val)

def create_dataloader(X, Y, batch_size=512):
    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_dataloader(X_train_proc, Y_train)
val_loader = create_dataloader(X_val_proc, Y_val, batch_size=1024)

# Neural network classifier
class ParticleClassifier(nn.Module):
    def __init__(self, input_size):
        super(ParticleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).view(-1)

# Model setup
input_size = X_train_proc.shape[1]
model = ParticleClassifier(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
        
        # Validation AUC
        model.eval()
        y_preds, y_trues = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                y_preds.extend(outputs.cpu().numpy())
                y_trues.extend(labels.cpu().numpy())
        
        auc = roc_auc_score(y_trues, y_preds)
        print(f'Epoch {epoch+1}, Validation AUC: {auc:.4f}')

train_model(model, train_loader, val_loader, epochs=15)