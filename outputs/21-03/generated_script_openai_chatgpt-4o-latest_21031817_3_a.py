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

# Data Preprocessing
def preprocess_data(X):
    # Extracting weight and missing transverse energy components
    weights = X[:, 0]
    Et_miss = X[:, 1:3]
    
    # Extracting object features (excluding weight and Et_miss)
    obj_features = X[:, 3:]
    
    # Normalize non-zero features using log scaling to account for large energy variance
    mask = obj_features != 0
    obj_features[mask] = np.log1p(obj_features[mask])
    
    # Normalize Et_miss similarly
    Et_miss = np.log1p(Et_miss)
    
    # Combine preprocessed features
    X_processed = np.concatenate([Et_miss, obj_features], axis=1)
    return torch.tensor(X_processed, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32)

X_train_proc, train_weights = preprocess_data(X_train.numpy())
X_val_proc, val_weights = preprocess_data(X_val.numpy())

dataset_train = TensorDataset(X_train_proc, Y_train)
dataset_val = TensorDataset(X_val_proc, Y_val)
train_loader = DataLoader(dataset_train, batch_size=1024, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=1024, shuffle=False)

# Defining the Model
class ParticleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ParticleClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        
        return self.model(x)

model = ParticleClassifier(input_dim=X_train_proc.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            Y_pred = model(X_batch).squeeze()
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()
    
        model.eval()
        Y_pred_val = []
        Y_true_val = []
        with torch.no_grad():
            for X_val_batch, Y_val_batch in val_loader:
                Y_pred_val.append(model(X_val_batch).squeeze().numpy())
                Y_true_val.append(Y_val_batch.numpy())
        
        Y_pred_val = np.concatenate(Y_pred_val)
        Y_true_val = np.concatenate(Y_true_val)
        auc = roc_auc_score(Y_true_val, Y_pred_val)
        print(f"Epoch {epoch + 1}: AUC = {auc:.5f}")

train_model(model, train_loader, val_loader, epochs=20)