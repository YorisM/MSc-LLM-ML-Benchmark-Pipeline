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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Data Preprocessing
def preprocess_data(X_train, X_val):
    # Normalize features while ignoring zero-padded entries
    nonzero_mask = (X_train != 0).float()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform((X_train * nonzero_mask).cpu().numpy())
    X_val_scaled = scaler.transform((X_val * (X_val != 0).float()).cpu().numpy())
    return torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(X_val_scaled, dtype=torch.float32)

X_train, X_val = preprocess_data(X_train, X_val)

# Define Neural Network Model
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
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Training Function
def train_model(model, X_train, Y_train, X_val, Y_val, epochs=50, batch_size=1024):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_data = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    best_auc = 0
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).squeeze().cpu().numpy()
            auc = roc_auc_score(Y_val.cpu().numpy(), val_preds)
            if auc > best_auc:
                best_auc = auc
                best_model_state = model.state_dict()
        
        print(f"Epoch {epoch + 1}, AUC: {auc:.4f}")
    
    model.load_state_dict(best_model_state)
    return model

# Model Training
input_dim = X_train.shape[1]
model = ParticleClassifier(input_dim)
model = train_model(model, X_train, Y_train, X_val, Y_val)