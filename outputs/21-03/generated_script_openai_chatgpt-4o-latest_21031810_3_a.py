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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

# Define a simple deep neural network
class ParticlePhysicsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ParticlePhysicsClassifier, self).__ninit__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
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

# Prepare data
def prepare_data(X, Y, batch_size=1024):
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# Training function
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
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
            val_auc = roc_auc_score(Y_val.cpu().numpy(), val_preds)
            print(f"Epoch {epoch+1}, Validation AUC: {val_auc:.4f}")

# Initialize and train the model
input_dim = X_train.shape[1]
model = ParticlePhysicsClassifier(input_dim)

train_loader = prepare_data(X_train, Y_train)
val_loader = prepare_data(X_val, Y_val)

train_model(model, train_loader, val_loader, epochs=20, lr=1e-3)