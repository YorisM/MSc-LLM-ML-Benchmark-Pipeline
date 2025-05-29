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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Data Preprocessing
def preprocess_data(X):
    # Mask padding values (zero-padding strategy)
    mask = (X != 0).float()
    # Normalize energy and momenta
    X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(dim=0)) / (X[:, 1:].std(dim=0) + 1e-8)
    return X, mask

X_train, mask_train = preprocess_data(X_train)
X_val, mask_val = preprocess_data(X_val)

dataset_train = TensorDataset(X_train, Y_train, mask_train)
dataset_val = TensorDataset(X_val, Y_val, mask_val)

dataloader_train = DataLoader(dataset_train, batch_size=512, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=512, shuffle=False)

# Model Definition
class ParticleNet(nn.Module):
    def __init__(self, input_dim):
        super(ParticleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc_out = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()
    
    def forward(self, x, mask):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.fc_out(x)
        return torch.sigmoid(x)

model = ParticleNet(X_train.shape[1]).to("cuda")

# Training Setup
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for X_batch, Y_batch, mask_batch in dataloader_train:
        X_batch, Y_batch, mask_batch = X_batch.to("cuda"), Y_batch.to("cuda"), mask_batch.to("cuda")
        optimizer.zero_grad()
        outputs = model(X_batch, mask_batch).squeeze()
        loss = criterion(outputs, Y_batch.float())
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, Y_batch, mask_batch in dataloader_val:
            X_batch, Y_batch, mask_batch = X_batch.to("cuda"), Y_batch.to("cuda"), mask_batch.to("cuda")
            outputs = model(X_batch, mask_batch).squeeze()
            y_true.extend(Y_batch.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    
    auc = roc_auc_score(y_true, y_pred)
    print(f"Epoch {epoch + 1}, AUC: {auc:.4f}")