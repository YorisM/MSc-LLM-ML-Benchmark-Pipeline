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
from sklearn.metrics import roc_auc_score
import numpy as np


class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x



def train_model(X_train, Y_train, X_val, Y_val, epochs=10, learning_rate=0.001, batch_size = 32):
    # Flatten the input tensors, handling potential ragged tensors by padding with zeros
    input_size = X_train.shape[1] * X_train.shape[2]
    X_train_flat = X_train.reshape(-1, input_size)
    X_val_flat = X_val.reshape(-1, input_size)

    # Handle NaN values by replacing them with 0
    X_train_flat = torch.nan_to_num(X_train_flat, nan=0.0)  
    X_val_flat = torch.nan_to_num(X_val_flat, nan=0.0)
    
    model = SimpleNN(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train_flat), batch_size):
            X_batch = X_train_flat[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size].float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_flat)
            val_loss = criterion(val_outputs, Y_val.float().unsqueeze(1))
            auc = roc_auc_score(Y_val.cpu().numpy(), val_outputs.cpu().numpy())

        print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss.item():.4f}, AUC: {auc:.4f}')

    return model


# Example usage (assuming X_train, Y_train, X_val, Y_val are already loaded as tensors)
#X_train = torch.randn(1000, 10, 20)  # Example data
#Y_train = torch.randint(0, 2, (1000,)) # Example labels
#X_val = torch.randn(500, 10, 20)    # Example data
#Y_val = torch.randint(0, 2, (500,))   # Example labels

#Ensure data types are correct
X_train = X_train.float()
X_val = X_val.float()
Y_train = Y_train.long()
Y_val = Y_val.long()

model = train_model(X_train, Y_train, X_val, Y_val)