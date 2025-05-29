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


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
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


# Preprocessing function
def preprocess_data(X):
    # Convert to numpy array if it's a tensor
    if isinstance(X, torch.Tensor):
        X = X.numpy()

    # Replace NaN values with 0
    X = np.nan_to_num(X, nan=0.0)
    
    # Flatten the input
    X = X.reshape(X.shape[0], -1)
    return X


# Training function
def train_model(model, X_train, Y_train, X_val, Y_val, epochs=10, learning_rate=0.001, batch_size = 32):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).reshape(-1, 1)

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            Y_batch = Y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, Y_val_tensor)
            predicted = val_outputs.cpu().numpy()
            auc = roc_auc_score(Y_val, predicted)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, AUC: {auc:.4f}')


# Apply Preprocessing
X_train_processed = preprocess_data(X_train)
X_val_processed = preprocess_data(X_val)

# Define input dimension
input_dim = X_train_processed.shape[1]

# Initialize the model
model = SimpleNN(input_dim)

# Train the model
train_model(model, X_train_processed, Y_train, X_val_processed, Y_val)


# Function to evaluate the model's AUC
def evaluate_model(model, X, Y):
    X_processed = preprocess_data(X)
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor)
        predicted = outputs.cpu().numpy()
        auc = roc_auc_score(Y, predicted)
    return auc


# Example usage (evaluate the model on the validation set):
auc_val = evaluate_model(model, X_val, Y_val)
print(f'Validation AUC: {auc_val:.4f}')