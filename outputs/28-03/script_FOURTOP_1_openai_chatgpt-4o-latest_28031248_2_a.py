import pandas as pd
import torch

X_train_df = pd.read_csv('./data/fourtops/X_train.csv')
Y_train_df = pd.read_csv('./data/fourtops/Y_train.csv')
X_val_df   = pd.read_csv('./data/fourtops/X_val.csv')
Y_val_df   = pd.read_csv('./data/fourtops/Y_val.csv')

X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()
print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)


#LLM Response below

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define the neural network architecture
class EventClassifier(nn.Module):
    def __init__(self, input_dim):
        super(EventClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.output = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        return torch.sigmoid(self.output(x))

# Preprocessing: Replace NaNs and ensure float32
def preprocess(X):
    X = torch.where(torch.isnan(X), torch.tensor(0.0, dtype=X.dtype), X)
    return X.float()

X_train = preprocess(X_train)
X_val = preprocess(X_val)
Y_train = Y_train.float()
Y_val = Y_val.float()

# Define datasets and loaders
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

# Initialize model, optimizer, and loss function
input_dim = X_train.shape[1]
model = EventClassifier(input_dim)
model = model.cuda() if torch.cuda.is_available() else model

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# Training function
def train(model, loader):
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)
        optimizer.zero_grad()
        preds = model(X_batch).squeeze()
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

# Evaluation function
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(model.device)
            preds = model(X_batch).squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    auc = roc_auc_score(all_labels, all_preds)
    return auc, all_labels, all_preds

# Add .device property to model
model.device = next(model.parameters()).device

# Train the model
best_auc = 0.0
for epoch in range(20):
    train(model, train_loader)
    auc, _, _ = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}, AUC = {auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        best_model_state = model.state_dict()

# Load best model
model.load_state_dict(best_model_state)
final_auc, _, _ = evaluate(model, val_loader)
print(f"Best AUC on validation set: {final_auc:.4f}")