import pandas as pd
import torch

X_train_df = pd.read_csv('./data/fourtops/X_train.csv')
Y_train_df  = pd.read_csv('./data/fourtops/Y_train.csv')
X_val_df   = pd.read_csv('./data/fourtops/X_val.csv')
Y_val_df   = pd.read_csv('./data/fourtops/Y_val.csv')

X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
Y_train  = torch.tensor(Y_train_df.values, dtype=torch.long)
X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long)


#LLM Response below

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

# Define a simple feed-forward neural network
class ParticleNet(nn.Module):
    def __init__(self, input_size):
        super(ParticleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
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

# Prepare DataLoader for efficient batch processing
batch_size = 512
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
model = ParticleNet(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_Y.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation performance
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            preds = model(batch_X).squeeze()
            val_preds.extend(preds.numpy())
            val_targets.extend(batch_Y.numpy())
    
    auc_score = roc_auc_score(val_targets, val_preds)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, AUC: {auc_score:.4f}')