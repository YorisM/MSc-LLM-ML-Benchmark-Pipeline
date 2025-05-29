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
from torch.utils.data import DataLoader, TensorDataset

def preprocess(X):
    # Normalize physical quantities to improve stability
    X[:, 0] = torch.log1p(X[:, 0])  # log-scaling weight
    X[:, 1] = torch.log1p(X[:, 1])  # log-scaling E_T_miss
    X[:, 2] = X[:, 2] / torch.pi  # Normalize phi_{E_T_miss}
    X[:, 3:] = torch.log1p(X[:, 3:])  # log scaling for 4-vectors
    return X

class ParticleClassifier(nn.Module):
    def __init__(self, input_size):
        super(ParticleClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).squeeze()

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_auc = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y.float())
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).cpu().numpy()
            val_auc = roc_auc_score(Y_val.cpu().numpy(), val_preds)
            print(f'Epoch {epoch+1}, AUC: {val_auc:.5f}')
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), 'best_model.pth')

def evaluate_model(model, X_val, Y_val):
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).cpu().numpy()
        return roc_auc_score(Y_val.cpu().numpy(), val_preds)

# Data Preprocessing
X_train = preprocess(X_train)
X_val = preprocess(X_val)

# Dataset and DataLoader
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# Model Training
model = ParticleClassifier(input_size=X_train.shape[1]).to(X_train.device)
train_model(model, train_loader, val_loader, epochs=20, lr=1e-3)

# Load best model and evaluate
model.load_state_dict(torch.load('best_model.pth'))
final_auc = evaluate_model(model, X_val, Y_val)
print(f'Final AUC: {final_auc:.5f}')