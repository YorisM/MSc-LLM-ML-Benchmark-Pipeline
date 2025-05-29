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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

def compute_pairwise_features(X):
    batch_size, seq_len = X.shape[0], (X.shape[1] - 3) // 5
    pairwise_features = []
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            E1, px1, py1, eta1, phi1 = X[:, 3 + 5*i : 3 + 5*(i+1)].T
            E2, px2, py2, eta2, phi2 = X[:, 3 + 5*j : 3 + 5*(j+1)].T
            dphi = torch.abs(phi1 - phi2)
            deta = eta1 - eta2
            dR = torch.sqrt(deta**2 + dphi**2)
            m_inv = torch.sqrt((E1 + E2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2)
            pairwise_features.append(torch.stack((dR, m_inv), dim=1))
    return torch.cat(pairwise_features, dim=1)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=128, num_heads=4, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return torch.sigmoid(self.fc(x))

def train_model(model, X_train, Y_train, X_val, Y_val, epochs=10, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train).squeeze()
        loss = criterion(y_pred, Y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val).squeeze()
            auc = roc_auc_score(Y_val.cpu().numpy(), y_val_pred.cpu().numpy())
        
        print(f"Epoch {epoch+1}: Loss = {loss.item()}, AUC = {auc}")
    
    return model

# Compute pairwise features
X_train_extended = torch.cat((X_train, compute_pairwise_features(X_train)), dim=1)
X_val_extended = torch.cat((X_val, compute_pairwise_features(X_val)), dim=1)

# Define model and train
input_dim = X_train_extended.shape[1]
seq_len = (X_train.shape[1] - 3) // 5
model = TransformerClassifier(input_dim=input_dim, seq_len=seq_len)

trained_model = train_model(model, X_train_extended, Y_train, X_val_extended, Y_val, epochs=10)