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


class PairwiseInteraction(nn.Module):
    def __init__(self, input_dim):
        super(PairwiseInteraction, self).__init__()
        self.input_dim = input_dim

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        # Create all pairwise combinations of objects
        obj_pairs = torch.combinations(torch.arange(seq_len), r=2)
        num_pairs = obj_pairs.shape[0]

        # Gather the features for each object in the pairs
        obj1_features = x[:, obj_pairs[:, 0]]  # (batch_size, num_pairs, input_dim)
        obj2_features = x[:, obj_pairs[:, 1]]  # (batch_size, num_pairs, input_dim)

        # Compute pairwise kinematic features
        delta_eta = obj1_features[:, :, 2] - obj2_features[:, :, 2] # index 2 corresponds to eta
        delta_phi = obj1_features[:, :, 3] - obj2_features[:, :, 3] # index 3 corresponds to phi
        delta_r = torch.sqrt(delta_eta**2 + delta_phi**2)
        inv_mass = torch.sqrt(2 * obj1_features[:, :, 1] * obj2_features[:, :, 1] * (torch.cosh(delta_eta) - torch.cos(delta_phi))) #index 1 corresponds to pt

        # Concatenate pairwise features
        pairwise_features = torch.stack([delta_eta, delta_phi, delta_r, inv_mass], dim=-1)

        return pairwise_features


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, hidden_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout), num_layers)
        self.pairwise_interaction = PairwiseInteraction(input_dim)
        self.fc = nn.Linear(hidden_dim , 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # (batch_size, seq_len, hidden_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        x = self.transformer_encoder(x)  # (seq_len, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)
        x = torch.mean(x, dim=1) # (batch_size, hidden_dim), average over sequence length

        pairwise_features = self.pairwise_interaction(x)

        x = self.dropout(x)
        x = self.fc(x) # (batch_size, 1)
        x = self.sigmoid(x)
        return x


def train_model(model, X_train, Y_train, X_val, Y_val, epochs=10, lr=0.001, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_auc = roc_auc_score(Y_val.cpu().numpy(), val_outputs.cpu().numpy())
            print(f'Epoch {epoch+1}, Validation AUC: {val_auc}')


# Example Usage (replace with your actual data loading and preprocessing)
# Assuming X_train, Y_train, X_val, Y_val are already loaded as PyTorch tensors

# Example shape of X:
# print(X_train.shape) #torch.Size([241657, 132, 4])
# print(Y_train.shape) #torch.Size([241657])


input_dim = X_train.shape[2] #4
num_layers = 2
num_heads = 4
hidden_dim = 64

model = TransformerModel(input_dim, num_layers, num_heads, hidden_dim)

train_model(model, X_train, Y_train, X_val, Y_val, epochs=5, lr=0.001, batch_size=64)