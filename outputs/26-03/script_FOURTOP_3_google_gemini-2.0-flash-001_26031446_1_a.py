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


class PairwiseKinematicFeatures(nn.Module):
    def __init__(self):
        super(PairwiseKinematicFeatures, self).__init__()

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        batch_size, seq_len, feature_dim = x.shape

        # Extract E, pt, eta, phi from the input tensor
        E = x[:, :, 1]
        pt = x[:, :, 2]
        eta = x[:, :, 3]
        phi = x[:, :, 4]

        # Create pairwise combinations of objects
        obj_pairs = torch.combinations(torch.arange(seq_len), r=2)

        # Initialize lists to store pairwise features
        m_inv_list = []
        delta_eta_list = []
        delta_phi_list = []
        delta_R_list = []

        # Calculate pairwise features
        for pair in obj_pairs:
            i, j = pair[0], pair[1]

            # Invariant mass
            E_i, E_j = E[:, i], E[:, j]
            pt_i, pt_j = pt[:, i], pt[:, j]
            eta_i, eta_j = eta[:, i], eta[:, j]
            phi_i, phi_j = phi[:, i], phi[:, j]

            m_inv_squared = (E_i + E_j)**2 - (pt_i * torch.cos(phi_i) + pt_j * torch.cos(phi_j))**2 - \
                              (pt_i * torch.sin(phi_i) + pt_j * torch.sin(phi_j))**2 - (pt_i * torch.sinh(eta_i) + pt_j * torch.sinh(eta_j))**2

            # numerical stability
            m_inv_squared = torch.clamp(m_inv_squared, min=0)

            m_inv = torch.sqrt(m_inv_squared)
            m_inv_list.append(m_inv.unsqueeze(1))

            # Delta eta and delta phi
            delta_eta = eta_i - eta_j
            delta_phi = phi_i - phi_j
            delta_eta_list.append(delta_eta.unsqueeze(1))
            delta_phi_list.append(delta_phi.unsqueeze(1))

            # Delta R
            delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)
            delta_R_list.append(delta_R.unsqueeze(1))

        # Concatenate pairwise features
        m_inv_pairwise = torch.cat(m_inv_list, dim=1)
        delta_eta_pairwise = torch.cat(delta_eta_list, dim=1)
        delta_phi_pairwise = torch.cat(delta_phi_list, dim=1)
        delta_R_pairwise = torch.cat(delta_R_list, dim=1)

        # Concatenate all features. Handle empty pairwise features if seq_len < 2
        if seq_len > 1:
            pairwise_features = torch.cat([m_inv_pairwise.unsqueeze(2), delta_eta_pairwise.unsqueeze(2), delta_phi_pairwise.unsqueeze(2), delta_R_pairwise.unsqueeze(2)], dim=2)
        else:
            # If there are fewer than 2 objects, return zero pairwise features
             pairwise_features = torch.zeros((batch_size, 0, 4), device=x.device)

        return pairwise_features


class TransformerModel(nn.Module):
    def __init__(self, feature_dim, num_encoder_layers, num_heads, hidden_dim, dropout):
        super(TransformerModel, self).__init__()

        self.pairwise_features = PairwiseKinematicFeatures()
        num_pairwise = (8*(8 -1)) // 2 if 8 > 1 else 0


        self.embedding = nn.Linear(feature_dim, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout), num_encoder_layers)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        batch_size, seq_len, feature_dim = x.shape

        # Apply pairwise kinematic features
        pairwise_features = self.pairwise_features(x)

        # Concatenate original features and pairwise features
        x = self.embedding(x)

        # Transformer encoder
        x = x.transpose(0, 1)  # (seq_len, batch_size, feature_dim)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, feature_dim)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Linear layer and sigmoid activation
        x = self.linear(x)
        x = self.sigmoid(x)
        return x.squeeze()


# Example Usage (Assuming X_train, Y_train, X_val, Y_val are already defined as PyTorch tensors)

# Hyperparameters
FEATURE_DIM = 5  # weight, E_T_miss, phi_Et_miss, E, pt, eta, phi
NUM_ENCODER_LAYERS = 2
NUM_HEADS = 4
HIDDEN_DIM = 64
DROPOUT = 0.1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(FEATURE_DIM, NUM_ENCODER_LAYERS, NUM_HEADS, HIDDEN_DIM, DROPOUT).to(device)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
model.train()
for epoch in range(NUM_EPOCHS):
    for i in range(0, X_train.shape[0], BATCH_SIZE):
        X_batch = X_train[i:i + BATCH_SIZE].to(device)
        Y_batch = Y_train[i:i + BATCH_SIZE].to(device)

        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch.float())
        loss.backward()
        optimizer.step()

        if (i / BATCH_SIZE) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i / BATCH_SIZE}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    Y_pred_val = model(X_val.to(device))
    auc = roc_auc_score(Y_val.cpu().numpy(), Y_pred_val.cpu().numpy())
    print(f'Validation AUC: {auc:.4f}')