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
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
import numpy as np

def compute_pairwise_features(event):
    nonzero_mask = event[:, 0] != 0
    event = event[nonzero_mask]
    pairwise_features = []
    for i in range(len(event)):
        for j in range(i+1, len(event)):
            delta_phi = np.abs(event[i, 3] - event[j, 3])
            delta_eta = np.abs(event[i, 2] - event[j, 2])
            invariant_mass = np.sqrt(2 * event[i, 1] * event[j, 1] * (np.cosh(delta_eta) - np.cos(delta_phi)))
            pairwise_features.append(invariant_mass)
    return np.pad(pairwise_features, (0, max_objects**2 - len(pairwise_features)), 'constant')

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers, num_classes=1):
        super(TransformerClassifier, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return self.sigmoid(x)

max_objects = int((X_train.shape[1] - 3) // 4)
pairwise_train = np.array([compute_pairwise_features(X_train[i].reshape(max_objects, 4)) for i in range(len(X_train))])
pairwise_val = np.array([compute_pairwise_features(X_val[i].reshape(max_objects, 4)) for i in range(len(X_val))])

X_train_extended = np.concatenate((X_train, pairwise_train), axis=1)
X_val_extended = np.concatenate((X_val, pairwise_val), axis=1)

train_dataset = data.TensorDataset(torch.tensor(X_train_extended, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
val_dataset = data.TensorDataset(torch.tensor(X_val_extended, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))

train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=128, shuffle=False)

model = TransformerClassifier(input_dim=X_train_extended.shape[1], num_heads=4, ff_dim=256, num_layers=3).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(torch.device("cuda")), batch_y.to(torch.device("cuda"))
        optimizer.zero_grad()
        output = model(batch_X).squeeze()
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_targets = []
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(torch.device("cuda")), batch_y.to(torch.device("cuda"))
            output = model(batch_X).squeeze().cpu().numpy()
            val_preds.extend(output)
            val_targets.extend(batch_y.cpu().numpy())
        auc = roc_auc_score(val_targets, val_preds)
        print(f"Epoch {epoch+1}, Validation AUC: {auc:.4f}")