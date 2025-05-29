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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

# Preprocessing: Calculate normalization parameters and embedding dimensions
features_train = X_train[:, 1:]

# Global feature normalization
global_mean = features_train[:, :2].mean(dim=0)
global_std = features_train[:, :2].std(dim=0)

# Object feature normalization
objects_all = features_train[:, 2:].reshape(-1, 5)
non_zero_mask = objects_all[:, 0] != 0
cont_features = objects_all[non_zero_mask][:, 1:5]
cont_mean = cont_features.mean(dim=0) if len(cont_features) > 0 else torch.zeros(4)
cont_std = cont_features.std(dim=0) if len(cont_features) > 0 else torch.ones(4)

# Calculate maximum object type for embeddings
non_zero_obj_types = objects_all[non_zero_mask][:, 0]
max_obj_type = int(non_zero_obj_types.max().item()) if len(non_zero_obj_types) > 0 else 0
num_embeddings = max_obj_type + 1

# Define physics-informed neural network
class ParticleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('global_mean', global_mean)
        self.register_buffer('global_std', global_std)
        self.register_buffer('cont_mean', cont_mean)
        self.register_buffer('cont_std', cont_std)
        
        self.obj_embedding = nn.Embedding(num_embeddings, 16, padding_idx=0)
        self.cont_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=80,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(144, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Split features
        global_feats = (x[:, :2] - self.global_mean) / (self.global_std + 1e-8)
        objects = x[:, 2:].view(x.size(0), -1, 5)
        
        # Process objects
        obj_types = objects[:, :, 0].long()
        mask = obj_types == 0
        cont_feats = (objects[:, :, 1:5] - self.cont_mean) / (self.cont_std + 1e-8)
        
        embedded = self.obj_embedding(obj_types)
        cont_encoded = self.cont_encoder(cont_feats)
        obj_features = torch.cat([embedded, cont_encoded], dim=-1)
        
        # Transformer processing
        transformed = self.transformer(obj_features, src_key_padding_mask=mask)
        aggregated = transformed.mean(dim=1)
        
        # Combine features
        global_encoded = self.global_encoder(global_feats)
        combined = torch.cat([aggregated, global_encoded], dim=1)
        return self.classifier(combined).squeeze()

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ParticleTransformer().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Create DataLoader with sample weights
train_dataset = TensorDataset(X_train[:, 1:], Y_train, X_train[:, 0])
val_dataset = TensorDataset(X_val[:, 1:], Y_val, X_val[:, 0])
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048)

# Training loop
for epoch in range(30):
    model.train()
    for features, labels, weights in train_loader:
        features, labels, weights = features.to(device), labels.to(device), weights.to(device)
        
        optimizer.zero_grad()
        logits = model(features)
        loss = (F.binary_cross_entropy_with_logits(logits, labels, reduction='none') * weights).mean()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for features, labels, _ in val_loader:
            features, labels = features.to(device), labels.to(device)
            preds = torch.sigmoid(model(features)).cpu()
            val_preds.append(preds)
            val_labels.append(labels.cpu())
    
    val_preds = torch.cat(val_preds).numpy()
    val_labels = torch.cat(val_labels).numpy()
    auc = roc_auc_score(val_labels, val_preds)
    print(f'Epoch {epoch+1}, Val AUC: {auc:.4f}')