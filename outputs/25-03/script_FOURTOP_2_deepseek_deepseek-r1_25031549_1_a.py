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
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score

# Data preprocessing
weights_train = X_train[:, 0]
features_train = X_train[:, 1:]

# Extract features and calculate normalization parameters
max_objects = (features_train.shape[1] - 2) // 5
all_objects_train = features_train[:, 2:].reshape(-1, max_objects, 5)
mask_train = all_objects_train[:, :, 0] != 0
real_objects_train = all_objects_train[mask_train]

# Calculate normalization statistics
stats = {
    'et_miss': (features_train[:, 0].float().mean().item(), features_train[:, 0].float().std().item()),
    'phi_miss': (features_train[:, 1].float().mean().item(), features_train[:, 1].float().std().item())
}

if len(real_objects_train) > 0:
    stats.update({
        'E': (real_objects_train[:, 1].float().mean().item(), real_objects_train[:, 1].float().std().item()),
        'pT': (real_objects_train[:, 2].float().mean().item(), real_objects_train[:, 2].float().std().item()),
        'eta': (real_objects_train[:, 3].float().mean().item(), real_objects_train[:, 3].float().std().item()),
        'phi': (real_objects_train[:, 4].float().mean().item(), real_objects_train[:, 4].float().std().item())
    })
else:
    stats.update({'E': (0,1), 'pT': (0,1), 'eta': (0,1), 'phi': (0,1)})

obj_type_vocab_size = int(all_objects_train[:, 0].max().item()) + 1 if len(real_objects_train) > 0 else 1

def normalize_features(features):
    processed = features.clone()
    processed[:, 0] = (processed[:, 0] - stats['et_miss'][0]) / stats['et_miss'][1]
    processed[:, 1] = (processed[:, 1] - stats['phi_miss'][0]) / stats['phi_miss'][1]
    
    objects = processed[:, 2:].reshape(-1, max_objects, 5)
    objects[:, :, 1] = (objects[:, :, 1] - stats['E'][0]) / stats['E'][1]
    objects[:, :, 2] = (objects[:, :, 2] - stats['pT'][0]) / stats['pT'][1]
    objects[:, :, 3] = (objects[:, :, 3] - stats['eta'][0]) / stats['eta'][1]
    objects[:, :, 4] = (objects[:, :, 4] - stats['phi'][0]) / stats['phi'][1]
    
    pad_mask = objects[:, :, 0] == 0
    for i in [1,2,3,4]:
        objects[:, :, i][pad_mask] = 0
    
    return torch.cat([processed[:, :2], objects.reshape(-1, max_objects*5)], dim=1)

# Create normalized datasets
features_train_norm = normalize_features(features_train)
features_val_norm = normalize_features(X_val[:, 1:])

train_dataset = TensorDataset(features_train_norm, Y_train, weights_train)
val_dataset = TensorDataset(features_val_norm, Y_val, X_val[:, 0])

# Define model architecture
class ParticleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_embed = nn.Embedding(obj_type_vocab_size, 32)
        self.obj_encoder = nn.Sequential(
            nn.Linear(32+4, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        global_features = x[:, :2]
        objects = x[:, 2:].reshape(batch_size, max_objects, 5)
        
        obj_types = objects[:, :, 0].long()
        obj_cont = objects[:, :, 1:5]
        
        embedded = self.obj_embed(obj_types)
        obj_features = torch.cat([embedded, obj_cont], dim=-1)
        obj_encoded = self.obj_encoder(obj_features)
        
        mask = (obj_types != 0).unsqueeze(-1)
        obj_pooled = (obj_encoded * mask).sum(dim=1)
        
        global_encoded = self.global_encoder(global_features)
        combined = torch.cat([obj_pooled, global_encoded], dim=1)
        return self.classifier(combined).squeeze()

# Training configuration
model = ParticleClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss(reduction='none')
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# Training loop
for epoch in range(20):
    model.train()
    for x, y, w in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = (criterion(pred, y.float()) * w).mean()
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_preds, val_labels, val_weights = [], [], []
    with torch.no_grad():
        for x, y, w in val_loader:
            val_preds.append(torch.sigmoid(model(x)).cpu())
            val_labels.append(y.cpu())
            val_weights.append(w.cpu())
    auc = roc_auc_score(np.concatenate(val_labels), np.concatenate(val_preds),
                        sample_weight=np.concatenate(val_weights))
    print(f'Epoch {epoch+1}, Val AUC: {auc:.4f}')