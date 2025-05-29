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
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score

class ParticleClassifier(torch.nn.Module):
    def __init__(self, max_obj_id=20):
        super().__init__()
        self.embedding = torch.nn.Embedding(max_obj_id + 1, 8, padding_idx=0)
        self.global_bn = torch.nn.BatchNorm1d(2)
        self.obj_net = torch.nn.Sequential(
            torch.nn.Linear(12, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(18, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        global_features = self.global_bn(x[:, 1:3])
        n_features = x.size(1)
        max_objects = (n_features - 3) // 5
        objects = x[:, 3:].view(batch_size, max_objects, 5)
        obj_ids = objects[:, :, 0].long()
        obj_kinematics = objects[:, :, 1:]
        mask = obj_ids != 0

        obj_emb = self.embedding(obj_ids)
        obj_features = torch.cat([obj_emb, obj_kinematics], dim=-1)
        obj_features_flat = obj_features.view(-1, 12)
        processed_flat = self.obj_net(obj_features_flat)
        processed = processed_flat.view(batch_size, max_objects, -1)

        mask_expanded = mask.unsqueeze(-1).float()
        processed_masked = processed * mask_expanded
        aggregated = processed_masked.sum(dim=1)
        combined = torch.cat([aggregated, global_features], dim=1)
        logits = self.classifier(combined).squeeze(-1)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ParticleClassifier().to(device)

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    model.train()
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        event_weights = X_batch[:, 0]
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, Y_batch.float(), weight=event_weights
        )
        loss.backward()
        optimizer.step()

    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true.extend(Y_batch.cpu().numpy())
            y_score.extend(probs)
    auc = roc_auc_score(y_true, y_score)
    print(f"Epoch {epoch}, Validation AUC: {auc:.4f}")