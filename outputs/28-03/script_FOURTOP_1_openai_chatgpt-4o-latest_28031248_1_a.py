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
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Assume X_train, Y_train, X_val, Y_val are preloaded PyTorch tensors

# Define Dataset wrappers
batch_size = 512
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# Neural network definition
class EventClassifier(nn.Module):
    def __init__(self, input_dim):
        super(EventClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))
        return x.view(-1)

# Preprocessing helper: mask out zero-padded objects (assuming E for object = 0 if padded)
def preprocess_inputs(X):
    with torch.no_grad():
        # Find non-zero object indicators (E values start at index 3 + 1 after obj_id)
        mask = X[:, 3::5][:, ::4] > 0
        object_mask = mask.repeat_interleave(5, dim=1)
        object_mask = F.pad(object_mask, (3, 0), value=True)  # Keep weight, ETmiss, phi_miss
        X_masked = X * object_mask.float()
        return X_masked

X_train = preprocess_inputs(X_train)
X_val = preprocess_inputs(X_val)

# Prepare model
input_dim = X_train.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EventClassifier(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_auc = 0.0
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Evaluate on validation set
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds.append(outputs.cpu().numpy())
            targets.append(batch_y.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    auc = roc_auc_score(targets, preds)

    if auc > best_auc:
        best_auc = auc
        # You could save best model here

    print(f"Epoch {epoch+1}/{num_epochs}: Validation AUC = {auc:.4f}")

print(f"Best Validation AUC: {best_auc:.4f}")