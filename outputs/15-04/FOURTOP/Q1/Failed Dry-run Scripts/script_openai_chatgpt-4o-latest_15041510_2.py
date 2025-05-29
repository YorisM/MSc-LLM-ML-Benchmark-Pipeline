import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Define the neural network architecture
class AUCModel(nn.Module):
    def __init__(self, input_dim):
        super(AUCModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Define function to train the model
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_auc = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x).view(-1)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

        # Evaluate on validation data
        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                output = model(val_x).view(-1)
                preds.extend(output.cpu().numpy())
                targets.extend(val_y.cpu().numpy())

        auc_score = roc_auc_score(targets, preds)
        print(f"Epoch {epoch+1}, AUC: {auc_score:.4f}")

        if auc_score > best_auc:
            best_auc = auc_score
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    return model, best_auc

# Normalize data (important for neural networks)
def normalize_tensor(tensor):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True) + 1e-8
    return (tensor - mean) / std

# Prepare datasets
X_train = normalize_tensor(X_train)
X_val = normalize_tensor(X_val)

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

# Instantiate and train the model
input_dim = X_train.shape[1]
model = AUCModel(input_dim)
model, final_auc = train_model(model, train_loader, val_loader)

print(f"Best Validation AUC: {final_auc:.5f}")