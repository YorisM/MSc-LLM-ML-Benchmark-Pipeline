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

# Data Preprocessing - Normalize kinematic variables
def preprocess_data(X):
    mask = X != 0  # Mask to avoid normalizing zero-padded values
    mean = X[mask].mean()
    std = X[mask].std()
    X = (X - mean) / std
    X[~mask] = 0  # Keep zero-padding unchanged
    return X

X_train = preprocess_data(X_train)
X_val = preprocess_data(X_val)

# Define the Neural Network model
class ParticleNet(nn.Module):
    def __init__(self, input_size):
        super(ParticleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x.squeeze()

# Initialize model, loss function and optimizer.
input_size = X_train.shape[1]
model = ParticleNet(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
batch_size = 1024
dataset = torch.utils.data.TensorDataset(X_train, Y_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_Y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y.float())
        loss.backward()
        optimizer.step()
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).detach().numpy()
        auc_score = roc_auc_score(Y_val.numpy(), val_preds)
    print(f"Epoch {epoch+1}/{num_epochs}, AUC: {auc_score:.4f}")

# Save the trained model
torch.save(model.state_dict(), "particle_net.pth")