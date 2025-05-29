import pandas as pd
import torch

X_train_df = pd.read_csv('./data/fourtops/X_train.csv')
Y_train_df  = pd.read_csv('./data/fourtops/Y_train.csv')
X_val_df   = pd.read_csv('./data/fourtops/X_val.csv')
Y_val_df   = pd.read_csv('./data/fourtops/Y_val.csv')

X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()

print(f"X_train shape: {X_train.shape}\nY_train shape: {Y_train.shape}\nX_val shape: {X_val.shape}\nY_val shape: {Y_val.shape}")

# LLM Reponse Below
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse
import os

# Define the classifier architecture
class ParticleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ParticleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.2)

        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        x = torch.sigmoid(self.output(x))
        return x

# Training function
def train_model(model, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=512, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    return model

# Evaluation function
def evaluate_model(model, X_val, Y_val):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        X_val = X_val.to(device)
        outputs = model(X_val).cpu().numpy().ravel()
        auc = roc_auc_score(Y_val.numpy(), outputs)
    return auc

def main(dryrun=False):
    input_dim = 106

    if dryrun:
        X_dummy = torch.rand((200, input_dim), dtype=torch.float32)
        Y_dummy = torch.randint(0, 2, (200,), dtype=torch.float32)
        X_val_dummy = torch.rand((50, input_dim), dtype=torch.float32)
        Y_val_dummy = torch.randint(0, 2, (50,), dtype=torch.float32)

        model = ParticleClassifier(input_dim)
        model = train_model(model, X_dummy, Y_dummy, X_val_dummy, Y_val_dummy, epochs=2)
        auc = evaluate_model(model, X_val_dummy, Y_val_dummy)
        print(f"[DRYRUN] AUC: {auc:.5f}")
    else:
        # Assume these are pre-imported globally available
        global X_train, Y_train, X_val, Y_val
        model = ParticleClassifier(input_dim)
        model = train_model(model, X_train, Y_train, X_val, Y_val, epochs=10)
        auc = evaluate_model(model, X_val, Y_val)
        print(f"Final AUC: {auc:.5f}")
        torch.save(model.state_dict(), os.path.splitext(__file__)[0] + '_model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a binary classifier on rare signal events')
    parser.add_argument('--dryrun', action='store_true', help='Run a dry run with dummy data')
    args = parser.parse_args()
    main(dryrun=args.dryrun)