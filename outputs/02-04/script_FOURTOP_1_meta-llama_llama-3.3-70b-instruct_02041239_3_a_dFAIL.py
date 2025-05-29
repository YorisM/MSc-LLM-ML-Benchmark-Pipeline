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
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse

# Define the binary classifier model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(106, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Define the training function
def train(model, device, X_train, Y_train, optimizer, criterion):
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Define the evaluation function
def evaluate(model, device, X_val, Y_val):
    model.eval()
    with torch.no_grad():
        output = model(X_val)
        auc = roc_auc_score(Y_val.cpu().numpy(), output.cpu().numpy())
        return auc

# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dryrun', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train = torch.load('X_train.pth', map_location=device)
    Y_train = torch.load('Y_train.pth', map_location=device)
    X_val = torch.load('X_val.pth', map_location=device)
    Y_val = torch.load('Y_val.pth', map_location=device)

    if args.dryrun:
        print('Dry run, no training or evaluation performed')
    else:
        model = BinaryClassifier().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, device, X_train, Y_train, optimizer, criterion)
        auc = evaluate(model, device, X_val, Y_val)
        print(f'AUC: {auc:.4f}')
        torch.save(model.state_dict(), 'binary_classifier_model.pth')