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

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(106, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Train the model
def train(X_train, Y_train, model, device, dry_run):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        if not dry_run:
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, Y_train.view(-1, 1))
            loss.backward()
            optimizer.step()
        else:
            print("Dry run: epoch", epoch)

# Evaluate the model
def evaluate(X_val, Y_val, model, device):
    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        _, predicted = torch.max(outputs, 1)
        auc = roc_auc_score(Y_val.numpy(), outputs.numpy())
        return auc

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun", action='store_true')
    args = parser.parse_args()
    dry_run = args.dryrun

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.from_numpy(np.load('X_train.npy'))
    Y_train = torch.from_numpy(np.load('Y_train.npy'))
    X_val = torch.from_numpy(np.load('X_val.npy'))
    Y_val = torch.from_numpy(np.load('Y_val.npy'))

    model = Net()
    train(X_train, Y_train, model, device, dry_run)
    auc = evaluate(X_val, Y_val, model, device)
    print("AUC: ", auc)
    torch.save(model.state_dict(), 'classifier_model.pth')