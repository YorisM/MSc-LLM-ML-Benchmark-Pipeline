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

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(106, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train(model, device, X_train, Y_train, optimizer, criterion):
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        print('Epoch {}: Loss = {:.4f}'.format(epoch+1, loss.item()))

def evaluate(model, device, X_val, Y_val):
    model.eval()
    output = model(X_val)
    prediction = (output >= 0.5).int()
    auc = roc_auc_score(Y_val.cpu().numpy(), output.cpu().detach().numpy())
    return auc

def dry_run(model, device, X_val):
    model.eval()
    output = model(X_val[:10])
    print(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dryrun', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BinaryClassifier().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train = torch.tensor(np.random.rand(241657, 106), dtype=torch.float32).to(device)
    Y_train = torch.tensor(np.random.randint(0, 2, 241657), dtype=torch.float32).to(device)
    X_val = torch.tensor(np.random.rand(30272, 106), dtype=torch.float32).to(device)
    Y_val = torch.tensor(np.random.randint(0, 2, 30272), dtype=torch.float32).to(device)

    if args.dryrun:
        dry_run(model, device, X_val)
    else:
        train(model, device, X_train, Y_train, optimizer, criterion)
        auc = evaluate(model, device, X_val, Y_val)
        print('AUC: {:.4f}'.format(auc))
        torch.save(model.state_dict(), 'binary_classifier_model.pth')

if __name__ == '__main__':
    main()