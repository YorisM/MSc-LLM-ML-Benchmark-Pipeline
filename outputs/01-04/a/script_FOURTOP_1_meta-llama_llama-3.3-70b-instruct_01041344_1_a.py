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
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the function to train the model
def train(model, device, X_train, Y_train, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(zip(X_train, Y_train)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long().view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(X_train)

# Define the function to evaluate the model
def evaluate(model, device, X_val, Y_val):
    model.eval()
    total_loss = 0
    predicted = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(zip(X_val, Y_val)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target.long().view(-1))
            total_loss += loss.item()
            predicted.extend(torch.argmax(output, dim=1).cpu().numpy())
    predicted = np.array(predicted)
    auc = roc_auc_score(Y_val.cpu().numpy(), predicted)
    return total_loss / len(X_val), auc

# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dryrun', action='store_true', help='dry run, do not save the model')
    args = parser.parse_args()

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    X_train = torch.randn(241657, 106)
    Y_train = torch.randn(241657)
    X_val = torch.randn(30272, 106)
    Y_val = torch.randn(30272)

    # Initialize the model, optimizer and criterion
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(10):
        train_loss = train(model, device, X_train, Y_train, optimizer, criterion)
        val_loss, auc = evaluate(model, device, X_val, Y_val)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, AUC: {auc:.4f}')

    # Save the model
    if not args.dryrun:
        torch.save(model.state_dict(), 'classifier_model.pth')
        print(f'Model saved as classifier_model.pth')
    print(f'Final AUC score: {auc:.4f}')