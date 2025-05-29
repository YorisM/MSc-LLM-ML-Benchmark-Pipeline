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
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse

# Define the classifier model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(106, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Set up the device (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the dataset
X_train = torch.load("X_train.pt")
Y_train = torch.load("Y_train.pt")
X_val = torch.load("X_val.pt")
Y_val = torch.load("Y_val.pt")

# Define the data loaders
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Initialize the classifier model, optimizer, and loss function
model = Classifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# Train the model
def train(model, device, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device).view(-1, 1)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluate the model
def evaluate(model, device, loader):
    model.eval()
    total_correct = 0
    total_loss = 0
    predictions = np.array([])
    targets = np.array([])
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device).view(-1, 1)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            predictions = np.concatenate((predictions, output.cpu().numpy()))
            targets = np.concatenate((targets, target.cpu().numpy()))
    auc = roc_auc_score(targets, predictions)
    return auc

# Dry run functionality
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()
    if args.dryrun:
        print("Dry run: Model will not be trained or saved.")
    else:
        print("Training model...")
        for epoch in range(10):
            train_loss = train(model, device, train_loader, optimizer, loss_fn)
            auc = evaluate(model, device, val_loader)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val AUC: {auc:.4f}")
        torch.save(model.state_dict(), "classifier_model.pth")
        print(f"Final AUC: {auc:.4f}")