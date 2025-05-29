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

# Define the model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(106, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Define the training function
def train(model, device, X_train, Y_train, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(zip(X_train, Y_train)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(X_train)

# Define the evaluation function
def evaluate(model, device, X_val, Y_val):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        outputs = model(X_val)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == Y_val).sum().item()
    return total_correct / len(Y_val)

# Define the AUC calculation function
def calculate_auc(model, device, X_val, Y_val):
    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        auc = roc_auc_score(Y_val.cpu().numpy(), outputs.cpu().numpy())
    return auc

# Main function
def main(dry_run=False):
    # Set the seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load the data
    X_train = torch.load('X_train.pth')
    Y_train = torch.load('Y_train.pth')
    X_val = torch.load('X_val.pth')
    Y_val = torch.load('Y_val.pth')

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model, optimizer, and criterion
    model = BinaryClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Train the model
    if not dry_run:
        for epoch in range(10):
            loss = train(model, device, X_train, Y_train, optimizer, criterion)
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    # Evaluate the model
    auc = calculate_auc(model, device, X_val, Y_val)
    print(f'AUC: {auc:.4f}')

    # Save the model
    if not dry_run:
        torch.save(model.state_dict(), 'binary_classifier_model.pth')

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dryrun', action='store_true', help='Dry run mode')
args = parser.parse_args()

# Run the main function
main(args.dryrun)