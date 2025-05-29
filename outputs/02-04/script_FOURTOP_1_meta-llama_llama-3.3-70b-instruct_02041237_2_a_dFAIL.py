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
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import argparse

# Define the binary classifier model

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(106, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        selfActivation = nn.ReLU()
    
    def forward(self, x):
        x = selfActivation(self.fc1(x))
        x = selfActivation(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the training function

def train(model, device, X_train, Y_train, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(zip(X_train.split(100), Y_train.split(100))):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.type(torch.long).to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('Train Loss: {:.4f}'.format(total_loss / len(X_train.split(100))))

# Define the evaluation function

def evaluate(model, device, X_val, Y_val):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        data, target = X_val.to(device), Y_val.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total_correct += (predicted == target.type(torch.long).to(device)).sum().item()
    accuracy = total_correct / len(Y_val)
    return accuracy

# Define the main function

def main(dryrun=False):
    parser = argparse.ArgumentParser(description='Binary Classifier for Particle Physics')
    parser.add_argument('--dryrun', action='store_true', help='Dry run mode')
    args = parser.parse_args()
    if dryrun:
        print('Dry run mode. Not training the model.')
        return
    
    # Load the data
    X_train = torch.load('X_train.pth')
    X_val = torch.load('X_val.pth')
    Y_train = torch.load('Y_train.pth')
    Y_val = torch.load('Y_val.pth')
    
    # Define the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model, optimizer, and criterion
    model = BinaryClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    for epoch in range(10):
        train(model, device, X_train, Y_train, optimizer, criterion)
    
    # Evaluate the model
    accuracy = evaluate(model, device, X_val, Y_val)
    
    # Calculate the AUC score
    with torch.no_grad():
        output = model(X_val.to(device))
        _, predicted = torch.max(output.data, 1)
        auc_score = roc_auc_score(Y_val.numpy(), predicted.cpu().numpy())
    
    # Save the trained model
    torch.save(model.state_dict(), 'binary_classifier_model.pth')
    
    # Print the final AUC score
    print('Final AUC Score: {:.4f}'.format(auc_score))

# Run the main function
if __name__ == '__main__':
    main()

# Dry run functionality
if __name__ == '__main__':
    import sys
    if '--dryrun' in sys.argv:
        main(dryrun=True)