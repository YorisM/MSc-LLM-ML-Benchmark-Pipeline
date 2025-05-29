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
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score

# Define the neural network architecture
class ParticleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ParticleClassifier, self).__init__()
        self.dropout_rate = 0.3
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Training function
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).squeeze().cpu().numpy()
            preds.extend(outputs)
            true_labels.extend(batch_y.numpy())
    return roc_auc_score(true_labels, preds)

# Main function
def main(dryrun=False):
    # Assume X_train, Y_train, X_val, Y_val are already loaded as torch tensors
    # They are assumed to be in scope
    global X_train, Y_train, X_val, Y_val

    input_dim = X_train.shape[1]
    batch_size = 512
    epochs = 10 if dryrun else 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ParticleClassifier(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(epochs):
        train_model(model, train_loader, criterion, optimizer, device)

    # Evaluate model
    auc_score = evaluate_model(model, val_loader, device)
    print(f"Final AUC Score: {auc_score:.5f}")

    # Save model
    torch.save(model.state_dict(), "classifier_model.pth")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun", action="store_true", help="Run a dry run with fewer epochs")
    args = parser.parse_args()

    # Placeholders for this runnable code
    # Normally these would be loaded externally or passed in
    # For test/dry run purpose, we generate random tensors matching the expected shape
    X_train = torch.randn(241657, 106)
    Y_train = torch.randint(0, 2, (241657,), dtype=torch.float32)
    X_val = torch.randn(30272, 106)
    Y_val = torch.randint(0, 2, (30272,), dtype=torch.float32)

    main(dryrun=args.dryrun)