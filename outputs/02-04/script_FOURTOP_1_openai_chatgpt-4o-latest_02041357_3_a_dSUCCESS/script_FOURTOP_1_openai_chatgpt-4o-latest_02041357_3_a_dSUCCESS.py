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
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

# Simple feedforward neural network
class ParticleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ParticleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)

def compute_auc(true_labels, predictions):
    return roc_auc_score(true_labels, predictions)

def main(dryrun=False):
    # Assume these are pre-loaded torch tensors
    global X_train, Y_train, X_val, Y_val

    # Create torch.utils.data.Dataset and DataLoader
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # Initialize model, loss, optimizer
    model = ParticleClassifier(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if dryrun:
        print("Dry run complete. Model architecture:")
        print(model)
        return

    # Training loop
    model.train()
    for epoch in range(10):  # Number of epochs can be extended
        epoch_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            pred = model(inputs).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(labels.numpy())

    auc_score = compute_auc(all_labels, all_preds)
    print(f"Final AUC score: {auc_score:.6f}")

    # Save model
    torch.save(model.state_dict(), "particle_classifier_model.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dryrun', action='store_true', help='Perform a dry run')
    args = parser.parse_args()

    # Dummy placeholders for imports – replace with actual data
    try:
        X_train
        Y_train
        X_val
        Y_val
    except NameError:
        # For testing the dryrun without loading actual data
        X_train = torch.zeros((241657, 106), dtype=torch.float32)
        Y_train = torch.zeros(241657, dtype=torch.float32)
        X_val = torch.zeros((30272, 106), dtype=torch.float32)
        Y_val = torch.zeros(30272, dtype=torch.float32)

    main(dryrun=args.dryrun)