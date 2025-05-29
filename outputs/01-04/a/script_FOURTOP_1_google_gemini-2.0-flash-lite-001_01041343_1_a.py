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
import argparse
import os
import numpy as np

# Define the neural network model
class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def train_model(X_train, Y_train, X_val, Y_val, learning_rate=0.001, batch_size=64, num_epochs=50, dryrun=False, filename="classifier"):
    """
    Trains a binary classifier model.

    Args:
        X_train (torch.Tensor): Training data features.
        Y_train (torch.Tensor): Training data labels.
        X_val (torch.Tensor): Validation data features.
        Y_val (torch.Tensor): Validation data labels.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        dryrun (bool): If True, performs a dry run (no training).
        filename (str): Filename for saving the model.

    Returns:
        float: AUC score on the validation set.
    """
    input_size = X_train.shape[1]
    model = Classifier(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if dryrun:
        print("Dry run enabled. No training will be performed.")
        return 0.5  # Return a default AUC score for dry run

    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size].reshape(-1, 1)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_auc = roc_auc_score(Y_val.cpu().numpy(), val_outputs.cpu().numpy())
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation AUC: {val_auc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), f"{filename}_model.pth")
    print(f"Trained model saved as {filename}_model.pth")

    return val_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a binary classifier for particle physics data.')
    parser.add_argument('--dryrun', action='store_true', help='Enable dry run mode (no training).')
    args = parser.parse_args()

    # Generate dummy data if the environment variable is not set
    if not os.environ.get("X_train") or not os.environ.get("Y_train") or not os.environ.get("X_val") or not os.environ.get("Y_val"):
        print("Generating dummy data...")
        X_train_np = np.random.rand(241657, 106).astype(np.float32)
        Y_train_np = np.random.randint(0, 2, 241657).astype(np.float32)
        X_val_np = np.random.rand(30272, 106).astype(np.float32)
        Y_val_np = np.random.randint(0, 2, 30272).astype(np.float32)

        X_train = torch.tensor(X_train_np)
        Y_train = torch.tensor(Y_train_np)
        X_val = torch.tensor(X_val_np)
        Y_val = torch.tensor(Y_val_np)


    else:
        # Assuming data is already loaded as pytorch tensors
        X_train = torch.load(os.environ.get("X_train"))
        Y_train = torch.load(os.environ.get("Y_train"))
        X_val = torch.load(os.environ.get("X_val"))
        Y_val = torch.load(os.environ.get("Y_val"))

    final_auc = train_model(X_train, Y_train, X_val, Y_val, dryrun=args.dryrun)
    print(f"Final AUC: {final_auc:.4f}")