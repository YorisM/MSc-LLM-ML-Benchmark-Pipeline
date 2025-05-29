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
import os

# Define the neural network model
class Classifier(nn.Module):
    def __init__(self, num_features, dropout_rate=0.1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x


def train_model(X_train, Y_train, X_val, Y_val, learning_rate=0.001, epochs=10, batch_size=64, dropout_rate=0.1, model_filename="model.pth"):
    # Determine the number of features
    num_features = X_train.shape[1]

    # Initialize the model, optimizer, and loss function
    model = Classifier(num_features, dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Convert data to tensors if not already
    if not isinstance(X_train, torch.Tensor):  # handles numpy array, etc
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if not isinstance(Y_train, torch.Tensor):  # handles numpy array, etc
        Y_train = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)
    if not isinstance(X_val, torch.Tensor):  # handles numpy array, etc
        X_val = torch.tensor(X_val, dtype=torch.float32)
    if not isinstance(Y_val, torch.Tensor):  # handles numpy array, etc
        Y_val = torch.tensor(Y_val, dtype=torch.float32).reshape(-1, 1)

    # Training loop
    for epoch in range(epochs):
        # Mini-batch training
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, Y_val)
            Y_pred_prob = val_outputs.cpu().numpy().flatten()
            auc = roc_auc_score(Y_val.cpu().numpy(), Y_pred_prob)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val AUC: {auc:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), model_filename)
    print(f'Model saved to {model_filename}')
    return model, auc

def evaluate_model(model, X_val, Y_val):
    if not isinstance(X_val, torch.Tensor):
        X_val = torch.tensor(X_val, dtype=torch.float32)
    if not isinstance(Y_val, torch.Tensor):
        Y_val = torch.tensor(Y_val, dtype=torch.float32).reshape(-1, 1)

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        Y_pred_prob = val_outputs.cpu().numpy().flatten()
        auc = roc_auc_score(Y_val.cpu().numpy(), Y_pred_prob)
    print(f'Final AUC: {auc:.4f}')
    return auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a binary classifier.')
    parser.add_argument('--dryrun', action='store_true', help='Enable dry run mode (simulate training)')
    args = parser.parse_args()

    # Generate dummy data if dryrun is enabled or data is not provided.
    if args.dryrun:
        print("Running in dry run mode...")
        X_train = np.random.rand(241657, 106).astype(np.float32)
        Y_train = np.random.randint(0, 2, 241657).astype(np.float32)
        X_val = np.random.rand(30272, 106).astype(np.float32)
        Y_val = np.random.randint(0, 2, 30272).astype(np.float32)
        epochs = 2
        learning_rate = 0.001
        batch_size = 128
        dropout_rate = 0.2
        model_filename = "dryrun_model.pth"
    else: # Assume data is provided
        # Load your dataset here. Example using numpy.  Replace with actual data loading.
        try:
            X_train = np.load("X_train.npy") # replace "X_train.npy" with the correct path if needed
            Y_train = np.load("Y_train.npy")
            X_val = np.load("X_val.npy")
            Y_val = np.load("Y_val.npy")
            epochs = 10
            learning_rate = 0.001
            batch_size = 64
            dropout_rate = 0.1
            # Determine the filename based on the script name
            script_name = os.path.basename(__file__)
            model_filename = f"{script_name.split('.')[0]}_model.pth"
        except FileNotFoundError:
            print("Error: One or more data files not found.  Please ensure X_train.npy, Y_train.npy, X_val.npy, and Y_val.npy exist in the current directory or that the correct data loading is implemented.")
            exit()

    # Train the model
    model, final_auc = train_model(X_train, Y_train, X_val, Y_val, learning_rate, epochs, batch_size, dropout_rate, model_filename)

    # Evaluate the model
    evaluate_model(model, X_val, Y_val)