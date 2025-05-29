import numpy as np
import pandas as pd
import math
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler

# ----- FIXED SECTION: Data Loading -----
def load_data():
    X_train_df = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv')
    Y_train_df = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv')
    X_val_df   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv')
    Y_val_df   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv')

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
    X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
    Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()
    return X_train, Y_train, X_val, Y_val

# ----- FREE SECTION: Data Preprocessing -----
def preprocess_data(X_train, Y_train, X_val, Y_val):
    # Preprocess data by scaling it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.detach().numpy())
    X_val_scaled = scaler.transform(X_val.detach().numpy())
    X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_scaled = torch.tensor(X_val_scaled, dtype=torch.float32)

    # Create data loaders for training and validation
    train_dataset = TensorDataset(X_train_scaled, Y_train)
    val_dataset = TensorDataset(X_val_scaled, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    return train_loader, val_loader

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            predicted = torch.round(outputs)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)
        training_loss.append(epoch_loss / len(train_loader))
        training_acc.append(correct / total)
        model.eval()
        epoch_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                epoch_loss += loss.item()
                predicted = torch.round(outputs)
                correct += (predicted == labels.unsqueeze(1)).sum().item()
                total += labels.size(0)
        validation_loss.append(epoch_loss / len(val_loader))
        validation_acc.append(correct / total)
    return model, training_loss, validation_loss, training_acc, validation_acc

# ----- FIXED SECTION: Plotting and Saving Outputs -----
def plot_and_save(metric_train, metric_val, metric_name, filename):
    plt.figure()
    plt.plot(metric_train, label=f'Training {metric_name}')
    plt.plot(metric_val, label=f'Validation {metric_name}')
    plt.title(f'{metric_name} per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# ----- FIXED SECTION: Main Function -----
def main(dryrun=False):
    # Data Loading
    X_train, Y_train, X_val, Y_val = load_data()

    # Preprocessing
    train_loader, val_loader = preprocess_data(X_train, Y_train, X_val, Y_val)

    # Model Initialization
    model = Classifier(input_dim=X_train.shape[1])

    # Training
    epochs = 1 if dryrun else 10
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)

    if not dryrun:
        # Save Model
        model_filename = sys.argv[0].replace(".py", "") + "_model.pth"
        torch.save(trained_model.state_dict(), model_filename)

        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, "Loss", "training_loss.png")
        plot_and_save(training_acc, validation_acc, "Accuracy", "training_accuracy.png")

        print("Full run complete. Outputs and model saved successfully.")
    else:
        print("Dry-run complete. No outputs saved.")

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)