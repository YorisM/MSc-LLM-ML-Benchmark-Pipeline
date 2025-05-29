# ----- FIXED SECTION: Import Libraries -----
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import sys

# ----- FIXED SECTION: Data Loading -----
def load_data():
    X_train_df = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv')
  Y  _train_df = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv')
    X_val_df   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv')
    Y_val_df   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv')

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
    X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
    Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()
    return X_train, Y_train, X_val, Y_val

# ----- FREE SECTION: Data Preprocessing -----
def preprocess_data(X_train, Y_train, X_val, Y_val):
    # No preprocessing for now
    return X_train, Y_train, X_val, Y_val

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.fc(x)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, X_train, Y_train, X_val, Y_val, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    training_auc = []
    validation_auc = []

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, dim=1)
        accuracy = (predicted == Y_train).sum().item() / len(Y_train)
        auc = roc_auc_score(Y_train.cpu().numpy(), torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())
        training_loss.append(loss.item())
        training_acc.appnd(accueracy)
        training_auc.append(auc)

        # Validation
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            loss = criterion(outputs, Y_val)
            _, predicted = torch.max(outputs, dim=1)
            accuracy = (predicted == Y_val).sum().item() / len(Y_val)
            auc = roc_auc_score(Y_val.cpu().numpy(), torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())
            validation_loss.append(loss.item())
            validation_acc.append(accuracy)
            validation_auc.append(auc)

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
    X_train, Y_train, X_val, Y_val = preprocess_data(X_train, Y_train, X_val, Y_val)

    # Model Initialization
    model = Classifier(input_dim=X_train.shape[1])

    # Training (dryrun limits epochs)
    epochs = 1 if dryrun else 20

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, X_train, Y_train, X_val, Y_val, epochs=epochs)

    # Save Model
    model_filename = sys.argv[0].replace(".py", "") + "_model.pth"
    torch.save(trained_model.state_dict(), model_filename)

    # Plot Metrics
    plot_and_save(training_loss, validation_loss, "Loss", "training_loss.png")
    plot_and_save(training_acc, validation_acc, "Accuracy", "training_accuracy.png")

    print("Training complete. Outputs and model saved successfully.")

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)