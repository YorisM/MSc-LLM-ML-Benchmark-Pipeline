# ----- FIXED SECTION: Import Libraries -----
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ----- FIXED SECTION: Data Loading -----
def load_data():
    X_train_df = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv')
    Y_train_df = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv')
    X_val_df   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv')
    Y_val_df   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv')

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.float32).squeeze()
    X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
    Y_val   = torch.tensor(Y_val_df.values, dtype=torch.float32).squeeze()
    return X_train, Y_train, X_val, Y_val

# ----- FREE SECTION: Data Preprocessing -----
def preprocess_data(X_train, Y_train, X_val, Y_val):
    # Scale the data to have zero mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    return X_train, Y_train, X_val, Y_val

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, X_train, Y_train, X_val, Y_val, epochs):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())
        predicted = (outputs > 0.5).float()
        train_accuracy = (predicted == Y_train).sum().item() / Y_train.size(0)
        training_acc.append(train_accuracy)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_loss = criterion(val_outputs, Y_val)
            validation_loss.append(val_loss.item())
            val_predicted = (val_outputs > 0.5).float()
            val_accuracy = (val_predicted == Y_val).sum().item() / Y_val.size(0)
            validation_acc.append(val_accuracy)

        print(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

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