# ----- FREE SECTION: Import Libraries -----
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
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

torch.manual_seed(42)
np.random.seed(42)
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
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert dataframes to tensors
    X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_scaled = torch.tensor(X_val_scaled, dtype=torch.float32)

    # Create data loaders
    train_dataset = TensorDataset(X_train_scaled, Y_train)
    val_dataset = TensorDataset(X_val_scaled, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    validation_auc = []
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            running_corrects += (predicted == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / total
        training_loss.append(epoch_loss)
        training_acc.append(epoch_acc)
        # Validation
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        predictions = []
        labels = []
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, target = data
                inputs, target = inputs.to(device), target.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, target)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                running_corrects += (predicted == target).sum().item()
                total += target.size(0)
                predictions.extend(nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                labels.extend(target.cpu().numpy())
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects / total
        validation_loss.append(epoch_loss)
        validation_acc.append(epoch_acc)
        epoch_auc = roc_auc_score(labels, predictions)
        validation_auc.append(epoch_auc)
        print(f'Epoch {epoch+1}, Training Loss: {epoch_loss}, Validation Loss: {epoch_loss}, Validation AUC: {epoch_auc}')
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

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)

    if not dryrun:
        # Save Model
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        model_filename = os.path.join(script_dir, f"{base}_model.pth")
        os.makedirs(script_dir, exist_ok=True)

        torch.save(trained_model.state_dict(), f"{base}_model.pth")

        scripted = torch.jit.script(trained_model)
        scripted_filename = os.path.join(script_dir, f"{base}_scripted.pt")
        scripted.save(scripted_filename)
        print(f"Saved TorchScript module to {scripted_filename}")

        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, f"Loss - {base}", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, f"Accuracy - {base}", os.path.join(script_dir, f"{base}_accuracy.png"))

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)