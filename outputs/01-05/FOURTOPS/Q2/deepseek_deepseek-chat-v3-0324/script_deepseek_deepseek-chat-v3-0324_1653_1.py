import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

# Data Loading
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

# Preprocessing
class PreprocessModule(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.register_buffer("mean", kwargs["mean"])
        self.register_buffer("std", kwargs["std"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=128):
    # Compute mean and std for normalization
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    std[std == 0] = 1.0  # Avoid division by zero
    
    preproc = PreprocessModule(mean=mean, std=std)
    
    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)
    
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    
    return train_loader, val_loader, preproc

# Lorentz Equivariant Network
class LorentzNet(nn.Module):
    def __init__(self, input_dim):
        super(LorentzNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# Training Loop
def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_Y.float())
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            predicted = (outputs > 0.5).long()
            correct += (predicted == batch_Y).sum().item()
            total += batch_Y.size(0)
        
        training_loss.append(epoch_loss / len(train_loader))
        training_acc.append(correct / total)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_outputs = []
        val_labels = []
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_Y.float())
                val_loss += loss.item()
                predicted = (outputs > 0.5).long()
                val_correct += (predicted == batch_Y).sum().item()
                val_total += batch_Y.size(0)
                val_outputs.extend(outputs.cpu().numpy())
                val_labels.extend(batch_Y.cpu().numpy())
        
        validation_loss.append(val_loss / len(val_loader))
        validation_acc.append(val_correct / val_total)
        auc = roc_auc_score(val_labels, val_outputs)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {training_loss[-1]:.4f}, Val Loss: {validation_loss[-1]:.4f}, Acc: {training_acc[-1]:.4f}, Val Acc: {validation_acc[-1]:.4f}, AUC: {auc:.4f}')
    
    return model, training_loss, validation_loss, training_acc, validation_acc

# Main Function
def main(dryrun=False):
    # Data Loading
    X_train, Y_train, X_val, Y_val = load_data()
    
    # Preprocessing
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=128)
    
    # Model Initialization
    sample_X, _ = next(iter(train_loader))
    model = LorentzNet(input_dim=sample_X.shape[1])
    
    # Training
    epochs = 1 if dryrun else 10
    
    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)
    
    if not dryrun:
        # determine base name & script directory
        base       = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.makedirs(script_dir, exist_ok=True)
        
        # save model
        model_path = os.path.join(script_dir, f"{base}_model.pth")
        torch.save(trained_model.state_dict(), model_path)
        
        # save scripted model
        scripted_path = os.path.join(script_dir, f"{base}_scripted.pt")
        torch.jit.script(trained_model).save(scripted_path)
        
        # save preprocessor
        scripted_preproc = torch.jit.script(preproc)
        scripted_preproc.save(os.path.join(script_dir, f"{base}_preproc.pt"))
        
        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, f"Loss - {base}", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, f"Accuracy - {base}", os.path.join(script_dir, f"{base}_accuracy.png"))

# Plotting and Saving Outputs
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

# Entry Point with Dry-run
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)