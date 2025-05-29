import numpy as np
import pandas as pd
import math
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import roc_auc_score


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


def preprocess_data(X_train, Y_train, X_val, Y_val):
    train_weights = X_train[:, 0]
    val_weights = X_val[:, 0]

    X_train = X_train[:, 1:]
    X_val = X_val[:, 1:]

    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_dataset = TensorDataset(X_train, Y_train, train_weights)
    val_dataset = TensorDataset(X_val, Y_val, val_weights)

    train_sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=4096, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=4096)
    
    return train_loader, val_loader


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x).squeeze()


def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    training_loss = []
    validation_loss = []
    training_auc = []
    validation_auc = []
    
    best_val_auc = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        y_true_train = []
        y_pred_train = []
        
        for x, y, w in train_loader:
            x, y, w = x.to(device), y.to(device).float(), w.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = (criterion(outputs, y) * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            y_true_train.extend(y.cpu().detach().numpy().tolist())
            y_pred_train.extend(outputs.cpu().detach().numpy().tolist())
        
        training_loss.append(epoch_loss / len(train_loader))
        train_auc = roc_auc_score(y_true_train, y_pred_train)
        training_auc.append(train_auc)
        
        model.eval()
        val_loss = 0
        y_true_val = []
        y_pred_val = []
        with torch.no_grad():
            for x, y, w in val_loader:
                x, y = x.to(device), y.to(device).float()
                outputs = model(x)
                loss = (criterion(outputs, y) * w.to(device)).mean()
                val_loss += loss.item()
                y_true_val.extend(y.cpu().numpy().tolist())
                y_pred_val.extend(outputs.cpu().numpy().tolist())
        
        validation_loss.append(val_loss / len(val_loader))
        val_auc = roc_auc_score(y_true_val, y_pred_val)
        validation_auc.append(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}: '
              f'Train Loss: {training_loss[-1]:.4f}, Val Loss: {validation_loss[-1]:.4f}, '
              f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')
    
    model.load_state_dict(torch.load('best_model.pth'))
    return model, training_loss, validation_loss, training_auc, validation_auc


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


def main(dryrun=False):
    # Data Loading
    X_train, Y_train, X_val, Y_val = load_data()

    # Preprocessing
    train_loader, val_loader = preprocess_data(X_train, Y_train, X_val, Y_val)

    # Model Initialization
    input_dim = X_train.shape[1] - 1
    model = Classifier(input_dim)

    # Training
    epochs = 1 if dryrun else 30

    # Train the model
    trained_model, training_loss, validation_loss, training_auc, validation_auc = train_model(
        model, train_loader, val_loader, epochs=epochs)

    if not dryrun:
        # Save Model
        model_filename = sys.argv[0].replace(".py", "") + "_model.pth"
        torch.save(trained_model.state_dict(), model_filename)

        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, "Loss", "training_loss.png")
        plot_and_save(training_auc, validation_auc, "AUC", "training_auc.png")

        print("Full run complete. Outputs and model saved successfully.")
    else:
        print("Dry-run complete. No outputs saved.")


if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)