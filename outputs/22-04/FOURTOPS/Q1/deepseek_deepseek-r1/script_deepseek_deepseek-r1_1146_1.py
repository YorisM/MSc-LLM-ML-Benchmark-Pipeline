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
# ---------------------

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
    weights_train = X_train[:, 0]
    features_train = X_train[:, 1:]
    weights_val = X_val[:, 0]
    features_val = X_val[:, 1:]
    
    mean = torch.mean(features_train, dim=0)
    std = torch.std(features_train, dim=0)
    std[std == 0] = 1.0
    
    features_train = (features_train - mean) / std
    features_val = (features_val - mean) / std
    
    train_dataset = TensorDataset(features_train, Y_train, weights_train)
    val_dataset = TensorDataset(features_val, Y_val, weights_val)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, pin_memory=True)
    
    return train_loader, val_loader

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.layers(x).squeeze()

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    training_loss, validation_loss = [], []
    training_acc, validation_acc = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for features, labels, weights in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = (criterion(outputs, labels.float()) * weights).mean()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= train_total
        training_loss.append(train_loss)
        training_acc.append(train_correct / train_total)
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels, weights in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                weights = weights.to(device)
                
                outputs = model(features)
                loss = (criterion(outputs, labels.float()) * weights).mean()
                
                val_loss += loss.item() * features.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).int()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        validation_loss.append(val_loss)
        validation_acc.append(val_correct / val_total)
        
        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_correct/train_total:.4f} | Val Loss: {val_loss:.4f} Acc: {val_correct/val_total:.4f}')
    
    return model, training_loss, validation_loss, training_acc, validation_acc

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
    X_train, Y_train, X_val, Y_val = load_data()
    train_loader, val_loader = preprocess_data(X_train, Y_train, X_val, Y_val)
    model = Classifier(input_dim=X_train.shape[1]-1)
    
    epochs = 1 if dryrun else 10
    trained_model, tloss, vloss, tacc, vacc = train_model(model, train_loader, val_loader, epochs)
    
    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        model_filename = f"{base}_model.pth"
        torch.save(trained_model.state_dict(), model_filename)
        scripted = torch.jit.script(trained_model)
        scripted.save(f"{base}_scripted.pt")
        plot_and_save(tloss, vloss, "Loss", f"{base}_loss.png")
        plot_and_save(tacc, vacc, "Accuracy", f"{base}_accuracy.png")

if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)