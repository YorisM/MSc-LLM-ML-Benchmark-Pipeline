import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def load_data():
    X_train_df = pd.read_csv('./data/fourtops/X_train.csv')
    Y_train_df = pd.read_csv('./data/fourtops/Y_train.csv')
    X_val_df   = pd.read_csv('./data/fourtops/X_val.csv')
    Y_val_df   = pd.read_csv('./data/fourtops/Y_val.csv')

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
    X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
    Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()
    return X_train, Y_train, X_val, Y_val

def preprocess_data(X_train, Y_train, X_val, Y_val):
    scaler = StandardScaler()
    X_train_np = X_train.numpy()
    X_val_np   = X_val.numpy()

    scaler.fit(X_train_np)
    X_train_scaled = scaler.transform(X_train_np)
    X_val_scaled   = scaler.transform(X_val_np)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val   = torch.tensor(X_val_scaled, dtype=torch.float32)
    return X_train, Y_train, X_val, Y_val

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x).squeeze()

def train_model(model, X_train, Y_train, X_val, Y_val, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    batch_size = 1024
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    training_loss, validation_loss = [], []
    training_auc, validation_auc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        all_preds, all_targets = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
        epoch_loss /= len(train_loader.dataset)
        auc_train = roc_auc_score(all_targets, all_preds)
        training_loss.append(epoch_loss)
        training_auc.append(auc_train)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_preds, val_targets = [], []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())
            val_loss /= len(val_loader.dataset)
            auc_val = roc_auc_score(val_targets, val_preds)
            validation_loss.append(val_loss)
            validation_auc.append(auc_val)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Train AUC: {auc_train:.4f}, Val AUC: {auc_val:.4f}")
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
    X_train, Y_train, X_val, Y_val = load_data()
    X_train, Y_train, X_val, Y_val = preprocess_data(X_train, Y_train, X_val, Y_val)
    model = Classifier(input_dim=X_train.shape[1])
    epochs = 1 if dryrun else 20

    trained_model, training_loss, validation_loss, training_auc, validation_auc = train_model(
        model, X_train, Y_train, X_val, Y_val, epochs=epochs)

    model_filename = sys.argv[0].replace(".py", "") + "_model.pth"
    torch.save(trained_model.state_dict(), model_filename)

    plot_and_save(training_loss, validation_loss, "Loss", "training_loss.png")
    plot_and_save(training_auc, validation_auc, "AUC", "training_auc.png")

    print("Training complete. Outputs and model saved successfully.")

if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)