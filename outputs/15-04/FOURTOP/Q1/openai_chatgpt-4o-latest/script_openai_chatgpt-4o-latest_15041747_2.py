import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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
    scaler = StandardScaler()
    X_train_np = X_train.numpy()
    X_val_np = X_val.numpy()
    X_train_np = scaler.fit_transform(X_train_np)
    X_val_np = scaler.transform(X_val_np)
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    return X_train, Y_train, X_val, Y_val

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

def train_model(model, X_train, Y_train, X_val, Y_val, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    batch_size = 1024
    train_loss_list = []
    val_loss_list = []
    auc_train_list = []
    auc_val_list = []
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0
        model_outputs = []
        model_targets = []

        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], Y_train[indices].float()

            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            model_outputs.append(torch.sigmoid(outputs).detach())
            model_targets.append(batch_y.detach())

        model_outputs = torch.cat(model_outputs).numpy()
        model_targets = torch.cat(model_targets).numpy()
        epoch_auc = roc_auc_score(model_targets, model_outputs)
        auc_train_list.append(epoch_auc)
        avg_loss = epoch_loss / X_train.size()[0]
        train_loss_list.append(avg_loss)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_loss = criterion(val_outputs, Y_val.float()).item()
            val_loss_list.append(val_loss)

            val_probs = torch.sigmoid(val_outputs).numpy()
            val_targets = Y_val.numpy()
            val_auc = roc_auc_score(val_targets, val_probs)
            auc_val_list.append(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - AUC: {epoch_auc:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f}")

    return model, train_loss_list, val_loss_list, auc_train_list, auc_val_list

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