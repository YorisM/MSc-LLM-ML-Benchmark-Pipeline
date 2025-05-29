import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# ----- FIXED SECTION: Data Loading -----
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

# ----- FREE SECTION: Data Preprocessing -----
def preprocess_data(X_train, Y_train, X_val, Y_val):
    scaler = StandardScaler()
    X_train_np = X_train.numpy()
    X_val_np = X_val.numpy()
    X_train_np = scaler.fit_transform(X_train_np)
    X_val_np = scaler.transform(X_val_np)
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    return X_train, Y_train, X_val, Y_val

# ----- FREE SECTION: Binary Classifier Definition -----
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
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x).squeeze()

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, X_train, Y_train, X_val, Y_val, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 128
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    training_loss, validation_loss = [], []
    training_auc, validation_auc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        y_true_train, y_pred_train = [], []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float()

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)
            y_true_train.extend(y_batch.detach().cpu().numpy())
            y_pred_train.extend(outputs.detach().cpu().numpy())

        avg_loss = epoch_loss / len(train_loader.dataset)
        training_loss.append(avg_loss)
        train_auc = roc_auc_score(y_true_train, y_pred_train)
        training_auc.append(train_auc)

        model.eval()
        val_loss = 0
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).float()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * x_batch.size(0)
                y_true_val.extend(y_batch.cpu().numpy())
                y_pred_val.extend(outputs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        validation_loss.append(avg_val_loss)
        val_auc = roc_auc_score(y_true_val, y_pred_val)
        validation_auc.append(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

    return model, training_loss, validation_loss, training_auc, validation_auc

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

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)