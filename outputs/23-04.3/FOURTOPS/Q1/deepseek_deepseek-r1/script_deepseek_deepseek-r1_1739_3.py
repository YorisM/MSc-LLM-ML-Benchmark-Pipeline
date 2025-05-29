import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

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
class PreprocessModule(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.register_buffer('means', kwargs['means'])
        self.register_buffer('stds', kwargs['stds'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = (x - self.means) / torch.clamp(self.stds, min=1e-7)
        return x_normalized


def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size):
    n_features = X_train.shape[1]
    means = []
    stds = []

    for i in range(n_features):
        if i < 3:
            feature = X_train[:, i]
            mean = torch.mean(feature)
            std = torch.std(feature, unbiased=True)
        else:
            feature = X_train[:, i]
            mask = feature != 0
            if mask.any():
                mean = torch.mean(feature[mask])
                std = torch.std(feature[mask], unbiased=True)
            else:
                mean = torch.tensor(0.0)
                std = torch.tensor(1.0)
        std = std if std > 0 else torch.tensor(1.0)
        means.append(mean)
        stds.append(std)

    means = torch.stack(means)
    stds = torch.stack(stds)

    preproc = PreprocessModule(means=means, stds=stds)

    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
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
        return self.net(x).squeeze(-1)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []

    for epoch in range(epochs):
        model.train()
        train_preds, train_targets = [], []
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            train_preds.extend(torch.sigmoid(outputs.detach().cpu()).numpy())
            train_targets.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        training_loss.append(epoch_loss)
        train_auc = roc_auc_score(train_targets, train_preds)
        training_acc.append(train_auc)

        model.eval()
        val_preds, val_targets = [], []
        running_vloss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_vloss += loss.item() * inputs.size(0)
                val_preds.extend(torch.sigmoid(outputs.cpu()).numpy())
                val_targets.extend(labels.cpu().numpy())

        vloss = running_vloss / len(val_loader.dataset)
        validation_loss.append(vloss)
        val_auc = roc_auc_score(val_targets, val_preds)
        validation_acc.append(val_auc)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} AUC: {train_auc:.4f} | Val Loss: {vloss:.4f} AUC: {val_auc:.4f}')

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
    X_train, Y_train, X_val, Y_val = load_data()

    batch_size = 512 if not dryrun else 64
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size)

    sample_X, _ = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[1])

    epochs = 1 if dryrun else 20
    trained_model, t_loss, v_loss, t_acc, v_acc = train_model(model, train_loader, val_loader, epochs)

    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].replace('script_', '')
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        
        model_path = os.path.join(script_dir, f'{base}_model.pth')
        torch.save(trained_model.state_dict(), model_path)

        scripted_model = torch.jit.script(trained_model)
        scripted_path = os.path.join(script_dir, f'{base}_scripted.pt')
        scripted_model.save(scripted_path)

        scripted_preproc = torch.jit.script(preproc)
        scripted_preproc.save(os.path.join(script_dir, f'{base}_preproc.pt'))

        plot_and_save(t_loss, v_loss, 'Loss', os.path.join(script_dir, f'{base}_loss.png'))
        plot_and_save(t_acc, v_acc, 'AUC', os.path.join(script_dir, f'{base}_accuracy.png'))

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)