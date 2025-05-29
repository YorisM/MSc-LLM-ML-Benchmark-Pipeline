import pandas as pd
import torch

X_train_df = pd.read_csv('./data/fourtops/X_train.csv')
Y_train_df  = pd.read_csv('./data/fourtops/Y_train.csv')
X_val_df   = pd.read_csv('./data/fourtops/X_val.csv')
Y_val_df   = pd.read_csv('./data/fourtops/Y_val.csv')

X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()

print(f"X_train shape: {X_train.shape}\nY_train shape: {Y_train.shape}\nX_val shape: {X_val.shape}\nY_val shape: {Y_val.shape}")

# LLM Reponse Below
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import argparse
import os

# Define model architecture
class ParticleClassifier(nn.Module):
    def __init__(self, input_dim=106):
        super(ParticleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.output(x))
        return x

# Train function
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu'):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze().cpu()
            all_preds.extend(outputs.numpy())
            all_labels.extend(labels.numpy())
    auc = roc_auc_score(all_labels, all_preds)
    return model, auc

# Main execution block
def main(dryrun=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Simulated placeholders that would be removed except for --dryrun
    if dryrun:
        X_train = torch.rand((1000, 106), dtype=torch.float32)
        Y_train = torch.randint(0, 2, (1000,), dtype=torch.float32)
        X_val = torch.rand((500, 106), dtype=torch.float32)
        Y_val = torch.randint(0, 2, (500,), dtype=torch.float32)
    else:
        from experiment_data import X_train, Y_train, X_val, Y_val  # Assuming this is defined externally

    # Create datasets
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # Model initialization
    model = ParticleClassifier(input_dim=106).to(device)
    model, auc = train_model(model, train_loader, val_loader, epochs=15, lr=0.001, device=device)

    # Save model
    filename = os.path.splitext(os.path.basename(__file__))[0]
    model_path = f"{filename}_model.pth"
    torch.save(model.state_dict(), model_path)

    print(f"Final AUC score: {auc:.5f}")

# Argument parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dryrun', action='store_true', help='Run a dry run with fake data')
    args = parser.parse_args()
    main(dryrun=args.dryrun)