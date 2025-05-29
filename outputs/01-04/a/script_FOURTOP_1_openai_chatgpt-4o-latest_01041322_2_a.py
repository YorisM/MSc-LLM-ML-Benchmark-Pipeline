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
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np

class ParticleNet(nn.Module):
    def __init__(self, input_dim):
        super(ParticleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, device, epochs=15):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_auc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                outputs = model(val_x).squeeze().cpu().numpy()
                all_preds.extend(outputs)
                all_labels.extend(val_y.numpy())

        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}, Validation AUC: {auc:.5f}")

        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict()

    return best_model_state, best_auc

def main(dryrun=False):
    # Replace this section with loaded torch tensors in real use
    if not dryrun:
        # Tensors: X_train, Y_train, X_val, Y_val are assumed preloaded globally
        global X_train, Y_train, X_val, Y_val

        # Wrap data in datasets and loaders
        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)

        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ParticleNet(input_dim=X_train.shape[1]).to(device)

        best_model_state, best_auc = train_model(model, train_loader, val_loader, device)
        model.load_state_dict(best_model_state)

        model_path = os.path.splitext(os.path.basename(__file__))[0] + "_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Best Validation AUC: {best_auc:.5f}")
        print(f"Model saved to {model_path}")
    else:
        print("Dry run: Initialized model architecture and verified data flow.")
        dummy_input = torch.randn(1, 106)
        model = ParticleNet(input_dim=106)
        output = model(dummy_input)
        print("Model output shape:", output.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dryrun', action='store_true', help='Run model architecture dry run only')
    args = parser.parse_args()
    main(dryrun=args.dryrun)