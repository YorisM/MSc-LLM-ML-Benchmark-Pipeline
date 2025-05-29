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
from sklearn.metrics import roc_auc_score
import numpy as np
import sys
import argparse

# Assume tensors are already imported in the environment
# X_train, Y_train, X_val, Y_val
# Shapes:
# X_train: [241657, 106], torch.float32
# Y_train: [241657], torch.float32
# X_val: [30272, 106], torch.float32
# Y_val: [30272], torch.float32

def get_data():
    # Assuming tensors are preloaded from global scope as provided
    global X_train, Y_train, X_val, Y_val
    return X_train, Y_train, X_val, Y_val

class FourTopClassifier(nn.Module):
    def __init__(self, input_dim=106):
        super(FourTopClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, X_train, Y_train, X_val, Y_val, epochs=15, batch_size=2048, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4096, shuffle=False)

    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = []
            val_targets = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb).squeeze(1).cpu()
                val_logits.append(logits)
                val_targets.append(yb)

            val_logits = torch.cat(val_logits)
            val_targets = torch.cat(val_targets)
            probs = torch.sigmoid(val_logits).numpy()
            val_auc = roc_auc_score(val_targets.numpy(), probs)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), "four_top_classifier_model.pth")

        print(f"Epoch {epoch+1}/{epochs} - Validation AUC: {val_auc:.5f}")
    
    print(f"Final Best Validation AUC: {best_val_auc:.5f}")
    return best_val_auc

def dry_run():
    print("Performing dry run with zero-weight model:")
    model = FourTopClassifier()
    with torch.no_grad():
        x_sample = torch.zeros((1, 106))
        output = model(x_sample)
        print("Output shape:", output.shape)
        print("Output:", output)

def main():
    parser = argparse.ArgumentParser(description="Four Top Quark Classifier")
    parser.add_argument("--dryrun", action="store_true", help="Perform dry run only")
    args = parser.parse_args()

    if args.dryrun:
        dry_run()
        return

    X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor = get_data()
    model = FourTopClassifier()
    auc = train_model(model, X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor)
    print(f"AUC: {auc:.6f}")

if __name__ == '__main__':
    main()