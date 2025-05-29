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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import argparse
import numpy as np

# Define the model architecture
class ParticleClassifier(nn.Module):
    def __init__(self, emb_dim=8, n_objs_embed=100):
        super().__init__()
        self.embedding = nn.Embedding(n_objs_embed, emb_dim)
        self.obj_mlp = nn.Sequential(
            nn.Linear(emb_dim + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.combined_mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        globals = x[:, :3]
        objs = x[:,3:3+100].view(-1, 20, 5)

        obj_ids = objs[:, :, 0].long()
        mask = obj_ids != 0
        kin_features = objs[:, :, 1:]

        embs = self.embedding(obj_ids)
        combined = torch.cat([embs, kin_features], dim=-1)
        processed = self.obj_mlp(combined)

        processed_masked = processed * mask.unsqueeze(-1)
        obj_sum = processed_masked.sum(dim=1)
        valid_counts = mask.sum(dim=1).clamp(min=1)
        obj_avg = obj_sum / valid_counts.unsqueeze(-1)

        global_out = self.global_mlp(globals)
        total = torch.cat([global_out, obj_avg], dim=1)
        return self.combined_mlp(total)

# Training and evaluation functions
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    best_auc = 0.0
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true, y_score = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                outputs = model(x).squeeze().cpu()
                y_true.extend(y.numpy())
                y_score.extend(torch.sigmoid(outputs).numpy())
        auc = roc_auc_score(y_true, y_score)
        print(f'Epoch {epoch+1}, AUC: {auc:.4f}')
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'particle_model.pth')
    print(f'Final Validation AUC: {best_auc:.4f}')
    return best_auc

# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dryrun', action='store_true')
    args = parser.parse_args()

    # Assuming X_train, Y_train, X_val, Y_val are loaded as described
    # Dummy data for testing dry run
    if args.dryrun:
        X_train = torch.randn(1000, 106)
        Y_train = torch.randint(0, 2, (1000,)).float()
        X_val = torch.randn(200, 106)
        Y_val = torch.randint(0, 2, (200,)).float()
    else:
        # In real usage, load actual tensors here
        pass

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    batch_size = 128 if not args.dryrun else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParticleClassifier()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10 if not args.dryrun else 2

    best_auc = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)
    print(f'Best AUC: {best_auc}')