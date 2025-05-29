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

class ParticleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 8)
        self.obj_net = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32)
        )
        self.combined_net = nn.Sequential(
            nn.Linear(35, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        global_feat = x[:, :3]
        objects = x[:, 3:103].view(-1, 20, 5)
        obj_ids = objects[:, :, 0].long()
        mask = obj_ids != 0

        embedded = self.embed(obj_ids)
        kinematic = objects[:, :, 1:]
        combined = torch.cat([embedded, kinematic], dim=2)
        processed = self.obj_net(combined)

        masked_features = processed * mask.unsqueeze(-1)
        aggregated = masked_features.sum(dim=1)
        out = torch.cat([aggregated, global_feat], dim=1)
        return torch.sigmoid(self.combined_net(out))

def train_model(X_train, Y_train, X_val, Y_val, dryrun):
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    batch_size = 256 if not dryrun else 32

    model = ParticleClassifier()
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    for epoch in range(10 if not dryrun else 2):
        model.train()
        for X, y in train_loader:
            opt.zero_grad()
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            opt.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X).squeeze()
                all_preds.append(pred.cpu())
                all_labels.append(y.cpu())
        auc = roc_auc_score(torch.cat(all_labels), torch.cat(all_preds))
        print(f"Epoch {epoch+1}: Validation AUC = {auc:.4f}")

    torch.save(model.state_dict(), 'particle_model.pth')
    return auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dryrun', action='store_true')
    args = parser.parse_args()

    if args.dryrun:
        X_train = X_train[:1000]
        Y_train = Y_train[:1000]
        X_val = X_val[:100]
        Y_val = Y_val[:100]

    final_auc = train_model(X_train, Y_train, X_val, Y_val, args.dryrun)
    print(f"Final AUC Score: {final_auc:.4f}")