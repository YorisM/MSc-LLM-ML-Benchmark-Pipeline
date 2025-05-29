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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

class ParticleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.object_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.combined = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        global_features = x[:, :3]
        objects = x[:, 3:].view(x.size(0), -1, 5)
        mask = (objects[:, :, 0] != 0.0).float()

        obj_embeds = self.object_mlp(objects)
        agg_objects = (obj_embeds * mask.unsqueeze(-1)).sum(dim=1)

        global_embeds = self.global_mlp(global_features)
        combined = torch.cat([agg_objects, global_embeds], dim=1)
        return self.combined(combined).squeeze()

def main(dryrun):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParticleClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 1 if dryrun else 10

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            weights = x[:, 0]
            loss = (nn.functional.binary_cross_entropy(preds, y, reduction='none') * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).squeeze()
                y_true.append(y.cpu())
                y_pred.append(pred.cpu())
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        auc = roc_auc_score(y_true, y_pred)
        print(f'Epoch {epoch+1}, Val AUC: {auc:.4f}')

    torch.save(model.state_dict(), 'model_model.pth')
    print(f'Final AUC: {auc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dryrun', action='store_true')
    args = parser.parse_args()
    # Assume X_train, Y_train, X_val, Y_val are preloaded
    main(args.dryrun)