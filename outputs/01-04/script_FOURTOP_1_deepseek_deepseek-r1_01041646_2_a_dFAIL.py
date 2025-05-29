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
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse

class TTBarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_emb = nn.Sequential(
            nn.Linear(5, 64),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128)
        )
        self.global_emb = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(64),
            nn.Linear(64, 64)
        )
        self.combined = nn.Sequential(
            nn.Linear(192, 256),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        global_features = x[:, :2]
        objects = x[:, 2:2+100].view(batch_size, 20, 5)
        mask = (objects[:, :, 0] != 0).float().unsqueeze(-1)
        
        obj_flat = objects.view(-1, 5)
        obj_emb = self.obj_emb(obj_flat).view(batch_size, 20, -1)
        obj_emb = obj_emb * mask
        obj_sum = obj_emb.sum(dim=1)
        
        global_emb = self.global_emb(global_features)
        combined = torch.cat([obj_sum, global_emb], dim=1)
        return self.combined(combined).squeeze()

def main(args):
        # Dummy data for illustration (assumes real data is loaded)
    X_train, Y_train = torch.randn(241657, 106), torch.randint(0,2,(241657,))
    X_val, Y_val = torch.randn(30272, 106), torch.randint(0,2,(30272,))
    
    if args.dryrun:
        X_train, Y_train = X_train[:1000], Y_train[:1000]
        X_val, Y_val = X_val[:500], Y_val[:500]
    
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size)
    
    model = TTBarClassifier()
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=2)
    best_auc = 0
    
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            x, y = batch
            x_processed = x[:, 1:].float()
            weights = x[:, 0].float()
            y = y.float()
            opt.zero_grad()
            pred = model(x_processed)
            loss = F.binary_cross_entropy_with_logits(pred, y, weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        
        model.eval()
        y_true, y_pred, val_weights = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x_processed = x[:, 1:].float()
                val_weights.extend(x[:,0].cpu().numpy())
                y_true.extend(y.cpu().numpy())
                logits = model(x_processed).cpu()
                y_pred.extend(torch.sigmoid(logits).numpy())
        
        auc = roc_auc_score(y_true, y_pred, sample_weight=val_weights)
        print(f'Epoch {epoch+1}, Val AUC: {auc:.4f}')
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'ttbar_model.pth')
        scheduler.step(auc)
    
    print(f'Final AUC: {best_auc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dryrun', action='store_true')
    args = parser.parse_args()
    args.epochs = 2 if args.dryrun else 10
    args.batch_size = 32 if args.dryrun else 512
    main(args)