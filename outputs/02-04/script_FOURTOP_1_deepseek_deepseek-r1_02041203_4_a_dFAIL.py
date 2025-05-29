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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

# Define the classifier model
class ParticleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_embedding = nn.Embedding(1000, 16)  # Assuming obj ids <1000
        
        self.kin_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        self.attention = nn.Linear(16 + 64, 1)
        
        self.global_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        
        self.combined_mlp = nn.Sequential(
            nn.Linear(16 + 64 + 32, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        global_feats = x[:, :3]
        
        # Process objects
        obj_feats = x[:, 3:103].view(batch_size, 20, 5)
        obj_ids = obj_feats[:, :, 0].long()
        mask = (obj_ids != 0).float()
        
        # Embed object types
        obj_emb = self.obj_embedding(obj_ids)
        
        # Process kinematic features
        kin_in = obj_feats[:, :, 1:5].view(-1,4)
        kin_out = self.kin_mlp(kin_in).view(batch_size, 20, -1)
        
        # Combine obj_emb and kin
        obj_combined = torch.cat([obj_emb, kin_out], dim=2)
        
        # Attention
        attn_scores = self.attention(obj_combined).squeeze(2)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        aggregated = (obj_combined * attn_weights.unsqueeze(2)).sum(1)
        
        # Process global features
        global_processed = self.global_mlp(global_feats)
        
        # Combine
        combined = torch.cat([aggregated, global_processed], dim=1)
        logits = self.combined_mlp(combined).squeeze()
        return logits

def main(args):
    # Dummy data for demonstration (assuming data is loaded)
    # In practice, use provided X_train, Y_train etc.
    # X_train, Y_train = ...
    
    # Create loaders
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512)
    
    # Initialize the model
    model = ParticleClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    # Training loop
    epochs = 1 if args.dryrun else 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs).cpu().sigmoid()
                val_preds.append(outputs)
                val_true.append(targets)
        val_preds = torch.cat(val_preds)
        val_true = torch.cat(val_true)
        auc = roc_auc_score(val_true.numpy(), val_preds.numpy())
        
        print(f'Epoch {epoch+1} | Loss: {total_loss/len(train_loader.dataset):.4f} | Val AUC: {auc:.4f}')
        scheduler.step(auc)
    
    # Save model
    torch.save(model.state_dict(), 'particle_model.pth')
    print(f'Final Val AUC: {auc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dryrun', action='store_true')
    args = parser.parse_args()
    main(args)