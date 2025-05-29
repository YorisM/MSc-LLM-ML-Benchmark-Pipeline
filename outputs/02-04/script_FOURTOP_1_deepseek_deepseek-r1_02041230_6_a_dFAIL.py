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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse

# Define the model
class ParticleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 8)  # Assume max 20 different particle types
        
        # Object feature processing
        self.obj_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 64)
        )
        
        # Global feature processing
        self.global_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract global features
        etmiss = x[:, 0].unsqueeze(1)
        phi_etmiss = x[:, 1].unsqueeze(1)
        objects = x[:, 2:102].view(-1, 20, 5)
        
        # Process object features
        obj_ids = objects[:, :, 0].long()
        obj_features = objects[:, :, 1:]
        
        embedding = self.embedding(obj_ids)
        encoded = self.obj_encoder(obj_features)
        combined = torch.cat([embedding, encoded], dim=-1)
        
        # Mask padded objects
        mask = (obj_ids != 0).unsqueeze(-1).float()
        combined = combined * mask
        obj_pooled = combined.sum(dim=1)
        
        # Process global features
        global_features = self.global_encoder(torch.cat([etmiss, phi_etmiss], dim=1))
        
        # Combine and classify
        combined = torch.cat([obj_pooled, global_features], dim=1)
        return self.classifier(combined).squeeze()

# Dataset class
class ParticleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        features = self.X[idx, 1:]  # Exclude weight
        weight = self.X[idx, 0].item()
        return features, self.y[idx].item(), weight

# Training function
def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = ParticleClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # Create data loaders
    train_dataset = ParticleDataset(X_train, Y_train)
    val_dataset = ParticleDataset(X_val, Y_val)
    
    batch_size = 256 if not args.dryrun else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048)

    # Training loop
    num_epochs = 20 if not args.dryrun else 2
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            features, labels, weights = batch
            features = features.float().to(device)
            labels = torch.tensor(labels).float().to(device)
            weights = torch.tensor(weights).float().to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = (F.binary_cross_entropy(outputs, labels, reduction='none') * weights).mean()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        all_weights = []
        
        with torch.no_grad():
            for batch in val_loader:
                features, labels, weights = batch
                features = features.to(device).float()
                outputs = model(features).cpu().numpy()
                
                all_preds.extend(outputs)
                all_labels.extend(labels.numpy())
                all_weights.extend(weights.numpy())
        
        auc = roc_auc_score(all_labels, all_preds, sample_weight=all_weights)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Val AUC: {auc:.4f}')
    
    # Save model
    torch.save(model.state_dict(), '4top_model.pth')
    return auc

# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dryrun', action='store_true', help='Run with reduced settings')
    args = parser.parse_args()
    
    # Dummy data for demonstration - replace with actual data tensors
    if 'X_train' not in globals():
        X_train = torch.randn(241657, 106)
        Y_train = torch.randint(0, 2, (241657,)).float()
        X_val = torch.randn(30272, 106)
        Y_val = torch.randint(0, 2, (30272,)).float()
    
    final_auc = train_model(args)
    print(f'Final AUC: {final_auc:.4f}')