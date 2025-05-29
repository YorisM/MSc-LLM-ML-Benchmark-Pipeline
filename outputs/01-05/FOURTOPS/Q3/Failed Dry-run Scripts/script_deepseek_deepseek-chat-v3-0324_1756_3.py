import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

# Data Loading
X_train, Y_train, X_val, Y_val = map(lambda x: torch.tensor(pd.read_csv(f'./challenges/FOURTOPS/data/{x}.csv').values, dtype=torch.float32 if 'X' in x else torch.long), ['X_train', 'Y_train', 'X_val', 'Y_val'])
Y_train = Y_train.squeeze()
Y_val = Y_val.squeeze()

# Preprocessing
class PreprocessModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.mean(X_train, dim=0))
        self.register_buffer('std', torch.std(X_train, dim=0, unbiased=False) + 1e-8)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

# Slot Attention Transformer
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_logsigma)
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        self.gru = nn.GRU(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, inputs, mask=None):
        b, n, d = inputs.shape
        slots = self.slots_mu.expand(b, self.num_slots, -1) + torch.exp(self.slots_logsigma).expand(b, self.num_slots, -1) * torch.randn(b, self.num_slots, d, device=inputs.device)
        
        inputs = self.to_v(inputs)
        k, v = self.to_k(inputs), inputs
        
        for _ in range(self.iters):
            slots_prev = slots
            
            q = self.to_q(slots)
            
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            if mask is not None:
                dots.masked_fill_(mask, -float('inf'))
            
            attn = dots.softmax(dim=1) + self.eps
            updates = torch.einsum('bjd,bij->bid', v, attn)
            
            slots, _ = self.gru(updates.reshape(1, b*self.num_slots, d), slots_prev.reshape(1, b*self.num_slots, d))
            slots = slots.reshape(b, self.num_slots, d)
            slots = slots + self.mlp(slots)
            
        return slots

# Augmented features including mass and delta R
class FeatureAugmentation(nn.Module):
    def forward(self, x):
        # Assuming x is [batch, seq_len, 6] where 6 is [obj_type, E, pT, eta, phi, weight]
        # Augment with mass (assuming mass can be computed from E and pT)
        mass = torch.sqrt((x[..., 1]**2 - x[..., 2]**2).clip(min=0))
        return torch.cat([x, mass.unsqueeze(-1)], dim=-1)

# Model
class Classifier(nn.Module):
    def __init__(self, input_dim, num_slots=4, slot_dim=64, hidden_dim=128):
        super().__init__()
        self.feature_aug = FeatureAugmentation()
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=slot_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for mass
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim * num_slots, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # Reshape input [batch, flat_features] to [batch, particles, features]
        x = x.view(x.size(0), -1, 6)  # Assuming 6 features per particle
        
        # Augment features
        x = self.feature_aug(x)
        
        # Encode
        x = self.encoder(x)
        
        # Slot Attention
        slots = self.slot_attention(x)
        
        # Classify
        slots = slots.flatten(1)
        return torch.sigmoid(self.decoder(slots)).squeeze(-1)

# Training
def train_model(model, train_loader, val_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    model.train()
    
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y.float())
            loss.backward()
            optimizer.step()
            
        # Validate
        model.eval()
        with torch.no_grad():
            val_preds, val_trues = [], []
            for x, y in val_loader:
                val_preds.append(model(x))
                val_trues.append(y)
            val_preds = torch.cat(val_preds)
            val_trues = torch.cat(val_trues)
            val_auc = roc_auc_score(val_trues.cpu().numpy(), val_preds.cpu().numpy())
            print(f'Epoch {epoch}, Validation AUC: {val_auc:.4f}')
        model.train()
    
    return model, [], [], [], []

# Main
def main(dryrun=False):
    # Preprocessing
    preproc = PreprocessModule()
    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)
    
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    # Model
    model = Classifier(input_dim=6)  # Assuming 6 features per particle
    
    # Train
    model, *_ = train_model(model, train_loader, val_loader, epochs=1 if dryrun else 10)
    
if __name__ == '__main__':
    main(dryrun='--dryrun' in sys.argv)