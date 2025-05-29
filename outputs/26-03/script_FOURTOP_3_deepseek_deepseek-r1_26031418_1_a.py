import pandas as pd
import torch

X_train_df = pd.read_csv('./data/fourtops/X_train.csv')
Y_train_df = pd.read_csv('./data/fourtops/Y_train.csv')
X_val_df   = pd.read_csv('./data/fourtops/X_val.csv')
Y_val_df   = pd.read_csv('./data/fourtops/Y_val.csv')

X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
Y_train = torch.tensor(Y_train_df.values, dtype=torch.long)
X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long)


#LLM Response below

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score

class ParticleTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, hidden_dim=256, num_layers=4, max_objects=20):
        super().__init__()
        self.obj_embedding = nn.Embedding(10, embed_dim, padding_idx=0)
        self.global_proj = nn.Linear(3, embed_dim)
        self.combined_proj = nn.Linear(embed_dim+4+embed_dim+4, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.max_objects = max_objects

    def forward(self, x):
        batch_size = x.shape[0]
        global_features = x[:, :3]
        objects = x[:, 3:].view(batch_size, self.max_objects, 5)
        
        obj_ids = objects[:, :, 0].long()
        obj_embeds = self.obj_embedding(obj_ids)
        
        E = objects[:, :, 1]/1e3
        pT = objects[:, :, 2]/1e3
        eta = objects[:, :, 3]
        phi = objects[:, :, 4]
        obj_features = torch.stack([E, pT, eta, phi], dim=2)
        
        mask = obj_ids != 0
        mask_ij = mask.unsqueeze(2) & mask.unsqueeze(1)
        mask_ij = mask_ij.float() * (1 - torch.eye(self.max_objects, device=mask.device).unsqueeze(0))
        
        # Pairwise features
        eta_i = eta.unsqueeze(2)
        eta_j = eta.unsqueeze(1)
        delta_eta = eta_i - eta_j
        
        phi_i = phi.unsqueeze(2)
        phi_j = phi.unsqueeze(1)
        delta_phi = (phi_i - phi_j + np.pi) % (2 * np.pi) - np.pi
        delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)
        
        p_x = pT * torch.cos(phi)
        p_y = pT * torch.sin(phi)
        p_z = pT * torch.sinh(eta)
        E_pairs = E.unsqueeze(2) + E.unsqueeze(1)
        p_x_pairs = p_x.unsqueeze(2) + p_x.unsqueeze(1)
        p_y_pairs = p_y.unsqueeze(2) + p_y.unsqueeze(1)
        p_z_pairs = p_z.unsqueeze(2) + p_z.unsqueeze(1)
        inv_mass_sq = E_pairs**2 - (p_x_pairs**2 + p_y_pairs**2 + p_z_pairs**2)
        inv_mass = torch.sqrt(torch.clamp(inv_mass_sq, min=0))
        
        sum_delta_eta = torch.sum(delta_eta * mask_ij, dim=2)
        sum_delta_phi = torch.sum(delta_phi * mask_ij, dim=2)
        sum_delta_R = torch.sum(delta_R * mask_ij, dim=2)
        sum_inv_mass = torch.sum(inv_mass * mask_ij, dim=2)
        count = torch.sum(mask_ij, dim=2) + 1e-8
        mean_delta_eta = sum_delta_eta/count
        mean_delta_phi = sum_delta_phi/count
        mean_delta_R = sum_delta_R/count
        mean_inv_mass = sum_inv_mass/count
        pairwise_features = torch.stack([mean_delta_eta, mean_delta_phi, mean_delta_R, mean_inv_mass], dim=2)
        
        global_proj = self.global_proj(global_features).unsqueeze(1).expand(-1, self.max_objects, -1)
        combined = torch.cat([obj_embeds, obj_features, global_proj, pairwise_features], dim=2)
        combined = self.combined_proj(combined)
        
        padding_mask = ~mask
        combined = combined.permute(1, 0, 2)
        output = self.transformer(combined, src_key_padding_mask=padding_mask)
        output = output.permute(1, 0, 2)
        output_masked = output * mask.unsqueeze(-1)
        mean_output = torch.sum(output_masked, dim=1)/(torch.sum(mask, dim=1, keepdim=True)+1e-8)
        logits = self.fc(mean_output).squeeze(-1)
        return logits

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ParticleTransformer(max_objects=(X_train.shape[1]-3)//5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

# Create DataLoader
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# Training loop
best_auc = 0.0
for epoch in range(50):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            val_preds.append(outputs.sigmoid().cpu())
            val_labels.append(labels.cpu())
    val_preds = torch.cat(val_preds).numpy()
    val_labels = torch.cat(val_labels).numpy()
    auc = roc_auc_score(val_labels, val_preds)
    print(f"Epoch {epoch}, Val AUC: {auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), 'best_model.pth')