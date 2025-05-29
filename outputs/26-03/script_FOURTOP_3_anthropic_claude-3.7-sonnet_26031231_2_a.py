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
import numpy as np
import math
from sklearn.metrics import roc_auc_score

# Define constants
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 30
DROPOUT = 0.1
EMBED_DIM = 128
NHEAD = 8
DIM_FEEDFORWARD = 512
NLAYERS = 4

# Physics-informed feature engineering
class PhysicsFeatureExtractor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
    def forward(self, x):
        # Extract features
        batch_size = x.shape[0]
        
        # The first three elements are weight, E_T_miss, phi_{E_t}_miss
        weights = x[:, 0].unsqueeze(1)  # Event weights
        et_miss = x[:, 1].unsqueeze(1)   # Missing transverse energy
        phi_et_miss = x[:, 2].unsqueeze(1)  # Phi of missing ET
        
        # The rest are objects with their properties
        # Each object has 5 values: obj_id, E, pT, eta, phi
        object_features = x[:, 3:]
        object_features = object_features.reshape(batch_size, -1, 5)
        
        # Create mask for padding (0 values)
        mask = (object_features[:, :, 0] != 0).float()
        
        # Basic features
        object_E = object_features[:, :, 1]
        object_pT = object_features[:, :, 2]
        object_eta = object_features[:, :, 3]
        object_phi = object_features[:, :, 4]
        
        # Physics-informed features
        # 1. Calculate transverse mass (mT) for each object
        m_T = torch.sqrt(object_E**2 - object_pT**2 + 1e-8)  # add epsilon to avoid NaN
        
        # 2. Calculate delta phi between objects and missing ET
        delta_phi_miss = torch.abs(object_phi.unsqueeze(-1) - phi_et_miss.unsqueeze(1))
        # Correct for phi periodicity
        delta_phi_miss = torch.min(delta_phi_miss, 2*math.pi - delta_phi_miss)
        
        # 3. Calculate pairwise features between all objects (Lorentz invariants)
        num_objects = object_features.shape[1]
        
        # Initialize tensors to hold pairwise features
        delta_R = torch.zeros((batch_size, num_objects, num_objects), device=x.device)
        invariant_mass = torch.zeros((batch_size, num_objects, num_objects), device=x.device)
        
        for i in range(num_objects):
            for j in range(i+1, num_objects):
                # Calculate delta eta
                delta_eta = object_eta[:, i] - object_eta[:, j]
                
                # Calculate delta phi (accounting for periodicity)
                delta_phi = torch.abs(object_phi[:, i] - object_phi[:, j])
                delta_phi = torch.min(delta_phi, 2*math.pi - delta_phi)
                
                # Calculate delta R = sqrt(delta_eta^2 + delta_phi^2)
                dR = torch.sqrt(delta_eta**2 + delta_phi**2 + 1e-8)
                delta_R[:, i, j] = dR
                delta_R[:, j, i] = dR  # Symmetric
                
                # Calculate invariant mass between particle pairs (E1+E2)^2 - (p1+p2)^2
                # For simplicity using a rough approximation
                px1 = object_pT[:, i] * torch.cos(object_phi[:, i])
                py1 = object_pT[:, i] * torch.sin(object_phi[:, i])
                pz1 = object_pT[:, i] * torch.sinh(object_eta[:, i])
                
                px2 = object_pT[:, j] * torch.cos(object_phi[:, j])
                py2 = object_pT[:, j] * torch.sin(object_phi[:, j])
                pz2 = object_pT[:, j] * torch.sinh(object_eta[:, j])
                
                E1 = object_E[:, i]
                E2 = object_E[:, j]
                
                m_squared = (E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2
                m_squared = torch.clamp(m_squared, min=0)  # Avoid negative values due to numerical issues
                inv_mass = torch.sqrt(m_squared + 1e-8)
                
                invariant_mass[:, i, j] = inv_mass
                invariant_mass[:, j, i] = inv_mass  # Symmetric
                
        # Combine features for each object
        # We'll create object-level features
        object_level_features = torch.cat([
            object_E.unsqueeze(-1),               # Energy
            object_pT.unsqueeze(-1),              # Transverse momentum
            object_eta.unsqueeze(-1),             # Pseudorapidity
            object_phi.unsqueeze(-1),             # Azimuthal angle
            m_T.unsqueeze(-1),                    # Transverse mass
            delta_phi_miss.squeeze(-1).unsqueeze(-1)  # Delta phi with missing ET
        ], dim=-1)
        
        # Add global event features
        global_features = torch.cat([
            et_miss,                  # Missing transverse energy
            phi_et_miss                # Phi of missing ET
        ], dim=1)
        
        # Add pairwise features - we'll flatten these as global features
        # For each object, we'll sum its interactions with other objects
        pairwise_sum_dR = torch.sum(delta_R * mask.unsqueeze(2), dim=2)
        pairwise_sum_mass = torch.sum(invariant_mass * mask.unsqueeze(2), dim=2)
        
        pairwise_features = torch.cat([
            pairwise_sum_dR.unsqueeze(-1),
            pairwise_sum_mass.unsqueeze(-1)
        ], dim=-1)
        
        combined_obj_features = torch.cat([object_level_features, pairwise_features], dim=-1)
        
        return combined_obj_features, global_features, mask, weights

# Multi-head self-attention with physics knowledge
class ParticleTransformer(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super().__init__()
        self.input_dim = input_dim
        
        # Physics feature extractor
        self.feature_extractor = PhysicsFeatureExtractor(feature_dim=input_dim)
        
        # Feature dimensions after extraction
        self.obj_feature_dim = 8  # 6 object + 2 pairwise features
        self.global_feature_dim = 2  # ET_miss and phi
        
        # Embedding layers
        self.obj_embedding = nn.Linear(self.obj_feature_dim, EMBED_DIM)
        self.global_embedding = nn.Linear(self.global_feature_dim, EMBED_DIM)
        self.pos_encoder = PositionalEncoding(EMBED_DIM, dropout=DROPOUT)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, 
            nhead=NHEAD, 
            dim_feedforward=DIM_FEEDFORWARD, 
            dropout=DROPOUT, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, NLAYERS)
        
        # Output layers
        self.global_pooling = GlobalAttentionPooling(EMBED_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM + self.global_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        obj_features, global_features, mask, weights = self.feature_extractor(x)
        
        # Embed object features
        obj_embedding = self.obj_embedding(obj_features)
        
        # Add positional encoding
        obj_embedding = self.pos_encoder(obj_embedding)
        
        # Create attention mask from padding mask (1 = ignore, 0 = attend)
        attn_mask = (1 - mask).bool()
        
        # Apply transformer
        transformer_out = self.transformer_encoder(obj_embedding, src_key_padding_mask=attn_mask)
        
        # Global pooling with attention
        pooled = self.global_pooling(transformer_out, mask)
        
        # Concatenate with global features
        global_feat_embedded = global_features
        combined = torch.cat([pooled, global_feat_embedded], dim=1)
        
        # Final classification
        output = self.classifier(combined)
        
        return output, weights

# Positional encoding for transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

# Global attention pooling
class GlobalAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, mask):
        # x: [batch_size, seq_len, hidden_dim]
        # mask: [batch_size, seq_len]
        attention_weights = self.attention(x).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask (set padding attention to -inf)
        attention_weights = attention_weights.masked_fill(mask.eq(0), -1e9)
        attention_weights = torch.softmax(attention_weights, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # Apply attention weights
        pooled = torch.bmm(attention_weights, x).squeeze(1)  # [batch_size, hidden_dim]
        return pooled

# Weighted loss function
class WeightedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets, weights):
        bce_loss = nn.BCELoss(reduction='none')(predictions, targets)
        weighted_loss = (bce_loss * weights).mean()
        return weighted_loss

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_auc = 0.0
    best_model = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs, weights = model(inputs)
            loss = criterion(outputs, targets.float().unsqueeze(1), weights)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_targets.extend(targets.detach().cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_auc = roc_auc_score(train_targets, train_predictions)
        
        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs, _ = model(inputs)
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_auc = roc_auc_score(val_targets, val_predictions)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')
        
        # Save the best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict().copy()
    
    print(f'Best validation AUC: {best_auc:.4f}')
    return best_model, best_auc

# Create dataloaders
def create_dataloaders(X_train, Y_train, X_val, Y_val, batch_size):
    # Convert to PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Main execution
def main(X_train, Y_train, X_val, Y_val):
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(X_train, Y_train, X_val, Y_val, batch_size=BATCH_SIZE)
    
    # Determine input dimensionality
    input_dim = X_train.shape[1]
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParticleTransformer(input_dim=input_dim).to(device)
    
    # Define loss and optimizer
    criterion = WeightedBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    
    # Train model
    best_model_state, best_auc = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS)
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    model.eval()
    
    # Final evaluation on validation set
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs, _ = model(inputs)
            val_predictions.extend(outputs.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())
    
    final_auc = roc_auc_score(val_targets, val_predictions)
    print(f'Final AUC on validation set: {final_auc:.4f}')
    
    return model, final_auc

# Example of how to run
# model, auc = main(X_train, Y_train, X_val, Y_val)