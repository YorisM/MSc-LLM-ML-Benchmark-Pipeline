import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import math

class PreprocessModule(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.register_buffer("mean", kwargs["mean"])
        self.register_buffer("std", kwargs["std"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=128):
    # Compute mean and std for normalization
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    std[std == 0] = 1.0  # Avoid division by zero
    
    preproc = PreprocessModule(mean=mean, std=std)
    
    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)
    
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, preproc

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_log_sigma)
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        self.gru = nn.GRUCell(dim, dim)
        
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, inputs, mask=None):
        b, n, d = inputs.shape
        
        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(b, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            q = self.to_q(slots)
            
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            if mask is not None:
                dots.masked_fill_(~mask, -1e9)
            attn = dots.softmax(dim=1) + self.eps
            
            updates = torch.einsum('bjd,bij->bid', v, attn)
            
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            ).reshape(b, self.num_slots, d)
            
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        
        return slots

class ParticleTransformer(nn.Module):
    def __init__(self, input_dim, num_slots=4, slot_dim=64, num_heads=4, num_layers=3, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        
        # Feature augmentation
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim)
        )
        
        self.slot_attention = SlotAttention(num_slots, slot_dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=slot_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(slot_dim * num_slots, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # Reshape input to (batch_size, num_particles, particle_features)
        # Assuming input is (batch_size, 105) where 105 is flattened features
        # We need to reconstruct the particle structure
        batch_size = x.size(0)
        
        # Each particle has 4 features (E, pT, eta, phi)
        # First two features are E_T_miss and phi_E_T_miss
        # Then each particle has 4 features
        num_particles = (x.size(1) - 2) // 4
        
        # Extract E_T_miss and phi_E_T_miss
        et_miss = x[:, 0].unsqueeze(1)
        phi_et_miss = x[:, 1].unsqueeze(1)
        
        # Extract particle features
        particles = x[:, 2:].reshape(batch_size, num_particles, 4)
        
        # Create mask for zero-padded particles
        mask = (particles.sum(dim=-1) != 0).unsqueeze(1)  # (batch_size, 1, num_particles)
        
        # Augment particle features with delta phi to missing ET
        particle_phi = particles[:, :, 3]
        delta_phi = torch.abs(particle_phi - phi_et_miss)
        delta_phi = torch.minimum(delta_phi, 2 * math.pi - delta_phi)
        
        # Create augmented features: [E, pT, eta, phi, delta_phi, pT/ET_miss]
        augmented = torch.cat([
            particles,
            delta_phi.unsqueeze(-1),
            particles[:, :, 1] / (et_miss + 1e-6).unsqueeze(-1)
        ], dim=-1)
        
        # Embed augmented features
        embedded = self.embedding(augmented)
        
        # Apply slot attention to group particles
        slots = self.slot_attention(embedded, mask)
        
        # Process slots with transformer
        slots = self.transformer(slots)
        
        # Classify
        out = self.classifier(slots.reshape(batch_size, -1))
        return torch.sigmoid(out.squeeze(-1))

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.model = ParticleTransformer(input_dim)
    
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    training_loss, validation_loss = [], []
    training_acc, validation_acc = [], []
    training_auc, validation_auc = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        all_preds, all_labels = [], []
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
            
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            all_preds.append(outputs.detach().cpu())
            all_labels.append(batch_y.detach().cpu())
        
        # Training metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        train_loss = epoch_loss / len(train_loader)
        train_acc = accuracy_score(all_labels.numpy() > 0.5, all_preds.numpy() > 0.5)
        train_auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())
        
        training_loss.append(train_loss)
        training_acc.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_preds.append(outputs.cpu())
                val_labels.append(batch_y.cpu())
        
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels.numpy() > 0.5, val_preds.numpy() > 0.5)
        val_auc = roc_auc_score(val_labels.numpy(), val_preds.numpy())
        
        validation_loss.append(val_loss)
        validation_acc.append(val_acc)
        
        scheduler.step(val_auc)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
        print(f'Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}')
        print('-' * 50)
    
    return model, training_loss, validation_loss, training_acc, validation_acc

def main(dryrun=False):
    # Data Loading
    X_train, Y_train, X_val, Y_val = load_data()
    
    # Preprocessing
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=128)
    
    # Model Initialization
    sample_X, _ = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[1])
    
    # Training
    epochs = 1 if dryrun else 10
    
    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)
    
    if not dryrun:
        # determine base name & script directory
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.makedirs(script_dir, exist_ok=True)
        
        # save model
        model_path = os.path.join(script_dir, f"{base}_model.pth")
        torch.save(trained_model.state_dict(), model_path)
        
        # save scripted model
        scripted_path = os.path.join(script_dir, f"{base}_scripted.pt")
        torch.jit.script(trained_model).save(scripted_path)
        
        # save preprocessor
        scripted_preproc = torch.jit.script(preproc)
        scripted_preproc.save(os.path.join(script_dir, f"{base}_preproc.pt"))
        
        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, f"Loss - {base}", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, f"Accuracy - {base}", os.path.join(script_dir, f"{base}_accuracy.png"))

if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)