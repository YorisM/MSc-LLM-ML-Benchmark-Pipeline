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

# ----- FIXED SECTION: Data Loading -----
def load_data():
    X_train_df = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv')
    Y_train_df = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv')
    X_val_df   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv')
    Y_val_df   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv')

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
    X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
    Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()
    return X_train, Y_train, X_val, Y_val

# ----- FREE SECTION: Data Preprocessing -----
class PreprocessModule(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Register mean and std for normalization
        self.register_buffer('mean', kwargs.get('mean', torch.zeros(1)))
        self.register_buffer('std', kwargs.get('std', torch.ones(1)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize the data
        x = (x - self.mean) / (self.std + 1e-8)
        return x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=128):
    # Compute mean and std for normalization
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    
    preproc = PreprocessModule(mean=mean, std=std)
    
    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)
    
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    
    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        
        # Initialize slots randomly
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_log_sigma)
        
        # Projections for Q, K, V
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        # GRU for slot updates
        self.gru = nn.GRUCell(dim, dim)
        
    def forward(self, inputs):
        b, n, d = inputs.shape
        
        # Initialize slots
        slots_init = torch.randn((b, self.num_slots, d), device=inputs.device)
        slots_init = slots_init * torch.exp(self.slots_log_sigma) + self.slots_mu
        slots = slots_init
        
        # Project inputs to keys and values
        k = self.to_k(inputs)  # (b, n, d)
        v = self.to_v(inputs)  # (b, n, d)
        
        # Iterative attention
        for _ in range(self.iters):
            slots_prev = slots
            
            # Project slots to queries
            q = self.to_q(slots)  # (b, num_slots, d)
            
            # Dot product attention
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            
            # Weighted sum of values
            updates = torch.einsum('bjd,bij->bid', v, attn)
            
            # Update slots with GRU
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            ).reshape(b, self.num_slots, d)
            
        return slots

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_slots=4, hidden_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        
        # Feature augmentation
        self.augment = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Slot attention
        self.slot_attention = SlotAttention(num_slots, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_slots, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # Reshape input to (batch_size, num_particles, features)
        # Assuming the first two features are E_T_miss and phi_E_T_miss
        # and the rest are particle features in groups of 4 (E, pT, eta, phi)
        batch_size = x.size(0)
        num_particles = (x.size(1) - 2) // 4
        
        # Extract particle features
        particles = x[:, 2:].reshape(batch_size, num_particles, 4)
        
        # Augment features
        particles_aug = self.augment(particles)
        
        # Apply slot attention
        slots = self.slot_attention(particles_aug)
        
        # Process slots with transformer
        slots = slots.permute(1, 0, 2)  # (num_slots, batch_size, hidden_dim)
        slots = self.transformer(slots)
        slots = slots.permute(1, 0, 2)  # (batch_size, num_slots, hidden_dim)
        
        # Classify
        out = self.classifier(slots.reshape(batch_size, -1))
        return torch.sigmoid(out.squeeze(-1))

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    
    best_auc = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
            
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            preds = (outputs > 0.5).long()
            correct += (preds == batch_y.long()).sum().item()
            total += batch_x.size(0)
        
        train_loss = epoch_loss / total
        train_acc = correct / total
        training_loss.append(train_loss)
        training_acc.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                preds = (outputs > 0.5).long()
                correct += (preds == batch_y.long()).sum().item()
                total += batch_x.size(0)
                
                all_outputs.append(outputs.cpu())
                all_labels.append(batch_y.cpu())
        
        val_loss = val_loss / total
        val_acc = correct / total
        validation_loss.append(val_loss)
        validation_acc.append(val_acc)
        
        # Calculate AUC
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        auc = roc_auc_score(all_labels.numpy(), all_outputs.numpy())
        
        scheduler.step(auc)
        
        if auc > best_auc:
            best_auc = auc
            
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | AUC: {auc:.4f}')
    
    return model, training_loss, validation_loss, training_acc, validation_acc

# ----- FIXED SECTION: Plotting and Saving Outputs -----
def plot_and_save(metric_train, metric_val, metric_name, filename):
    plt.figure()
    plt.plot(metric_train, label=f'Training {metric_name}')
    plt.plot(metric_val, label=f'Validation {metric_name}')
    plt.title(f'{metric_name} per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# ----- FIXED SECTION: Main Function -----
def main(dryrun=False):
    # Data Loading
    X_train, Y_train, X_val, Y_val = load_data()
    
    # Preprocessing
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=128)
    
    # Model Initialization
    sample_X, _ = next(iter(train_loader))
    model = TransformerClassifier(input_dim=4)  # Each particle has 4 features (E, pT, eta, phi)
    
    # Training
    epochs = 1 if dryrun else 10
    
    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)
    
    if not dryrun:
        # determine base name & script directory
        base       = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix('script_')
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.makedirs(script_dir, exist_ok=True)
        
        # save model
        model_path = os.path.join(script_dir, f'{base}_model.pth')
        torch.save(trained_model.state_dict(), model_path)
        
        # save scripted model
        scripted_path = os.path.join(script_dir, f'{base}_scripted.pt')
        torch.jit.script(trained_model).save(scripted_path)
        
        # save preprocessor
        scripted_preproc = torch.jit.script(preproc)
        scripted_preproc.save(os.path.join(script_dir, f'{base}_preproc.pt'))
        
        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, f'Loss - {base}', os.path.join(script_dir, f'{base}_loss.png'))
        plot_and_save(training_acc, validation_acc, f'Accuracy - {base}', os.path.join(script_dir, f'{base}_accuracy.png'))

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)