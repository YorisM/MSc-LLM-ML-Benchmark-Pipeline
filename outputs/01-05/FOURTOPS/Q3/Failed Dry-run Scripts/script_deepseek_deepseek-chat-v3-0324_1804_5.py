import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F
import math

class PreprocessModule(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.register_buffer("max_objects", torch.tensor(kwargs["max_objects"]))
        self.register_buffer("energy_mean", torch.tensor(kwargs["energy_mean"]))
        self.register_buffer("energy_std", torch.tensor(kwargs["energy_std"]))
        self.register_buffer("pt_mean", torch.tensor(kwargs["pt_mean"]))
        self.register_buffer("pt_std", torch.tensor(kwargs["pt_std"]))
        self.register_buffer("eta_mean", torch.tensor(kwargs["eta_mean"]))
        self.register_buffer("eta_std", torch.tensor(kwargs["eta_std"]))
        self.register_buffer("phi_mean", torch.tensor(kwargs["phi_mean"]))
        self.register_buffer("phi_std", torch.tensor(kwargs["phi_std"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, -1, 7)  # Assuming 7 features per object
        
        # Normalize kinematic features
        x_reshaped[..., 1] = (x_reshaped[..., 1] - self.energy_mean) / self.energy_std
        x_reshaped[..., 2] = (x_reshaped[..., 2] - self.pt_mean) / self.pt_std
        x_reshaped[..., 3] = (x_reshaped[..., 3] - self.eta_mean) / self.eta_std
        x_reshaped[..., 4] = (x_reshaped[..., 4] - self.phi_mean) / self.phi_std
        
        return x_reshaped.view(batch_size, -1)

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=32):
    # Compute statistics from training data
    max_objects = X_train.shape[1] // 7
    
    # Reshape and compute statistics
    X_train_reshaped = X_train.view(-1, max_objects, 7)
    
    energy_mean = X_train_reshaped[..., 1].mean()
    energy_std = X_train_reshaped[..., 1].std()
    
    pt_mean = X_train_reshaped[..., 2].mean()
    pt_std = X_train_reshaped[..., 2].std()
    
    eta_mean = X_train_reshaped[..., 3].mean()
    eta_std = X_train_reshaped[..., 3].std()
    
    phi_mean = X_train_reshaped[..., 4].mean()
    phi_std = X_train_reshaped[..., 4].std()
    
    preproc = PreprocessModule(
        max_objects=max_objects,
        energy_mean=energy_mean,
        energy_std=energy_std,
        pt_mean=pt_mean,
        pt_std=pt_std,
        eta_mean=eta_mean,
        eta_std=eta_std,
        phi_mean=phi_mean,
        phi_std=phi_std
    )
    
    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)
    
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, preproc

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output

class SlotAttention(nn.Module):
    def __init__(self, num_slots, embed_dim, num_iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.embed_dim = embed_dim
        self.num_iters = num_iters
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Slot initialization (learnable parameters)
        self.slots = nn.Parameter(torch.randn(1, num_slots, embed_dim))
        
    def forward(self, inputs):
        batch_size = inputs.size(0)
        slots = self.slots.expand(batch_size, -1, -1)
        
        for _ in range(self.num_iters):
            # Compute attention scores
            q = self.layer_norm(slots)
            k = self.layer_norm(inputs)
            
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim)
            attn_probs = F.softmax(attn_scores, dim=-1)
            
            # Update slots
            updates = torch.matmul(attn_probs, inputs)
            slots = slots + self.mlp(self.layer_norm(updates))
        
        return slots

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PhysicsClassifier(nn.Module):
    def __init__(self, input_dim, num_slots=4, embed_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        # Feature extraction
        self.object_embedding = nn.Linear(7, embed_dim)
        self.slot_attention = SlotAttention(num_slots, embed_dim)
        
        # Transformer backbone
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        
        # Physics-aware classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_slots * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        max_objects = x.size(1) // 7
        
        # Reshape and embed object features
        x = x.view(batch_size, max_objects, 7)
        x = self.object_embedding(x)
        
        # Apply slot attention to group particles
        slots = self.slot_attention(x)
        
        # Process slots with transformer
        for transformer in self.transformer_blocks:
            slots = transformer(slots)
        
        # Classify based on slot representations
        slots = slots.view(batch_size, -1)
        output = self.classifier(slots)
        
        return output.squeeze()

def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    
    best_val_auc = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        
        train_loss = epoch_train_loss / total_train
        train_acc = correct_train / total_train
        training_loss.append(train_loss)
        training_acc.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels.float()).item() * inputs.size(0)
                
                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
                all_outputs.append(outputs)
                all_labels.append(labels)
        
        val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        validation_loss.append(val_loss)
        validation_acc.append(val_acc)
        
        # Calculate AUC
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        val_auc = roc_auc_score(all_labels.cpu().numpy(), torch.sigmoid(all_outputs).cpu().numpy())
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        print("-" * 50)
    
    return model, training_loss, validation_loss, training_acc, validation_acc

def main(dryrun=False):
    # Data Loading
    X_train, Y_train, X_val, Y_val = load_data()
    
    # Preprocessing
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=32)
    
    # Model Initialization
    sample_X, _ = next(iter(train_loader))
    model = PhysicsClassifier(input_dim=sample_X.shape[1])
    
    # Training
    epochs = 1 if dryrun else 10
    
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)
    
    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.makedirs(script_dir, exist_ok=True)
        
        model_path = os.path.join(script_dir, f"{base}_model.pth")
        torch.save(trained_model.state_dict(), model_path)
        
        scripted_path = os.path.join(script_dir, f"{base}_scripted.pt")
        torch.jit.script(trained_model).save(scripted_path)
        
        scripted_preproc = torch.jit.script(preproc)
        scripted_preproc.save(os.path.join(script_dir, f"{base}_preproc.pt"))
        
        plot_and_save(training_loss, validation_loss, f"Loss - {base}", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, f"Accuracy - {base}", os.path.join(script_dir, f"{base}_accuracy.png"))

if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)