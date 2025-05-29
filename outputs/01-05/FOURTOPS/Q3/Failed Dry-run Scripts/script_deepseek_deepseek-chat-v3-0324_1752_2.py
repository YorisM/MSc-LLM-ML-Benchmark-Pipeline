import os, sys, torch
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=256):
    preproc = PreprocessModule()
    
    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)
    
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, preproc

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
        
        self.gru = nn.GRUCell(dim, dim)
        
    def forward(self, inputs):
        b, n, d = inputs.shape
        
        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(b, self.num_slots, -1)
        slots = torch.randn_like(mu) * sigma + mu
        
        inputs = self.to_k(inputs)
        
        for _ in range(self.iters):
            slots_prev = slots
            
            q = self.to_q(slots)
            dots = torch.einsum('bid,bjd->bij', q, inputs) * self.scale
            attn = torch.softmax(dots, dim=1)
            
            updates = torch.einsum('bjd,bij->bid', self.to_v(inputs), attn)
            updates = updates / (attn.sum(dim=-1, keepdim=True) + self.eps)
            
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
        
        return slots

class PhysicsAugmentation(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.obj_embedding = nn.Embedding(100, hidden_dim)
        
    @staticmethod
    def calc_delta_phi(phi1, phi2):
        dphi = phi1 - phi2
        dphi = torch.where(dphi > math.pi, dphi - 2*math.pi, dphi)
        dphi = torch.where(dphi <= -math.pi, dphi + 2*math.pi, dphi)
        return dphi
    
    @staticmethod
    def calc_delta_r(eta1, phi1, eta2, phi2):
        deta = eta1 - eta2
        dphi = PhysicsAugmentation.calc_delta_phi(phi1, phi2)
        return torch.sqrt(deta**2 + dphi**2)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features (assuming format is consistent)
        et_miss = x[:, 0]
        phi_et_miss = x[:, 1]
        
        # Extract object features (groups of 5 columns starting at index 2)
        obj_features = x[:, 2:].reshape(batch_size, -1, 5)
        
        # Get object identifiers (assuming first column is obj_id)
        obj_ids = obj_features[:, :, 0].long()
        
        # Kinematic features
        energies = obj_features[:, :, 1]
        pt = obj_features[:, :, 2]
        eta = obj_features[:, :, 3]
        phi = obj_features[:, :, 4]
        
        # Calculate pairwise features between objects
        dr_matrix = torch.zeros(batch_size, obj_ids.size(1), obj_ids.size(1), device=x.device)
        for i in range(obj_ids.size(1)):
            for j in range(obj_ids.size(1)):
                dr_matrix[:, i, j] = self.calc_delta_r(eta[:, i], phi[:, i], eta[:, j], phi[:, j])
        
        # Augment features with physics-inspired combinations
        obj_pt_sum = pt.sum(dim=1)
        obj_dr_min = dr_matrix.view(batch_size, -1).min(dim=1)[0]
        
        # Combine features
        base_features = torch.stack([energies.sum(dim=1), 
                                   et_miss, 
                                   obj_pt_sum, 
                                   obj_dr_min], dim=1)
        
        # Object embeddings
        obj_emb = self.obj_embedding(obj_ids)
        
        # Concatenate all features
        combined = torch.cat([base_features.unsqueeze(1).expand(-1, obj_ids.size(1), -1), 
                            obj_emb, 
                            pt.unsqueeze(-1), 
                            eta.unsqueeze(-1), 
                            phi.unsqueeze(-1)], dim=-1)
        
        return combined

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_slots=6):
        super().__init__()
        self.physics_aug = PhysicsAugmentation()
        
        # Calculate post-augmentation dimension
        aug_dim = 64 + 5 + 4  # embedding + kinematic + global features
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=aug_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=aug_dim, 
            nhead=4, 
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.mlp = nn.Sequential(
            nn.Linear(aug_dim * num_slots, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # Physics-based feature augmentation
        x_aug = self.physics_aug(x)
        
        # Slot attention to group relevant particles
        slots = self.slot_attention(x_aug)
        
        # Transformer processing
        encoded = self.encoder(slots)
        
        # Global pooling via concatenation
        pooled = encoded.view(encoded.size(0), -1)
        
        # Final classification
        out = self.mlp(pooled)
        return torch.sigmoid(out)

def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    
    training_loss, validation_loss = [], []
    training_acc, validation_acc = [], []
    best_auc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        all_train_preds, all_train_labels = [], []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).long()
            train_correct += (preds == labels.long()).sum().item()
            train_total += labels.size(0)
            
            all_train_preds.extend(outputs.detach().cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        
        train_auc = roc_auc_score(all_train_labels, all_train_preds)
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_val_preds, all_val_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
                val_total += labels.size(0)
                
                all_val_preds.extend(outputs.detach().cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        val_auc = roc_auc_score(all_val_labels, all_val_preds)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step(val_auc)
        
        # Save metrics
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        training_acc.append(train_acc)
        validation_acc.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model, training_loss, validation_loss, training_acc, validation_acc

def main(dryrun=False):
    # Data Loading
    X_train, Y_train, X_val, Y_val = load_data()
    
    # Preprocessing
    batch_size = 128 if dryrun else 256
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=batch_size)
    
    # Model Initialization
    sample_X, _ = next(iter(train_loader))
    model = TransformerClassifier(input_dim=sample_X.shape[1], num_slots=6)
    
    # Training
    epochs = 1 if dryrun else 30
    
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