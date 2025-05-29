import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
torch.manual_seed(42)

# Fixed Data Loading
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

# Enhanced preprocessing with masked normalization
class PreprocessModule(torch.nn.Module):
    def __init__(self, feature_means, feature_stds):
        super().__init__()
        self.register_buffer('feature_means', feature_means)
        self.register_buffer('feature_stds', feature_stds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x[:, 1:]  # Exclude weights
        mask = (features != 0).float()  # Filter padding
        normed = (features - self.feature_means) / torch.clamp(self.feature_stds, min=1e-6)
        return normed * mask  # Apply normalization with padding mask


def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=2048):
    # Split weights temporarily for feature stats calculation
    train_features = X_train[:, 1:]
    
    # Calculate stats ignoring padding zeros
    non_zero_mask = (train_features != 0)
    feature_means = torch.sum(train_features * non_zero_mask, dim=0) / torch.clamp(non_zero_mask.sum(dim=0), min=1)
    feature_stds = torch.sqrt(torch.sum(((train_features - feature_means) * non_zero_mask)**2, dim=0) / torch.clamp(non_zero_mask.sum(dim=0)-1, min=1)))  

    preproc = PreprocessModule(feature_means, feature_stds)

    return (
        DataLoader(TensorDataset(preproc(X_train), Y_train), batch_size=batch_size, shuffle=True, pin_memory=True),
        DataLoader(TensorDataset(preproc(X_val), Y_val), batch_size=batch_size*2, pin_memory=True),
        preproc
    )

# Attention-based classifier
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        return self.attention(x, x, x, need_weights=False)[0]

class Classifier(nn.Module):
    def __init__(self, input_dim=105):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        
        # Attention processing (sequential and global features)
        self.attn_1 = MultiHeadSelfAttention(input_dim, 8)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
        # Classification head
        self.final = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.norm(x)
        attn_out = self.attn_1(x.unsqueeze(1)).squeeze(1)
        x = x + attn_out
        x = x + self.ffn(x)
        return self.final(x).squeeze()


# Optimized training with weighted BCE loss and early stopping
def train_model(model, train_loader, val_loader, epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    best_auc = 0
    
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            loss = nn.functional.binary_cross_entropy_with_logits(outputs, batch_Y.float())
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * batch_Y.size(0)
            pred = (torch.sigmoid(outputs) > 0.5).int()
            correct += (pred == batch_Y).sum().item()
            total += batch_Y.size(0)
            
            all_preds.append(torch.sigmoid(outputs).detach().cpu())
            all_labels.append(batch_Y.cpu())
        
        train_loss.append(epoch_loss / total)
        train_acc.append(correct / total)
        auc_train = roc_auc_score(torch.cat(all_labels), torch.cat(all_preds))

        # Validation phase
        model.eval()
        val_epoch_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                outputs = model(batch_X)
                loss = nn.functional.binary_cross_entropy_with_logits(outputs, batch_Y.float())
                
                val_epoch_loss += loss.item() * batch_Y.size(0)
                pred = (torch.sigmoid(outputs) > 0.5).int()
                val_correct += (pred == batch_Y).sum().item()
                val_total += batch_Y.size(0)
                
                val_preds.append(torch.sigmoid(outputs).cpu())
                val_labels.append(batch_Y.cpu())
        
        val_loss.append(val_epoch_loss / val_total)
        val_acc.append(val_correct / val_total)
        auc_val = roc_auc_score(torch.cat(val_labels), torch.cat(val_preds))
        scheduler.step(auc_val)

        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss[-1]:.4f} Val Loss: {val_loss[-1]:.4f} | '
              f'AUC Train/Val: {auc_train:.4f}/{auc_val:.4f}')
        
        # Save best model
        if auc_val > best_auc:
            best_auc = auc_val
            best_model = model.state_dict()
        
    model.load_state_dict(best_model)
    return model, train_loss, val_loss, train_acc, val_acc


# Modified main function
def main(dryrun=False):
    X_train, Y_train, X_val, Y_val = load_data()
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=2048 if not dryrun else 512)
    
    model = Classifier(input_dim=X_train.shape[1]-1)
    epochs = 2 if dryrun else 35
    
    trained_model, training_loss, val_loss, train_acc, val_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)
    
    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].replace('script_', '')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(script_dir, exist_ok=True)
        
        torch.save(trained_model.state_dict(), os.path.join(script_dir, f'{base}_model.pth'))
        torch.jit.script(trained_model).save(os.path.join(script_dir, f'{base}_scripted.pt'))
        torch.jit.script(preproc).save(os.path.join(script_dir, f'{base}_preproc.pt'))
        
        plot_and_save(training_loss, val_loss, f'Loss - {base}', os.path.join(script_dir, f'{base}_loss.png'))
        plot_and_save(train_acc, val_acc, f'Accuracy - {base}', os.path.join(script_dir, f'{base}_accuracy.png'))

if __name__ == '__main__':
    main(dryrun='--dryrun' in sys.argv)