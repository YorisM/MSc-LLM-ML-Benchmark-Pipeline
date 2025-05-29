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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import time
import os

# Parse arguments
parser = argparse.ArgumentParser(description='Train a classifier for particle physics events')
parser.add_argument('--dryrun', action='store_true', help='Run a quick test with fewer iterations')
args = parser.parse_args()

class EventDataset(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.X[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.y is not None:
            return sample, self.y[idx]
        else:
            return sample

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(FeatureExtractor, self).__init__()
        
        # Process the first three features (weight, E_T_miss, phi_{E_t}_miss)
        self.global_features = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Process each object's features (5 features per object: obj_n, E_n, p_Tn, eta_n, phi_n)
        self.object_features = nn.Sequential(
            nn.Linear(4, 64),  # We'll process each object's E, pT, eta, phi (ignore obj_n)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Attention mechanism
        self.query = nn.Linear(64, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.attention_scale = np.sqrt(64)
        
        # Combine features
        self.combiner = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.feature_dim = 64
    
    def forward(self, x):
        # Extract global features (first 3)
        global_feats = self.global_features(x[:, :3])
        
        # Reshape and process object features
        # Skip every 5th value (obj_n) and reshape to [batch, objects, 4]
        # Starting from index 3 to skip the globals
        object_data = []
        for i in range(4, x.shape[1], 5):
            if i+3 < x.shape[1]:  # Ensure we have full object data
                # Take E, pT, eta, phi for each object
                object_data.append(x[:, i:i+4])
        
        if not object_data:  # Handle case with no objects
            return self.combiner(torch.cat([global_feats, torch.zeros_like(global_feats)], dim=1))
        
        # Stack all objects
        object_tensor = torch.stack(object_data, dim=1)
        batch_size, num_objects, obj_features = object_tensor.shape
        
        # Create a mask for zero-padded objects (objects with all zeros)
        mask = (object_tensor.sum(dim=2) != 0).float().unsqueeze(2)  # [batch, objects, 1]
        
        # Apply object feature extractor to each object
        object_feats_flat = object_tensor.reshape(-1, obj_features)
        processed_objects = self.object_features(object_feats_flat)
        processed_objects = processed_objects.reshape(batch_size, num_objects, -1)
        
        # Apply mask to zero out padding
        processed_objects = processed_objects * mask
        
        # Self-attention mechanism
        q = self.query(processed_objects)  # [batch, objects, 64]
        k = self.key(processed_objects)  # [batch, objects, 64]
        v = self.value(processed_objects)  # [batch, objects, 64]
        
        # Calculate attention scores
        attention = torch.matmul(q, k.transpose(1, 2)) / self.attention_scale  # [batch, objects, objects]
        
        # Apply mask to attention scores
        mask_expanded = mask.squeeze(2).unsqueeze(1)  # [batch, 1, objects]
        attention = attention * mask_expanded
        attention = torch.softmax(attention, dim=2)  # [batch, objects, objects]
        
        # Weighted sum of values
        object_context = torch.matmul(attention, v)  # [batch, objects, 64]
        
        # Combine with original features
        object_feats = processed_objects + object_context
        
        # Aggregate across objects (with attention to importance)
        object_weights = torch.softmax(object_feats.sum(dim=2), dim=1).unsqueeze(2)  # [batch, objects, 1]
        weighted_objects = object_feats * object_weights * mask
        aggregated_objects = weighted_objects.sum(dim=1)  # [batch, 64]
        
        # Combine global and object features
        combined = torch.cat([global_feats, aggregated_objects], dim=1)
        features = self.combiner(combined)
        
        return features

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out.squeeze()

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, scheduler=None, dryrun=False):
    best_auc = 0.0
    best_model = None
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    
    if dryrun:
        epochs = 2  # Reduced epochs for dry run
    
    for epoch in range(epochs):
        start_time = time.time()
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds = []
        train_true = []
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            train_preds.extend(outputs.detach().cpu().numpy())
            train_true.extend(labels.cpu().numpy())
            
            if dryrun and len(train_preds) > 1000:  # Process only a subset in dry run
                break
        
        train_loss = running_loss / (len(train_true) if dryrun else len(train_loader.dataset))
        train_auc = roc_auc_score(train_true, train_preds)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                val_preds.extend(outputs.detach().cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                
                if dryrun and len(val_preds) > 1000:  # Process only a subset in dry run
                    break
        
        val_loss = running_loss / (len(val_true) if dryrun else len(val_loader.dataset))
        val_auc = roc_auc_score(val_true, val_preds)
        
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | '
              f'Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}')
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()
    
    # Load best model
    model.load_state_dict(best_model)
    
    return model, history, best_auc

def plot_history(history):
    plt.figure(figsize=(14, 6))
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    # Plot training & validation AUC values
    plt.subplot(1, 2, 2)
    plt.plot(history['train_auc'], label='Train')
    plt.plot(history['val_auc'], label='Validation')
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def normalize_data(X_train, X_val):
    # Create masks for non-zero elements
    train_mask = (X_train != 0)
    
    # Calculate mean and std for non-zero elements (per feature)
    train_mean = np.zeros(X_train.shape[1])
    train_std = np.ones(X_train.shape[1])
    
    # First 3 features: weight, E_T_miss, phi_{E_t}_miss
    for i in range(3):
        non_zero_vals = X_train[:, i][X_train[:, i] != 0]
        if len(non_zero_vals) > 0:
            train_mean[i] = non_zero_vals.mean()
            train_std[i] = non_zero_vals.std() if non_zero_vals.std() > 0 else 1.0
    
    # For each object feature type (E, pT, eta, phi)
    for offset in range(4, 8):  # E, pT, eta, phi
        feat_positions = range(offset, X_train.shape[1], 5)  # Skip obj_n
        for pos in feat_positions:
            if pos < X_train.shape[1]:
                non_zero_vals = X_train[:, pos][X_train[:, pos] != 0]
                if len(non_zero_vals) > 0:
                    train_mean[pos] = non_zero_vals.mean()
                    train_std[pos] = non_zero_vals.std() if non_zero_vals.std() > 0 else 1.0
    
    # Normalize, preserving zeros
    X_train_norm = X_train.copy()
    X_val_norm = X_val.copy()
    
    for i in range(X_train.shape[1]):
        X_train_norm[:, i] = np.where(train_mask[:, i], (X_train[:, i] - train_mean[i]) / train_std[i], 0)
        X_val_norm[:, i] = np.where(X_val[:, i] != 0, (X_val[:, i] - train_mean[i]) / train_std[i], 0)
    
    return X_train_norm, X_val_norm

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # For this script, let's generate synthetic data to simulate the expected format
    # In a real scenario, you would load your data from files
    try:
        # Try to load the data - we assume it's already available as described in the problem
        print("Looking for existing data tensors...")
        X_train = torch.load('X_train.pt') if os.path.exists('X_train.pt') else None
        Y_train = torch.load('Y_train.pt') if os.path.exists('Y_train.pt') else None
        X_val = torch.load('X_val.pt') if os.path.exists('X_val.pt') else None
        Y_val = torch.load('Y_val.pt') if os.path.exists('Y_val.pt') else None
        
        if None in [X_train, Y_train, X_val, Y_val]:
            raise FileNotFoundError("One or more required data files not found")
            
        print(f"Data loaded successfully!")
        print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic data for demonstration...")
        
        # Create synthetic data for demonstration
        X_train = torch.randn(1000, 106)
        Y_train = torch.randint(0, 2, (1000,), dtype=torch.float32)
        X_val = torch.randn(200, 106)
        Y_val = torch.randint(0, 2, (200,), dtype=torch.float32)
        
        print("Synthetic data created for demonstration purposes.")
        print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
    
    # Convert to numpy for preprocessing
    X_train_np = X_train.numpy()
    X_val_np = X_val.numpy()
    
    # Normalize the data
    X_train_norm, X_val_norm = normalize_data(X_train_np, X_val_np)
    
    # Convert back to torch tensors
    X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    Y_train_tensor = Y_train.float()
    X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32)
    Y_val_tensor = Y_val.float()
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    
    batch_size = 64 if not args.dryrun else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = Classifier(input_dim).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Train model
    epochs = 15 if not args.dryrun else 2
    model, history, best_auc = train_model(
        model, train_loader, val_loader, criterion, optimizer, epochs, device, scheduler, args.dryrun
    )
    
    # Plot training history
    plot_history(history)
    
    # Save the model
    model_filename = os.path.basename(__file__).replace('.py', '_model.pth')
    torch.save(model.state_dict(), model_filename)
    print(f"\nModel saved as {model_filename}")
    
    # Final evaluation on validation set
    model.eval()
    val_preds = []
    val_true = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            val_preds.extend(outputs.cpu().numpy())
            val_true.extend(labels.numpy())
    
    final_auc = roc_auc_score(val_true, val_preds)
    print(f"\nFinal Validation AUC: {final_auc:.6f}")

if __name__ == "__main__":
    main()