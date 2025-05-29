# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F
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
    # TorchScript-compatible module applying pre-fitted transformations.
    def __init__(self, **kwargs):
        super().__init__()
        # Register statistics for normalization
        for key in kwargs:
            self.register_buffer(key, kwargs[key])
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply mean and std normalization
        x_normalized = (x - self.mean) / (self.std + 1e-8)
        
        # Replace NaN and Inf values with zeros
        x_normalized = torch.where(torch.isnan(x_normalized), torch.zeros_like(x_normalized), x_normalized)
        x_normalized = torch.where(torch.isinf(x_normalized), torch.zeros_like(x_normalized), x_normalized)
        
        # Extract structured features
        # Missing transverse energy features (first two columns)
        et_miss = x_normalized[:, 0:2]
        
        # Rest of data is organized as objects
        rest_data = x_normalized[:, 2:]
        batch_size = rest_data.shape[0]
        
        # Reshape to extract objects (each object has 5 values: obj_id, E, pT, eta, phi)
        num_objects = (rest_data.shape[1]) // 5
        rest_data = rest_data.reshape(batch_size, num_objects, 5)
        
        # Mask to identify real objects vs padding (obj_id != 0)
        object_mask = (rest_data[:, :, 0] != 0)
        
        # Apply mask to zero out padding
        masked_data = rest_data * object_mask.unsqueeze(-1)
        
        # Compute object features
        # Energy
        total_energy = torch.sum(masked_data[:, :, 1], dim=1, keepdim=True)
        
        # Transverse momentum
        total_pt = torch.sum(masked_data[:, :, 2], dim=1, keepdim=True)
        
        # Count non-zero objects
        obj_count = torch.sum(object_mask, dim=1, keepdim=True)
        
        # Extract specific object types (simplify for script compatibility)
        # Calculate features for different object types
        type_features = []
        
        for obj_type in range(1, 6):  # Assuming object types 1-5
            # Create mask for this object type
            type_mask = (masked_data[:, :, 0] == obj_type)
            # Count objects of this type
            type_count = torch.sum(type_mask, dim=1, keepdim=True)
            # Sum energy and pt for this type
            type_energy = torch.sum(masked_data[:, :, 1] * type_mask, dim=1, keepdim=True)
            type_pt = torch.sum(masked_data[:, :, 2] * type_mask, dim=1, keepdim=True)
            # Combine features for this type
            type_features.append(type_count)
            type_features.append(type_energy)
            type_features.append(type_pt)
        
        # Flatten rest_data for the transformer encoder input
        flattened_data = masked_data.reshape(batch_size, -1)
        
        # Combine all engineered features
        engineered_features = torch.cat([et_miss, total_energy, total_pt, obj_count] + type_features, dim=1)
        
        # Combine engineered features with flattened original data
        result = torch.cat([engineered_features, flattened_data], dim=1)
        
        return result

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=256):
    # Calculate mean and std for normalization (exclude zero-padded values)
    mask = X_train != 0
    mean = torch.zeros(X_train.shape[1])
    std = torch.ones(X_train.shape[1])
    
    for i in range(X_train.shape[1]):
        col_data = X_train[:, i][mask[:, i]]
        if len(col_data) > 0:
            mean[i] = col_data.mean()
            std[i] = col_data.std() if col_data.std() > 0 else 1.0
    
    # Initialize preprocessor with calculated statistics
    preproc = PreprocessModule(mean=mean, std=std)
    
    # Apply preprocessing
    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)
    
    # Create datasets and dataloaders
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        return attn_output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = SelfAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output = self.attn(x)
        x = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        return self.layernorm2(x + self.dropout(ffn_output))
        
class Classifier(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=4, ff_dim=256, num_transformer_blocks=2, dropout=0.3):
        super(Classifier, self).__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # Project to embedding dimension for transformer
        self.projection = nn.Linear(256, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_transformer_blocks)]
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Project to embedding dimension
        embeddings = self.projection(features)
        
        # Reshape for transformer (add sequence dimension of 1)
        transformer_input = embeddings.unsqueeze(1)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            transformer_input = block(transformer_input)
        
        # Squeeze sequence dimension and apply classification head
        transformer_output = transformer_input.squeeze(1)
        logits = self.classification_head(transformer_output).squeeze(-1)
        
        return logits

# ----- FREE SECTION: Training Loop Implementation -----
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()

def train_model(model, train_loader, val_loader, epochs, learning_rate=3e-4, weight_decay=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    
    # Use a loss function that works well for imbalanced data
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Metrics tracking
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_auc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Store predictions and targets for metrics
            train_losses.append(loss.item())
            train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch metrics for training
        epoch_train_loss = np.mean(train_losses)
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        epoch_train_auc = roc_auc_score(train_targets, train_preds)
        epoch_train_acc = accuracy_score(train_targets, train_preds > 0.5)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass only (no backprop in validation)
                outputs = model(inputs)
                loss = criterion(outputs, targets.float())
                
                # Store predictions and targets for metrics
                val_losses.append(loss.item())
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch metrics for validation
        epoch_val_loss = np.mean(val_losses)
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        epoch_val_auc = roc_auc_score(val_targets, val_preds)
        epoch_val_acc = accuracy_score(val_targets, val_preds > 0.5)
        
        # Store metrics for plotting
        training_loss.append(epoch_train_loss)
        validation_loss.append(epoch_val_loss)
        training_acc.append(epoch_train_acc)
        validation_acc.append(epoch_val_acc)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {epoch_train_loss:.4f}, Train AUC: {epoch_train_auc:.4f}, Train Acc: {epoch_train_acc:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f}, Val AUC: {epoch_val_auc:.4f}, Val Acc: {epoch_val_acc:.4f}')
        
        # Save best model based on validation AUC
        if epoch_val_auc > best_auc:
            best_auc = epoch_val_auc
            best_model = model.state_dict()
            print(f'New best model saved with validation AUC: {best_auc:.4f}')
    
    # Load the best model before returning
    model.load_state_dict(best_model)
    
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
    batch_size = 256 if not dryrun else 32
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size)

    # Model Initialization
    sample_X, _ = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[1])

    # Training
    epochs = 1 if dryrun else 15

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)

    if not dryrun:
        # determine base name & script directory
        base       = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
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

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)