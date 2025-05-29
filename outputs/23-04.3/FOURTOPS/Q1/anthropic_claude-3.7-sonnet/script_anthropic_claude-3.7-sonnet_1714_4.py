# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import math
from typing import List, Tuple

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
        # Register necessary buffers for preprocessing
        self.register_buffer("means", kwargs.get("means", torch.zeros(1)))
        self.register_buffer("stds", kwargs.get("stds", torch.ones(1)))
        self.register_buffer("nonzero_mask", kwargs.get("nonzero_mask", torch.ones(1, dtype=torch.bool)))
        self.register_buffer("object_ids", kwargs.get("object_ids", torch.zeros(1)))
        self.register_buffer("valid_mask", kwargs.get("valid_mask", torch.ones(1, dtype=torch.bool)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply preprocessing transformations
        # 1. Extract header features (weight, E_T_miss, phi_miss)
        header_features = x[:, :3]
        
        # 2. Extract physics objects and their properties
        object_data = x[:, 3:]
        
        # 3. Apply normalization using stored statistics
        normalized_x = torch.zeros_like(x)
        normalized_x[:, self.valid_mask] = (x[:, self.valid_mask] - self.means) / (self.stds + 1e-8)
        
        # 4. Filter out zero-padded data using the nonzero mask
        filtered_x = normalized_x[:, self.nonzero_mask]
        
        return filtered_x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=256):
    # Split the data into header and physics objects
    # The first 3 features are weight, E_T_miss, and phi_miss
    header_features = X_train[:, :3]
    object_data = X_train[:, 3:]
    
    # Identify features that have non-zero variance (filter out padding)
    feature_variance = torch.var(X_train, dim=0)
    nonzero_mask = feature_variance > 1e-8
    
    # Identify valid features (avoid constant features)
    valid_mask = (feature_variance > 1e-8) & ~torch.isnan(torch.mean(X_train, dim=0))
    
    # Calculate normalization parameters for valid features
    means = torch.mean(X_train[:, valid_mask], dim=0)
    stds = torch.std(X_train[:, valid_mask], dim=0)
    
    # Create object IDs for the physics objects
    # Every 5 values represent a new object: (object_id, E, p_T, eta, phi)
    n_features = X_train.shape[1]
    n_objects = (n_features - 3) // 5  # Excluding header features
    object_ids = torch.zeros(n_features - 3)
    
    for i in range(n_objects):
        start_idx = i * 5
        object_ids[start_idx:start_idx + 5] = i
    
    # Instantiate the preprocessing module with our derived statistics
    preproc = PreprocessModule(
        means=means,
        stds=stds,
        nonzero_mask=nonzero_mask,
        object_ids=object_ids,
        valid_mask=valid_mask
    )
    
    # Apply preprocessing
    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)
    
    # Create pytorch datasets and dataloaders
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        # Transpose for MultiheadAttention which expects [seq_len, batch_size, embed_dim]
        x = x.transpose(0, 1)
        attn_output, _ = self.attention(x, x, x)
        # Transpose back
        attn_output = attn_output.transpose(0, 1)
        return self.norm(x.transpose(0, 1) + attn_output)

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        
        # Feature extraction layers
        self.embed_dim = 128
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Self-attention mechanism to capture relationships between physics objects
        self.attention = SelfAttention(self.embed_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Initial feature extraction
        x = self.feature_extractor(x)
        
        # Add a sequence dimension for attention (treating each sample as a sequence of length 1)
        x = x.unsqueeze(1)
        
        # Apply self-attention
        x = self.attention(x)
        
        # Flatten and classify
        x = x.squeeze(1)
        x = self.classifier(x)
        
        return x.squeeze(-1)  # Remove last dimension for BCE loss

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Binary Cross Entropy Loss for binary classification
    criterion = nn.BCELoss()
    
    # Adam optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler to reduce LR when training plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Metrics tracking
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_val_auc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_predictions = []
        train_targets = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            
            # Calculate loss
            loss = criterion(y_pred, y_batch.float())
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Store batch loss and predictions
            train_losses.append(loss.item())
            train_predictions.extend(y_pred.detach().cpu().numpy())
            train_targets.extend(y_batch.cpu().numpy())
            
        # Calculate training metrics for this epoch
        epoch_train_loss = np.mean(train_losses)
        train_predictions = np.array(train_predictions)
        train_targets = np.array(train_targets)
        train_acc = accuracy_score(train_targets, train_predictions > 0.5)
        train_auc = roc_auc_score(train_targets, train_predictions)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                y_pred = model(X_batch)
                
                # Calculate loss
                val_loss = criterion(y_pred, y_batch.float())
                
                # Store batch loss and predictions
                val_losses.append(val_loss.item())
                val_predictions.extend(y_pred.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
        
        # Calculate validation metrics for this epoch
        epoch_val_loss = np.mean(val_losses)
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        val_acc = accuracy_score(val_targets, val_predictions > 0.5)
        val_auc = roc_auc_score(val_targets, val_predictions)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Store metrics for plotting
        training_loss.append(epoch_train_loss)
        validation_loss.append(epoch_val_loss)
        training_acc.append(train_acc)
        validation_acc.append(val_acc)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            print(f"New best model with validation AUC: {val_auc:.4f}")
    
    # Restore best model before returning
    if best_val_auc > 0:
        model.load_state_dict(best_model_state)
    
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
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val)

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