# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        
        # Register preprocessing constants
        if 'mean' in kwargs and 'std' in kwargs:
            self.register_buffer("mean", kwargs["mean"])
            self.register_buffer("std", kwargs["std"])
        
        if 'feature_mask' in kwargs:
            self.register_buffer("feature_mask", kwargs["feature_mask"])
        
        # Register metadata about how to read the event structure
        self.register_buffer("object_start_indices", kwargs.get("object_start_indices", None))
        self.register_buffer("is_event_data", kwargs.get("is_event_data", None))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply feature mask if it exists
        if hasattr(self, "feature_mask"):
            x = x[:, self.feature_mask]
            
        # Normalize using mean and std
        if hasattr(self, "mean") and hasattr(self, "std"):
            # Replace zeros with small values to avoid division by zero
            safe_std = torch.where(self.std == 0, torch.tensor(1e-8, device=self.std.device), self.std)
            x = (x - self.mean) / safe_std
            
        # Handle NaN and Inf values
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        return x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=1024):
    # Parse the dataset structure
    # First 3 columns: weight, E_T_miss, phi_{E_t}_miss
    # Then groups of 5 columns: obj_n, E_n, p_Tn, eta_n, phi_n
    
    # Create a feature mask to exclude unnecessary columns and zero-padded data
    non_zero_mask = (X_train != 0).sum(dim=0) > 0
    
    # Identify columns with real physics info (exclude padding)
    is_event_data = non_zero_mask.clone()
    
    # Calculate statistics for normalization (on non-zero values only)
    masked_X_train = X_train[:, is_event_data]
    
    # Replace zeros with NaN for proper mean/std calculation
    temp_data = masked_X_train.clone()
    temp_data[temp_data == 0] = float('nan')
    
    # Calculate mean and std, ignoring NaN values
    mean = torch.nanmean(temp_data, dim=0)
    std = torch.nanstd(temp_data, dim=0)
    
    # Find indices where each object information starts
    object_indices = []
    for i in range(3, X_train.shape[1], 5):  # Start after weight, ET_miss, phi_ET_miss
        if i < X_train.shape[1] and non_zero_mask[i]:
            object_indices.append(i)
    
    # Create the preprocessor with calculated statistics
    preproc = PreprocessModule(
        mean=mean,
        std=std,
        feature_mask=is_event_data,
        object_start_indices=torch.tensor(object_indices, dtype=torch.long),
        is_event_data=is_event_data
    )

    # Apply preprocessing
    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)

    # Create dataloaders
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        
        # Define network architecture
        hidden1 = 256
        hidden2 = 128
        hidden3 = 64
        
        # Main layers
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc_out = nn.Linear(hidden3, 1)
        
        # Initialize weights for better convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Forward pass through the network
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Output layer - no activation (handled by loss function)
        x = self.fc_out(x).squeeze(1)
        
        return x

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Use BCEWithLogitsLoss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam optimizer with learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    # Metric tracking
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    validation_auc = []
    best_auc = 0.0
    
    # EarlyStopping parameters
    patience = 5
    early_stop_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.float())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            # Apply gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item() * X_batch.size(0)
            predicted = (outputs > 0.0).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch.float()).sum().item()
            
            all_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            all_targets.extend(y_batch.cpu().numpy())
        
        epoch_loss /= total
        epoch_acc = correct / total
        epoch_auc = roc_auc_score(all_targets, all_preds)
        
        training_loss.append(epoch_loss)
        training_acc.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.float())
                
                val_loss += loss.item() * X_batch.size(0)
                predicted = (outputs > 0.0).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch.float()).sum().item()
                
                all_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        val_loss /= total
        val_acc = correct / total
        val_auc = roc_auc_score(all_targets, all_preds)
        
        validation_loss.append(val_loss)
        validation_acc.append(val_acc)
        validation_auc.append(val_auc)
        
        # Update learning rate based on AUC
        scheduler.step(val_auc)
        
        # Track best model and implement early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Train AUC: {epoch_auc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
        
        # Early stopping check
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_auc > 0:
        model.load_state_dict(best_state)
    
    print(f"Best validation AUC: {best_auc:.4f}")
    
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
    epochs = 1 if dryrun else 20

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