# ----- FIXED SECTION: Import Libraries -----
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

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
def preprocess_data(X_train, Y_train, X_val, Y_val):
    # Extract weights from first column
    weights_train = X_train[:, 0].clone()
    weights_val = X_val[:, 0].clone()
    
    # Find features that are mostly zero (padding features)
    non_zero_cols = (X_train != 0).float().mean(dim=0) > 0.01
    
    # Keep only features with sufficient non-zero values
    X_train_filtered = X_train[:, non_zero_cols]
    X_val_filtered = X_val[:, non_zero_cols]
    
    # Calculate mean and std for normalization (excluding weights, E_T_miss, and phi_E_t_miss)
    feature_means = X_train_filtered.mean(dim=0)
    feature_stds = X_train_filtered.std(dim=0)
    feature_stds[feature_stds == 0] = 1.0  # Prevent division by zero
    
    # Normalize data
    X_train_norm = (X_train_filtered - feature_means) / feature_stds
    X_val_norm = (X_val_filtered - feature_means) / feature_stds
    
    # Replace NaNs and infinities with 0
    X_train_norm = torch.nan_to_num(X_train_norm, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_norm = torch.nan_to_num(X_val_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Add weights as a separate feature that algorithms can use
    X_train_with_weights = torch.cat([weights_train.unsqueeze(1), X_train_norm], dim=1)
    X_val_with_weights = torch.cat([weights_val.unsqueeze(1), X_val_norm], dim=1)
    
    return X_train_with_weights, Y_train, X_val_with_weights, Y_val, weights_train, weights_val

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        
        # A deep architecture with residual connections
        self.bn_input = nn.BatchNorm1d(input_dim)
        
        # First block
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        # Second block
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        # Third block with residual connection
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)
        
        # Fourth block with residual connection
        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.3)
        
        # Fifth block with residual connection
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.2)
        
        # Output layer
        self.fc_output = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.bn_input(x)
        
        # First block
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Second block
        identity = x
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Third block with residual connection
        x = x + identity
        identity = x
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Fourth block with residual connection
        x = x + identity
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        # Fifth block
        x = F.leaky_relu(self.bn5(self.fc5(x)))
        x = self.dropout5(x)
        
        # Output layer
        x = self.fc_output(x)
        
        return x.squeeze()

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, X_train, Y_train, X_val, Y_val, epochs, weights_train=None, weights_val=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Use BCEWithLogitsLoss which combines sigmoid and BCE for stability
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Use a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Create data loaders with weighted sampling
    batch_size = 512
    train_dataset = TensorDataset(X_train, Y_train.float())
    val_dataset = TensorDataset(X_val, Y_val.float())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    training_auc = []
    validation_auc = []
    best_val_auc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        y_true_train = []
        y_score_train = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Get batch weights from the first column
            batch_weights = X_batch[:, 0].clone()
            
            # Forward pass
            outputs = model(X_batch)
            
            # Calculate loss with sample weights
            loss = criterion(outputs, y_batch)
            weighted_loss = (loss * batch_weights).mean()
            
            # Backward and optimize
            optimizer.zero_grad()
            weighted_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate metrics
            epoch_loss += weighted_loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            # Collect predictions for AUC calculation
            y_true_train.extend(y_batch.cpu().numpy())
            y_score_train.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        
        # Calculate epoch metrics
        epoch_loss /= len(train_loader)
        accuracy = correct / total
        auc = roc_auc_score(y_true_train, y_score_train)
        
        training_loss.append(epoch_loss)
        training_acc.append(accuracy)
        training_auc.append(auc)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        y_true_val = []
        y_score_val = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Get batch weights from the first column
                batch_weights = X_batch[:, 0].clone()
                
                # Forward pass
                outputs = model(X_batch)
                
                # Calculate loss
                loss = criterion(outputs, y_batch)
                weighted_loss = (loss * batch_weights).mean()
                
                # Calculate metrics
                val_loss += weighted_loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
                
                # Collect predictions for AUC calculation
                y_true_val.extend(y_batch.cpu().numpy())
                y_score_val.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_auc = roc_auc_score(y_true_val, y_score_val)
        
        validation_loss.append(val_loss)
        validation_acc.append(val_accuracy)
        validation_auc.append(val_auc)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Early stopping based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {accuracy:.4f}, Train AUC: {auc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")
    
    # Load the best model
    if best_model_state is not None:
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
    X_train, Y_train, X_val, Y_val, weights_train, weights_val = preprocess_data(X_train, Y_train, X_val, Y_val)

    # Model Initialization
    model = Classifier(input_dim=X_train.shape[1])

    # Training (dryrun limits epochs)
    epochs = 1 if dryrun else 30

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, X_train, Y_train, X_val, Y_val, epochs=epochs, weights_train=weights_train, weights_val=weights_val)

    # Save Model
    model_filename = sys.argv[0].replace(".py", "") + "_model.pth"
    torch.save(trained_model.state_dict(), model_filename)

    # Plot Metrics
    plot_and_save(training_loss, validation_loss, "Loss", "training_loss.png")
    plot_and_save(training_acc, validation_acc, "Accuracy", "training_accuracy.png")

    print("Training complete. Outputs and model saved successfully.")

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)