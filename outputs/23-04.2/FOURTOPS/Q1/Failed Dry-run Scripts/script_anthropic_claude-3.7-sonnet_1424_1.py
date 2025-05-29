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
    def __init__(self, **kwargs):
        super().__init__()
        
        # Register means and stds for normalization
        if "means" in kwargs and "stds" in kwargs:
            self.register_buffer("means", kwargs["means"])
            self.register_buffer("stds", kwargs["stds"])
        
        # Register mask for non-zero features
        if "non_zero_mask" in kwargs:
            self.register_buffer("non_zero_mask", kwargs["non_zero_mask"])
            
        # Register object type embeddings
        if "obj_positions" in kwargs:
            self.register_buffer("obj_positions", kwargs["obj_positions"])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract the weight, missing ET and missing ET phi (first 3 features)
        weight = x[:, 0:1]
        missing_et = x[:, 1:2]
        missing_et_phi = x[:, 2:3]
        
        # Extract object data
        obj_data = x[:, 3:]
        
        # Create feature matrix
        batch_size = x.shape[0]
        features_list = []
        
        # Add the first three features
        features_list.append(missing_et)
        features_list.append(missing_et_phi)
        
        # Calculate number of objects
        num_objects = (obj_data.shape[1] // 5)
        
        # Process each object
        object_features = []
        
        for i in range(num_objects):
            start_idx = i * 5
            # Check if object exists (non-zero values)
            obj_mask = obj_data[:, start_idx] != 0
            
            if torch.any(obj_mask):
                # Extract object features
                obj_type = obj_data[:, start_idx:start_idx+1]
                obj_E = obj_data[:, start_idx+1:start_idx+2]
                obj_pT = obj_data[:, start_idx+2:start_idx+3]
                obj_eta = obj_data[:, start_idx+3:start_idx+4]
                obj_phi = obj_data[:, start_idx+4:start_idx+5]
                
                # Create per-object features
                obj_features = torch.cat([
                    obj_type, obj_E, obj_pT, obj_eta, obj_phi,
                    # Add log-transformed features for E and pT
                    torch.log1p(obj_E),
                    torch.log1p(obj_pT)
                ], dim=1)
                
                # Set features to zero for non-existent objects
                zeros_tensor = torch.zeros_like(obj_features)
                obj_features = torch.where(
                    obj_mask.unsqueeze(1).expand_as(obj_features),
                    obj_features,
                    zeros_tensor
                )
                
                object_features.append(obj_features)
        
        # Stack all object features
        if object_features:
            all_obj_features = torch.cat(object_features, dim=1)
            features_list.append(all_obj_features)
        
        # Create global features
        obj_data_reshaped = obj_data.reshape(batch_size, num_objects, 5)
        
        # Count non-zero objects
        obj_mask = obj_data_reshaped[:, :, 0] != 0
        object_count = obj_mask.sum(dim=1, keepdim=True).float()
        features_list.append(object_count)
        
        # Calculate sums of E, pT for all objects
        valid_E = torch.where(obj_mask, obj_data_reshaped[:, :, 1], torch.zeros_like(obj_data_reshaped[:, :, 1]))
        valid_pT = torch.where(obj_mask, obj_data_reshaped[:, :, 2], torch.zeros_like(obj_data_reshaped[:, :, 2]))
        
        total_E = valid_E.sum(dim=1, keepdim=True)
        total_pT = valid_pT.sum(dim=1, keepdim=True)
        features_list.extend([total_E, total_pT, torch.log1p(total_E), torch.log1p(total_pT)])
        
        # Calculate mean and std of object properties
        for prop_idx in range(1, 5):  # E, pT, eta, phi
            valid_prop = torch.where(
                obj_mask, 
                obj_data_reshaped[:, :, prop_idx], 
                torch.zeros_like(obj_data_reshaped[:, :, prop_idx])
            )
            prop_sum = valid_prop.sum(dim=1, keepdim=True)
            prop_mean = torch.where(
                object_count > 0,
                prop_sum / torch.clamp(object_count, min=1.0),
                torch.zeros_like(prop_sum)
            )
            features_list.append(prop_mean)
            
            # Calculate standard deviation
            squared_diff = torch.where(
                obj_mask,
                (valid_prop - prop_mean)**2,
                torch.zeros_like(valid_prop)
            )
            prop_var = torch.where(
                object_count > 1,
                squared_diff.sum(dim=1, keepdim=True) / torch.clamp(object_count - 1, min=1.0),
                torch.zeros_like(prop_sum)
            )
            prop_std = torch.sqrt(torch.clamp(prop_var, min=1e-8))
            features_list.append(prop_std)
        
        # Concatenate all features
        processed_features = torch.cat(features_list, dim=1)
        
        # Normalize features using stored means and stds
        if hasattr(self, "means") and hasattr(self, "stds"):
            processed_features = (processed_features - self.means) / (self.stds + 1e-8)
        
        return processed_features

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=512):
    # Analyze the structure of the data to extract object information
    batch_size = X_train.shape[0]
    n_features = X_train.shape[1]
    
    # Calculate the number of objects in the data
    n_objects = (n_features - 3) // 5  # First 3 features are weight, missing ET, and missing ET phi
    
    # Prepare the preprocessing module without computing final statistics yet
    temp_preproc = PreprocessModule()
    
    # Get sample processed features to determine the output shape
    sample_processed = temp_preproc(X_train[:10])
    feature_dim = sample_processed.shape[1]
    
    # Process all training data to compute statistics
    all_processed = temp_preproc(X_train)
    
    # Calculate mean and std for each feature
    means = all_processed.mean(dim=0)
    stds = all_processed.std(dim=0)
    stds[stds < 1e-8] = 1.0  # Prevent division by zero
    
    # Find features that are non-zero across most samples
    non_zero_counts = (all_processed != 0).float().mean(dim=0)
    non_zero_mask = (non_zero_counts > 0.05).float()  # Keep features that are non-zero in at least 5% of samples
    
    # Find object positions in the input data
    obj_positions = []
    for i in range(n_objects):
        obj_start = 3 + i * 5
        obj_positions.append(obj_start)
    obj_positions = torch.tensor(obj_positions, dtype=torch.long)
    
    # Create the preprocessor with the computed statistics
    preproc = PreprocessModule(
        means=means,
        stds=stds,
        non_zero_mask=non_zero_mask,
        obj_positions=obj_positions
    )
    
    # Apply preprocessing to both training and validation data
    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)
    
    # Create datasets and dataloaders
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(2)).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.squeeze(2)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # If dimensions don't match, use a projection shortcut
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        
        self.se = SELayer(out_features, reduction=8)

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = F.selu(out)
        out = self.dropout1(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        out = self.se(out)
        
        out += identity
        out = F.selu(out)
        out = self.dropout2(out)
        
        return out

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        
        # Network architecture
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SELU(),
            nn.Dropout(0.3),
            
            ResidualBlock(256, 256, dropout_rate=0.3),
            ResidualBlock(256, 256, dropout_rate=0.3),
            ResidualBlock(256, 128, dropout_rate=0.3),
            ResidualBlock(128, 128, dropout_rate=0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SELU(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.layers(x).squeeze(-1)

# ----- FREE SECTION: Training Loop Implementation -----
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, weight_decay=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # Initialize history tracking
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_val_auc = 0.0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_epoch_loss = 0.0
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss (combination of BCE and Focal loss)
            loss = criterion(outputs, targets.float()) + focal_loss(outputs, targets.float())
            
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            train_epoch_loss += loss.item() * inputs.size(0)
            train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch-level metrics for training
        train_epoch_loss /= len(train_loader.dataset)
        training_loss.append(train_epoch_loss)
        
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        train_acc = accuracy_score(train_targets, train_preds > 0.5)
        train_auc = roc_auc_score(train_targets, train_preds)
        training_acc.append(train_acc)
        
        # Validation phase
        model.eval()
        val_epoch_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets.float())
                
                # Track metrics
                val_epoch_loss += loss.item() * inputs.size(0)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch-level metrics for validation
        val_epoch_loss /= len(val_loader.dataset)
        validation_loss.append(val_epoch_loss)
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_acc = accuracy_score(val_targets, val_preds > 0.5)
        val_auc = roc_auc_score(val_targets, val_preds)
        validation_acc.append(val_acc)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Print metrics
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Train AUC: {train_auc:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}')
    
    # Load best model weights
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
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=512)

    # Model Initialization
    sample_X, _ = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[1])

    # Training
    epochs = 1 if dryrun else 15

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs, lr=3e-4, weight_decay=1e-4)

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