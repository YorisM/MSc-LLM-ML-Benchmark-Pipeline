# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
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
        # Constants for feature extraction and normalization
        self.register_buffer("feature_mask", kwargs.get("feature_mask"))
        self.register_buffer("mean", kwargs.get("mean"))
        self.register_buffer("std", kwargs.get("std"))
        self.register_buffer("non_zero_mask", kwargs.get("non_zero_mask"))
        self.register_buffer("max_objects", torch.tensor(kwargs.get("max_objects")))
        # Number of features per object (E, pT, eta, phi)
        self.register_buffer("features_per_obj", torch.tensor(4))
        # Mask for which elements correspond to actual physics objects
        self.pad_value = -999.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract missing ET and phi
        et_miss = x[:, 0:1]
        phi_et_miss = x[:, 1:2]
        
        # Get the object data
        object_data = x[:, 2:]
        
        # Reshape to identify object blocks (every 5th element is object ID)
        batch_size = x.shape[0]
        max_objects = int((x.shape[1] - 2) // 5)
        
        # Initialize tensor for reorganized data
        # Format: [batch, objects, features] where features = [E, pT, eta, phi]
        reshaped = torch.zeros((batch_size, max_objects, 4), device=x.device)
        
        # Extract object data (ignoring object IDs)
        for i in range(max_objects):
            start_idx = i * 5 + 2
            # Skip the object ID and take the 4 features: E, pT, eta, phi
            reshaped[:, i, :] = x[:, start_idx+1:start_idx+5]
        
        # Create a mask for valid objects (non-zero and non-padded)
        valid_mask = (reshaped[:, :, 0] != 0) & (reshaped[:, :, 0] != self.pad_value)
        
        # Replace zeros or pad values with NaNs to avoid them affecting statistics
        cleaned = reshaped.clone()
        cleaned[~valid_mask.unsqueeze(-1).expand_as(cleaned)] = float('nan')
        
        # Normalize each feature
        normalized = (cleaned - self.mean) / self.std
        
        # Replace NaNs with zeros for further processing
        normalized[torch.isnan(normalized)] = 0.0
        
        # Compute Lorentz 4-vectors: [E, px, py, pz]
        px = normalized[:, :, 1] * torch.cos(normalized[:, :, 3])  # pT * cos(phi)
        py = normalized[:, :, 1] * torch.sin(normalized[:, :, 3])  # pT * sin(phi)
        pz = normalized[:, :, 1] * torch.sinh(normalized[:, :, 2]) # pT * sinh(eta)
        
        # Stack into a tensor of shape [batch, objects, 4]
        lorentz_vectors = torch.stack([normalized[:, :, 0], px, py, pz], dim=2)
        
        # Calculate object validity mask (1 for valid objects)
        object_mask = valid_mask.float().unsqueeze(-1)  # [batch, objects, 1]
        
        # Combine features: Lorentz vectors and object mask
        features = torch.cat([lorentz_vectors, object_mask], dim=2)  # [batch, objects, 5]
        
        # Flatten for the model input
        features_flat = features.reshape(batch_size, -1)  # [batch, objects * 5]
        
        # Append missing ET and phi
        # Normalize missing ET
        et_miss_norm = (et_miss - self.mean[0, 0]) / self.std[0, 0]
        
        # Return the final preprocessed features
        output = torch.cat([et_miss_norm, phi_et_miss, features_flat], dim=1)
        return output

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=64):
    # Calculate the maximum number of objects in any event
    event_length = X_train.shape[1]
    max_objects = (event_length - 2) // 5  # First 2 elements are ET_miss and phi
    
    # Extract just the object data for normalization
    object_data_train = X_train[:, 2:]
    
    # Reshape to separate object IDs from features
    batch_size_train = X_train.shape[0]
    reshaped_train = torch.zeros((batch_size_train, max_objects, 4))
    
    for i in range(max_objects):
        start_idx = i * 5
        # Skip object ID, take E, pT, eta, phi
        reshaped_train[:, i, :] = object_data_train[:, start_idx+1:start_idx+5]
    
    # Create a mask for valid objects
    valid_mask_train = (reshaped_train[:, :, 0] != 0) & (reshaped_train[:, :, 0] != -999.0)
    
    # Calculate mean and std for normalization (ignoring zeros and padding)
    cleaned_train = reshaped_train.clone()
    cleaned_train[~valid_mask_train.unsqueeze(-1).expand_as(cleaned_train)] = float('nan')
    
    # Calculate mean and std for each feature
    mean = torch.nanmean(cleaned_train, dim=(0, 1), keepdim=True)
    std = torch.nanstd(cleaned_train, dim=(0, 1), keepdim=True)
    # Replace any zero std with 1 to avoid division by zero
    std[std == 0] = 1.0
    
    # Calculate a mask for non-zero elements
    non_zero_mask = ~torch.isnan(cleaned_train)
    
    # Define feature mask if needed
    feature_mask = torch.ones_like(reshaped_train[0, 0])
    
    # Create preprocessor module with calculated statistics
    preproc = PreprocessModule(
        feature_mask=feature_mask,
        mean=mean,
        std=std,
        non_zero_mask=non_zero_mask,
        max_objects=max_objects
    )
    
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
class LorentzEquivariantLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(LorentzEquivariantLayer, self).__init__()
        self.hidden_dim = hidden_dim
        # Define transformations that respect Lorentz symmetry
        self.W_scalar = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.W_vector = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
    def forward(self, x, vectors):
        # x shape: [batch_size, num_objects, hidden_dim]
        # vectors shape: [batch_size, num_objects, 4] (Lorentz vectors as E, px, py, pz)
        
        batch_size, num_objects, _ = x.shape
        
        # Scalar transformation
        scalar_out = torch.matmul(x, self.W_scalar)
        
        # Vector transformation that preserves Lorentz symmetry
        # Calculate Minkowski inner product for each pair of objects
        expanded_vectors_i = vectors.unsqueeze(2)  # [batch, objects, 1, 4]
        expanded_vectors_j = vectors.unsqueeze(1)  # [batch, 1, objects, 4]
        
        # Minkowski metric tensor: diag(1, -1, -1, -1)
        minkowski_product = expanded_vectors_i[:,:,:,0] * expanded_vectors_j[:,:,:,0] \
                          - expanded_vectors_i[:,:,:,1] * expanded_vectors_j[:,:,:,1] \
                          - expanded_vectors_i[:,:,:,2] * expanded_vectors_j[:,:,:,2] \
                          - expanded_vectors_i[:,:,:,3] * expanded_vectors_j[:,:,:,3]
        
        # Use this Lorentz-invariant quantity in the message passing
        messages = torch.matmul(x.unsqueeze(1), self.W_vector).squeeze(1)
        messages = messages.unsqueeze(1).expand(-1, num_objects, -1, -1)
        
        # Weight messages by Minkowski inner product
        weighted_messages = messages * minkowski_product.unsqueeze(-1)
        
        # Aggregate messages across all objects
        aggregated_messages = weighted_messages.sum(dim=2) + self.bias
        
        # Combine scalar and vector outputs
        output = scalar_out + aggregated_messages
        return output

class MessagePassingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MessagePassingLayer, self).__init__()
        self.fc_message = nn.Linear(input_dim * 2, output_dim)
        self.fc_update = nn.Linear(input_dim + output_dim, output_dim)
        
    def forward(self, x, mask):
        # x: [batch, objects, features]
        # mask: [batch, objects, 1]
        batch_size, num_objects, feature_dim = x.shape
        
        # Expand x for pairwise comparison
        x_i = x.unsqueeze(2).expand(-1, -1, num_objects, -1)  # [batch, objects, objects, features]
        x_j = x.unsqueeze(1).expand(-1, num_objects, -1, -1)  # [batch, objects, objects, features]
        
        # Concatenate features from both objects
        pairs = torch.cat([x_i, x_j], dim=-1)  # [batch, objects, objects, 2*features]
        
        # Generate messages
        messages = torch.relu(self.fc_message(pairs))  # [batch, objects, objects, output_dim]
        
        # Mask out invalid objects
        mask_i = mask.squeeze(-1).unsqueeze(2).expand(-1, -1, num_objects)  # [batch, objects, objects]
        mask_j = mask.squeeze(-1).unsqueeze(1).expand(-1, num_objects, -1)  # [batch, objects, objects]
        valid_pairs = (mask_i * mask_j).unsqueeze(-1)
        
        # Apply mask to messages
        masked_messages = messages * valid_pairs
        
        # Aggregate messages (sum)
        aggregated = masked_messages.sum(dim=2)  # [batch, objects, output_dim]
        
        # Update node features
        combined = torch.cat([x, aggregated], dim=-1)  # [batch, objects, features+output_dim]
        updated = torch.relu(self.fc_update(combined))  # [batch, objects, output_dim]
        
        return updated

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        # Parameters
        self.max_objects = (input_dim - 2) // 5
        self.lorentz_dim = 4  # E, px, py, pz
        self.hidden_dim = 64
        
        # Preprocessing layers
        self.fc_et_miss = nn.Linear(2, 16)  # For missing ET and phi
        self.ln_et_miss = nn.LayerNorm(16)
        
        # Initial projection for object features
        self.fc_init = nn.Linear(5, self.hidden_dim)  # 5 = 4 (Lorentz) + 1 (mask)
        self.ln_init = nn.LayerNorm(self.hidden_dim)
        
        # Equivariant message passing layers
        self.lorentz_layer1 = LorentzEquivariantLayer(self.hidden_dim)
        self.message_layer1 = MessagePassingLayer(self.hidden_dim, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        
        self.lorentz_layer2 = LorentzEquivariantLayer(self.hidden_dim)
        self.message_layer2 = MessagePassingLayer(self.hidden_dim, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        
        # Global attention for object aggregation
        self.query = nn.Linear(self.hidden_dim, 1)
        
        # Final classification layers
        self.fc1 = nn.Linear(self.hidden_dim + 16, 128)
        self.ln_fc1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.ln_fc2 = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Process missing ET and phi (first two elements)
        et_miss = x[:, :2]
        et_miss_features = torch.relu(self.ln_et_miss(self.fc_et_miss(et_miss)))
        
        # Process object data
        object_data = x[:, 2:].reshape(batch_size, self.max_objects, 5)
        
        # Extract mask and Lorentz vectors
        mask = object_data[:, :, 4:5]  # Last dimension is the mask
        lorentz_vectors = object_data[:, :, :4]  # First 4 dimensions are Lorentz vectors
        
        # Initial object feature encoding
        obj_features = torch.relu(self.ln_init(self.fc_init(object_data)))
        
        # Apply Lorentz-equivariant layers and message passing
        obj_features = self.lorentz_layer1(obj_features, lorentz_vectors)
        obj_features = self.message_layer1(obj_features, mask)
        obj_features = self.ln1(obj_features)
        obj_features = torch.relu(obj_features)
        
        obj_features = self.lorentz_layer2(obj_features, lorentz_vectors)
        obj_features = self.message_layer2(obj_features, mask)
        obj_features = self.ln2(obj_features)
        obj_features = torch.relu(obj_features)
        
        # Attention-based pooling
        attention_weights = torch.softmax(self.query(obj_features), dim=1)
        attention_weights = attention_weights * mask  # Apply object mask
        attention_weights = attention_weights / (torch.sum(attention_weights, dim=1, keepdim=True) + 1e-8)
        
        # Weighted sum of object features
        global_features = torch.sum(obj_features * attention_weights, dim=1)
        
        # Combine with missing ET features
        combined = torch.cat([global_features, et_miss_features], dim=1)
        
        # Final classification layers
        x = torch.relu(self.ln_fc1(self.fc1(combined)))
        x = self.dropout1(x)
        
        x = torch.relu(self.ln_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Output logits
        logits = self.fc3(x)
        
        return torch.sigmoid(logits).squeeze(-1)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    # Tracking metrics
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_truths = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.float())
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            train_losses.append(loss.item())
            train_preds.extend(y_pred.detach().cpu().numpy())
            train_truths.extend(y_batch.detach().cpu().numpy())
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        train_auc = roc_auc_score(train_truths, train_preds)
        train_accuracy = accuracy_score(train_truths, [1 if pred > 0.5 else 0 for pred in train_preds])
        
        training_loss.append(avg_train_loss)
        training_acc.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_truths = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.float())
                
                # Track metrics
                val_losses.append(loss.item())
                val_preds.extend(y_pred.detach().cpu().numpy())
                val_truths.extend(y_batch.detach().cpu().numpy())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        val_auc = roc_auc_score(val_truths, val_preds)
        val_accuracy = accuracy_score(val_truths, [1 if pred > 0.5 else 0 for pred in val_preds])
        
        validation_loss.append(avg_val_loss)
        validation_acc.append(val_accuracy)
        
        # Update scheduler based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train AUC: {train_auc:.4f} - "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation AUC: {best_auc:.4f}")
    
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
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=64)

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