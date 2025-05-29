# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import math
import torch.nn.functional as F

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
        # Register normalization statistics
        if "mean" in kwargs and "std" in kwargs:
            self.register_buffer("mean", kwargs["mean"])
            self.register_buffer("std", kwargs["std"])
        
        # Register object masks for feature extraction
        if "obj_mask" in kwargs:
            self.register_buffer("obj_mask", kwargs["obj_mask"])
        
        # Number of objects to consider
        if "num_objects" in kwargs:
            self.register_buffer("num_objects", torch.tensor(kwargs["num_objects"], dtype=torch.int64))
        else:
            self.register_buffer("num_objects", torch.tensor(25, dtype=torch.int64))
            
        # Max features per object
        if "features_per_obj" in kwargs:
            self.register_buffer("features_per_obj", torch.tensor(kwargs["features_per_obj"], dtype=torch.int64))
        else:
            self.register_buffer("features_per_obj", torch.tensor(4, dtype=torch.int64))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get the ET_miss and phi_ET_miss features (first two columns)
        et_miss = x[:, 0:2]
        
        # Extract object features - reshape to have [batch, objects, features]
        batch_size = x.shape[0]
        features_start = 2  # Start after ET_miss and phi_ET_miss
        
        # Calculate max objects from the data shape
        max_objects = (x.shape[1] - features_start) // (self.features_per_obj + 1)  # +1 for obj_id
        
        # Use only up to num_objects
        max_objects = min(max_objects, self.num_objects.item())
        
        # Initialize tensor to store reshaped objects
        objects_tensor = torch.zeros(batch_size, max_objects, self.features_per_obj, device=x.device)
        
        # Extract each object's features
        for i in range(max_objects):
            # Each object has obj_id + 4 features (E, pT, eta, phi)
            start_idx = features_start + i * (self.features_per_obj + 1)
            end_idx = start_idx + (self.features_per_obj + 1)
            
            # Skip the object ID and take the 4 features
            objects_tensor[:, i, :] = x[:, start_idx+1:end_idx]
        
        # Create object validity mask - identify zero-padded objects
        # An object is considered valid if its energy (first feature) is non-zero
        valid_mask = objects_tensor[:, :, 0] > 0
        
        # Normalize the ET_miss
        if hasattr(self, 'mean') and hasattr(self, 'std'):
            et_miss_normalized = (et_miss - self.mean[:2]) / (self.std[:2] + 1e-8)
            
            # Normalize the object features - we apply this only to the energy (0) and pT (1) features
            # Eta (2) and phi (3) are angles and have intrinsic ranges
            objects_tensor[:, :, 0] = (objects_tensor[:, :, 0] - self.mean[2]) / (self.std[2] + 1e-8)
            objects_tensor[:, :, 1] = (objects_tensor[:, :, 1] - self.mean[3]) / (self.std[3] + 1e-8)
        else:
            et_miss_normalized = et_miss
        
        # Convert to 4-vectors [E, px, py, pz]
        four_vectors = torch.zeros(batch_size, max_objects, 4, device=x.device)
        
        # E is already available
        four_vectors[:, :, 0] = objects_tensor[:, :, 0]  # Energy
        
        # Calculate px, py, pz from pT, eta, phi
        pt = objects_tensor[:, :, 1]  # pT
        eta = objects_tensor[:, :, 2]  # eta
        phi = objects_tensor[:, :, 3]  # phi
        
        # px = pT * cos(phi)
        four_vectors[:, :, 1] = pt * torch.cos(phi)
        # py = pT * sin(phi)
        four_vectors[:, :, 2] = pt * torch.sin(phi)
        # pz = pT * sinh(eta)
        four_vectors[:, :, 3] = pt * torch.sinh(eta)
        
        # Compute the missing transverse energy vector components
        et_miss_x = et_miss[:, 0] * torch.cos(et_miss[:, 1])
        et_miss_y = et_miss[:, 0] * torch.sin(et_miss[:, 1])
        et_miss_components = torch.stack([et_miss[:, 0], et_miss_x, et_miss_y, torch.zeros_like(et_miss[:, 0])], dim=1)
        
        # Prepare final features
        # Include: normalized ET_miss, four_vectors, valid_mask
        processed_features = {
            'et_miss': et_miss_normalized,
            'four_vectors': four_vectors,
            'valid_mask': valid_mask,
            'et_miss_components': et_miss_components
        }
        
        return processed_features

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=128):
    # Calculate statistics for normalization from training data
    # First, let's identify feature positions
    et_miss = X_train[:, 0]
    phi_et_miss = X_train[:, 1]
    
    # Process objects to calculate statistics
    features_per_obj = 4  # E, pT, eta, phi
    obj_features = []
    
    # Process each event to extract valid object features
    for event in X_train:
        idx = 2  # Start after ET_miss and phi_ET_miss
        while idx < len(event):
            if event[idx] != 0:  # Check if there's a valid object ID
                # Extract E, pT values to compute statistics
                energy = event[idx + 1]
                pt = event[idx + 2]
                if energy > 0:  # Valid object
                    obj_features.append([energy, pt])
            idx += features_per_obj + 1  # Move to the next object
    
    obj_features = torch.tensor(obj_features)
    
    # Compute mean and standard deviation
    et_miss_mean = et_miss.mean()
    et_miss_std = et_miss.std()
    phi_et_miss_mean = phi_et_miss.mean()
    phi_et_miss_std = phi_et_miss.std()
    energy_mean = obj_features[:, 0].mean()
    energy_std = obj_features[:, 0].std()
    pt_mean = obj_features[:, 1].mean()
    pt_std = obj_features[:, 1].std()
    
    # Create the normalization statistics tensor
    mean = torch.tensor([et_miss_mean, phi_et_miss_mean, energy_mean, pt_mean])
    std = torch.tensor([et_miss_std, phi_et_miss_std, energy_std, pt_std])
    
    # Determine the maximum number of objects to keep
    num_objects = 25  # Set a reasonable limit
    
    # Create preprocessing module
    preproc = PreprocessModule(
        mean=mean,
        std=std,
        num_objects=num_objects,
        features_per_obj=features_per_obj
    )
    
    # Create dataset and dataloader
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class LorentzLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(LorentzLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Layers for computing interaction tensors
        self.W_g = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.W_out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, four_vectors, mask):
        # x: [batch, num_objects, hidden_dim]
        # four_vectors: [batch, num_objects, 4]
        # mask: [batch, num_objects] - indicates which objects are valid
        
        batch_size, num_objects, _ = x.shape
        
        # Compute queries and keys for attention
        q = self.W_q(x)  # [batch, num_objects, hidden_dim]
        k = self.W_k(x)  # [batch, num_objects, hidden_dim]
        g = self.W_g(x)  # [batch, num_objects, hidden_dim]
        
        # Compute Lorentz-invariant products between four-vectors
        # p_i · p_j = E_i*E_j - px_i*px_j - py_i*py_j - pz_i*pz_j
        v_i = four_vectors.unsqueeze(2)  # [batch, num_objects, 1, 4]
        v_j = four_vectors.unsqueeze(1)  # [batch, 1, num_objects, 4]
        
        # Minkowski metric
        minkowski = torch.tensor([1, -1, -1, -1], device=x.device)
        
        # Compute Lorentz inner products: p_i · p_j
        products = (v_i * v_j * minkowski).sum(dim=3)  # [batch, num_objects, num_objects]
        
        # Scale the Lorentz products and apply softmax for attention
        scale = 1.0 / math.sqrt(self.hidden_dim)
        attn_logits = scale * products
        
        # Apply mask to exclude padding
        mask_i = mask.unsqueeze(2)  # [batch, num_objects, 1]
        mask_j = mask.unsqueeze(1)  # [batch, 1, num_objects]
        mask_combined = mask_i & mask_j  # [batch, num_objects, num_objects]
        
        # Apply mask by setting invalid positions to a large negative number
        attn_logits = torch.where(mask_combined, attn_logits, torch.tensor(-1e9, device=x.device))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_logits, dim=2)  # [batch, num_objects, num_objects]
        
        # Combine with queries and keys
        qk_similarity = torch.bmm(q, k.transpose(1, 2))  # [batch, num_objects, num_objects]
        qk_similarity = scale * qk_similarity
        qk_similarity = torch.where(mask_combined, qk_similarity, torch.tensor(-1e9, device=x.device))
        qk_weights = F.softmax(qk_similarity, dim=2)  # [batch, num_objects, num_objects]
        
        # Final attention is a combination of Lorentz products and query-key similarity
        combined_attn = attn_weights * qk_weights
        
        # Apply attention to values
        messages = torch.bmm(combined_attn, g)  # [batch, num_objects, hidden_dim]
        
        # Project to output dimension
        output = self.W_out(messages)
        
        return output

class ParticleFeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ParticleFeatureEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: [batch, num_objects, input_dim]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.ln(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        
        # Define dimensions
        self.hidden_dim = 128
        self.num_lorentz_layers = 3
        
        # Initial feature encoding for particle four-vectors
        self.particle_encoder = ParticleFeatureEncoder(4, self.hidden_dim)
        
        # ET_miss encoder
        self.et_miss_encoder = nn.Sequential(
            nn.Linear(2, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )
        
        # Lorentz-equivariant message passing layers
        self.lorentz_layers = nn.ModuleList([
            LorentzLayer(self.hidden_dim) for _ in range(self.num_lorentz_layers)
        ])
        
        # Layer norm after each Lorentz layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_lorentz_layers)
        ])
        
        # Global pooling MLP
        self.global_pool_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Final classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x is now a dictionary containing preprocessed features
        four_vectors = x['four_vectors']  # [batch, num_objects, 4]
        valid_mask = x['valid_mask']      # [batch, num_objects]
        et_miss = x['et_miss']            # [batch, 2]
        
        # Embed particle features
        particle_features = self.particle_encoder(four_vectors)  # [batch, num_objects, hidden_dim]
        
        # Process through Lorentz-equivariant layers with residual connections
        for i in range(self.num_lorentz_layers):
            lorentz_out = self.lorentz_layers[i](particle_features, four_vectors, valid_mask)
            particle_features = self.layer_norms[i](particle_features + lorentz_out)
        
        # Global pooling with mask
        # Compute the sum of features for valid particles
        mask_expanded = valid_mask.unsqueeze(-1).expand_as(particle_features)
        masked_features = particle_features * mask_expanded
        
        # Sum pooling
        summed_features = masked_features.sum(dim=1)  # [batch, hidden_dim]
        
        # Count valid particles per event
        valid_counts = valid_mask.sum(dim=1, keepdim=True)  # [batch, 1]
        # Avoid division by zero
        valid_counts = torch.clamp(valid_counts, min=1.0)
        
        # Average pooling
        avg_features = summed_features / valid_counts
        
        # Apply MLP to pooled features
        global_features = self.global_pool_mlp(avg_features)  # [batch, hidden_dim]
        
        # Process ET_miss
        et_miss_features = self.et_miss_encoder(et_miss)  # [batch, hidden_dim]
        
        # Combine global particle features with ET_miss features
        combined_features = torch.cat([global_features, et_miss_features], dim=1)  # [batch, 2*hidden_dim]
        
        # Final classification
        logits = self.classifier(combined_features).squeeze(-1)
        
        return logits

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_val_auc = 0.0
    best_model = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            
            # Forward pass through preprocessor
            preproc_inputs = train_loader.dataset.tensors[0][0:1]
            sample_output = model.preprocess(preproc_inputs)
            expected_keys = list(sample_output.keys())
            
            # Process current batch
            processed_inputs = model.preprocess(inputs)
            
            # Forward pass through model
            optimizer.zero_grad()
            outputs = model(processed_inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Store predictions and labels for accuracy calculation
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        training_loss.append(epoch_loss)
        
        # Calculate AUC and accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        train_auc = roc_auc_score(all_labels, all_preds)
        train_acc = accuracy_score(all_labels, all_preds > 0.5)
        training_acc.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                
                # Process inputs
                processed_inputs = model.preprocess(inputs)
                
                # Forward pass
                outputs = model(processed_inputs)
                
                # Compute loss
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                # Store predictions and labels
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_val_preds.extend(preds)
                all_val_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        validation_loss.append(val_epoch_loss)
        
        # Calculate AUC and accuracy
        all_val_preds = np.array(all_val_preds)
        all_val_labels = np.array(all_val_labels)
        val_auc = roc_auc_score(all_val_labels, all_val_preds)
        val_acc = accuracy_score(all_val_labels, all_val_preds > 0.5)
        validation_acc.append(val_acc)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
    
    # Load best model
    if best_model is not None:
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
    batch_size = 64 if not dryrun else 16
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size)

    # Model Initialization
    model = Classifier(input_dim=X_train.shape[1])
    model.preprocess = preproc  # Add preprocessor to model for convenience

    # Training
    epochs = 1 if dryrun else 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs, device=device)

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