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
        self.register_buffer("means", kwargs["means"] if "means" in kwargs else None)
        self.register_buffer("stds", kwargs["stds"] if "stds" in kwargs else None)
        self.register_buffer("feat_mask", kwargs["feat_mask"] if "feat_mask" in kwargs else None)
        self.register_buffer("object_type_offset", torch.tensor(1))
        self.register_buffer("particle_dims_per_obj", torch.tensor(5))
        self.register_buffer("max_objects", torch.tensor(21))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract basic features: ETmiss, phi_ETmiss
        et_miss = x[:, 0:1]  # ETmiss
        phi_et_miss = x[:, 1:2]  # phi_ETmiss
        
        # Normalize these basic features
        if self.means is not None and self.stds is not None:
            et_miss = (et_miss - self.means[0]) / self.stds[0]
            phi_et_miss = (phi_et_miss - self.means[1]) / self.stds[1]
        
        # Prepare tensor for objects (using the maximum number of objects in the dataset)
        batch_size = x.shape[0]
        max_objects = self.max_objects
        
        # Create object-centric tensor with shape [batch_size, max_objects, 5]
        # Each object has 5 features: [obj_type, E, pT, eta, phi]
        object_features = torch.zeros((batch_size, max_objects, 5), device=x.device)
        
        # Extract object features from the flat representation
        for i in range(max_objects):
            # Each object occupies 5 positions in the original tensor
            start_idx = 2 + i * 5
            if start_idx + 4 < x.shape[1]:
                # Map the object type to a numerical value
                # The object identifier is at position 0 of each 5-element chunk
                obj_type = x[:, start_idx:start_idx+1]
                
                # Fill object features: [obj_type, E, pT, eta, phi]
                object_features[:, i, 0] = obj_type.squeeze()
                object_features[:, i, 1:5] = x[:, start_idx+1:start_idx+5]
        
        # Create mask for valid objects (obj_type > 0)
        valid_mask = (object_features[:, :, 0] > 0).float().unsqueeze(-1)
        
        # Apply normalization to object features if needed
        if self.means is not None and self.stds is not None:
            for j in range(1, 5):  # E, pT, eta, phi
                feat_idx = 2 + j  # offset in the original mean/std vectors
                object_features[:, :, j] = (object_features[:, :, j] - self.means[feat_idx]) / self.stds[feat_idx]
        
        # Physics-informed features
        # 1. Transverse mass MT: sqrt(E^2 - pT^2)
        mt = torch.sqrt(torch.clamp(object_features[:, :, 1]**2 - object_features[:, :, 2]**2, min=1e-8))
        mt = mt.unsqueeze(-1)
        
        # 2. Delta R between objects (computed only for valid objects)
        delta_r_matrix = torch.zeros((batch_size, max_objects, max_objects), device=x.device)
        for i in range(max_objects):
            for j in range(i+1, max_objects):
                # Calculate ΔR = sqrt((Δη)^2 + (Δφ)^2)
                delta_eta = object_features[:, i, 3] - object_features[:, j, 3]
                delta_phi = torch.abs(object_features[:, i, 4] - object_features[:, j, 4])
                # Adjust delta_phi to consider the circular nature of phi
                delta_phi = torch.min(delta_phi, 2*torch.tensor(math.pi, device=x.device) - delta_phi)
                delta_r = torch.sqrt(delta_eta**2 + delta_phi**2)
                delta_r_matrix[:, i, j] = delta_r
                delta_r_matrix[:, j, i] = delta_r  # Symmetric
        
        # Create a flattened version with nearest neighbor distances
        k_nearest = 3  # Keep K nearest neighbors for each object
        knn_distances = torch.zeros((batch_size, max_objects, k_nearest), device=x.device)
        
        for i in range(max_objects):
            # For each object, get its distances to all others
            distances = delta_r_matrix[:, i, :]
            # Replace self-distance (0) with a large value to exclude it
            distances[:, i] = 1e9
            # Get indices of k nearest neighbors
            _, indices = torch.topk(distances, k=k_nearest, dim=1, largest=False)
            # Get the corresponding distances
            for k in range(k_nearest):
                batch_indices = torch.arange(batch_size, device=x.device)
                knn_distances[:, i, k] = delta_r_matrix[batch_indices, i, indices[:, k]]
        
        # 3. Augment object features with mass and KNN distances
        augmented_object_features = torch.cat([
            object_features,  # [batch_size, max_objects, 5]
            mt,  # [batch_size, max_objects, 1]
            knn_distances,  # [batch_size, max_objects, k_nearest]
        ], dim=-1)
        
        # 4. Add global features
        # Concatenate ETmiss and phi_ETmiss as global features
        global_features = torch.cat([et_miss, phi_et_miss], dim=-1)  # [batch_size, 2]
        
        # Construct the final features
        # Return [object_features, valid_mask, global_features] as a list
        return {
            "object_features": augmented_object_features,  # [batch_size, max_objects, 5+1+k_nearest]
            "valid_mask": valid_mask,  # [batch_size, max_objects, 1]
            "global_features": global_features,  # [batch_size, 2]
            "delta_r_matrix": delta_r_matrix,  # [batch_size, max_objects, max_objects]
        }

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=128):
    # Calculate means and standard deviations for normalization
    # Using only non-zero values for means and stds calculation
    non_zero_mask = (X_train != 0)
    # Calculate means for each feature across non-zero entries
    means = torch.zeros(X_train.shape[1])
    stds = torch.ones(X_train.shape[1])
    
    for j in range(X_train.shape[1]):
        non_zero_values = X_train[:, j][non_zero_mask[:, j]]
        if len(non_zero_values) > 0:
            means[j] = non_zero_values.mean()
            stds[j] = non_zero_values.std()
            # Handle zero std
            if stds[j] < 1e-8:
                stds[j] = 1.0
    
    # Create a mask for actual feature columns
    feat_mask = torch.ones(X_train.shape[1], dtype=torch.bool)
    
    # Initialize preprocessor with calculated statistics
    preproc = PreprocessModule(means=means, stds=stds, feat_mask=feat_mask)

    # Create datasets and dataloaders
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Slot Attention Implementation -----
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, num_slots, dim))
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        
        hidden_dim = max(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, mask=None):
        b, n, d = inputs.shape
        slots = self.slots_mu.expand(b, -1, -1) + self.slots_sigma.expand(b, -1, -1) * torch.randn(b, self.num_slots, d, device=inputs.device)
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            if mask is not None:
                # Apply mask to attention (mask is [batch, num_inputs, 1])
                mask = mask.squeeze(-1)  # [batch, num_inputs]
                dots = dots.masked_fill(~mask.unsqueeze(1).bool(), -1e9)
            
            attn = dots.softmax(dim=2) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bij,bjd->bid', attn, v)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            ).reshape(b, self.num_slots, d)

            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn  # Return attention weights for visualization

# ----- FREE SECTION: Transformer Encoder -----
class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, dropout=dropout),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]))

    def forward(self, x, mask=None):
        # Expected input shape: [batch_size, seq_len, dim]
        # Need to reorder for torch's MultiheadAttention: [seq_len, batch_size, dim]
        x = x.transpose(0, 1)
        
        # If mask provided, convert it for MultiheadAttention
        attention_mask = None
        if mask is not None:
            # Create attention mask from boolean mask
            # mask shape: [batch_size, seq_len, 1]
            attention_mask = ~(mask.squeeze(-1).bool())  # Invert because PyTorch uses True to MASK
        
        for norm1, attn, norm2, ff in self.layers:
            # Self-attention block
            x_norm = norm1(x)
            attn_out, _ = attn(x_norm, x_norm, x_norm, key_padding_mask=attention_mask)
            x = x + attn_out
            
            # Feedforward block
            x = x + ff(norm2(x))
        
        # Return to original shape [batch_size, seq_len, dim]
        return x.transpose(0, 1)

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim=105):
        super(Classifier, self).__init__()
        # Model dimensions
        self.hidden_dim = 256
        self.num_slots = 8  # 4 top quarks + potential extra slots for background
        self.num_heads = 8
        self.transformer_depth = 3
        self.slot_dim = 256
        self.object_feature_dim = 9  # 5 basic features + MT + 3 KNN distances
        
        # Object feature embedding
        self.object_embedding = nn.Sequential(
            nn.Linear(self.object_feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Global feature embedding
        self.global_embedding = nn.Sequential(
            nn.Linear(2, self.hidden_dim // 2),  # ETmiss and phi_ETmiss
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )
        
        # Transformer for learning contextual object representations
        self.transformer = TransformerEncoder(
            dim=self.hidden_dim,
            depth=self.transformer_depth,
            heads=self.num_heads,
            dim_head=self.hidden_dim // self.num_heads,
            mlp_dim=self.hidden_dim * 4,
            dropout=0.1
        )
        
        # Slot Attention for grouping objects
        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=self.hidden_dim,
            iters=3,
            hidden_dim=self.hidden_dim * 2
        )
        
        # Slot post-processing
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # Final classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * (self.num_slots + 1), self.hidden_dim),  # +1 for global features
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, x):
        if isinstance(x, dict):
            # Using preprocessed input
            preprocessed = x
        else:
            # Process raw input through preprocessor
            # Note: This is expected to be handled by the PreprocessModule
            raise ValueError("Input should be preprocessed")
        
        # Extract components from preprocessed input
        object_feats = preprocessed["object_features"]  # [batch_size, max_objects, 9]
        valid_mask = preprocessed["valid_mask"]  # [batch_size, max_objects, 1]
        global_feats = preprocessed["global_features"]  # [batch_size, 2]
        
        batch_size = object_feats.shape[0]
        
        # Embed object features
        obj_embeddings = self.object_embedding(object_feats)  # [batch_size, max_objects, hidden_dim]
        
        # Apply transformer to get contextual object representations
        # Use valid_mask to ignore padding
        obj_embeddings = self.transformer(obj_embeddings, mask=valid_mask)
        
        # Apply Slot Attention to group objects
        slots, attn = self.slot_attention(obj_embeddings, mask=valid_mask)  # [batch_size, num_slots, hidden_dim]
        
        # Process slots with MLP
        slots = self.slot_mlp(slots)  # [batch_size, num_slots, hidden_dim]
        
        # Embed global features
        global_embedding = self.global_embedding(global_feats)  # [batch_size, hidden_dim]
        
        # Combine slot representations with global features
        slots_flat = slots.reshape(batch_size, -1)  # [batch_size, num_slots * hidden_dim]
        combined = torch.cat([slots_flat, global_embedding], dim=1)  # [batch_size, (num_slots+1) * hidden_dim]
        
        # Final classification
        logits = self.classifier(combined).squeeze(-1)
        
        return logits

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    # Metric tracking
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_val_auc = 0.0
    
    # Extract preprocessor from the first batch
    preproc = next(iter(train_loader))[0]
    preproc = model.preprocess if hasattr(model, 'preprocess') else None
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()
            
            # Preprocess data if not already done
            if preproc is not None:
                batch_x = preproc(batch_x)
                
            # Forward pass
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            preds = torch.sigmoid(logits) > 0.5
            correct = (preds == batch_y.bool()).sum().item()
            epoch_loss += loss.item() * len(batch_y)
            epoch_correct += correct
            total_samples += len(batch_y)
            
            # Store predictions for AUC calculation
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_loss /= total_samples
        epoch_acc = epoch_correct / total_samples
        training_loss.append(epoch_loss)
        training_acc.append(epoch_acc)
        
        # Calculate AUC for training
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        train_auc = roc_auc_score(all_labels, all_preds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()
                
                # Preprocess data if not already done
                if preproc is not None:
                    batch_x = preproc(batch_x)
                    
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                # Track metrics
                preds = torch.sigmoid(logits) > 0.5
                correct = (preds == batch_y.bool()).sum().item()
                val_loss += loss.item() * len(batch_y)
                val_correct += correct
                val_samples += len(batch_y)
                
                # Store predictions for AUC calculation
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_labels.append(batch_y.cpu().numpy())
        
        # Calculate validation metrics
        val_loss /= val_samples
        val_acc = val_correct / val_samples
        validation_loss.append(val_loss)
        validation_acc.append(val_acc)
        
        # Calculate AUC for validation
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Train AUC: {train_auc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
    
    # Load best model
    if best_val_auc > 0:
        model.load_state_dict(best_model_state)
        model = model.to(device)
    
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
    batch_size = 64  # Use a smaller batch size to accommodate the transformer
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size)

    # Model Initialization
    model = Classifier()
    
    # Register preprocessor with model for inference
    model.preprocess = preproc

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