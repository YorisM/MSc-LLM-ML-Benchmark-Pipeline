# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

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
        if 'num_objects' in kwargs:
            self.register_buffer("num_objects", kwargs["num_objects"])
        if 'feature_means' in kwargs:
            self.register_buffer("feature_means", kwargs["feature_means"])
        if 'feature_stds' in kwargs:
            self.register_buffer("feature_stds", kwargs["feature_stds"])
        if 'feature_mask' in kwargs:
            self.register_buffer("feature_mask", kwargs["feature_mask"])
        if 'obj_mask' in kwargs:
            self.register_buffer("obj_mask", kwargs["obj_mask"])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract MET and phi MET (first two features)
        met_features = x[:, :2].clone()
        
        # Normalize MET features
        met_features = (met_features - self.feature_means[:2]) / (self.feature_stds[:2] + 1e-8)
        
        # Restructure the remaining features into objects
        batch_size = x.shape[0]
        objects = []
        
        # Process each object (starting from index 2, with 5 features per object)
        for i in range(self.num_objects):
            idx = 2 + i * 5
            obj_id = x[:, idx].clone().unsqueeze(1)  # Object identifier
            
            # Skip if this is a padding object (all zeros)
            if not self.obj_mask[i]:
                continue
                
            # Extract the object's features (E, pT, eta, phi)
            obj_features = x[:, idx+1:idx+5].clone()
            
            # Normalize object features
            obj_features = (obj_features - self.feature_means[idx+1:idx+5]) / (self.feature_stds[idx+1:idx+5] + 1e-8)
            
            # Compute 4-momentum components (px, py, pz, E)
            pt = obj_features[:, 1]  # pT
            eta = obj_features[:, 2]  # eta
            phi = obj_features[:, 3]  # phi
            e = obj_features[:, 0]    # E
            
            # Convert to Cartesian coordinates
            px = pt * torch.cos(phi)
            py = pt * torch.sin(phi)
            pz = pt * torch.sinh(eta)
            
            # Create 4-vector [E, px, py, pz]
            four_vector = torch.stack([e, px, py, pz], dim=1)
            
            # Include object ID and 4-vector
            obj_data = torch.cat([obj_id, four_vector], dim=1)
            objects.append(obj_data)
        
        # Stack all objects
        if objects:
            all_objects = torch.stack(objects, dim=1)  # [batch_size, n_objects, 5]
            
            # Sort objects by pT (descending)
            pt_values = all_objects[:, :, 2]**2 + all_objects[:, :, 3]**2  # px^2 + py^2
            _, indices = torch.sort(pt_values, dim=1, descending=True)
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand_as(indices)
            all_objects = all_objects[batch_indices, indices]
            
            # Flatten for output
            all_objects = all_objects.reshape(batch_size, -1)
            return torch.cat([met_features, all_objects], dim=1)
        else:
            return met_features

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=128):
    # Calculate the number of objects in the data
    num_features = X_train.shape[1]
    num_objects = (num_features - 2) // 5  # First two are MET and phi_MET
    
    # Calculate mean and std for normalization
    # Replace extreme values for stability
    X_train_clean = X_train.clone()
    mask = torch.abs(X_train_clean) > 1e6
    X_train_clean[mask] = 0.0
    
    feature_means = X_train_clean.mean(dim=0)
    feature_stds = X_train_clean.std(dim=0)
    feature_stds[feature_stds < 1e-8] = 1.0
    
    # Create a mask for valid features
    feature_mask = (feature_stds > 1e-8).float()
    
    # Identify non-padding objects (objects with non-zero values)
    obj_mask = torch.zeros(num_objects, dtype=torch.bool)
    for i in range(num_objects):
        idx = 2 + i * 5
        obj_data = X_train[:, idx:idx+5]
        # Object is valid if it has non-zero data in any sample
        obj_mask[i] = torch.any(torch.abs(obj_data).sum(dim=1) > 1e-8)
    
    # Create preprocessor
    preproc = PreprocessModule(
        num_objects=torch.tensor(num_objects),
        feature_means=feature_means,
        feature_stds=feature_stds,
        feature_mask=feature_mask,
        obj_mask=obj_mask
    )

    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class LorentzLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LorentzLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias

class LorentzInvariantNetwork(nn.Module):
    def __init__(self, hidden_dim=64):
        super(LorentzInvariantNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Projections for each 4-vector component
        self.project_e = nn.Linear(1, hidden_dim)
        self.project_p = nn.Linear(3, hidden_dim)
        
        # MLP for further processing
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )
        
    def forward(self, four_vectors):
        # four_vectors shape: [batch_size, n_particles, 4] where each vector is [E, px, py, pz]
        e = four_vectors[:, :, 0:1]  # Energy component
        p = four_vectors[:, :, 1:4]  # Momentum components
        
        # Project each component
        h_e = self.project_e(e)
        h_p = self.project_p(p)
        
        # Combine with Lorentz-invariant structure
        h = h_e + h_p
        
        # Apply MLP
        h = self.mlp(h)
        
        return h

class ParticleMessagePassing(nn.Module):
    def __init__(self, hidden_dim=64, num_message_passing=3):
        super(ParticleMessagePassing, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_message_passing = num_message_passing
        
        # Edge network to compute interactions between particles
        self.edge_network = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )
        
        # Node update network
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, n_particles, hidden_dim]
        # mask shape: [batch_size, n_particles] - True for real particles, False for padding
        batch_size, n_particles, _ = x.shape
        
        # Initialize hidden state
        h = x
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [batch_size, n_particles, 1]
            h = h * mask
        
        # Message passing iterations
        for _ in range(self.num_message_passing):
            # Compute all pairwise interactions
            h_i = h.unsqueeze(2).expand(batch_size, n_particles, n_particles, self.hidden_dim)
            h_j = h.unsqueeze(1).expand(batch_size, n_particles, n_particles, self.hidden_dim)
            
            # Concatenate features for edge network
            edge_input = torch.cat([h_i, h_j], dim=-1)
            
            # Compute edge features
            edge_features = self.edge_network(edge_input.view(-1, 2 * self.hidden_dim))
            edge_features = edge_features.view(batch_size, n_particles, n_particles, self.hidden_dim)
            
            # Apply mask if provided
            if mask is not None:
                edge_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # [batch_size, n_particles, n_particles, 1]
                edge_features = edge_features * edge_mask
            
            # Aggregate messages (sum over neighbors)
            messages = edge_features.sum(dim=2)  # [batch_size, n_particles, hidden_dim]
            
            # Update node features
            node_input = torch.cat([h, messages], dim=-1)  # [batch_size, n_particles, 2*hidden_dim]
            h_new = self.node_update(node_input.view(-1, 2 * self.hidden_dim))
            h_new = h_new.view(batch_size, n_particles, self.hidden_dim)
            
            # Apply mask if provided
            if mask is not None:
                h_new = h_new * mask
            
            # Residual connection
            h = h + h_new
        
        return h

class LorentzEquivariantLayer(nn.Module):
    def __init__(self, hidden_dim=64):
        super(LorentzEquivariantLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable weights for Lorentz-equivariant transformations
        self.weights = nn.Parameter(torch.randn(hidden_dim, 4, 4))
        
        # Initialize weights to be close to identity in Lorentz space
        for i in range(hidden_dim):
            # Minkowski metric-like initialization
            self.weights.data[i, 0, 0] = 1.0  # Time-time component
            self.weights.data[i, 1:, 1:] = -0.1 * torch.eye(3)  # Space-space components
        
    def forward(self, vectors):
        # vectors shape: [batch_size, n_particles, 4]
        batch_size, n_particles, _ = vectors.shape
        
        # Expand vectors and weights for broadcasting
        vectors_expanded = vectors.unsqueeze(-1)  # [batch_size, n_particles, 4, 1]
        
        # Apply Lorentz transformations for each dimension in hidden_dim
        transformed_vectors = []
        for i in range(self.hidden_dim):
            # Apply Lorentz transformation
            transform = self.weights[i]  # [4, 4]
            transformed = torch.matmul(transform, vectors_expanded).squeeze(-1)  # [batch_size, n_particles, 4]
            transformed_vectors.append(transformed)
        
        # Stack along new dimension to get [batch_size, n_particles, hidden_dim, 4]
        transformed_vectors = torch.stack(transformed_vectors, dim=2)
        
        return transformed_vectors

class MetFeatureProcessor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(MetFeatureProcessor, self).__init__()
        self.processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, met_features):
        return self.processor(met_features)

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_objects=20):
        super(Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        
        # Process MET features
        self.met_processor = MetFeatureProcessor(input_dim=2, hidden_dim=hidden_dim)
        
        # Lorentz-invariant network for object features
        self.lin = LorentzInvariantNetwork(hidden_dim=hidden_dim)
        
        # Lorentz-equivariant layer
        self.equivariant = LorentzEquivariantLayer(hidden_dim=hidden_dim//4)
        
        # Message passing between particles
        self.message_passing = ParticleMessagePassing(hidden_dim=hidden_dim, num_message_passing=3)
        
        # Global pooling MLP
        self.global_pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def extract_objects(self, x):
        # First 2 features are MET-related
        met_features = x[:, :2]
        
        # Process objects - each object has 5 features (id + 4-vector)
        # Objects are flattened in the input, so we reshape
        batch_size = x.shape[0]
        obj_data = x[:, 2:]
        
        # Maximum number of objects based on input dimension
        max_objects = (x.shape[1] - 2) // 5
        
        # Initialize tensors to store objects and mask
        objects = torch.zeros(batch_size, max_objects, 5, device=x.device)
        mask = torch.zeros(batch_size, max_objects, dtype=torch.bool, device=x.device)
        
        # Fill in the objects and determine which are real (non-padding)
        for i in range(max_objects):
            if 2 + i*5 < x.shape[1]:
                # Extract object features
                obj = obj_data[:, i*5:(i+1)*5]
                
                if obj.shape[1] == 5:  # Ensure we have complete object data
                    objects[:, i, :] = obj
                    # Object is valid if it has non-zero magnitude
                    mask[:, i] = (obj.abs().sum(dim=1) > 1e-8)
        
        return met_features, objects, mask
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract MET features and objects
        met_features, objects, mask = self.extract_objects(x)
        
        # Process MET features
        met_repr = self.met_processor(met_features)
        
        # Split objects into ID and 4-vectors [E, px, py, pz]
        object_ids = objects[:, :, 0:1]  # [batch_size, n_objects, 1]
        four_vectors = objects[:, :, 1:5]  # [batch_size, n_objects, 4]
        
        # Apply Lorentz-invariant network to 4-vectors
        particle_repr = self.lin(four_vectors)  # [batch_size, n_objects, hidden_dim]
        
        # Apply mask to zero out padding
        particle_repr = particle_repr * mask.unsqueeze(-1).float()
        
        # Apply message passing between particles
        particle_repr = self.message_passing(particle_repr, mask)
        
        # Apply Lorentz-equivariant transformations
        equivariant_repr = self.equivariant(four_vectors)  # [batch_size, n_objects, hidden_dim//4, 4]
        
        # Contract the last dimension to get [batch_size, n_objects, hidden_dim//4]
        # Using the Minkowski metric (-1, 1, 1, 1) for contraction
        minkowski_metric = torch.tensor([1, -1, -1, -1], device=x.device)
        equivariant_repr = (equivariant_repr * minkowski_metric).sum(dim=-1)
        
        # Apply mask to zero out padding
        equivariant_repr = equivariant_repr * mask.unsqueeze(-1).float()
        
        # Concatenate to regular representation
        particle_repr = torch.cat([particle_repr, equivariant_repr], dim=-1)
        
        # Global pooling: sum over particles
        global_repr = particle_repr.sum(dim=1)  # [batch_size, hidden_dim]
        
        # Process global representation
        global_repr = self.global_pool_mlp(global_repr)
        
        # Combine with MET representation
        combined_repr = torch.cat([global_repr, met_repr], dim=1)
        
        # Final classification
        logits = self.classifier(combined_repr).squeeze(-1)
        return logits

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use Binary Cross Entropy with Logits Loss
    criterion = nn.BCEWithLogitsLoss()
    
    # AdamW optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5, verbose=True
    )
    
    # Initialize metric tracking
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_val_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_true = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch.float())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record batch metrics
            train_losses.append(loss.item())
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_true.extend(y_batch.detach().cpu().numpy())
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                
                # Forward pass
                val_logits = model(X_val)
                val_loss = criterion(val_logits, y_val.float())
                
                # Record batch metrics
                val_losses.append(val_loss.item())
                val_preds.extend(torch.sigmoid(val_logits).cpu().numpy())
                val_true.extend(y_val.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        
        # Convert predictions to binary for accuracy calculation
        train_binary_preds = (np.array(train_preds) > 0.5).astype(int)
        val_binary_preds = (np.array(val_preds) > 0.5).astype(int)
        
        epoch_train_acc = accuracy_score(train_true, train_binary_preds)
        epoch_val_acc = accuracy_score(val_true, val_binary_preds)
        
        # Calculate AUC score for validation set
        val_auc = roc_auc_score(val_true, val_preds)
        
        # Update LR scheduler based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Record epoch metrics
        training_loss.append(epoch_train_loss)
        validation_loss.append(epoch_val_loss)
        training_acc.append(epoch_train_acc)
        validation_acc.append(epoch_val_acc)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
              f"Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}, "
              f"Val AUC: {val_auc:.4f}")
    
    # Load best model
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
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=256)

    # Model Initialization
    sample_X, _ = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[1], hidden_dim=128)

    # Training
    epochs = 1 if dryrun else 15

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)

    if not dryrun:
        # determine base name & script directory
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
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