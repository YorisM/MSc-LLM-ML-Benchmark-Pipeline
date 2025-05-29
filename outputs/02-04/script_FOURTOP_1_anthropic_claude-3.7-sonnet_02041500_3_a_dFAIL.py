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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a binary classifier for particle physics data')
parser.add_argument('--dryrun', action='store_true', help='Run with a small subset of data for testing')
args = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Function to preprocess data
def preprocess_data(X):
    # X shape: [batch_size, 106]
    # First 3 features: weight, E_T_miss, phi_{E_t}_miss
    # Rest: obj_id, E, p_T, eta, phi, ...
    
    batch_size = X.shape[0]
    
    # Extract global features (first 3)
    global_features = X[:, :3]
    
    # Extract object features (remaining features)
    object_features = X[:, 3:]
    
    # Reshape to have [batch_size, max_objects, 4]
    # Each object has 4 features: E, p_T, eta, phi
    max_objects = (object_features.shape[1]) // 4
    object_features = object_features.reshape(batch_size, max_objects, 4)
    
    # Create a mask for valid objects (non-zero energy)
    mask = (object_features[:, :, 0] != 0).float()
    
    # Sort objects by pT (descending) within each event
    pT_values = object_features[:, :, 1]  # pT is the second feature
    sorted_indices = torch.argsort(pT_values, dim=1, descending=True)
    
    # Get batch indices
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, max_objects)
    
    # Sort object features
    sorted_features = object_features[batch_indices, sorted_indices]
    sorted_mask = mask[batch_indices, sorted_indices]
    
    # Count valid objects per event
    object_counts = torch.sum(sorted_mask, dim=1).int()
    
    # Calculate derived features
    derived_features = calculate_derived_features(global_features, sorted_features, sorted_mask)
    
    return global_features, sorted_features, sorted_mask, object_counts, derived_features

# Function to calculate derived features
def calculate_derived_features(global_features, object_features, mask):
    batch_size = object_features.shape[0]
    
    # Initialize tensor for derived features
    derived_features = torch.zeros(batch_size, 20, device=object_features.device)
    
    # Extract E_T_miss and phi_E_T_miss from global features
    E_T_miss = global_features[:, 1]
    phi_E_T_miss = global_features[:, 2]
    
    # Total energy and momentum
    valid_objects = object_features * mask.unsqueeze(-1)
    total_energy = torch.sum(valid_objects[:, :, 0], dim=1)  # Sum of E
    total_pT = torch.sum(valid_objects[:, :, 1], dim=1)  # Sum of pT
    
    # Jet multiplicity (using objects with pT > 30000 MeV)
    jet_mask = (object_features[:, :, 1] > 30000).float() * mask
    jet_count = torch.sum(jet_mask, dim=1)
    
    # b-jet multiplicity (assuming objects with pT > 25000 and |eta| < 2.5)
    b_jet_mask = ((object_features[:, :, 1] > 25000) & 
                 (torch.abs(object_features[:, :, 2]) < 2.5)).float() * mask
    b_jet_count = torch.sum(b_jet_mask, dim=1)
    
    # Transverse mass of the system
    px_sum = torch.sum(valid_objects[:, :, 1] * torch.cos(valid_objects[:, :, 3]), dim=1)
    py_sum = torch.sum(valid_objects[:, :, 1] * torch.sin(valid_objects[:, :, 3]), dim=1)
    mT = torch.sqrt(px_sum**2 + py_sum**2)
    
    # HT (scalar sum of jet pT)
    HT = torch.sum(valid_objects[:, :, 1] * jet_mask, dim=1)
    
    # Missing ET ratio to HT
    MET_HT_ratio = E_T_miss / (HT + 1e-8)  # Avoid division by zero
    
    # Angular differences between objects
    delta_phi = torch.zeros(batch_size)
    delta_eta = torch.zeros(batch_size)
    delta_R = torch.zeros(batch_size)
    
    # Consider only the top 4 pT objects (if available)
    for i in range(min(4, object_features.shape[1])):
        for j in range(i+1, min(4, object_features.shape[1])):
            if i < j:
                # Phi difference (ensure it's in [-π, π])
                dphi = torch.abs(object_features[:, i, 3] - object_features[:, j, 3])
                dphi = torch.min(dphi, 2*np.pi - dphi)
                
                # Eta difference
                deta = torch.abs(object_features[:, i, 2] - object_features[:, j, 2])
                
                # ΔR = √(Δη² + Δφ²)
                dr = torch.sqrt(deta**2 + dphi**2)
                
                # Average these differences (weighted by whether both objects are valid)
                both_valid = mask[:, i] * mask[:, j]
                delta_phi += dphi * both_valid
                delta_eta += deta * both_valid
                delta_R += dr * both_valid
    
    # Record derived features
    derived_features[:, 0] = total_energy
    derived_features[:, 1] = total_pT
    derived_features[:, 2] = jet_count
    derived_features[:, 3] = b_jet_count
    derived_features[:, 4] = HT
    derived_features[:, 5] = MET_HT_ratio
    derived_features[:, 6] = mT
    derived_features[:, 7] = delta_phi
    derived_features[:, 8] = delta_eta
    derived_features[:, 9] = delta_R
    
    # Add statistics of the top 5 objects (if available)
    top_count = min(5, object_features.shape[1])
    derived_features[:, 10:10+top_count] = object_features[:, :top_count, 1]  # pT of top objects
    
    # 4-object invariant mass (if at least 4 objects)
    if object_features.shape[1] >= 4:
        # Calculate four-vectors for top 4 objects
        E = object_features[:, :4, 0]
        pT = object_features[:, :4, 1]
        eta = object_features[:, :4, 2]
        phi = object_features[:, :4, 3]
        
        # Convert to px, py, pz
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)
        
        # Sum of four-vectors
        E_sum = torch.sum(E, dim=1)
        px_sum = torch.sum(px, dim=1)
        py_sum = torch.sum(py, dim=1)
        pz_sum = torch.sum(pz, dim=1)
        
        # Invariant mass
        m_inv = torch.sqrt(E_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2)
        derived_features[:, 15] = m_inv
    
    return derived_features

# Define RNN model for processing sequence data
class ParticleRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, global_size=3, derived_size=20):
        super(ParticleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # RNN for processing object sequences
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Process global features
        self.global_fc = nn.Sequential(
            nn.Linear(global_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.ReLU()
        )
        
        # Process derived features
        self.derived_fc = nn.Sequential(
            nn.Linear(derived_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Combine all features and classify
        combined_size = hidden_size*2 + hidden_size//2 + hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, combined_size//2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(combined_size//2, combined_size//4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(combined_size//4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, global_features, object_features, mask, object_counts, derived_features):
        batch_size = object_features.shape[0]
        
        # Process global features
        global_out = self.global_fc(global_features)
        
        # Process derived features
        derived_out = self.derived_fc(derived_features)
        
        # Process object features with RNN
        # Pack padded sequence
        packed_input = pack_padded_sequence(
            object_features, 
            object_counts.cpu(),  # counts need to be on CPU for packing
            batch_first=True, 
            enforce_sorted=False
        )
        
        packed_output, _ = self.rnn(packed_input)
        
        # Unpack sequence
        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Attention mechanism
        attention_weights = self.attention(rnn_output)  # shape: [batch_size, seq_len, 1]
        context_vector = torch.sum(attention_weights * rnn_output, dim=1)  # shape: [batch_size, hidden_size*2]
        
        # Concatenate all feature representations
        combined = torch.cat([context_vector, global_out, derived_out], dim=1)
        
        # Final classification
        output = self.classifier(combined)
        
        return output.squeeze()

# Create data loaders
def create_dataloaders(X_train, Y_train, X_val, Y_val, batch_size=128, dryrun=False):
    # If dryrun, use a small subset of data
    if dryrun:
        X_train = X_train[:1000]
        Y_train = Y_train[:1000]
        X_val = X_val[:200]
        Y_val = Y_val[:200]
    
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=30, patience=5, dryrun=False):
    if dryrun:
        num_epochs = 2
    
    model.to(device)
    best_auc = 0.0
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_aucs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            # Preprocess data
            global_features, object_features, mask, object_counts, derived_features = preprocess_data(batch_X)
            
            optimizer.zero_grad()
            outputs = model(global_features, object_features, mask, object_counts, derived_features)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                
                # Preprocess data
                global_features, object_features, mask, object_counts, derived_features = preprocess_data(batch_X)
                
                outputs = model(global_features, object_features, mask, object_counts, derived_features)
                
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(batch_Y.cpu().numpy())
        
        val_auc = roc_auc_score(val_true, val_preds)
        val_aucs.append(val_auc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val AUC: {val_auc:.4f}')
        
        # Save the best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_aucs)
    plt.title('Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    
    # Load the best model
    model.load_state_dict(best_model_state)
    
    return model, best_auc

# Evaluate model
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            # Preprocess data
            global_features, object_features, mask, object_counts, derived_features = preprocess_data(batch_X)
            
            outputs = model(global_features, object_features, mask, object_counts, derived_features)
            
            all_preds.extend(outputs.cpu().numpy())
            all_true.extend(batch_Y.cpu().numpy())
    
    auc = roc_auc_score(all_true, all_preds)
    return auc, all_preds, all_true

# Main function
def main():
    # For the purposes of this submission, we'll create dummy data
    # In a real scenario, X_train, Y_train, X_val, Y_val would be loaded from files
    if args.dryrun:
        # Create dummy data for testing
        X_train = torch.randn(1000, 106) 
        Y_train = torch.randint(0, 2, (1000,)).float()
        X_val = torch.randn(200, 106)
        Y_val = torch.randint(0, 2, (200,)).float()
        print("Running with dummy data in dryrun mode")
    else:
        # In a real scenario, these would be loaded from files
        # For this exercise, we assume these variables are already defined
        # and properly imported, as specified in the problem description
        try:
            print("Using pre-loaded tensors with shapes:")
            print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
            print(f"X_val: {X_val.shape}, Y_val: {Y_val.shape}")
        except NameError:
            print("Variables not found. Running with dummy data.")
            # Create dummy data for testing
            X_train = torch.randn(10000, 106) 
            Y_train = torch.randint(0, 2, (10000,)).float()
            X_val = torch.randn(2000, 106)
            Y_val = torch.randint(0, 2, (2000,)).float()
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(X_train, Y_train, X_val, Y_val, dryrun=args.dryrun)
    
    # Initialize model, criterion, and optimizer
    model = ParticleRNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model
    model, best_auc = train_model(model, train_loader, val_loader, optimizer, criterion, dryrun=args.dryrun)
    
    # Final evaluation
    final_auc, _, _ = evaluate_model(model, val_loader)
    print(f'Final AUC on validation set: {final_auc:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'particle_classifier_model.pth')
    print("Model saved as particle_classifier_model.pth")

if __name__ == "__main__":
    main()