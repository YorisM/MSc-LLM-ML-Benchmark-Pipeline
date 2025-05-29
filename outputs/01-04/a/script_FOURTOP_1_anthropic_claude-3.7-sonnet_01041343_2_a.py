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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import math

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a binary classifier for particle physics events')
parser.add_argument('--dryrun', action='store_true', help='Run a quick test with reduced data')
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic data for dry run
def generate_synthetic_data(n_samples=5000, n_features=106):
    X = torch.randn(n_samples, n_features)
    # First column is weight, second is E_T_miss, third is phi_E_T_miss
    # Make weights positive
    X[:, 0] = torch.abs(X[:, 0])
    # Create binary labels
    y = torch.randint(0, 2, (n_samples,)).float()
    return X, y

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature Engineering
def process_features(X):
    # Extract basic features
    weight = X[:, 0].reshape(-1, 1)  # Event weight
    et_miss = X[:, 1].reshape(-1, 1)  # Missing transverse energy
    phi_et_miss = X[:, 2].reshape(-1, 1)  # Phi of missing transverse energy
    
    # The rest of the data contains objects with their properties
    # Each object has 5 values: obj_id, E, p_T, eta, phi
    object_data = X[:, 3:]
    
    # Reshape to get objects and their properties
    n_samples = X.shape[0]
    n_objects = (X.shape[1] - 3) // 5  # Number of potential objects
    
    # Initialize arrays to hold derived features
    total_energy = torch.zeros(n_samples, 1, device=X.device)
    total_pt = torch.zeros(n_samples, 1, device=X.device)
    n_jets = torch.zeros(n_samples, 1, device=X.device)
    n_leptons = torch.zeros(n_samples, 1, device=X.device)
    ht = torch.zeros(n_samples, 1, device=X.device)  # Sum of jet pT
    
    # Top-level features for the first 6 highest pT jets and leptons
    max_objects_to_keep = 6
    jet_pts = torch.zeros(n_samples, max_objects_to_keep, device=X.device)
    jet_etas = torch.zeros(n_samples, max_objects_to_keep, device=X.device)
    jet_phis = torch.zeros(n_samples, max_objects_to_keep, device=X.device)
    jet_energies = torch.zeros(n_samples, max_objects_to_keep, device=X.device)
    
    lepton_pts = torch.zeros(n_samples, max_objects_to_keep, device=X.device)
    lepton_etas = torch.zeros(n_samples, max_objects_to_keep, device=X.device)
    lepton_phis = torch.zeros(n_samples, max_objects_to_keep, device=X.device)
    lepton_energies = torch.zeros(n_samples, max_objects_to_keep, device=X.device)
    
    # Process objects batch-wise
    for i in range(n_objects):
        idx = 3 + i * 5
        obj_id = X[:, idx]
        energy = X[:, idx + 1]
        pt = X[:, idx + 2]
        eta = X[:, idx + 3]
        phi = X[:, idx + 4]
        
        # Calculate total energy and pT only for valid objects (non-zero)
        valid_obj = (obj_id != 0)
        total_energy += torch.where(valid_obj, energy, torch.zeros_like(energy)).reshape(-1, 1)
        total_pt += torch.where(valid_obj, pt, torch.zeros_like(pt)).reshape(-1, 1)
        
        # Identify jets (obj_id = 1) and leptons (obj_id = 2 or 3)
        is_jet = (obj_id == 1)
        is_lepton = (obj_id == 2) | (obj_id == 3)
        
        n_jets += is_jet.float().reshape(-1, 1)
        n_leptons += is_lepton.float().reshape(-1, 1)
        
        # Add to HT if it's a jet
        ht += torch.where(is_jet, pt, torch.zeros_like(pt)).reshape(-1, 1)
        
        # Store properties of highest pT jets and leptons
        for j in range(n_samples):
            if is_jet[j] and pt[j] > 0:
                # Find the first empty slot or the slot with lower pT
                for k in range(max_objects_to_keep):
                    if jet_pts[j, k] == 0 or pt[j] > jet_pts[j, k]:
                        # Shift everything down to make room
                        if k < max_objects_to_keep - 1:
                            jet_pts[j, k+1:] = jet_pts[j, k:-1]
                            jet_etas[j, k+1:] = jet_etas[j, k:-1]
                            jet_phis[j, k+1:] = jet_phis[j, k:-1]
                            jet_energies[j, k+1:] = jet_energies[j, k:-1]
                        # Insert new value
                        jet_pts[j, k] = pt[j]
                        jet_etas[j, k] = eta[j]
                        jet_phis[j, k] = phi[j]
                        jet_energies[j, k] = energy[j]
                        break
            elif is_lepton[j] and pt[j] > 0:
                # Similar logic for leptons
                for k in range(max_objects_to_keep):
                    if lepton_pts[j, k] == 0 or pt[j] > lepton_pts[j, k]:
                        if k < max_objects_to_keep - 1:
                            lepton_pts[j, k+1:] = lepton_pts[j, k:-1]
                            lepton_etas[j, k+1:] = lepton_etas[j, k:-1]
                            lepton_phis[j, k+1:] = lepton_phis[j, k:-1]
                            lepton_energies[j, k+1:] = lepton_energies[j, k:-1]
                        lepton_pts[j, k] = pt[j]
                        lepton_etas[j, k] = eta[j]
                        lepton_phis[j, k] = phi[j]
                        lepton_energies[j, k] = energy[j]
                        break
    
    # Calculate delta R between highest pT jets and leptons (up to 3 pairs)
    deltaR_features = torch.zeros(n_samples, 9, device=X.device)  # 3 pairs x 3 features (dR, dEta, dPhi)
    for i in range(min(3, max_objects_to_keep)):
        for j in range(min(3, max_objects_to_keep)):
            if jet_pts[:, i].sum() > 0 and lepton_pts[:, j].sum() > 0:  # Only if both objects exist
                deta = jet_etas[:, i] - lepton_etas[:, j]
                dphi = torch.abs(jet_phis[:, i] - lepton_phis[:, j])
                # Adjust dphi to be in [-pi, pi]
                dphi = torch.where(dphi > math.pi, 2 * math.pi - dphi, dphi)
                dr = torch.sqrt(deta**2 + dphi**2)
                idx = i * 3 + j
                deltaR_features[:, idx] = dr
                deltaR_features[:, idx + 1] = deta
                deltaR_features[:, idx + 2] = dphi
    
    # Invariant mass of leading 2 jets and leading 2 leptons
    mass_features = torch.zeros(n_samples, 2, device=X.device)
    
    # Invariant mass of jet 0 and jet 1
    if (jet_pts[:, 0].sum() > 0) and (jet_pts[:, 1].sum() > 0):
        E1, E2 = jet_energies[:, 0], jet_energies[:, 1]
        px1, px2 = jet_pts[:, 0] * torch.cos(jet_phis[:, 0]), jet_pts[:, 1] * torch.cos(jet_phis[:, 1])
        py1, py2 = jet_pts[:, 0] * torch.sin(jet_phis[:, 0]), jet_pts[:, 1] * torch.sin(jet_phis[:, 1])
        pz1 = jet_pts[:, 0] * torch.sinh(jet_etas[:, 0])
        pz2 = jet_pts[:, 1] * torch.sinh(jet_etas[:, 1])
        
        m_squared = (E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2
        mass_features[:, 0] = torch.sqrt(torch.clamp(m_squared, min=0))
    
    # Invariant mass of lepton 0 and lepton 1
    if (lepton_pts[:, 0].sum() > 0) and (lepton_pts[:, 1].sum() > 0):
        E1, E2 = lepton_energies[:, 0], lepton_energies[:, 1]
        px1 = lepton_pts[:, 0] * torch.cos(lepton_phis[:, 0])
        px2 = lepton_pts[:, 1] * torch.cos(lepton_phis[:, 1])
        py1 = lepton_pts[:, 0] * torch.sin(lepton_phis[:, 0])
        py2 = lepton_pts[:, 1] * torch.sin(lepton_phis[:, 1])
        pz1 = lepton_pts[:, 0] * torch.sinh(lepton_etas[:, 0])
        pz2 = lepton_pts[:, 1] * torch.sinh(lepton_etas[:, 1])
        
        m_squared = (E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2
        mass_features[:, 1] = torch.sqrt(torch.clamp(m_squared, min=0))
    
    # Combine all features
    features = torch.cat([
        weight,  # Event weight
        et_miss,  # Missing transverse energy
        phi_et_miss,  # Phi of missing transverse energy
        total_energy,  # Total energy of the event
        total_pt,  # Total pT of the event
        n_jets,  # Number of jets
        n_leptons,  # Number of leptons
        ht,  # HT (scalar sum of jet pT)
        jet_pts,  # pT of leading jets
        jet_etas,  # eta of leading jets
        jet_phis,  # phi of leading jets
        jet_energies,  # energy of leading jets
        lepton_pts,  # pT of leading leptons
        lepton_etas,  # eta of leading leptons
        lepton_phis,  # phi of leading leptons
        lepton_energies,  # energy of leading leptons
        deltaR_features,  # deltaR, deltaEta, deltaPhi between leading objects
        mass_features,  # Invariant masses
    ], dim=1)
    
    return features

# Define the neural network architecture
class TTTTClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64]):
        super(TTTTClassifier, self).__init__()
        
        # Sequential layers with batch normalization and dropout
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            current_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5):
    best_val_auc = 0.0
    epochs_no_improve = 0
    best_model_state = model.state_dict()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)  # Average loss per batch
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                
                val_predictions.extend(outputs.squeeze().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)  # Average loss per batch
        val_auc = roc_auc_score(val_labels, val_predictions)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}')
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs. Best validation AUC: {best_val_auc:.4f}')
                model.load_state_dict(best_model_state)  # Restore best model
                break
    
    return model, best_val_auc

# Main function
def main():
    # Load or generate data
    if args.dryrun:
        print("Running in dry run mode with synthetic data...")
        X_train, Y_train = generate_synthetic_data(n_samples=5000)
        X_val, Y_val = generate_synthetic_data(n_samples=1000)
    else:
        # Assume the data is already loaded in these variables
        # In a real scenario, you would load the data here
        try:
            # Try to load the data variables that are assumed to exist
            X_train
            Y_train
            X_val
            Y_val
            print(f"Data loaded. Shapes: X_train {X_train.shape}, Y_train {Y_train.shape}, X_val {X_val.shape}, Y_val {Y_val.shape}")
        except NameError:
            print("Data variables not found. Switching to dry run mode with synthetic data.")
            X_train, Y_train = generate_synthetic_data(n_samples=5000)
            X_val, Y_val = generate_synthetic_data(n_samples=1000)
    
    # Move data to device
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    
    # Process features
    print("Processing features...")
    X_train_proc = process_features(X_train)
    X_val_proc = process_features(X_val)
    
    # Create data loaders
    batch_size = 128 if not args.dryrun else 32
    train_dataset = TensorDataset(X_train_proc, Y_train)
    val_dataset = TensorDataset(X_val_proc, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = X_train_proc.shape[1]
    model = TTTTClassifier(input_size=input_size).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    print("Training model...")
    model, best_val_auc = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer,
        num_epochs=15 if args.dryrun else 50
    )
    
    # Final evaluation
    model.eval()
    val_predictions = []
    
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            val_predictions.extend(outputs.squeeze().cpu().numpy())
    
    final_auc = roc_auc_score(Y_val.cpu().numpy(), val_predictions)
    print(f'Final validation AUC: {final_auc:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'tttt_classifier_model.pth')
    print("Model saved as tttt_classifier_model.pth")

if __name__ == "__main__":
    main()