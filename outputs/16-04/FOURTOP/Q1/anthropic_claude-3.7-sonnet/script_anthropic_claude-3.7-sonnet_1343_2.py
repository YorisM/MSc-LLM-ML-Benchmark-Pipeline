# ----- FREE SECTION: Import Libraries -----
import numpy as np
import pandas as pd
import math
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.nn.utils import weight_norm

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
    # Feature engineering: Extract physics-relevant features
    X_train_processed = extract_features(X_train)
    X_val_processed = extract_features(X_val)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = torch.tensor(scaler.fit_transform(X_train_processed), dtype=torch.float32)
    X_val_scaled = torch.tensor(scaler.transform(X_val_processed), dtype=torch.float32)
    
    # Create DataLoader objects
    batch_size = 256
    train_dataset = TensorDataset(X_train_scaled, Y_train)
    val_dataset = TensorDataset(X_val_scaled, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def extract_features(X):
    # Convert to numpy for easier manipulation
    X_np = X.numpy()
    
    # Extract basic features (weight, ET_miss, phi_ET_miss)
    basic_features = X_np[:, :3]
    
    # The rest of the tensor contains object data in groups of 5 values
    # (obj_id, E, pT, eta, phi)
    n_samples = X_np.shape[0]
    
    # Extract particle features
    engineered_features = []
    
    for i in range(n_samples):
        # Basic features from the event
        weight = X_np[i, 0]
        et_miss = X_np[i, 1] 
        phi_et_miss = X_np[i, 2]
        
        # Initialize particle counters and kinematic sums
        leptons = []
        jets = []
        bjets = []
        event_features = []
        
        # Process each object in the event
        for j in range(3, X_np.shape[1], 5):
            obj_id = X_np[i, j]
            # Skip if we've reached padding (zeros)
            if obj_id == 0 and X_np[i, j+1] == 0 and X_np[i, j+2] == 0:
                continue
                
            # Extract kinematic properties
            E = X_np[i, j+1]
            pT = X_np[i, j+2]
            eta = X_np[i, j+3]
            phi = X_np[i, j+4]
            
            # Skip invalid objects
            if E <= 0 or pT <= 0:
                continue
                
            # Categorize objects based on obj_id
            # Assuming: 1=electron, 2=muon, 3=jet, 4=b-jet
            # (These categories are assumptions based on typical physics conventions)
            if obj_id in [1, 2]:  # leptons (e or μ)
                leptons.append((E, pT, eta, phi))
            elif obj_id == 3:     # jets
                jets.append((E, pT, eta, phi))
            elif obj_id == 4:     # b-jets
                bjets.append((E, pT, eta, phi))
        
        # Count objects
        n_leptons = len(leptons)
        n_jets = len(jets)
        n_bjets = len(bjets)
        
        # Calculate event-level features
        total_jet_pt = sum([j[1] for j in jets + bjets]) if (jets or bjets) else 0
        total_lepton_pt = sum([l[1] for l in leptons]) if leptons else 0
        
        # HT (scalar sum of jet pT)
        ht = total_jet_pt
        
        # Calculate missing ET significance
        et_miss_significance = et_miss / math.sqrt(total_jet_pt) if total_jet_pt > 0 else 0
        
        # Calculate angular features
        delta_phi_values = []
        for jet in jets + bjets:
            delta_phi = abs(jet[3] - phi_et_miss)
            # Normalize to [0, π]
            delta_phi = min(delta_phi, 2*math.pi - delta_phi) if delta_phi <= 2*math.pi else delta_phi
            delta_phi_values.append(delta_phi)
        
        min_delta_phi = min(delta_phi_values) if delta_phi_values else math.pi
        
        # Angular separation between leading objects
        delta_R_values = []
        all_objects = leptons + jets + bjets
        for i in range(len(all_objects)):
            for j in range(i+1, len(all_objects)):
                eta1, phi1 = all_objects[i][2], all_objects[i][3]
                eta2, phi2 = all_objects[j][2], all_objects[j][3]
                delta_eta = eta1 - eta2
                delta_phi = abs(phi1 - phi2)
                delta_phi = min(delta_phi, 2*math.pi - delta_phi) if delta_phi <= 2*math.pi else delta_phi
                delta_R = math.sqrt(delta_eta**2 + delta_phi**2)
                delta_R_values.append(delta_R)
        
        mean_delta_R = np.mean(delta_R_values) if delta_R_values else 0
        min_delta_R = min(delta_R_values) if delta_R_values else 0
        
        # Extract features from leading objects
        leading_lepton_pt = max([l[1] for l in leptons]) if leptons else 0
        leading_jet_pt = max([j[1] for j in jets]) if jets else 0
        leading_bjet_pt = max([b[1] for b in bjets]) if bjets else 0
        
        # Build feature vector
        event_features = [
            weight,
            et_miss,
            et_miss_significance,
            min_delta_phi,
            n_leptons,
            n_jets,
            n_bjets,
            ht,
            total_lepton_pt,
            mean_delta_R,
            min_delta_R,
            leading_lepton_pt,
            leading_jet_pt,
            leading_bjet_pt,
            n_leptons * n_bjets  # Interaction term
        ]
        
        engineered_features.append(event_features)
    
    return np.array(engineered_features)

# ----- FREE SECTION: Binary Classifier Definition -----
class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.linear1 = weight_norm(nn.Linear(in_features, hidden_features))
        self.norm2 = nn.LayerNorm(hidden_features)
        self.linear2 = weight_norm(nn.Linear(hidden_features, in_features))
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = F.selu(self.linear1(out))
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.linear2(out)
        return out + identity

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        
        # Expand input dimension to a larger hidden state
        self.input_layer = nn.Sequential(
            weight_norm(nn.Linear(input_dim, 128)),
            nn.SELU(),
            nn.Dropout(0.2)
        )
        
        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(128, 256) for _ in range(4)
        ])
        
        # Prediction head
        self.output_layer = nn.Sequential(
            nn.LayerNorm(128),
            weight_norm(nn.Linear(128, 64)),
            nn.SELU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(64, 1))
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Final prediction
        x = self.output_layer(x)
        return x.squeeze(1)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define loss function (BCEWithLogitsLoss for binary classification)
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam optimizer with learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    # Initialize metric tracking lists
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        train_pred_list = []
        train_target_list = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_train_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            epoch_train_correct += (predicted == targets).sum().item()
            epoch_train_total += targets.size(0)
            
            # Save predictions for AUC calculation
            train_pred_list.append(torch.sigmoid(outputs).detach().cpu().numpy())
            train_target_list.append(targets.detach().cpu().numpy())
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0
        val_pred_list = []
        val_target_list = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update metrics
                epoch_val_loss += loss.item() * inputs.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                epoch_val_correct += (predicted == targets).sum().item()
                epoch_val_total += targets.size(0)
                
                # Save predictions for AUC calculation
                val_pred_list.append(torch.sigmoid(outputs).detach().cpu().numpy())
                val_target_list.append(targets.detach().cpu().numpy())
        
        # Calculate epoch metrics
        train_loss = epoch_train_loss / epoch_train_total
        val_loss = epoch_val_loss / epoch_val_total
        train_acc = epoch_train_correct / epoch_train_total
        val_acc = epoch_val_correct / epoch_val_total
        
        # Calculate AUC
        train_preds = np.concatenate(train_pred_list)
        train_targets = np.concatenate(train_target_list)
        val_preds = np.concatenate(val_pred_list)
        val_targets = np.concatenate(val_target_list)
        
        train_auc = roc_auc_score(train_targets, train_preds)
        val_auc = roc_auc_score(val_targets, val_preds)
        
        # Update scheduler based on validation AUC
        scheduler.step(val_auc)
        
        # Save metrics
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        training_acc.append(train_acc)
        validation_acc.append(val_acc)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Train Acc: {train_acc:.4f} | '
              f'Val Acc: {val_acc:.4f} | '
              f'Train AUC: {train_auc:.4f} | '
              f'Val AUC: {val_auc:.4f}')
    
    # Load best model state
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
    train_loader, val_loader = preprocess_data(X_train, Y_train, X_val, Y_val)

    # Get input dimension from the first batch
    for inputs, _ in train_loader:
        input_dim = inputs.shape[1]
        break

    # Model Initialization
    model = Classifier(input_dim=input_dim)

    # Training
    epochs = 1 if dryrun else 10

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)

    if not dryrun:
        # Save Model
        model_filename = sys.argv[0].replace(".py", "") + "_model.pth"
        torch.save(trained_model.state_dict(), model_filename)

        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, "Loss", "training_loss.png")
        plot_and_save(training_acc, validation_acc, "Accuracy", "training_accuracy.png")

        print("Full run complete. Outputs and model saved successfully.")
    else:
        print("Dry-run complete. No outputs saved.")

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)