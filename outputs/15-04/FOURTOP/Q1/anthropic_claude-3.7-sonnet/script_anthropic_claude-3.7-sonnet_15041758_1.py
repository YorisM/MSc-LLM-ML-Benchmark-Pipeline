# ----- FIXED SECTION: Import Libraries -----
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

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
    # Convert to numpy for preprocessing
    X_train_np = X_train.numpy()
    X_val_np = X_val.numpy()
    
    # Extract physics-meaningful features
    n_features = X_train.shape[1]
    
    # First 3 columns are weight, ET_miss, phi_ET_miss
    # Rest are obj_ID, E, pT, eta, phi repeated for each particle
    # Create new features that capture physics information
    
    # Get number of objects per event (non-zero entries)
    def count_objects(X):
        # Starting from index 3 (after weight, ET_miss, phi_ET_miss)
        # Every 5 values represent: obj_ID, E, pT, eta, phi
        obj_count = np.zeros(len(X))
        
        for i in range(len(X)):
            count = 0
            # Start from index 3 and check every 5th value
            for j in range(3, n_features, 5):
                if j+4 < n_features and X[i, j] != 0:  # If obj_ID exists
                    count += 1
            obj_count[i] = count
        
        return obj_count
    
    train_obj_count = count_objects(X_train_np)
    val_obj_count = count_objects(X_val_np)
    
    # Calculate total energy and pT
    def calculate_sums(X):
        total_energy = np.zeros(len(X))
        total_pt = np.zeros(len(X))
        
        for i in range(len(X)):
            energy_sum = 0
            pt_sum = 0
            # Start from index 3 and process every 5 values
            for j in range(3, n_features, 5):
                if j+4 < n_features and X[i, j] != 0:  # Valid object
                    energy_sum += X[i, j+1]  # E is at j+1
                    pt_sum += X[i, j+2]      # pT is at j+2
            
            total_energy[i] = energy_sum
            total_pt[i] = pt_sum
        
        return total_energy, total_pt
    
    train_total_energy, train_total_pt = calculate_sums(X_train_np)
    val_total_energy, val_total_pt = calculate_sums(X_val_np)
    
    # Calculate spatial distribution features
    def calculate_spatial_features(X):
        # Calculate mean and std of eta and phi
        mean_eta = np.zeros(len(X))
        std_eta = np.zeros(len(X))
        mean_phi = np.zeros(len(X))
        std_phi = np.zeros(len(X))
        
        for i in range(len(X)):
            etas = []
            phis = []
            
            # Extract eta and phi values
            for j in range(3, n_features, 5):
                if j+4 < n_features and X[i, j] != 0:  # Valid object
                    etas.append(X[i, j+3])  # eta is at j+3
                    phis.append(X[i, j+4])  # phi is at j+4
            
            if etas:  # Non-empty list
                mean_eta[i] = np.mean(etas)
                std_eta[i] = np.std(etas) if len(etas) > 1 else 0
                mean_phi[i] = np.mean(phis)
                std_phi[i] = np.std(phis) if len(phis) > 1 else 0
        
        return mean_eta, std_eta, mean_phi, std_phi
    
    train_mean_eta, train_std_eta, train_mean_phi, train_std_phi = calculate_spatial_features(X_train_np)
    val_mean_eta, val_std_eta, val_mean_phi, val_std_phi = calculate_spatial_features(X_val_np)
    
    # Identify different particle types and count them
    def count_particle_types(X):
        # Assuming obj_ID represents particle type
        unique_types = set()
        
        # First identify all unique particle types
        for i in range(len(X)):
            for j in range(3, n_features, 5):
                if j < n_features and X[i, j] != 0:
                    unique_types.add(X[i, j])
        
        # Create counters for each type
        type_counts = {}
        for type_id in unique_types:
            type_counts[type_id] = np.zeros(len(X))
        
        # Count occurrences
        for i in range(len(X)):
            for j in range(3, n_features, 5):
                if j < n_features and X[i, j] != 0:
                    type_counts[X[i, j]][i] += 1
        
        return type_counts
    
    train_type_counts = count_particle_types(X_train_np)
    val_type_counts = count_particle_types(X_val_np)
    
    # Create new feature matrices
    def create_feature_matrix(X, obj_count, total_energy, total_pt, mean_eta, std_eta, mean_phi, std_phi, type_counts):
        # Original features + engineered features
        new_features = []
        
        # Keep original features
        new_features.append(X)
        
        # Add engineered features
        new_features.append(obj_count.reshape(-1, 1))
        new_features.append(total_energy.reshape(-1, 1))
        new_features.append(total_pt.reshape(-1, 1))
        new_features.append(mean_eta.reshape(-1, 1))
        new_features.append(std_eta.reshape(-1, 1))
        new_features.append(mean_phi.reshape(-1, 1))
        new_features.append(std_phi.reshape(-1, 1))
        
        # Add ratio of ET_miss to total pT
        et_miss_ratio = X[:, 1] / (total_pt + 1e-8)  # Avoid division by zero
        new_features.append(et_miss_ratio.reshape(-1, 1))
        
        # Add particle type counts
        for type_id, counts in type_counts.items():
            new_features.append(counts.reshape(-1, 1))
        
        # Concatenate all features
        feature_matrix = np.hstack(new_features)
        
        return feature_matrix
    
    X_train_new = create_feature_matrix(
        X_train_np, train_obj_count, train_total_energy, train_total_pt,
        train_mean_eta, train_std_eta, train_mean_phi, train_std_phi, train_type_counts
    )
    
    X_val_new = create_feature_matrix(
        X_val_np, val_obj_count, val_total_energy, val_total_pt,
        val_mean_eta, val_std_eta, val_mean_phi, val_std_phi, val_type_counts
    )
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_new)
    X_val_scaled = scaler.transform(X_val_new)
    
    # Convert back to torch tensors
    X_train_processed = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_processed = torch.tensor(X_val_scaled, dtype=torch.float32)
    
    return X_train_processed, Y_train, X_val_processed, Y_val

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        
        # Define network architecture
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x.squeeze(-1)  # Squeeze to get scalar output

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, X_train, Y_train, X_val, Y_val, epochs):
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create data loaders
    batch_size = 256
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy with logits
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                     patience=3, verbose=True)
    
    # Initialize metrics tracking
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    training_auc = []
    validation_auc = []
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_probs = []
        train_labels = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * inputs.size(0)
            train_probs.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Calculate accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).int()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                
                # Track metrics
                val_loss += loss.item() * inputs.size(0)
                val_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                # Calculate accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).int()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_train_acc = train_correct / train_total
        epoch_val_acc = val_correct / val_total
        
        # Calculate AUC scores
        epoch_train_auc = roc_auc_score(train_labels, train_probs)
        epoch_val_auc = roc_auc_score(val_labels, val_probs)
        
        # Update learning rate based on validation AUC
        scheduler.step(epoch_val_auc)
        
        # Store metrics
        training_loss.append(epoch_train_loss)
        validation_loss.append(epoch_val_loss)
        training_acc.append(epoch_train_acc)
        validation_acc.append(epoch_val_acc)
        training_auc.append(epoch_train_auc)
        validation_auc.append(epoch_val_auc)
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}')
        print(f'Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}')
        print(f'Train AUC: {epoch_train_auc:.4f} | Val AUC: {epoch_val_auc:.4f}')
        print('-' * 50)
    
    # Create additional plot for AUC
    plt.figure()
    plt.plot(training_auc, label='Training AUC')
    plt.plot(validation_auc, label='Validation AUC')
    plt.title('AUC per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig('training_auc.png')
    plt.close()
    
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
    X_train, Y_train, X_val, Y_val = preprocess_data(X_train, Y_train, X_val, Y_val)

    # Model Initialization
    model = Classifier(input_dim=X_train.shape[1])

    # Training (dryrun limits epochs)
    epochs = 1 if dryrun else 20

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, X_train, Y_train, X_val, Y_val, epochs=epochs)

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