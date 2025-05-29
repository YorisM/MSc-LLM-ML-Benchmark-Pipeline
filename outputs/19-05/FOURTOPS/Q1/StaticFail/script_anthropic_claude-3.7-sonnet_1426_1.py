
import os, sys, json, pickle, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}
                       
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

def make_loaders(X_train, Y_train, X_val, Y_val, batch=1024):
    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train))
    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val))
    return (DataLoader(train, batch_size=batch, shuffle=True),
            DataLoader(val, batch_size=batch))
                        
# ----------------  START OF LLM BLOCK  ----------------
# Imports
import os, sys, json, pickle, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
# Only import extra std-lib modules, torch.nn or sklearn sub-modules you actually use.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoded_features = None
        self.num_objects = 18  # Maximum number of objects in the dataset
        self.features_per_object = 5  # obj_id, E, pT, eta, phi
        self.missing_ET_features = 2  # E_T_miss, phi_ET_miss
        
    def fit(self, X, y=None):
        # Convert to numpy for easier manipulation if it's a torch tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        
        # Extract features
        processed_features = self._extract_features(X)
        
        # Fit scaler on the processed data
        self.scaler.fit(processed_features)
        
        # Save the feature shape
        self.encoded_features = processed_features.shape[1]
        
        return self
    
    def transform(self, X):
        # Convert to numpy for easier manipulation if it's a torch tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        
        # Extract features
        processed_features = self._extract_features(X)
        
        # Scale the data
        scaled_features = self.scaler.transform(processed_features)
        
        # Return as torch tensor for model input
        return torch.tensor(scaled_features, dtype=torch.float32)
    
    def _extract_features(self, X):
        # Number of samples
        n_samples = X.shape[0]
        
        # Initialize array for engineered features
        features = []
        
        # Process each sample
        for i in range(n_samples):
            sample = X[i]
            
            # Extract missing ET features (first two elements)
            et_miss = sample[0]
            phi_et_miss = sample[1]
            
            # Dictionary to store objects by type
            objects_by_type = defaultdict(list)
            
            # Process each object in the event
            for j in range(self.num_objects):
                start_idx = self.missing_ET_features + j * self.features_per_object
                
                # Check if this is a valid object (not padding)
                obj_id = sample[start_idx]
                if obj_id == 0:  # Assuming 0 is padding
                    continue
                
                # Extract object features
                obj_energy = sample[start_idx + 1]
                obj_pt = sample[start_idx + 2]
                obj_eta = sample[start_idx + 3]
                obj_phi = sample[start_idx + 4]
                
                # Skip if all values are 0 (padding)
                if obj_energy == 0 and obj_pt == 0 and obj_eta == 0 and obj_phi == 0:
                    continue
                
                # Store object by its type
                objects_by_type[int(obj_id)].append([obj_energy, obj_pt, obj_eta, obj_phi])
            
            # Event-level features
            event_features = [et_miss, phi_et_miss]
            
            # Calculate derived features for each object type
            for obj_type in sorted(objects_by_type.keys()):
                objs = np.array(objects_by_type[obj_type])
                
                # Count of this object type
                event_features.append(len(objs))
                
                if len(objs) > 0:
                    # Sum of energy, pT for this object type
                    event_features.append(np.sum(objs[:, 0]))  # Sum E
                    event_features.append(np.sum(objs[:, 1]))  # Sum pT
                    
                    # Mean and std of features for this object type
                    for feature_idx in range(4):  # E, pT, eta, phi
                        event_features.append(np.mean(objs[:, feature_idx]))
                        event_features.append(np.std(objs[:, feature_idx]) if len(objs) > 1 else 0)
                    
                    # Min and max values
                    for feature_idx in range(4):
                        event_features.append(np.min(objs[:, feature_idx]))
                        event_features.append(np.max(objs[:, feature_idx]))
                    
                    # Calculate deltaR between pairs of same object type if multiple objects exist
                    if len(objs) > 1:
                        delta_r_values = []
                        for idx1 in range(len(objs)):
                            for idx2 in range(idx1 + 1, len(objs)):
                                delta_eta = objs[idx1, 2] - objs[idx2, 2]
                                delta_phi = self._delta_phi(objs[idx1, 3], objs[idx2, 3])
                                delta_r = np.sqrt(delta_eta**2 + delta_phi**2)
                                delta_r_values.append(delta_r)
                        
                        if delta_r_values:
                            event_features.append(np.min(delta_r_values))
                            event_features.append(np.max(delta_r_values))
                            event_features.append(np.mean(delta_r_values))
                        else:
                            event_features.extend([0, 0, 0])  # Placeholders if no deltaR calculated
                    else:
                        event_features.extend([0, 0, 0])  # Placeholders if only one object
                else:
                    # Add placeholders for empty object types
                    event_features.extend([0] * 15)  # 2 sums + 8 mean/std + 2 min/max + 3 deltaR
            
            # Calculate cross-type features (e.g., between different particle types)
            all_objs = []
            for obj_list in objects_by_type.values():
                all_objs.extend(obj_list)
            
            if len(all_objs) > 0:
                all_objs = np.array(all_objs)
                
                # Global event features
                event_features.append(np.sum(all_objs[:, 0]))  # Total energy
                event_features.append(np.sum(all_objs[:, 1]))  # Total pT
                
                # Event shape features
                event_features.append(np.mean(all_objs[:, 2]))  # Mean eta
                event_features.append(np.std(all_objs[:, 2]) if len(all_objs) > 1 else 0)  # Std eta
                event_features.append(np.mean(all_objs[:, 3]))  # Mean phi
                event_features.append(np.std(all_objs[:, 3]) if len(all_objs) > 1 else 0)  # Std phi
            else:
                event_features.extend([0] * 6)  # Placeholders for global features
            
            features.append(event_features)
        
        # Convert to numpy array
        return np.array(features, dtype=np.float32)
    
    def _delta_phi(self, phi1, phi2):
        """Calculate the correct delta phi between two phi angles"""
        dphi = phi1 - phi2
        while dphi > np.pi:
            dphi -= 2 * np.pi
        while dphi < -np.pi:
            dphi += 2 * np.pi
        return dphi

def make_preprocessor():
    return MyPreprocessor()

class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out

class FourTopClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FourTopClassifier, self).__init__()
        
        # Network architecture
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        
        # Residual blocks
        self.res1 = ResidualBlock(256, 128)
        self.res2 = ResidualBlock(256, 128)
        
        # Final layers
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.fc4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.res1(x)
        x = self.res2(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        
        return x.squeeze()

def make_model(input_dim):
    return FourTopClassifier(input_dim)

EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    # Init optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    
    # Initialize lists to track metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_auc = 0.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            predicted = torch.sigmoid(outputs) >= 0.5
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            # Store predictions and targets for AUC calculation
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_auc = roc_auc_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets.float())
                
                running_loss += loss.item() * inputs.size(0)
                predicted = torch.sigmoid(outputs) >= 0.5
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                
                # Store predictions and targets for AUC calculation
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        epoch_val_loss = running_loss / total
        epoch_val_acc = correct / total
        val_auc = roc_auc_score(val_targets, val_preds)
        
        # Update scheduler based on validation AUC
        scheduler.step(val_auc)
        
        # Save metrics
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        train_acc.append(epoch_train_acc)
        val_acc.append(epoch_val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}, '
              f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')
        
        # Save best model (based on validation AUC)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
    
    # Load the best model state
    model.load_state_dict(best_model_state)
    
    return model, train_loss, val_loss, train_acc, val_acc

# Main execution function (would be part of script execution)
def main():
    # Load data
    X_train = torch.load('X_train.pt')
    Y_train = torch.load('Y_train.pt')
    X_val = torch.load('X_val.pt')
    Y_val = torch.load('Y_val.pt')
    
    # Preprocess data
    preprocessor = make_preprocessor()
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_processed, Y_train)
    val_dataset = TensorDataset(X_val_processed, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Create and train model
    input_dim = X_train_processed.shape[1]
    model = make_model(input_dim)
    
    model, train_loss, val_loss, train_acc, val_acc = train_model(
        model, train_loader, val_loader, epochs=EPOCHS
    )
    
    # Save model and preprocessor
    torch.save(model.state_dict(), 'four_top_model.pth')
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    
    # Evaluate final model on validation set
    model.eval()
    val_preds = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = model(inputs)
            val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
    
    final_auc = roc_auc_score(Y_val, val_preds)
    print(f'Final validation AUC: {final_auc:.4f}')

if __name__ == '__main__':
    main()
# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_tr, y_tr, X_va, y_va = load_data()
    pre = make_preprocessor();  pre.fit(X_tr, y_tr)
    X_tr = pre.transform(X_tr); X_va = pre.transform(X_va)
    tr_loader, va_loader = make_loaders(X_tr, y_tr, X_va, y_va)

    # 2. Build model
    model = make_model(input_dim=X_tr.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    trained, tr_loss, va_loss, tr_acc, va_acc = train_model(
        model, tr_loader, va_loader, epochs=n_epochs
    )

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy = torch.zeros(8, X_tr.shape[1])      # 8 fake events
        try:
            _ = trained(pre.transform(toy))
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return  # no files in dry-run

    # 4. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
    torch.save(trained.state_dict(), f"{base}_state.pt")
    with open(f"{base}_model.pkl", "wb") as f: pickle.dump(trained, f)
    with open(f"{base}_preproc.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

