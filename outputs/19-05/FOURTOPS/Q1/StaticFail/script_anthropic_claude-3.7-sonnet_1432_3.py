
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.n_objects = 18  # Maximum number of objects
        self.object_features = 5  # Number of features per object
        
    def fit(self, X, y=None):
        # Convert to numpy if it's a torch tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()
            
        # Extract and reshape the valid features (non-zero entries)
        valid_features = self._extract_valid_features(X)
        
        # Fit the scaler on valid features
        self.scaler.fit(valid_features)
        return self
    
    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X.copy()
        
        # Process and structure the data
        processed_data = np.zeros((X_np.shape[0], 132), dtype=np.float32)
        
        # First two features: Missing ET magnitude and phi
        processed_data[:, 0:2] = X_np[:, 0:2]
        
        # Extract object features and identify valid objects
        valid_obj_mask = np.zeros((X_np.shape[0], self.n_objects), dtype=bool)
        obj_features = np.zeros((X_np.shape[0], self.n_objects, 4), dtype=np.float32)
        
        for i in range(self.n_objects):
            start_idx = 2 + i * self.object_features
            obj_id_idx = start_idx
            feat_start_idx = start_idx + 1
            feat_end_idx = start_idx + self.object_features
            
            # Check if object is valid (obj_id != 0)
            valid_obj_mask[:, i] = X_np[:, obj_id_idx] != 0
            
            # Extract energy, pT, eta, phi
            obj_features[:, i, :] = X_np[:, feat_start_idx:feat_end_idx]
        
        # Calculate derived features for valid objects
        for i in range(self.n_objects):
            if np.any(valid_obj_mask[:, i]):
                obj_feat = obj_features[:, i]
                
                # Copy original features with scaling
                processed_data[:, 2 + i*6:2 + i*6 + 4] = obj_feat
                
                # Add derived features: pT/E ratio and delta phi with missing ET
                pt_e_ratio = np.where(obj_feat[:, 0] > 0, obj_feat[:, 1] / obj_feat[:, 0], 0)
                processed_data[:, 2 + i*6 + 4] = pt_e_ratio
                
                dphi = np.abs(obj_feat[:, 3] - X_np[:, 1])
                dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
                processed_data[:, 2 + i*6 + 5] = dphi
        
        # Calculate global features
        # Total energy, pT, and object count
        total_e = np.sum(np.where(valid_obj_mask[:, :, np.newaxis], obj_features[:, :, 0:1], 0), axis=1)
        total_pt = np.sum(np.where(valid_obj_mask[:, :, np.newaxis], obj_features[:, :, 1:2], 0), axis=1)
        obj_count = np.sum(valid_obj_mask, axis=1, keepdims=True)
        
        # Add global features to the processed data
        processed_data[:, -6] = total_e.flatten()
        processed_data[:, -5] = total_pt.flatten()
        processed_data[:, -4] = obj_count.flatten()
        
        # HT (scalar sum of pT)
        processed_data[:, -3] = total_pt.flatten()
        
        # MET to HT ratio
        processed_data[:, -2] = np.where(processed_data[:, -3] > 0, 
                                        processed_data[:, 0] / processed_data[:, -3], 0)
        
        # Average object pT
        processed_data[:, -1] = np.where(obj_count.flatten() > 0, 
                                        total_pt.flatten() / obj_count.flatten(), 0)
        
        # Scale the features
        processed_data_scaled = self._scale_features(processed_data)
        
        return torch.tensor(processed_data_scaled, dtype=torch.float32)
    
    def _extract_valid_features(self, X):
        """Extract non-zero features for scaling."""
        valid_features = []
        
        # Extract missing ET features
        valid_features.append(X[:, 0:2])
        
        # Extract object features for valid objects
        for i in range(self.n_objects):
            start_idx = 2 + i * self.object_features
            obj_id_idx = start_idx
            feat_start_idx = start_idx + 1
            feat_end_idx = start_idx + self.object_features
            
            mask = X[:, obj_id_idx] != 0
            if np.any(mask):
                valid_obj_feats = X[mask, feat_start_idx:feat_end_idx]
                valid_features.append(valid_obj_feats)
        
        return np.vstack(valid_features)
    
    def _scale_features(self, X):
        """Scale the features using the fitted scaler."""
        # Reshape to 2D for scaling
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        
        # Apply scaling
        X_scaled = self.scaler.transform(X_flat)
        
        # Reshape back to original shape
        return X_scaled.reshape(original_shape)

def make_preprocessor():
    return MyPreprocessor()

class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_features)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out

class FourTopClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FourTopClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.res_block1 = ResidualBlock(256, 128)
        self.res_block2 = ResidualBlock(256, 128)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x.squeeze()

def make_model(input_dim):
    return FourTopClassifier(input_dim)

EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
    
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    best_auc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(torch.sigmoid(outputs).detach().cpu().numpy())
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        # Concatenate all batches
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)
        train_auc = roc_auc_score(all_targets, all_predictions)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_val_targets = []
        all_val_predictions = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.float())
                
                running_loss += loss.item() * inputs.size(0)
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_val_targets.append(targets.cpu().numpy())
                all_val_predictions.append(torch.sigmoid(outputs).cpu().numpy())
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        val_loss.append(epoch_loss)
        val_acc.append(epoch_acc)
        
        # Calculate validation AUC
        all_val_targets = np.concatenate(all_val_targets)
        all_val_predictions = np.concatenate(all_val_predictions)
        val_auc = roc_auc_score(all_val_targets, all_val_predictions)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, Train AUC: {train_auc:.4f}, '
              f'Val Loss: {val_loss[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}, Val AUC: {val_auc:.4f}')
    
    return model, train_loss, val_loss, train_acc, val_acc

# Main execution function
def main():
    # Load data (assuming the data is already loaded in the environment)
    X_train = torch.load('X_train.pt')
    Y_train = torch.load('Y_train.pt')
    X_val = torch.load('X_val.pt')
    Y_val = torch.load('Y_val.pt')
    
    # Preprocess data
    preprocessor = make_preprocessor()
    preprocessor.fit(X_train, Y_train)
    
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train_processed, Y_train)
    val_dataset = TensorDataset(X_val_processed, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # Initialize model
    input_dim = X_train_processed.shape[1]
    model = make_model(input_dim)
    
    # Train model
    trained_model, train_loss, val_loss, train_acc, val_acc = train_model(
        model, train_loader, val_loader, EPOCHS)
    
    # Save model and preprocessor
    torch.save(trained_model.state_dict(), 'model.pt')
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Evaluate final model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.to(device)
    trained_model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = trained_model(inputs)
            probs = torch.sigmoid(outputs)
            
            all_predictions.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    final_auc = roc_auc_score(all_targets, all_predictions)
    print(f'Final validation AUC: {final_auc:.6f}')
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')

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

