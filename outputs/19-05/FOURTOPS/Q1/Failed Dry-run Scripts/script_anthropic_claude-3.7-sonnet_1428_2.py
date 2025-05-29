
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
from torch.optim import Adam
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import math

class MyPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.feature_indices = {}
        self.num_objects = 18  # Maximum number of objects in the dataset
        self.num_features_per_obj = 5
        self.global_features_count = 2  # E_T_miss and phi_E_t_miss
        
    def fit(self, X, y=None):
        # Convert to numpy if it's a tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        # Extract global features (E_T_miss, phi_E_t_miss)
        global_features = X[:, :self.global_features_count]
        
        # Scale global features
        self.scalers['global'] = StandardScaler().fit(global_features)
        
        # Process each object separately
        for obj_idx in range(self.num_objects):
            # For each object, we have 5 features: [obj_id, E, pT, eta, phi]
            start_idx = self.global_features_count + obj_idx * self.num_features_per_obj
            end_idx = start_idx + self.num_features_per_obj
            
            # Check if this object exists in the data (nonzero values)
            obj_features = X[:, start_idx:end_idx]
            if np.any(obj_features[:, 0] != 0):
                # Extract object type (first column)
                self.feature_indices[f'obj_type_{obj_idx}'] = start_idx
                
                # Scale continuous features (E, pT, eta, phi)
                continuous_features = obj_features[:, 1:]
                # Only fit scaler if we have non-zero values
                if np.any(continuous_features != 0):
                    self.scalers[f'obj_{obj_idx}'] = StandardScaler().fit(continuous_features)
        
        return self
    
    def transform(self, X):
        # Convert to numpy if it's a tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        
        # Initialize the transformed feature list
        transformed_features = []
        
        # Transform global features
        global_features = X[:, :self.global_features_count]
        transformed_global = self.scalers['global'].transform(global_features)
        transformed_features.append(transformed_global)
        
        # Lists to store features by object type
        electron_features = []
        muon_features = []
        jet_features = []
        bjet_features = []
        
        # Process each object
        for obj_idx in range(self.num_objects):
            start_idx = self.global_features_count + obj_idx * self.num_features_per_obj
            end_idx = start_idx + self.num_features_per_obj
            
            # Check if this object exists in the data
            if f'obj_{obj_idx}' in self.scalers:
                obj_features = X[:, start_idx:end_idx]
                obj_type = obj_features[:, 0]
                obj_kinematics = obj_features[:, 1:]
                
                # Scale the continuous features
                if np.any(obj_kinematics != 0):
                    scaled_kinematics = self.scalers[f'obj_{obj_idx}'].transform(obj_kinematics)
                    
                    # Create a mask for each object type
                    electron_mask = (obj_type == 11) | (obj_type == -11)
                    muon_mask = (obj_type == 13) | (obj_type == -13)
                    bjet_mask = (obj_type == 5) | (obj_type == -5)
                    jet_mask = ~(electron_mask | muon_mask | bjet_mask) & (obj_type != 0)
                    
                    # Create feature vectors with zeros for missing objects
                    batch_size = X.shape[0]
                    
                    # For electrons
                    if np.any(electron_mask):
                        e_features = np.zeros((batch_size, scaled_kinematics.shape[1]))
                        e_features[electron_mask] = scaled_kinematics[electron_mask]
                        electron_features.append(e_features)
                        
                    # For muons
                    if np.any(muon_mask):
                        m_features = np.zeros((batch_size, scaled_kinematics.shape[1]))
                        m_features[muon_mask] = scaled_kinematics[muon_mask]
                        muon_features.append(m_features)
                        
                    # For b-jets
                    if np.any(bjet_mask):
                        b_features = np.zeros((batch_size, scaled_kinematics.shape[1]))
                        b_features[bjet_mask] = scaled_kinematics[bjet_mask]
                        bjet_features.append(b_features)
                        
                    # For jets
                    if np.any(jet_mask):
                        j_features = np.zeros((batch_size, scaled_kinematics.shape[1]))
                        j_features[jet_mask] = scaled_kinematics[jet_mask]
                        jet_features.append(j_features)
        
        # Calculate summary statistics for each object type
        obj_type_summaries = []
        
        # Process electrons
        if electron_features:
            e_stack = np.stack(electron_features, axis=1)
            e_count = np.sum(np.any(e_stack != 0, axis=2), axis=1, keepdims=True)
            e_sum = np.sum(e_stack, axis=1)
            e_mean = np.mean(e_stack, axis=1, where=(e_stack != 0))
            e_mean = np.nan_to_num(e_mean)  # Replace NaNs with zeros
            obj_type_summaries.extend([e_count, e_sum, e_mean])
        else:
            # Add zeros if no electrons
            batch_size = X.shape[0]
            obj_type_summaries.extend([np.zeros((batch_size, 1)), 
                                    np.zeros((batch_size, 4)), 
                                    np.zeros((batch_size, 4))])
        
        # Process muons
        if muon_features:
            m_stack = np.stack(muon_features, axis=1)
            m_count = np.sum(np.any(m_stack != 0, axis=2), axis=1, keepdims=True)
            m_sum = np.sum(m_stack, axis=1)
            m_mean = np.mean(m_stack, axis=1, where=(m_stack != 0))
            m_mean = np.nan_to_num(m_mean)
            obj_type_summaries.extend([m_count, m_sum, m_mean])
        else:
            batch_size = X.shape[0]
            obj_type_summaries.extend([np.zeros((batch_size, 1)), 
                                    np.zeros((batch_size, 4)), 
                                    np.zeros((batch_size, 4))])
        
        # Process b-jets
        if bjet_features:
            b_stack = np.stack(bjet_features, axis=1)
            b_count = np.sum(np.any(b_stack != 0, axis=2), axis=1, keepdims=True)
            b_sum = np.sum(b_stack, axis=1)
            b_mean = np.mean(b_stack, axis=1, where=(b_stack != 0))
            b_mean = np.nan_to_num(b_mean)
            obj_type_summaries.extend([b_count, b_sum, b_mean])
        else:
            batch_size = X.shape[0]
            obj_type_summaries.extend([np.zeros((batch_size, 1)), 
                                    np.zeros((batch_size, 4)), 
                                    np.zeros((batch_size, 4))])
        
        # Process jets
        if jet_features:
            j_stack = np.stack(jet_features, axis=1)
            j_count = np.sum(np.any(j_stack != 0, axis=2), axis=1, keepdims=True)
            j_sum = np.sum(j_stack, axis=1)
            j_mean = np.mean(j_stack, axis=1, where=(j_stack != 0))
            j_mean = np.nan_to_num(j_mean)
            obj_type_summaries.extend([j_count, j_sum, j_mean])
        else:
            batch_size = X.shape[0]
            obj_type_summaries.extend([np.zeros((batch_size, 1)), 
                                    np.zeros((batch_size, 4)), 
                                    np.zeros((batch_size, 4))])
        
        # Add object count features
        transformed_features.extend(obj_type_summaries)
        
        # Concatenate all features
        result = np.concatenate(transformed_features, axis=1)
        
        return torch.tensor(result, dtype=torch.float32)

def make_preprocessor():
    return MyPreprocessor()

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class PhysicsAwareNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        # Initial projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Define the network layers
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2)
            ])
        self.layers = nn.Sequential(*layers)
        
        # Attention block
        self.use_attention = True
        if self.use_attention:
            self.attention = AttentionBlock(hidden_dims[-1], num_heads=4)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.layers(x)
        
        # Apply attention if enabled
        if self.use_attention:
            # Reshape for attention [batch_size, seq_len=1, hidden_dim]
            x = x.unsqueeze(1)
            x = self.attention(x)
            x = x.squeeze(1)  # Back to [batch_size, hidden_dim]
        
        # Output layer
        x = self.output_layer(x)
        return torch.sigmoid(x).squeeze(-1)

def make_model(input_dim):
    return PhysicsAwareNet(input_dim)

EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define optimizer and loss function
    optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, 
        min_lr=1e-6, threshold=1e-3
    )
    
    # Lists to store metrics
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    best_val_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        train_outputs = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
            
            # Store predictions and targets for AUC calculation
            train_outputs.extend(outputs.detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Calculate training metrics
        train_loss = train_loss / total_train
        train_acc = correct_train / total_train
        train_auc = roc_auc_score(train_targets, train_outputs)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_outputs = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track metrics
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()
                
                # Store predictions and targets for AUC calculation
                val_outputs.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        val_auc = roc_auc_score(val_targets, val_outputs)
        
        # Update learning rate
        lr_scheduler.step(val_auc)
        
        # Save the best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {key: value.cpu() for key, value in model.state_dict().items()}
        
        # Store history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f} - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
    
    # Load the best model weights
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history
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

