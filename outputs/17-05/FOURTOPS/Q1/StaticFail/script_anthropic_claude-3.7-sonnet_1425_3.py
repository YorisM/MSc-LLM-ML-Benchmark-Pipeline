
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

EPOCHS = 10
                        
def load_data():
    to_np = lambda path: pd.read_csv(path).values
    X_train = to_np(DATASET["X_train"])
    Y_train = to_np(DATASET["Y_train"]).ravel()
    X_val   = to_np(DATASET["X_val"])
    Y_val   = to_np(DATASET["Y_val"]).ravel()
    return X_train, Y_train, X_val, Y_val

def make_loaders(X_train, Y_train, X_val, Y_val, batch=1024):
    from torch.utils.data import TensorDataset, DataLoader
    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train))
    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val))
    return (DataLoader(train, batch_size=batch, shuffle=True),
            DataLoader(val, batch_size=batch))
                        
# ----------------  START OF LLM BLOCK  ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.num_objects = 18
        self.features_per_object = 5
        self.obj_id_idx = 0  # Index of object ID within each object's feature group

    def fit(self, X, y=None):
        # Identify non-zero values for normalization (excluding object IDs)
        mask = self._create_non_zero_mask(X)
        
        # Calculate mean and std for normalization
        self.mean = torch.mean(X[mask], dim=0)
        self.std = torch.std(X[mask], dim=0)
        self.std[self.std == 0] = 1.0  # Prevent division by zero
        
        return self
    
    def _create_non_zero_mask(self, X):
        # Create a mask for non-zero values and exclude object IDs
        # First two indices are E_T_miss and phi_E_t_miss
        mask = torch.ones_like(X, dtype=torch.bool)
        
        # For each object slot
        for i in range(self.num_objects):
            start_idx = 2 + i * self.features_per_object
            obj_id_idx = start_idx + self.obj_id_idx
            
            # If object_id is zero, mask all features of that object
            zero_objects = (X[:, obj_id_idx] == 0)
            for j in range(self.features_per_object):
                mask[zero_objects, start_idx + j] = False
        
        return mask

    def transform(self, X):
        # Convert to tensor if needed
        X = torch.tensor(X, dtype=torch.float32) if isinstance(X, np.ndarray) else X.clone()
        
        # Extract and normalize ET_miss and phi_ET_miss (first two features)
        et_miss = X[:, 0:2]
        et_miss_norm = (et_miss - self.mean[0:2]) / self.std[0:2]
        
        # Process each object
        object_features = []
        object_masks = []
        seq_lengths = []
        
        # Count valid objects in each event
        for i in range(X.shape[0]):
            valid_objects = 0
            for obj_idx in range(self.num_objects):
                start_idx = 2 + obj_idx * self.features_per_object
                obj_id = X[i, start_idx]
                if obj_id > 0:  # Valid object
                    valid_objects += 1
            seq_lengths.append(valid_objects)
        
        max_seq_len = max(seq_lengths)
        
        # Create feature tensors for each event
        batch_size = X.shape[0]
        object_features = torch.zeros(batch_size, max_seq_len, 4)  # E, pT, eta, phi
        object_types = torch.zeros(batch_size, max_seq_len, 1)     # Object ID
        object_masks = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        
        for i in range(batch_size):
            obj_count = 0
            for obj_idx in range(self.num_objects):
                start_idx = 2 + obj_idx * self.features_per_object
                obj_id = X[i, start_idx]
                
                if obj_id > 0 and obj_count < max_seq_len:  # Valid object
                    # Extract and normalize the 4 kinematic features
                    kinematic_features = X[i, start_idx+1:start_idx+5]
                    normalized_features = (kinematic_features - self.mean[start_idx+1:start_idx+5]) / self.std[start_idx+1:start_idx+5]
                    
                    object_features[i, obj_count] = normalized_features
                    object_types[i, obj_count, 0] = obj_id
                    object_masks[i, obj_count] = True
                    obj_count += 1
        
        # Construct the final feature representation
        et_miss_expanded = et_miss_norm.unsqueeze(1).expand(-1, max_seq_len, -1)
        
        # Concatenate all features
        combined_features = torch.cat([
            et_miss_expanded,      # Shape: [batch_size, max_seq_len, 2]
            object_features,       # Shape: [batch_size, max_seq_len, 4]
            object_types           # Shape: [batch_size, max_seq_len, 1]
        ], dim=2)  # Final shape: [batch_size, max_seq_len, 7]
        
        # Create a dictionary with all the necessary information
        result = {
            'features': combined_features,         # [batch_size, max_seq_len, 7]
            'masks': object_masks,                 # [batch_size, max_seq_len]
            'seq_lengths': torch.tensor(seq_lengths)  # [batch_size]
        }
        
        return result

def make_preprocessor():
    return MyPreprocessor()

# Custom attention mechanism
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, rnn_output, mask=None):
        # rnn_output shape: [batch, seq_len, hidden_dim]
        # mask shape: [batch, seq_len]
        
        # Calculate attention scores
        attention_scores = self.attention(rnn_output).squeeze(-1)  # [batch, seq_len]
        
        # Apply mask to set padding attention scores to a large negative value
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, -1e10)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len]
        
        # Apply attention weights to input
        context_vector = torch.bmm(attention_weights.unsqueeze(1), rnn_output).squeeze(1)  # [batch, hidden_dim]
        
        return context_vector, attention_weights

class FourTopClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, lstm_layers=2, dropout_rate=0.3):
        super().__init__()
        
        # LSTM to process variable-length sequences
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        
    def forward(self, x_dict):
        # Unpack the input dictionary
        x = x_dict['features']           # [batch_size, seq_len, feature_dim]
        mask = x_dict['masks']           # [batch_size, seq_len]
        seq_lengths = x_dict['seq_lengths']  # [batch_size]
        
        # Pack padded sequence for efficient computation
        packed_input = pack_padded_sequence(
            x, 
            seq_lengths.cpu().numpy(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Run LSTM
        packed_output, _ = self.lstm(packed_input)
        
        # Unpack the sequence
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply attention mechanism
        context_vector, _ = self.attention(lstm_output, mask)
        
        # Fully connected layers with activations and regularization
        x = self.dropout(F.relu(self.bn1(self.fc1(context_vector))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        return x.squeeze(-1)  # Return logits

def make_model(input_dim=7):  # Our preprocessor creates 7 features per object
    return FourTopClassifier(input_dim=input_dim)

EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use binary cross entropy loss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=False
    )
    
    # Track metrics
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    best_val_auc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_logits = []
        train_labels = []
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Move data to device
            X_batch = {k: v.to(device) for k, v in X_batch.items() if isinstance(v, torch.Tensor)}
            y_batch = y_batch.float().to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track training metrics
            train_loss += loss.item() * y_batch.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)
            
            # Store predictions and labels for AUC calculation
            train_logits.append(logits.detach().cpu())
            train_labels.append(y_batch.cpu())
        
        # Calculate epoch training metrics
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_logits = []
        val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Move data to device
                X_batch = {k: v.to(device) for k, v in X_batch.items() if isinstance(v, torch.Tensor)}
                y_batch = y_batch.float().to(device)
                
                # Forward pass
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                # Track validation metrics
                val_loss += loss.item() * y_batch.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
                
                # Store predictions and labels for AUC calculation
                val_logits.append(logits.cpu())
                val_labels.append(y_batch.cpu())
        
        # Calculate epoch validation metrics
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        # Calculate AUC for validation set
        val_probs = torch.sigmoid(torch.cat(val_logits)).numpy()
        val_true = torch.cat(val_labels).numpy()
        val_auc = roc_auc_score(val_true, val_probs)
        
        # Update learning rate
        scheduler.step(val_auc)
    
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

# Custom dataset class to handle preprocessed data
class FourTopDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, preprocessor):
        self.X_processed = preprocessor.transform(X)
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Extract this sample's data
        features = {k: v[idx] for k, v in self.X_processed.items() if isinstance(v, torch.Tensor)}
        label = self.y[idx]
        
        return features, label

# Collate function for batching
def collate_fn(batch):
    features_batch = {}
    labels = []
    
    # Initialize dictionaries for each feature type
    for k in batch[0][0].keys():
        features_batch[k] = []
    
    # Collect all features and labels
    for features, label in batch:
        for k, v in features.items():
            features_batch[k].append(v)
        labels.append(label)
    
    # Convert lists to tensors
    for k in features_batch:
        if k == 'seq_lengths':
            features_batch[k] = torch.stack(features_batch[k])
        else:
            features_batch[k] = torch.stack(features_batch[k])
    
    labels = torch.tensor(labels)
    
    return features_batch, labels
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
    torch.jit.script(trained).save(f"{base}_scripted.pt")
    torch.jit.script(pre).save(f"{base}_preproc.pt")
    with open(f"{base}_pre.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

