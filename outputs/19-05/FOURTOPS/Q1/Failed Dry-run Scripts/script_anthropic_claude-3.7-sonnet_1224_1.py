
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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MyPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.object_stats = {}
        self.n_objects = 18
        self.obj_features_per_object = 5
        self.global_features = 2  # E_T_miss, phi_Et_miss
        
    def fit(self, X, y=None):
        # Convert to numpy if it's a tensor
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        # Process global features (missing ET and phi)
        global_features = X[:, :self.global_features]
        self.scalers['global'] = StandardScaler().fit(global_features)
        
        # Extract and organize objects
        object_data = defaultdict(list)
        
        # For each object position
        for i in range(self.n_objects):
            start_idx = self.global_features + i * self.obj_features_per_object
            end_idx = start_idx + self.obj_features_per_object
            
            # Get object type and features
            obj_slice = X[:, start_idx:end_idx]
            obj_ids = obj_slice[:, 0]  # Object ID
            obj_features = obj_slice[:, 1:]  # E, pT, eta, phi
            
            # Find non-zero entries (actual objects, not padding)
            mask = obj_ids != 0
            
            if np.any(mask):
                # Store statistics about which object types appear and their count
                unique_objs = obj_ids[mask]
                for obj_id in np.unique(unique_objs):
                    if obj_id == 0:
                        continue
                    if obj_id not in self.object_stats:
                        self.object_stats[obj_id] = {'count': 0, 'positions': []}
                    self.object_stats[obj_id]['count'] += len(unique_objs[unique_objs == obj_id])
                    self.object_stats[obj_id]['positions'].append(i)
                
                # Create feature scalers for non-zero features
                valid_features = obj_features[mask]
                if len(valid_features) > 0:
                    # Create a scaler for each feature column of each object type
                    self.scalers[f'obj_{i}'] = StandardScaler().fit(valid_features)
        
        return self
    
    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X.copy()
            
        # Create output array to hold processed features
        batch_size = X_np.shape[0]
        
        # Transform global features
        global_scaled = self.scalers['global'].transform(X_np[:, :self.global_features])
        
        # Process each object
        object_features = []
        
        # Include scaled global features
        object_features.append(global_scaled)
        
        # We'll create features for each object position
        for i in range(self.n_objects):
            start_idx = self.global_features + i * self.obj_features_per_object
            end_idx = start_idx + self.obj_features_per_object
            
            obj_slice = X_np[:, start_idx:end_idx]
            obj_ids = obj_slice[:, 0]  # Object ID
            obj_features = obj_slice[:, 1:]  # E, pT, eta, phi
            
            # Create a mask for where objects exist (non-zero)
            mask = obj_ids != 0
            
            # Initialize features array
            scaled_features = np.zeros((batch_size, 5))  # obj_id + 4 scaled features
            
            # Set object IDs directly (no scaling)
            scaled_features[:, 0] = obj_ids
            
            # Scale non-zero features if we have a scaler for this object position
            if f'obj_{i}' in self.scalers and np.any(mask):
                valid_indices = np.where(mask)[0]
                valid_features = obj_features[mask]
                
                # Scale the valid features
                scaled_valid = self.scalers[f'obj_{i}'].transform(valid_features)
                
                # Put the scaled features back into the output array
                scaled_features[valid_indices, 1:] = scaled_valid
            
            # Append the features for this object
            object_features.append(scaled_features)
        
        # Calculate derived features
        
        # 1. Count objects per event
        obj_counts = np.zeros((batch_size, 1))
        for i in range(self.n_objects):
            start_idx = self.global_features + i * self.obj_features_per_object
            obj_ids = X_np[:, start_idx]
            obj_counts += (obj_ids != 0).astype(np.float32).reshape(-1, 1)
        object_features.append(obj_counts)
        
        # 2. Calculate sum of pT for all objects
        pt_sum = np.zeros((batch_size, 1))
        for i in range(self.n_objects):
            start_idx = self.global_features + i * self.obj_features_per_object
            obj_ids = X_np[:, start_idx]
            pt_vals = X_np[:, start_idx + 2]  # pT is at index 2 within each object
            pt_sum += np.where(obj_ids != 0, pt_vals, 0).reshape(-1, 1)
        object_features.append(pt_sum)
        
        # 3. Calculate HT (scalar sum of jet pT)
        ht = np.zeros((batch_size, 1))
        for i in range(self.n_objects):
            start_idx = self.global_features + i * self.obj_features_per_object
            obj_ids = X_np[:, start_idx]
            # Check if this is a jet (assuming jets have specific object IDs)
            # This is an approximation - adapt based on your understanding of the object IDs
            pt_vals = X_np[:, start_idx + 2]  # pT is at index 2 within each object
            ht += np.where(obj_ids != 0, pt_vals, 0).reshape(-1, 1)
        object_features.append(ht)
        
        # Combine all features
        result = np.hstack(object_features)
        
        # Convert back to tensor if input was tensor
        if isinstance(X, torch.Tensor):
            return torch.tensor(result, dtype=torch.float32)
        else:
            return result

def make_preprocessor():
    return MyPreprocessor()

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def forward(self, x):
        # Self attention with residual connection and normalization
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed forward with residual and normalization
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x

class FourTopClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # Parameters
        self.n_objects = 18
        self.features_per_object = 5
        self.global_features = 2
        self.extra_features = 3  # Number of derived features
        
        # Object embedding dims
        self.obj_embed_dim = 32
        
        # Embedding for object IDs
        self.obj_id_embedding = nn.Embedding(20, 8)  # Assuming max 20 different object types
        
        # Object feature processing
        self.obj_encoder = nn.Sequential(
            nn.Linear(self.features_per_object - 1 + 8, self.obj_embed_dim),  # -1 for obj_id, +8 for embedding
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Global feature processing
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_features, self.obj_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Attention layers for objects
        self.attention_layers = nn.ModuleList([
            AttentionBlock(self.obj_embed_dim) for _ in range(3)  # 3 attention blocks
        ])
        
        # Extra features processing
        self.extra_encoder = nn.Sequential(
            nn.Linear(self.extra_features, self.obj_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Final classifier - combines global features with attended object features
        self.classifier = nn.Sequential(
            nn.Linear((self.n_objects + 2) * self.obj_embed_dim, 128),  # +2 for global and extra features
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract global features (first 2)
        global_features = x[:, :self.global_features]
        
        # Process global features
        global_embedding = self.global_encoder(global_features)
        
        # Extract extra derived features (last 3)
        extra_features = x[:, -(self.extra_features):]
        extra_embedding = self.extra_encoder(extra_features)
        
        # Process each object
        object_embeddings = []
        
        for i in range(self.n_objects):
            start_idx = self.global_features + i * self.features_per_object
            end_idx = start_idx + self.features_per_object
            
            obj_features = x[:, start_idx:end_idx]
            obj_ids = obj_features[:, 0].long()  # Extract object IDs
            
            # Embed object IDs
            obj_id_embed = self.obj_id_embedding(obj_ids)
            
            # Get continuous features
            cont_features = obj_features[:, 1:]
            
            # Detect which objects are real (non-padding)
            is_real_obj = (obj_ids != 0).float().unsqueeze(-1)
            
            # Combine ID embedding with continuous features
            combined = torch.cat([obj_id_embed, cont_features], dim=1)
            
            # Encode object features
            obj_embedding = self.obj_encoder(combined)
            
            # Zero out padding objects
            obj_embedding = obj_embedding * is_real_obj
            
            # Add to list of embeddings
            object_embeddings.append(obj_embedding)
        
        # Stack object embeddings [batch_size, n_objects, embed_dim]
        object_tensor = torch.stack(object_embeddings, dim=1)
        
        # Apply attention blocks
        for attn in self.attention_layers:
            object_tensor = attn(object_tensor)
        
        # Flatten the object embeddings
        object_flat = object_tensor.reshape(batch_size, -1)
        
        # Concatenate global, object, and extra embeddings
        all_features = torch.cat([
            global_embedding,
            object_flat,
            extra_embedding
        ], dim=1)
        
        # Final classification
        logits = self.classifier(all_features).squeeze(-1)
        
        return logits

def make_model(input_dim):
    model = FourTopClassifier(input_dim)
    return model

EPOCHS = 15

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=False)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Lists to track metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_aucs = []
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            
            # Compute loss
            loss = criterion(outputs, y_batch.float())
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * X_batch.size(0)
            train_total += y_batch.size(0)
            
            # Convert outputs to binary predictions
            preds = torch.sigmoid(outputs) >= 0.5
            train_correct += (preds == y_batch).sum().item()
            
            # Save raw predictions for AUC calculation
            train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_labels.extend(y_batch.cpu().numpy())
        
        # Calculate training metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                outputs = model(X_batch)
                
                # Compute loss
                loss = criterion(outputs, y_batch.float())
                
                # Track statistics
                val_loss += loss.item() * X_batch.size(0)
                val_total += y_batch.size(0)
                
                # Convert outputs to binary predictions
                preds = torch.sigmoid(outputs) >= 0.5
                val_correct += (preds == y_batch).sum().item()
                
                # Save raw predictions for AUC calculation
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_labels, val_preds)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
    
    return model, train_losses, val_losses, train_accs, val_accs
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

