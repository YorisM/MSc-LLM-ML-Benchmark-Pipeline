
import os, sys, pickle, torch, gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train),
            torch.from_numpy(Y_train),
            torch.from_numpy(X_val),
            torch.from_numpy(Y_val))

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))
                        
# ----------------  START OF LLM BLOCK  ----------------

# 0. ---------- IMPORTS ----------
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # Scalers for numerical features
        self.global_scaler = StandardScaler()
        self.object_scaler = StandardScaler()
        self.n_objects = 18  # Max number of objects
        self.obj_features = 4  # E, pT, eta, phi

    def fit(self, X, y=None):
        # Extract and fit global features (missing-ET)
        global_features = X[:, :2].numpy()
        self.global_scaler.fit(global_features)
        
        # Extract object features (excluding object ID)
        object_features = []
        for i in range(self.n_objects):
            # Skip object ID, take only E, pT, eta, phi
            start_idx = 2 + i * 5 + 1  # Skip object ID
            end_idx = start_idx + 4
            # Only include objects that exist (non-zero energy)
            mask = X[:, start_idx] > 0  # Energy > 0
            if torch.any(mask):
                obj_feats = X[mask, start_idx:end_idx].numpy()
                object_features.append(obj_feats)
        
        # Combine all valid object features for scaling
        if object_features:
            combined_features = np.vstack(object_features)
            self.object_scaler.fit(combined_features)
        
        return self

    def transform(self, X):
        # Create new feature array
        batch_size = X.shape[0]
        
        # 1. Scale global features
        global_feats = self.global_scaler.transform(X[:, :2].numpy())
        global_feats = torch.tensor(global_feats, dtype=torch.float32)  # [batch_size, 2]
        
        # 2. Process each object
        all_object_features = []
        
        for i in range(self.n_objects):
            start_idx = 2 + i * 5  # Start of this object's features
            obj_id = X[:, start_idx].unsqueeze(1)  # object ID
            
            # Extract E, pT, eta, phi
            feats_start = start_idx + 1
            feats_end = feats_start + 4
            obj_feats = X[:, feats_start:feats_end].clone()  # [batch_size, 4]
            
            # Check if object exists (non-zero energy)
            mask = obj_feats[:, 0] > 0  # Energy > 0
            
            # Apply scaling only to valid objects
            if torch.any(mask):
                # Scale valid objects
                valid_feats = obj_feats[mask].numpy()
                scaled_valid = self.object_scaler.transform(valid_feats)
                obj_feats[mask] = torch.tensor(scaled_valid, dtype=torch.float32)
            
            # Create existence flag
            exists = mask.float().unsqueeze(1)  # [batch_size, 1]
            
            # Compute Lorentz 4-vector components (px, py, pz, E)
            E = obj_feats[:, 0].clone()  # Energy
            pT = obj_feats[:, 1].clone()  # Transverse momentum
            eta = obj_feats[:, 2].clone()  # Pseudorapidity
            phi = obj_feats[:, 3].clone()  # Azimuthal angle
            
            # Calculate px, py, pz from pT, eta, phi
            px = pT * torch.cos(phi)
            py = pT * torch.sin(phi)
            pz = pT * torch.sinh(eta)
            
            # Create Lorentz 4-vector
            lorentz_vec = torch.stack([E, px, py, pz], dim=1)  # [batch_size, 4]
            
            # Combine all features for this object
            obj_combined = torch.cat([exists, obj_feats, lorentz_vec], dim=1)  # [batch_size, 9]
            all_object_features.append(obj_combined)
        
        # Stack all object features
        stacked_objects = torch.stack(all_object_features, dim=1)  # [batch_size, n_objects, 9]
        
        # Flatten for output
        flattened = stacked_objects.reshape(batch_size, -1)  # [batch_size, n_objects * 9]
        
        # Combine with global features
        result = torch.cat([global_feats, flattened], dim=1)  # [batch_size, 2 + n_objects * 9]
        
        return result

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class LorentzLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, n_objects, features]
        return self.linear(x)

class LorentzAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, n_objects, feature_dim]
        batch_size, n_objects, _ = x.shape
        
        # Project queries, keys, values and reshape to separate attention heads
        q = self.q_proj(x).reshape(batch_size, n_objects, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, n_objects, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, n_objects, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            # mask shape: [batch_size, n_objects] - True for valid objects
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, n_objects]
            mask = mask & mask.transpose(-2, -1)   # [batch_size, 1, n_objects, n_objects]
            scores = scores.masked_fill(~mask, -1e9)
        
        # Apply softmax and matmul with values
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, n_objects, self.feature_dim)
        return self.out_proj(attn_output)

class LorentzEquivariantBlock(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.attention = LorentzAttention(feature_dim)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.ff = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.norm2 = nn.LayerNorm(feature_dim)
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, n_objects, feature_dim]
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x), mask)
        # Feed-forward with residual connection
        x = x + self.ff(self.norm2(x))
        return x

class ParticleTransformer(nn.Module):
    def __init__(self, input_dim, n_objects, obj_features, global_features=2, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.n_objects = n_objects
        self.obj_features = obj_features
        
        # Process global features
        self.global_embed = nn.Sequential(
            nn.Linear(global_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Process per-object features
        self.object_embed = nn.Linear(obj_features, hidden_dim)
        
        # Equivariant blocks
        self.transformer_blocks = nn.ModuleList([
            LorentzEquivariantBlock(hidden_dim, hidden_dim * 4) 
            for _ in range(num_layers)
        ])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Split input into global and object features
        global_feats = x[:, :2]  # [batch_size, 2]
        obj_feats = x[:, 2:].reshape(batch_size, self.n_objects, -1)  # [batch_size, n_objects, features_per_obj]
        
        # Object mask: first feature is existence flag
        obj_mask = obj_feats[:, :, 0] > 0  # [batch_size, n_objects]
        
        # Process global features
        global_emb = self.global_embed(global_feats)  # [batch_size, hidden_dim]
        
        # Process object features
        obj_emb = self.object_embed(obj_feats)  # [batch_size, n_objects, hidden_dim]
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            obj_emb = block(obj_emb, obj_mask)
        
        # Object pooling with mask
        mask_expanded = obj_mask.unsqueeze(-1).float()  # [batch_size, n_objects, 1]
        masked_obj_emb = obj_emb * mask_expanded
        obj_sum = masked_obj_emb.sum(dim=1)  # [batch_size, hidden_dim]
        obj_count = mask_expanded.sum(dim=1).clamp(min=1.0)  # [batch_size, 1]
        obj_avg = obj_sum / obj_count  # [batch_size, hidden_dim]
        
        # Combine global and object features
        combined = torch.cat([global_emb, obj_avg], dim=1)  # [batch_size, hidden_dim*2]
        
        # Final classification
        output = self.classifier(combined).squeeze(-1)  # [batch_size]
        return output

def make_model(input_dim):
    n_objects = 18
    obj_features = 9  # exists, E, pT, eta, phi, E, px, py, pz
    global_features = 2
    
    model = ParticleTransformer(
        input_dim=input_dim,
        n_objects=n_objects,
        obj_features=obj_features,
        global_features=global_features,
        hidden_dim=128,
        num_layers=3,
        dropout=0.1
    )
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Binary cross entropy loss
    criterion = nn.BCELoss()
    
    # Adam optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=False)
    
    # Initialize lists to store metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        train_preds = []
        train_targets = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.float().to(device)  # Convert to float for BCE
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_train_loss += loss.item() * X_batch.size(0)
            predictions = (outputs > 0.5).float()
            epoch_train_correct += (predictions == y_batch).sum().item()
            epoch_train_total += y_batch.size(0)
            
            # Store predictions and targets for AUC calculation
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(y_batch.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_train_loss /= epoch_train_total
        epoch_train_acc = epoch_train_correct / epoch_train_total
        try:
            train_auc = roc_auc_score(train_targets, train_preds)
        except:
            train_auc = 0.5
        
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.float().to(device)
                
                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Track metrics
                epoch_val_loss += loss.item() * X_batch.size(0)
                predictions = (outputs > 0.5).float()
                epoch_val_correct += (predictions == y_batch).sum().item()
                epoch_val_total += y_batch.size(0)
                
                # Store predictions and targets for AUC calculation
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_val_loss /= epoch_val_total
        epoch_val_acc = epoch_val_correct / epoch_val_total
        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except:
            val_auc = 0.5
        
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Train AUC: {train_auc:.4f} - '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val AUC: {val_auc:.4f}')
    
    return model, train_loss, val_loss, train_acc, val_acc

# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    pre = make_preprocessor()
    pre.fit(X_train, Y_train)
    X_train = pre.transform(X_train)
    X_val = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    model = make_model(input_dim=X_train.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy = torch.zeros(8, X_train.shape[1])      # 8 fake events
        try:
            _ = trained_model(pre.transform(toy))
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return  # no files in dry-run

    # 4. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

