
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
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import math

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.epsilon = 1e-8
        self.mean_Et = None
        self.std_Et = None
        self.mean_phi = None
        self.std_phi = None
        self.obj_types = None
        self.mean_E = None
        self.std_E = None
        self.mean_pT = None
        self.std_pT = None
        self.mean_eta = None
        self.std_eta = None
        
    def fit(self, X, y=None):
        # Extract statistics for normalization
        # Missing ET and its phi
        self.mean_Et = torch.mean(X[:, 0])
        self.std_Et = torch.std(X[:, 0]) + self.epsilon
        self.mean_phi = torch.mean(X[:, 1])
        self.std_phi = torch.std(X[:, 1]) + self.epsilon
        
        # For object features
        # We'll first reshape the data to extract per-object statistics
        max_objs = 18
        obj_slice_size = 5
        
        # Initialize masks for valid objects
        valid_mask = X[:, 2::obj_slice_size] != 0  # Object ID not zero
        
        # Extract object types and their statistics
        obj_types = X[:, 2::obj_slice_size][valid_mask]
        self.obj_types = torch.unique(obj_types).tolist()
        
        # Extract and compute statistics for E, pT, eta, phi
        valid_E = X[:, 3::obj_slice_size][valid_mask]
        valid_pT = X[:, 4::obj_slice_size][valid_mask]
        valid_eta = X[:, 5::obj_slice_size][valid_mask]
        valid_phi = X[:, 6::obj_slice_size][valid_mask]
        
        self.mean_E = torch.mean(valid_E)
        self.std_E = torch.std(valid_E) + self.epsilon
        self.mean_pT = torch.mean(valid_pT)
        self.std_pT = torch.std(valid_pT) + self.epsilon
        self.mean_eta = torch.mean(valid_eta)
        self.std_eta = torch.std(valid_eta) + self.epsilon
        self.mean_phi = torch.mean(valid_phi)
        self.std_phi = torch.std(valid_phi) + self.epsilon
        
        return self

    def transform(self, X):
        # Create a copy to avoid modifying the original data
        X_transformed = X.clone()
        
        # Normalize missing ET and its phi
        X_transformed[:, 0] = (X[:, 0] - self.mean_Et) / self.std_Et
        X_transformed[:, 1] = (X[:, 1] - self.mean_phi) / self.std_phi
        
        # Process object features
        max_objs = 18
        obj_slice_size = 5
        
        # Initialize new feature tensor
        batch_size = X.shape[0]
        
        # We'll create a tensor to hold our transformed objects
        # Format: [batch_size, max_objs, features_per_obj]
        # where features_per_obj includes normalized E, pT, eta, phi, cos(phi), sin(phi), and one-hot encoded object type
        obj_features = 7 + len(self.obj_types)  # normalized kinematics + trig + one-hot
        transformed_objects = torch.zeros((batch_size, max_objs, obj_features), dtype=torch.float32)
        
        for i in range(max_objs):
            # Get the slice indices for this object
            obj_idx = 2 + i * obj_slice_size
            E_idx = obj_idx + 1
            pT_idx = obj_idx + 2
            eta_idx = obj_idx + 3
            phi_idx = obj_idx + 4
            
            # Check if this is a valid object (obj_id != 0)
            valid_mask = X[:, obj_idx] != 0
            
            if valid_mask.sum() > 0:
                # Extract object types and create one-hot encoding
                obj_types = X[valid_mask, obj_idx]
                obj_one_hot = torch.zeros((valid_mask.sum(), len(self.obj_types)), dtype=torch.float32)
                for j, obj_type in enumerate(self.obj_types):
                    obj_one_hot[:, j] = (obj_types == obj_type).float()
                
                # Normalize E, pT, eta, phi
                norm_E = (X[valid_mask, E_idx] - self.mean_E) / self.std_E
                norm_pT = (X[valid_mask, pT_idx] - self.mean_pT) / self.std_pT
                norm_eta = (X[valid_mask, eta_idx] - self.mean_eta) / self.std_eta
                norm_phi = (X[valid_mask, phi_idx] - self.mean_phi) / self.std_phi
                
                # Add trigonometric features
                cos_phi = torch.cos(X[valid_mask, phi_idx])
                sin_phi = torch.sin(X[valid_mask, phi_idx])
                
                # Combine features
                obj_features = torch.cat([
                    norm_E.unsqueeze(1),
                    norm_pT.unsqueeze(1),
                    norm_eta.unsqueeze(1),
                    norm_phi.unsqueeze(1),
                    cos_phi.unsqueeze(1),
                    sin_phi.unsqueeze(1),
                    # Object mass (derived from E and pT)
                    ((X[valid_mask, E_idx]**2 - X[valid_mask, pT_idx]**2).clamp(min=0).sqrt() / self.std_E).unsqueeze(1),
                    obj_one_hot
                ], dim=1)
                
                # Store in the transformed tensor
                transformed_objects[valid_mask, i, :] = obj_features
        
        # Flatten the transformed objects tensor for the output
        flattened = transformed_objects.reshape(batch_size, -1)
        
        # Add missing ET features: normalized ET, cos(phi), sin(phi)
        missing_ET_features = torch.stack([
            X_transformed[:, 0],
            torch.cos(X[:, 1]),
            torch.sin(X[:, 1])
        ], dim=1)
        
        # Combine all features
        result = torch.cat([missing_ET_features, flattened], dim=1)
        
        return result

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class LorentzVector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
    def forward(self, E, px, py, pz):
        # Compute Lorentz invariant quantities
        # Mass squared: m^2 = E^2 - p^2
        p_squared = px**2 + py**2 + pz**2
        m_squared = E**2 - p_squared
        m_squared = torch.clamp(m_squared, min=0)  # Ensure non-negative
        mass = torch.sqrt(m_squared)
        
        # Transverse momentum
        pt = torch.sqrt(px**2 + py**2)
        
        # Return useful Lorentz invariant features
        return torch.cat([mass, pt, E], dim=-1)

class LorentzEquivariantBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Message passing networks
        self.msg_net = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Object feature update network
        self.update_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Lorentz-aware projection
        self.lorentz_proj = nn.Linear(input_dim, 4)  # Project to (E, px, py, pz)
        self.lorentz_vector = LorentzVector(input_dim)
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, num_objects, features]
        batch_size, num_objects, features = x.shape
        
        # For each pair of objects, compute interaction messages
        messages = []
        for i in range(num_objects):
            # Create pairs with object i and all objects
            obj_i = x[:, i:i+1].repeat(1, num_objects, 1)  # [batch, num_objects, features]
            pairs = torch.cat([obj_i, x], dim=-1)  # [batch, num_objects, 2*features]
            
            # Compute messages from all objects to object i
            msg_i = self.msg_net(pairs)  # [batch, num_objects, hidden]
            
            # Apply mask if provided
            if mask is not None:
                msg_i = msg_i * mask.unsqueeze(-1)
            
            # Aggregate messages (sum pooling)
            agg_msg = msg_i.sum(dim=1, keepdim=True)  # [batch, 1, hidden]
            messages.append(agg_msg)
        
        # Concatenate all messages
        all_messages = torch.cat(messages, dim=1)  # [batch, num_objects, hidden]
        
        # Update object features
        obj_and_msg = torch.cat([x, all_messages], dim=-1)  # [batch, num_objects, features+hidden]
        updated_x = x + self.update_net(obj_and_msg)  # Residual connection
        
        # Apply Lorentz-aware transformation
        lorentz_features = self.lorentz_proj(updated_x)  # [batch, num_objects, 4]
        E = lorentz_features[:, :, 0]
        px = lorentz_features[:, :, 1]
        py = lorentz_features[:, :, 2]
        pz = lorentz_features[:, :, 3]
        
        # Compute Lorentz invariant quantities
        lorentz_invariants = self.lorentz_vector(E, px, py, pz)  # [batch, num_objects, 3]
        
        # Combine with updated features
        result = torch.cat([updated_x, lorentz_invariants], dim=-1)  # [batch, num_objects, features+3]
        
        return result

class FourTopClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3, max_objects=18):
        super().__init__()
        self.max_objects = max_objects
        features_per_obj = (input_dim - 3) // max_objects  # -3 for missing ET features
        
        # Process missing ET features
        self.et_miss_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Process objects
        self.object_embedding = nn.Sequential(
            nn.Linear(features_per_obj, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Lorentz equivariant blocks for message passing
        self.blocks = nn.ModuleList([
            LorentzEquivariantBlock(hidden_dim, hidden_dim)
            for _ in range(num_blocks)
        ])
        
        # Global pooling and final classification
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for Lorentz features
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),  # *2 for concatenated pooled features
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Split input into missing ET and objects
        et_miss_features = x[:, :3]  # First 3 features are missing ET related
        object_features = x[:, 3:]  # Rest are object features
        
        # Reshape object features to [batch, max_objects, features_per_obj]
        features_per_obj = object_features.shape[1] // self.max_objects
        object_features = object_features.reshape(batch_size, self.max_objects, features_per_obj)
        
        # Create object mask (1 for valid objects, 0 for padding)
        # We assume an object is valid if any of its features is non-zero
        object_mask = (torch.sum(object_features != 0, dim=2) > 0).float()
        
        # Process missing ET
        et_features = self.et_miss_net(et_miss_features)
        
        # Embed object features
        obj_embeddings = self.object_embedding(object_features)
        
        # Apply Lorentz equivariant blocks
        x = obj_embeddings
        for block in self.blocks:
            x = block(x, object_mask)
        
        # Global pooling operations
        # Sum pooling
        sum_pooled = torch.sum(x * object_mask.unsqueeze(-1), dim=1)
        
        # Max pooling
        # First replace zeros with large negative values for proper max pooling
        mask_expanded = object_mask.unsqueeze(-1).expand_as(x)
        masked_x = torch.where(mask_expanded > 0, x, torch.tensor(-1e9, device=x.device))
        max_pooled = torch.max(masked_x, dim=1)[0]
        
        # Combine pooled features
        pooled_features = torch.cat([sum_pooled, max_pooled], dim=1)
        
        # Combine with missing ET features
        global_features = self.global_pool(torch.cat([pooled_features, et_features], dim=1))
        
        # Final classification
        logits = self.classifier(global_features).squeeze(-1)
        
        return logits

def make_model(input_dim):
    model = FourTopClassifier(input_dim, hidden_dim=128, num_blocks=3)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    
    # Track metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    best_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        train_samples = 0
        train_preds = []
        train_targets = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.float().to(device)
            
            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            epoch_train_loss += loss.item() * X_batch.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            epoch_train_correct += (preds == y_batch).sum().item()
            train_samples += y_batch.size(0)
            
            # Save predictions for AUC calculation
            train_preds.extend(probs.detach().cpu().numpy())
            train_targets.extend(y_batch.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_train_loss /= train_samples
        epoch_train_acc = epoch_train_correct / train_samples
        epoch_train_auc = roc_auc_score(train_targets, train_preds)
        
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        val_samples = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.float().to(device)
                
                # Forward pass
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                # Track metrics
                epoch_val_loss += loss.item() * X_batch.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                epoch_val_correct += (preds == y_batch).sum().item()
                val_samples += y_batch.size(0)
                
                # Save predictions for AUC calculation
                val_preds.extend(probs.detach().cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_val_loss /= val_samples
        epoch_val_acc = epoch_val_correct / val_samples
        epoch_val_auc = roc_auc_score(val_targets, val_preds)
        
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        # Update learning rate based on validation AUC
        scheduler.step(epoch_val_auc)
        
        # Save best model
        if epoch_val_auc > best_auc:
            best_auc = epoch_val_auc
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Train AUC: {epoch_train_auc:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val AUC: {epoch_val_auc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
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

