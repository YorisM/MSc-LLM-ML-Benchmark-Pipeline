
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
import math
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.max_objects = 18
        self.feature_dim = 4  # E, pT, eta, phi
        self.obj_indices = []
        for i in range(self.max_objects):
            start_idx = 2 + i * 5
            self.obj_indices.append((start_idx + 1, start_idx + 2, start_idx + 3, start_idx + 4))
        
        # For physics-informed features
        self.mass_mean = 0.0
        self.mass_std = 1.0
        self.dR_mean = 0.0
        self.dR_std = 1.0
        self.mt_mean = 0.0
        self.mt_std = 1.0
        
    def fit(self, X, y=None):
        # Extract features from all objects (skip the object ID)
        features = []
        
        # Extract missing ET features (first 2 elements)
        missing_et_features = X[:, :2].numpy()  # [N, 2]
        features.append(missing_et_features)
        
        # Extract object features
        valid_objects = []
        masses = []
        delta_Rs = []
        mt_values = []
        
        for batch_idx in range(X.shape[0]):
            batch_features = []
            batch_objects = []
            
            for i in range(self.max_objects):
                obj_id = X[batch_idx, 2 + i * 5].item()
                if obj_id != 0:  # Valid object
                    E = X[batch_idx, self.obj_indices[i][0]].item()
                    pt = X[batch_idx, self.obj_indices[i][1]].item()
                    eta = X[batch_idx, self.obj_indices[i][2]].item()
                    phi = X[batch_idx, self.obj_indices[i][3]].item()
                    
                    # Add standard features
                    batch_features.extend([E, pt, eta, phi])
                    
                    # Compute mass: M^2 = E^2 - |p|^2
                    px = pt * np.cos(phi)
                    py = pt * np.sin(phi)
                    pz = pt * np.sinh(eta)
                    p_squared = px**2 + py**2 + pz**2
                    mass_squared = E**2 - p_squared
                    mass = np.sqrt(max(0, mass_squared))
                    masses.append(mass)
                    
                    # Store valid object for delta R calculation
                    batch_objects.append((eta, phi))
                    
                    # Compute transverse mass with missing ET
                    met = missing_et_features[batch_idx, 0]
                    met_phi = missing_et_features[batch_idx, 1]
                    mt = np.sqrt(2 * pt * met * (1 - np.cos(phi - met_phi)))
                    mt_values.append(mt)
                else:
                    # Padding for non-existent objects
                    batch_features.extend([0, 0, 0, 0])
            
            # Calculate delta R between all pairs of objects
            for i in range(len(batch_objects)):
                for j in range(i+1, len(batch_objects)):
                    eta1, phi1 = batch_objects[i]
                    eta2, phi2 = batch_objects[j]
                    deta = eta1 - eta2
                    dphi = np.abs(phi1 - phi2)
                    # Handle the circular nature of phi
                    if dphi > np.pi:
                        dphi = 2 * np.pi - dphi
                    delta_Rs.append(np.sqrt(deta**2 + dphi**2))
            
            valid_objects.append(batch_features)
        
        # Convert to numpy arrays for scaling
        valid_objects = np.array(valid_objects)  # [N, max_objects*4]
        
        # Fit scaler on valid objects
        self.scaler.fit(valid_objects)
        
        # Compute statistics for the physics-informed features
        if masses:
            self.mass_mean = np.mean(masses)
            self.mass_std = np.std(masses) if np.std(masses) > 0 else 1.0
        
        if delta_Rs:
            self.dR_mean = np.mean(delta_Rs)
            self.dR_std = np.std(delta_Rs) if np.std(delta_Rs) > 0 else 1.0
        
        if mt_values:
            self.mt_mean = np.mean(mt_values)
            self.mt_std = np.std(mt_values) if np.std(mt_values) > 0 else 1.0
        
        return self

    def transform(self, X):
        # Extract and transform standard features
        batch_size = X.shape[0]
        transformed_features = []
        
        # Extract missing ET features (first 2 elements)
        missing_et_features = X[:, :2].numpy()  # [N, 2]
        transformed_features.append(torch.tensor(missing_et_features, dtype=torch.float32))
        
        # Process object features
        obj_features_list = []
        masses_list = []
        mt_values_list = []
        
        for batch_idx in range(batch_size):
            batch_objects = []
            batch_masses = []
            batch_mt_values = []
            
            # Process each object
            for i in range(self.max_objects):
                obj_id = X[batch_idx, 2 + i * 5].item()
                if obj_id != 0:  # Valid object
                    E = X[batch_idx, self.obj_indices[i][0]].item()
                    pt = X[batch_idx, self.obj_indices[i][1]].item()
                    eta = X[batch_idx, self.obj_indices[i][2]].item()
                    phi = X[batch_idx, self.obj_indices[i][3]].item()
                    
                    # Add standard features
                    features = [E, pt, eta, phi]
                    
                    # Compute mass: M^2 = E^2 - |p|^2
                    px = pt * np.cos(phi)
                    py = pt * np.sin(phi)
                    pz = pt * np.sinh(eta)
                    p_squared = px**2 + py**2 + pz**2
                    mass_squared = E**2 - p_squared
                    mass = np.sqrt(max(0, mass_squared))
                    batch_masses.append((mass - self.mass_mean) / self.mass_std)
                    
                    # Store object for delta R calculation
                    batch_objects.append(features)
                    
                    # Compute transverse mass with missing ET
                    met = missing_et_features[batch_idx, 0]
                    met_phi = missing_et_features[batch_idx, 1]
                    mt = np.sqrt(2 * pt * met * (1 - np.cos(phi - met_phi)))
                    batch_mt_values.append((mt - self.mt_mean) / self.mt_std)
                else:
                    batch_objects.append([0, 0, 0, 0])
                    batch_masses.append(0)
                    batch_mt_values.append(0)
            
            obj_features_list.append(batch_objects)
            masses_list.append(batch_masses)
            mt_values_list.append(batch_mt_values)
        
        # Convert to numpy arrays and apply scaling
        obj_features_array = np.array(obj_features_list)  # [batch_size, max_objects, 4]
        obj_features_reshaped = obj_features_array.reshape(batch_size, -1)  # [batch_size, max_objects*4]
        
        # Scale the features
        scaled_features = self.scaler.transform(obj_features_reshaped)
        
        # Reshape back
        scaled_features = scaled_features.reshape(batch_size, self.max_objects, 4)  # [batch_size, max_objects, 4]
        
        # Convert to tensor
        obj_features_tensor = torch.tensor(scaled_features, dtype=torch.float32)  # [batch_size, max_objects, 4]
        
        # Add object masses and transverse masses as additional features
        masses_tensor = torch.tensor(masses_list, dtype=torch.float32).unsqueeze(-1)  # [batch_size, max_objects, 1]
        mt_values_tensor = torch.tensor(mt_values_list, dtype=torch.float32).unsqueeze(-1)  # [batch_size, max_objects, 1]
        
        # Concatenate standard and physics-informed features for each object
        enhanced_features = torch.cat([obj_features_tensor, masses_tensor, mt_values_tensor], dim=2)  # [batch_size, max_objects, 6]
        
        # Create a tensor to indicate which positions have valid objects
        valid_mask = (enhanced_features.sum(dim=2) != 0).float().unsqueeze(2)  # [batch_size, max_objects, 1]
        
        # Combine features with mask
        final_object_features = torch.cat([enhanced_features, valid_mask], dim=2)  # [batch_size, max_objects, 7]
        
        return final_object_features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv_proj(x)  # [batch_size, seq_len, dim*3]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention scores shape
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)
        return self.out_proj(attn_output)

class SlotAttention(nn.Module):
    def __init__(self, slot_dim, num_slots, hidden_dim, num_iterations=3):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        
        # Initialize slots parameters (learnable)
        self.slots = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slot_mu = nn.Parameter(torch.zeros(1, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, slot_dim))
        
        # Attention layers
        self.norm_input = nn.LayerNorm(hidden_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(hidden_dim, slot_dim)
        self.to_v = nn.Linear(hidden_dim, slot_dim)
        
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim)
        )
        
    def forward(self, inputs, mask=None):
        batch_size, num_inputs, hidden_dim = inputs.shape
        slots = self.slots.repeat(batch_size, 1, 1)  # [batch_size, num_slots, slot_dim]
        
        # Apply mask to inputs
        if mask is not None:
            inputs = inputs * mask.unsqueeze(-1)
        
        inputs = self.norm_input(inputs)  # [batch_size, num_inputs, hidden_dim]
        k = self.to_k(inputs)  # [batch_size, num_inputs, slot_dim]
        v = self.to_v(inputs)  # [batch_size, num_inputs, slot_dim]
        
        for _ in range(self.num_iterations):
            slots_prev = slots
            
            # Slot attention
            slots = self.norm_slots(slots)
            q = self.to_q(slots)  # [batch_size, num_slots, slot_dim]
            
            # Compute attention weights
            scale = self.slot_dim ** -0.5
            attn_logits = torch.matmul(q, k.transpose(1, 2)) * scale  # [batch_size, num_slots, num_inputs]
            
            if mask is not None:
                # Apply mask to attention scores
                attn_mask = mask.squeeze(-1).unsqueeze(1)  # [batch_size, 1, num_inputs]
                attn_logits = attn_logits.masked_fill(attn_mask == 0, -1e9)
            
            attn = F.softmax(attn_logits, dim=-1)  # [batch_size, num_slots, num_inputs]
            attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)  # Normalize
            
            # Weighted mean of inputs according to attention
            updates = torch.matmul(attn, v)  # [batch_size, num_slots, slot_dim]
            
            # Update slots with GRU
            slots = slots_prev.reshape(-1, self.slot_dim)
            updates = updates.reshape(-1, self.slot_dim)
            slots = self.gru(updates, slots)
            slots = slots.reshape(batch_size, self.num_slots, self.slot_dim)
            
            slots = slots + self.mlp(self.norm_slots(slots))
        
        return slots

class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class PhysicsInformedTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_slots, num_heads, num_layers, ff_dim, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.slot_attention = SlotAttention(
            slot_dim=hidden_dim,
            num_slots=num_slots,
            hidden_dim=hidden_dim,
            num_iterations=3
        )
        
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, max_objects, input_dim = x.shape
        
        # Extract mask from input features (last dimension)
        mask = x[:, :, -1]  # [batch_size, max_objects]
        x = x[:, :, :-1]    # [batch_size, max_objects, input_dim-1]
        
        # Project inputs to hidden dimension
        x = self.input_projection(x)  # [batch_size, max_objects, hidden_dim]
        
        # Apply slot attention to group particles
        slots = self.slot_attention(x, mask.unsqueeze(-1))  # [batch_size, num_slots, hidden_dim]
        
        # Apply transformer layers to refine the slot representations
        x = slots
        for layer in self.transformer_layers:
            x = layer(x)  # [batch_size, num_slots, hidden_dim]
        
        # Global average pooling over slots
        x = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Classify
        output = self.classifier(x)  # [batch_size, 1]
        return output.squeeze(-1)  # [batch_size]

def make_model(input_dim: int):
    # For this task, input_dim is the number of features per event after preprocessing
    # Our preprocessing outputs: [batch_size, max_objects, 7]
    # where max_objects=18, and 7 features per object (4 standard + 2 physics + 1 mask)
    model = PhysicsInformedTransformer(
        input_dim=6,  # 6 features per object (excluding mask)
        hidden_dim=64,
        num_slots=4,  # 4 slots for the 4 top quarks
        num_heads=4,
        num_layers=3,
        ff_dim=128,
        dropout=0.1
    )
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
    
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    best_val_auc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).float()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * data.size(0)
            predicted = (output > 0.5).float()
            epoch_train_correct += (predicted == target).sum().item()
            epoch_train_total += target.size(0)
        
        epoch_train_loss = epoch_train_loss / epoch_train_total
        epoch_train_acc = epoch_train_correct / epoch_train_total
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0
        val_outputs = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device).float()
                
                output = model(data)
                loss = criterion(output, target)
                
                epoch_val_loss += loss.item() * data.size(0)
                predicted = (output > 0.5).float()
                epoch_val_correct += (predicted == target).sum().item()
                epoch_val_total += target.size(0)
                
                val_outputs.append(output.cpu().numpy())
                val_targets.append(target.cpu().numpy())
        
        epoch_val_loss = epoch_val_loss / epoch_val_total
        epoch_val_acc = epoch_val_correct / epoch_val_total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        # Compute AUC for validation
        val_outputs = np.concatenate(val_outputs)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_outputs)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train loss: {epoch_train_loss:.4f}, Train acc: {epoch_train_acc:.4f}, '
              f'Val loss: {epoch_val_loss:.4f}, Val acc: {epoch_val_acc:.4f}, Val AUC: {val_auc:.4f}')
    
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

