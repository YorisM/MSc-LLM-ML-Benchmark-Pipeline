
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
        self.mean = None
        self.std = None
        self.nonzero_mask = None
        self.num_objects = 18  # Max number of objects
        self.features_per_object = 5  # Number of features per object

    def fit(self, X, y=None):
        # Get indices for each object and its features
        object_indices = [2 + i * self.features_per_object for i in range(self.num_objects)]
        
        # Create a mask for non-zero object IDs (to identify actual particles vs padding)
        nonzero_mask = (X[:, object_indices] != 0)
        self.nonzero_mask = nonzero_mask
        
        # Extract valid features for normalization (ignoring padding)
        valid_features = []
        for i in range(self.num_objects):
            # Base index for this object
            base_idx = 2 + i * self.features_per_object
            
            # Only include non-zero objects
            mask = X[:, base_idx] != 0
            
            if mask.sum() > 0:
                # Add E, pT, eta, phi features for valid objects
                for j in range(1, self.features_per_object):
                    feat_idx = base_idx + j
                    valid_features.append(X[mask, feat_idx])
        
        # Also include missing ET and phi
        valid_features.append(X[:, 0])  # ET_miss
        valid_features.append(X[:, 1])  # phi_ET_miss
        
        # Concatenate all valid features for normalization
        all_valid = torch.cat(valid_features)
        
        # Compute mean and std for normalization
        self.mean = all_valid.mean()
        self.std = all_valid.std()
        
        return self

    def transform(self, X):
        # Create a copy to avoid modifying the original data
        X_transformed = X.clone()
        
        # Normalize ET_miss and phi_ET_miss
        X_transformed[:, 0] = (X[:, 0] - self.mean) / self.std
        X_transformed[:, 1] = torch.sin(X[:, 1])  # Sin transform for azimuthal angle
        X_transformed = torch.cat([X_transformed[:, :2], torch.cos(X[:, 1]).unsqueeze(1)], dim=1)  # Add cos transform
        
        # Create a container for all processed features
        processed_features = [X_transformed[:, :3]]  # [ET_miss, sin(phi), cos(phi)]
        
        # Process each object
        for i in range(self.num_objects):
            # Base index for this object
            base_idx = 2 + i * self.features_per_object
            
            # Extract object ID and features
            obj_id = X[:, base_idx].unsqueeze(1)
            E = X[:, base_idx + 1].unsqueeze(1)
            pT = X[:, base_idx + 2].unsqueeze(1)
            eta = X[:, base_idx + 3].unsqueeze(1)
            phi = X[:, base_idx + 4].unsqueeze(1)
            
            # Feature engineering: Add meaningful physics-based features
            obj_mask = (obj_id != 0).float()
            
            # Normalize E and pT
            E_norm = (E - self.mean) / self.std
            pT_norm = (pT - self.mean) / self.std
            
            # Process angular features
            sin_phi = torch.sin(phi)
            cos_phi = torch.cos(phi)
            
            # Calculate transverse mass: sqrt(E^2 - pT^2)
            mT = torch.sqrt(torch.clamp(E**2 - pT**2, min=1e-8))
            mT_norm = (mT - self.mean) / self.std
            
            # pT/E ratio as a feature
            pT_E_ratio = pT / torch.clamp(E, min=1e-8)
            
            # Object type one-hot encoding (handle as numerical for simplicity)
            # Objects are: 11: electron, 13: muon, 15: tau, 5: b-jet, 1: light jet
            # We'll normalize them to be in a reasonable range
            obj_type_norm = obj_id / 20.0
            
            # Combine features for this object
            obj_features = torch.cat([
                obj_type_norm, E_norm, pT_norm, eta, sin_phi, cos_phi, 
                mT_norm, pT_E_ratio, obj_mask
            ], dim=1)  # Shape: [batch_size, 9]
            
            processed_features.append(obj_features)
        
        # Concatenate all features
        return torch.cat(processed_features, dim=1)  # Shape: [batch_size, 3 + 18*9]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# Slot Attention module
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        
        # Parameters for slots
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, num_slots, dim))
        
        # Layer norm for inputs and slots
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        
        # Linear projections
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        # GRU for slot updates
        self.gru = nn.GRUCell(dim, dim)
        
        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, inputs, mask=None):
        b, n, d = inputs.shape
        n_s = self.num_slots
        
        # Initialize slots using reparameterization trick
        mu = self.slots_mu.expand(b, -1, -1)
        sigma = F.softplus(self.slots_sigma).expand(b, -1, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=mu.device)
        
        # Apply layer norm to inputs
        inputs = self.norm_input(inputs)
        
        # Main slot attention iterations
        for _ in range(self.iters):
            slots_prev = slots
            
            # Apply layer norm to slots
            slots = self.norm_slots(slots)
            
            # Compute queries, keys, and values
            q = self.to_q(slots)  # b, n_s, d
            k = self.to_k(inputs)  # b, n, d
            v = self.to_v(inputs)  # b, n, d
            
            # Compute attention scores
            attn = torch.einsum('bid,bjd->bij', q, k) * self.scale  # b, n_s, n
            
            # Apply mask if provided (to exclude padding)
            if mask is not None:
                attn = attn.masked_fill(~mask.unsqueeze(1), -1e9)
            
            # Softmax normalization across slots dimension
            attn = torch.softmax(attn, dim=1)  # b, n_s, n
            attn_weights = attn / (torch.sum(attn, dim=-1, keepdim=True) + self.eps)  # b, n_s, n
            
            # Weighted sum of values
            updates = torch.einsum('bij,bjd->bid', attn_weights, v)  # b, n_s, d
            
            # Update slots with GRU
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            ).reshape(b, n_s, d)
            
            # Apply MLP to refine slots
            slots = slots + self.mlp(slots)
        
        return slots, attn_weights

# Transformer encoder with positional encoding
class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]))
    
    def forward(self, x, mask=None):
        for norm1, attn, norm2, mlp in self.layers:
            # Self-attention block
            x_norm = norm1(x)
            if mask is not None:
                # Convert mask to attention mask
                attn_mask = ~mask.unsqueeze(1).repeat(1, mask.size(1), 1)
                attn_out, _ = attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
            else:
                attn_out, _ = attn(x_norm, x_norm, x_norm)
            x = x + attn_out
            
            # MLP block
            x = x + mlp(norm2(x))
        
        return self.norm(x)

# Positional encoding for transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # Add positional encoding
        return x + self.pe[:, :x.size(1)]

# 2. ---------- MODEL DEFINITION ----------
class TopQuarkClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim=256, slot_dim=128, num_slots=4,
                 transformer_dim=128, num_heads=4, transformer_depth=2, slot_iters=3):
        super().__init__()
        self.num_objects = 18
        self.features_per_obj = (input_dim - 3) // 18  # Subtract ET_miss features (3)
        
        # Embedding layers
        self.et_miss_embed = nn.Linear(3, embed_dim)  # ET_miss + sin/cos phi
        self.obj_embed = nn.Linear(self.features_per_obj, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            dim=embed_dim,
            depth=transformer_depth,
            heads=num_heads,
            dim_head=embed_dim // num_heads,
            mlp_dim=embed_dim * 4,
            dropout=0.1
        )
        
        # Slot attention module
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=embed_dim,
            iters=slot_iters,
            hidden_dim=slot_dim * 2
        )
        
        # Slot processor
        self.slot_processor = nn.Sequential(
            nn.Linear(embed_dim, slot_dim),
            nn.LayerNorm(slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(slot_dim * num_slots + embed_dim, slot_dim),
            nn.LayerNorm(slot_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(slot_dim, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Split input into ET_miss and object features
        et_miss_features = x[:, :3]  # [ET_miss, sin(phi), cos(phi)]
        object_features = x[:, 3:].view(batch_size, self.num_objects, self.features_per_obj)
        
        # Create masks for valid objects (using the mask feature at index 8)
        obj_mask = (object_features[:, :, 8] > 0)
        
        # Embed ET_miss features
        et_miss_embed = self.et_miss_embed(et_miss_features)  # [batch, embed_dim]
        
        # Embed object features
        obj_embed = self.obj_embed(object_features)  # [batch, num_objects, embed_dim]
        
        # Add positional encoding
        obj_embed = self.pos_encoding(obj_embed)
        
        # Prepend ET_miss embedding as a token
        et_miss_token = et_miss_embed.unsqueeze(1)  # [batch, 1, embed_dim]
        all_tokens = torch.cat([et_miss_token, obj_embed], dim=1)  # [batch, 1+num_objects, embed_dim]
        
        # Create mask including ET_miss token
        token_mask = torch.cat([torch.ones(batch_size, 1, device=x.device).bool(), obj_mask], dim=1)
        
        # Apply transformer encoder to all tokens
        transformed_tokens = self.transformer(all_tokens, token_mask)  # [batch, 1+num_objects, embed_dim]
        
        # Extract ET_miss transformed token and object tokens
        et_miss_token = transformed_tokens[:, 0]  # [batch, embed_dim]
        object_tokens = transformed_tokens[:, 1:]  # [batch, num_objects, embed_dim]
        
        # Apply slot attention to object tokens
        slots, _ = self.slot_attention(object_tokens, obj_mask)  # [batch, num_slots, embed_dim]
        
        # Process slots
        processed_slots = self.slot_processor(slots)  # [batch, num_slots, slot_dim]
        
        # Flatten slots and concatenate with ET_miss features
        flat_slots = processed_slots.reshape(batch_size, -1)  # [batch, num_slots * slot_dim]
        combined_features = torch.cat([flat_slots, et_miss_token], dim=1)
        
        # Pass through classifier
        logits = self.classifier(combined_features).squeeze(-1)
        
        return logits

def make_model(input_dim):
    return TopQuarkClassifier(input_dim=input_dim)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20  # Set epochs based on validation convergence

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5)
    
    # Initialize tracking metrics
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
        train_correct = 0
        train_total = 0
        train_pred_list = []
        train_true_list = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Calculate loss
            loss = criterion(outputs, y_batch.float())
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * X_batch.size(0)
            train_pred = torch.sigmoid(outputs) >= 0.5
            train_correct += (train_pred == y_batch).sum().item()
            train_total += y_batch.size(0)
            
            # Store predictions for AUC calculation
            train_pred_list.append(torch.sigmoid(outputs).cpu().detach().numpy())
            train_true_list.append(y_batch.cpu().numpy())
        
        # Calculate epoch metrics
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Concatenate all batches for AUC calculation
        train_preds = np.concatenate(train_pred_list)
        train_truth = np.concatenate(train_true_list)
        train_auc = roc_auc_score(train_truth, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_pred_list = []
        val_true_list = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                outputs = model(X_batch)
                
                # Calculate loss
                loss = criterion(outputs, y_batch.float())
                
                # Update metrics
                val_loss += loss.item() * X_batch.size(0)
                val_pred = torch.sigmoid(outputs) >= 0.5
                val_correct += (val_pred == y_batch).sum().item()
                val_total += y_batch.size(0)
                
                # Store predictions for AUC calculation
                val_pred_list.append(torch.sigmoid(outputs).cpu().numpy())
                val_true_list.append(y_batch.cpu().numpy())
        
        # Calculate epoch metrics
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Concatenate all batches for AUC calculation
        val_preds = np.concatenate(val_pred_list)
        val_truth = np.concatenate(val_true_list)
        val_auc = roc_auc_score(val_truth, val_preds)
        
        # Update scheduler based on validation AUC
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Store history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}')
    
    # Load best model
    if best_model_state is not None:
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

