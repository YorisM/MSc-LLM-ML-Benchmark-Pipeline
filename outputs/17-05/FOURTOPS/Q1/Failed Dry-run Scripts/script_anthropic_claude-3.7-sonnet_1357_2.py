
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
# Imports: torch, numpy, pandas, matplotlib, sklearn are already available
# via the wrapper. Only import extra std-lib modules or torch.nn sub-modules
# you actually use.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
import math

class MyPreprocessor:
    def __init__(self):
        # Number of features per object
        self.obj_features = 5
        # Maximum number of objects
        self.max_objects = 18
        # Constants for preprocessing
        self.means = None
        self.stds = None
        self.log_scale_features = [0, 2, 3]  # E_T_miss, E, p_T
        self.angle_features = [1, 4]  # phi_E_T_miss, phi
        # Feature mapping to convert object type to one-hot
        self.object_types = None
        
    def fit(self, X, y=None):
        # Extract object types
        obj_ids = []
        for i in range(self.max_objects):
            obj_col = X[:, 2 + i * self.obj_features].cpu().numpy() if isinstance(X, torch.Tensor) else X[:, 2 + i * self.obj_features]
            unique_ids = np.unique(obj_col)
            unique_ids = unique_ids[unique_ids != 0]  # Remove padding
            obj_ids.extend(unique_ids)
        
        self.object_types = sorted(list(set(obj_ids)))
        
        # Identify non-zero elements for normalization
        non_zero_mask = X != 0
        
        # Extract features for normalization (ignore object IDs)
        features_to_normalize = []
        for i in range(2):  # First 2 features: E_T_miss, phi_E_T_miss
            features_to_normalize.append(X[:, i])
        
        for i in range(self.max_objects):
            base_idx = 2 + i * self.obj_features
            for j in range(1, self.obj_features):  # Skip object ID
                features_to_normalize.append(X[:, base_idx + j])
        
        # Calculate mean and std for each feature
        self.means = []
        self.stds = []
        
        for i, feat in enumerate(features_to_normalize):
            if i % self.obj_features == 1 or i % self.obj_features == 4:  # Angular features
                self.means.append(0)
                self.stds.append(1)
            else:
                non_zero_vals = feat[feat != 0]
                if len(non_zero_vals) > 0:
                    self.means.append(float(np.mean(non_zero_vals)))
                    std = float(np.std(non_zero_vals))
                    self.stds.append(std if std > 1e-6 else 1.0)
                else:
                    self.means.append(0)
                    self.stds.append(1)
        
        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.clone().cpu().numpy()
        
        n_samples = X.shape[0]
        
        # Process E_T_miss and phi_E_T_miss
        transformed_features = []
        
        # Log-scale and normalize E_T_miss
        et_miss = X[:, 0]
        et_miss_processed = np.zeros_like(et_miss)
        non_zero_mask = et_miss != 0
        et_miss_processed[non_zero_mask] = np.log1p(et_miss[non_zero_mask])
        et_miss_normalized = (et_miss_processed - self.means[0]) / self.stds[0] if self.means is not None else et_miss_processed
        transformed_features.append(et_miss_normalized)
        
        # Handle phi_E_T_miss (circular feature)
        phi_miss = X[:, 1]
        transformed_features.append(np.sin(phi_miss))
        transformed_features.append(np.cos(phi_miss))
        
        # Process per-object features
        object_presence = np.zeros((n_samples, self.max_objects), dtype=np.float32)
        
        for i in range(self.max_objects):
            base_idx = 2 + i * self.obj_features
            
            # Object type one-hot encoding
            obj_id = X[:, base_idx]
            obj_mask = obj_id != 0
            object_presence[:, i] = obj_mask.astype(np.float32)
            
            if self.object_types:
                obj_type_onehot = np.zeros((n_samples, len(self.object_types)), dtype=np.float32)
                for j, obj_type in enumerate(self.object_types):
                    obj_type_onehot[:, j] = (obj_id == obj_type).astype(np.float32)
                transformed_features.extend([obj_type_onehot[:, j] for j in range(len(self.object_types))])
            
            # Process E, p_T, eta, phi
            for j in range(1, self.obj_features):
                feat_idx = base_idx + j
                feat = X[:, feat_idx].copy()
                
                # Log scale for energy and momentum
                if j in [1, 2]:  # E, p_T
                    non_zero_mask = feat != 0
                    feat_log = np.zeros_like(feat)
                    feat_log[non_zero_mask] = np.log1p(feat[non_zero_mask])
                    feat = feat_log
                
                # Normalize non-angular features
                if j != 4:  # Not phi
                    mean_idx = 1 + (i * (self.obj_features-1)) + (j-1)
                    if self.means is not None:
                        feat = (feat - self.means[mean_idx]) / self.stds[mean_idx] * obj_mask
                    else:
                        feat = feat * obj_mask
                    transformed_features.append(feat)
                else:  # phi - circular feature
                    transformed_features.append(np.sin(feat) * obj_mask)
                    transformed_features.append(np.cos(feat) * obj_mask)
        
        # Add object count as a feature
        obj_count = np.sum(object_presence, axis=1, keepdims=True)
        transformed_features.append(obj_count)
        
        # Stack all features
        result = np.column_stack(transformed_features)
        
        return torch.tensor(result, dtype=torch.float32)

def make_preprocessor():
    return MyPreprocessor()

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class PhysicsInformedNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # First process features with MLP
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Reshape for attention layers
        self.embed_dim = 64
        self.seq_len = 4
        self.reshape_linear = nn.Linear(256, self.seq_len * self.embed_dim)
        
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(self.embed_dim, num_heads=4) for _ in range(3)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # Initial feature processing
        x = self.input_layer(x)
        
        # Reshape for attention
        x = self.reshape_linear(x)
        x = x.view(-1, self.seq_len, self.embed_dim)
        
        # Apply attention blocks
        for attn_block in self.attention_blocks:
            x = attn_block(x)
        
        # Flatten and output
        x = x.reshape(x.shape[0], -1)
        x = self.output_layers(x)
        
        return x.squeeze(-1)

def make_model(input_dim):
    return PhysicsInformedNN(input_dim)

EPOCHS = 20

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Initialize focal loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Track metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_outputs_list = []
        train_targets_list = []
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets.float())
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            epoch_train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0).long()
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
            
            # Store outputs and targets for AUC calculation
            train_outputs_list.append(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets_list.append(targets.cpu().numpy())
        
        # Update learning rate
        scheduler.step()
        
        # Calculate training metrics
        epoch_train_loss /= len(train_loader.dataset)
        train_loss.append(epoch_train_loss)
        epoch_train_acc = train_correct / train_total
        train_acc.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_outputs_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets.float())
                
                # Track metrics
                epoch_val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0).long()
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
                
                # Store outputs and targets for AUC calculation
                val_outputs_list.append(torch.sigmoid(outputs).cpu().numpy())
                val_targets_list.append(targets.cpu().numpy())
        
        # Calculate validation metrics
        epoch_val_loss /= len(val_loader.dataset)
        val_loss.append(epoch_val_loss)
        epoch_val_acc = val_correct / val_total
        val_acc.append(epoch_val_acc)
        
        # Calculate AUC scores
        train_outputs_flat = np.concatenate(train_outputs_list)
        train_targets_flat = np.concatenate(train_targets_list)
        val_outputs_flat = np.concatenate(val_outputs_list)
        val_targets_flat = np.concatenate(val_targets_list)
        
        train_auc = roc_auc_score(train_targets_flat, train_outputs_flat)
        val_auc = roc_auc_score(val_targets_flat, val_outputs_flat)
    
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

