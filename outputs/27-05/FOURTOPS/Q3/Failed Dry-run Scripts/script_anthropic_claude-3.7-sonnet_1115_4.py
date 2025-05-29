
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
import torch.nn.functional as F

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # Define stateful components
        self.scaler = StandardScaler()
        self.max_objects = 18  # Maximum number of objects in the dataset
        self.feature_dim = 4   # E, pT, eta, phi for each object
        
    def fit(self, X, y=None):
        # Extract all particle features and fit scaler
        particle_features = []
        
        for i in range(X.shape[0]):
            # Extract particles (objects)
            for j in range(18):  # 18 max objects
                obj_idx = 2 + j * 5  # Start of each object's data
                # Check if this is a valid object (not padding)
                if obj_idx < X.shape[1] and X[i, obj_idx] != 0:
                    features = X[i, obj_idx+1:obj_idx+5].numpy()  # E, pT, eta, phi
                    particle_features.append(features)
        
        # Fit scaler on all valid particle features
        if particle_features:
            self.scaler.fit(np.vstack(particle_features))
        
        # Compute global statistics for ET_miss and phi_ET_miss
        et_miss = X[:, 0].numpy().reshape(-1, 1)
        phi_et_miss = X[:, 1].numpy().reshape(-1, 1)
        
        self.et_miss_mean = np.mean(et_miss)
        self.et_miss_std = np.std(et_miss)
        self.phi_et_miss_mean = np.mean(phi_et_miss)
        self.phi_et_miss_std = np.std(phi_et_miss)
        
        return self

    def transform(self, X):
        batch_size = X.shape[0]
        
        # Create vectors of transformed features
        transformed_data = []
        
        # Process each event
        for i in range(batch_size):
            event_features = []
            
            # Normalize ET_miss and add to features
            et_miss_norm = (X[i, 0].item() - self.et_miss_mean) / self.et_miss_std
            phi_et_miss_norm = (X[i, 1].item() - self.phi_et_miss_mean) / self.phi_et_miss_std
            event_features.extend([et_miss_norm, phi_et_miss_norm])
            
            # Count valid particles
            valid_particles = 0
            particles = []
            
            # Extract and normalize particle features
            for j in range(self.max_objects):
                obj_idx = 2 + j * 5
                if obj_idx < X.shape[1] and X[i, obj_idx] != 0:  # Valid object
                    # Extract object type and kinematics
                    obj_type = X[i, obj_idx].item()
                    E = X[i, obj_idx + 1].item()
                    pT = X[i, obj_idx + 2].item()
                    eta = X[i, obj_idx + 3].item()
                    phi = X[i, obj_idx + 4].item()
                    
                    # Physical features
                    px = pT * np.cos(phi)
                    py = pT * np.sin(phi)
                    pz = pT * np.sinh(eta)
                    mass = np.sqrt(max(0, E*E - px*px - py*py - pz*pz))
                    
                    # Normalize kinematics
                    features = np.array([[E, pT, eta, phi]])
                    scaled_features = self.scaler.transform(features)[0]
                    
                    # Create final feature vector for this particle
                    particle_vector = [
                        obj_type,          # Object type
                        scaled_features[0], # Scaled E
                        scaled_features[1], # Scaled pT
                        scaled_features[2], # Scaled eta
                        scaled_features[3], # Scaled phi
                        mass,               # Derived mass
                        E/pT,               # E/pT ratio
                        px,                 # Momentum x component
                        py,                 # Momentum y component
                        pz                  # Momentum z component
                    ]
                    
                    particles.append(particle_vector)
                    valid_particles += 1
            
            # Add global event features
            event_features.append(valid_particles)  # Number of particles
            
            # Create a fixed-size representation with padding
            padded_particles = np.zeros((self.max_objects, 10))
            for j in range(min(len(particles), self.max_objects)):
                padded_particles[j] = particles[j]
            
            # Flatten particle features and add to event features
            event_features.extend(padded_particles.flatten().tolist())
            
            transformed_data.append(event_features)
        
        # Convert to tensor
        return torch.tensor(transformed_data, dtype=torch.float32)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class SlotAttention(nn.Module):
    def __init__(self, input_dim, slot_dim, num_slots, num_iterations=3):
        super().__init__()
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.epsilon = 1e-8
        
        # Learned slot parameters
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, slot_dim)
        self.norm_input = nn.LayerNorm(slot_dim)
        
        # Slot projection and updates
        self.q_proj = nn.Linear(slot_dim, slot_dim)
        self.k_proj = nn.Linear(slot_dim, slot_dim)
        self.v_proj = nn.Linear(slot_dim, slot_dim)
        
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.slot_norm = nn.LayerNorm(slot_dim)
        
    def forward(self, inputs, batch_size=None):
        # inputs shape: [batch_size, num_objects, input_dim]
        if batch_size is None:
            batch_size = inputs.shape[0]
        num_objects = inputs.shape[1]
        
        # Project inputs to slot dimension
        inputs = self.input_proj(inputs)  # [batch_size, num_objects, slot_dim]
        inputs = self.norm_input(inputs)
        
        # Initialize slots
        slots_mean = self.slots_mu.expand(batch_size, -1, -1)
        slots_sigma = torch.exp(self.slots_log_sigma.expand(batch_size, -1, -1))
        noise = torch.randn_like(slots_mean)
        slots = slots_mean + slots_sigma * noise  # [batch_size, num_slots, slot_dim]
        
        # Iterative refinement
        for _ in range(self.num_iterations):
            # Attention
            slots_prev = slots
            
            slots = self.slot_norm(slots)
            q = self.q_proj(slots)  # [batch_size, num_slots, slot_dim]
            k = self.k_proj(inputs)  # [batch_size, num_objects, slot_dim]
            v = self.v_proj(inputs)  # [batch_size, num_objects, slot_dim]
            
            # Compute attention
            qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.slot_dim)  # [batch_size, num_slots, num_objects]
            attn = F.softmax(qk, dim=-1)  # [batch_size, num_slots, num_objects]
            attn_weighted_avg = torch.matmul(attn, v)  # [batch_size, num_slots, slot_dim]
            
            # Update slots with GRU
            slots = slots.reshape(-1, self.slot_dim)
            attn_weighted_avg = attn_weighted_avg.reshape(-1, self.slot_dim)
            slots = self.gru(attn_weighted_avg, slots)
            slots = slots.reshape(batch_size, self.num_slots, self.slot_dim)
            
        return slots, attn  # slots: [batch_size, num_slots, slot_dim], attn: [batch_size, num_slots, num_objects]

class ParticleTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, num_slots=4, slot_dim=64, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        
        # Process particles
        self.particle_embed = nn.Linear(10, embedding_dim)  # 10 features per particle
        
        # Process global features (ET_miss, phi_ET_miss, num_particles)
        self.global_embed = nn.Linear(3, embedding_dim)
        
        # Transformer layers for particle embedding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads,
            dim_feedforward=embedding_dim*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Slot attention to group particles
        self.slot_attention = SlotAttention(
            input_dim=embedding_dim,
            slot_dim=slot_dim,
            num_slots=num_slots,
            num_iterations=3
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_slots * slot_dim + embedding_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract global features and particles
        global_features = x[:, :3]  # ET_miss, phi_ET_miss, num_particles
        particles_flat = x[:, 3:]
        
        # Reshape to [batch_size, num_particles, 10]  
        particles = particles_flat.reshape(batch_size, 18, 10)
        
        # Embed particles 
        particle_embeddings = self.particle_embed(particles)  # [batch_size, 18, embedding_dim]
        
        # Create attention mask for valid particles
        mask = particles[:, :, 0] == 0  # Mask if object type is 0 (padding)
        
        # Process with transformer
        particle_embeddings = self.transformer(particle_embeddings, src_key_padding_mask=mask)  # [batch_size, 18, embedding_dim]
        
        # Apply slot attention (exclude padding)
        valid_embeddings = []
        for i in range(batch_size):
            # Get valid particles for this sample
            valid_idx = ~mask[i]
            if valid_idx.sum() > 0:
                valid_embeddings.append(particle_embeddings[i, valid_idx])
            else:
                # Handle edge case with no valid particles
                valid_embeddings.append(particle_embeddings[i, 0:1])
        
        # Process each event separately with slot attention
        all_slots = []
        for emb in valid_embeddings:
            # Add batch dimension for slot attention
            emb = emb.unsqueeze(0)  
            slots, _ = self.slot_attention(emb, batch_size=1) # [1, num_slots, slot_dim]
            all_slots.append(slots.squeeze(0))
        
        # Stack all slot results  
        slots = torch.stack(all_slots)  # [batch_size, num_slots, slot_dim]
        
        # Process global features
        global_embedding = self.global_embed(global_features)
        
        # Final classifier input: flatten slots + global features
        slots_flat = slots.reshape(batch_size, -1)  # [batch_size, num_slots * slot_dim]
        classifier_input = torch.cat([slots_flat, global_embedding], dim=1)
        
        # Final classification
        logits = self.classifier(classifier_input).squeeze(-1)
        return logits

def make_model(input_dim):
    # For our preprocessed data: 3 global features + 18 particles * 10 features = 183
    model = ParticleTransformer(input_dim)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Initialize tracking variables
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_auc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        train_total = 0
        train_y_true = []
        train_y_pred = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch.float())
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record metrics
            preds = (torch.sigmoid(logits) > 0.5).float()
            epoch_train_correct += (preds == y_batch.float()).sum().item()
            train_total += y_batch.size(0)
            epoch_train_loss += loss.item() * y_batch.size(0)
            
            # Save for AUC calculation
            train_y_true.extend(y_batch.cpu().numpy())
            train_y_pred.extend(torch.sigmoid(logits).detach().cpu().numpy())
        
        # Calculate training metrics
        epoch_train_loss /= train_total
        epoch_train_acc = epoch_train_correct / train_total
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_correct = 0
        val_total = 0
        val_y_true = []
        val_y_pred = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                logits = model(X_batch)
                loss = criterion(logits, y_batch.float())
                
                # Record metrics
                preds = (torch.sigmoid(logits) > 0.5).float()
                epoch_val_correct += (preds == y_batch.float()).sum().item()
                val_total += y_batch.size(0)
                epoch_val_loss += loss.item() * y_batch.size(0)
                
                # Save for AUC calculation
                val_y_true.extend(y_batch.cpu().numpy())
                val_y_pred.extend(torch.sigmoid(logits).detach().cpu().numpy())
        
        # Calculate validation metrics
        epoch_val_loss /= val_total
        epoch_val_acc = epoch_val_correct / val_total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_y_true, val_y_pred)
        
        # Update learning rate based on validation AUC
        scheduler.step(val_auc)
        
        # Print metrics
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | Val AUC: {val_auc:.4f}')
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
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

