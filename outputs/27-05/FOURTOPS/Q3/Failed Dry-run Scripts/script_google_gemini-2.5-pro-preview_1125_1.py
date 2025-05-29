
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

import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# 0. ---------- IMPORTS ----------
# (No new imports needed beyond what's provided and an F for functional)

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    # Constants for feature indices
    MET_E_IDX, MET_PHI_IDX = 0, 1
    OBJ_ID_IDX, OBJ_E_IDX, OBJ_PT_IDX, OBJ_ETA_IDX, OBJ_PHI_IDX = 0, 1, 2, 3, 4
    NUM_OBJ_FEATURES = 5 # obj_id, E, pT, eta, phi
    MAX_OBJECTS = 18

    # Features to be standardized for particles (after initial transformation)
    # obj_id_transformed, E_transformed, pT_transformed, eta_transformed
    PARTICLE_STANDARDIZE_INDICES = [0, 1, 2, 3]

    def __init__(self, k_val_for_obj_id=1.0, log_eps=1e-6, division_eps=1e-7):
        self.k_val_for_obj_id = k_val_for_obj_id # For obj_id transformation
        self.log_eps = log_eps # Epsilon for log transformation
        self.division_eps = division_eps # Epsilon for division in standardization

        # For MET features (E_T_miss_transformed)
        self.met_means = None
        self.met_stds = None

        # For particle features (obj_id_transformed, E_transformed, pT_transformed, eta_transformed)
        self.particle_means = None
        self.particle_stds = None
        
        self.fitted = False

    def _transform_features(self, X):
        # X shape: (N, 92)
        # MET features: E_T_miss, phi_Et_miss
        met_e = X[:, self.MET_E_IDX]
        met_phi = X[:, self.MET_PHI_IDX]

        # Transform MET features
        # Log-transform E_T_miss, convert phi to sin/cos
        # All energies are in MeV, scale them by 1000 (to GeV range)
        met_e_transformed = torch.log(met_e / 1000.0 + self.log_eps) 
        met_phi_sin = torch.sin(met_phi)
        met_phi_cos = torch.cos(met_phi)
        
        # Processed MET features: (N, 3) -> (E_T_miss_log_scaled, sin(phi_Et_miss), cos(phi_Et_miss))
        processed_met = torch.stack([met_e_transformed, met_phi_sin, met_phi_cos], dim=1)

        # Particle features
        # X[:, 2:] has shape (N, 90). Reshape to (N, 18, 5)
        particles = X[:, 2:].view(-1, self.MAX_OBJECTS, self.NUM_OBJ_FEATURES)
        
        # Create mask for valid particles (E > epsilon MeV)
        # Use a small threshold like 0.001 MeV to avoid issues with exact zero from data generation
        # Assuming E is at OBJ_E_IDX (index 1) within the 5 particle features
        particle_mask = (particles[:, :, self.OBJ_E_IDX] > 1e-3)

        # Transform particle features
        obj_id = particles[:, :, self.OBJ_ID_IDX]
        e = particles[:, :, self.OBJ_E_IDX]
        pt = particles[:, :, self.OBJ_PT_IDX]
        eta = particles[:, :, self.OBJ_ETA_IDX]
        phi = particles[:, :, self.OBJ_PHI_IDX]

        # Apply transformations
        # For obj_id: sign(x) * log1p(|x|/k) to handle various integer PIDs gracefully
        obj_id_transformed = torch.sign(obj_id) * torch.log1p(torch.abs(obj_id) / self.k_val_for_obj_id)
        e_transformed = torch.log(e / 1000.0 + self.log_eps) # Log-scale energy (GeV)
        pt_transformed = torch.log(pt / 1000.0 + self.log_eps) # Log-scale pT (GeV)
        eta_transformed = eta # Eta is usually fine as is
        phi_sin = torch.sin(phi)
        phi_cos = torch.cos(phi)

        # Combine into a tensor: (N, 18, 6 features: obj_id_T, E_T, pT_T, eta_T, sin(phi), cos(phi))
        # Features to be standardized are first, followed by non-standardized (sin/cos phi)
        unstd_particle_features = torch.stack([
            obj_id_transformed, e_transformed, pt_transformed, eta_transformed, 
            phi_sin, phi_cos
        ], dim=-1) # Shape: (N, 18, 6)
        
        return processed_met, unstd_particle_features, particle_mask

    def fit(self, X, y=None):
        processed_met, unstd_particle_features, particle_mask = self._transform_features(X)

        # Calculate mean/std for MET E_T_miss (transformed) (index 0 of processed_met)
        self.met_means = torch.mean(processed_met[:, 0], dim=0, keepdim=True) # Shape (1,)
        self.met_stds = torch.std(processed_met[:, 0], dim=0, keepdim=True)   # Shape (1,)

        # Calculate mean/std for particle features that need standardization
        # These are obj_id_T, E_T, pT_T, eta_T (indices 0,1,2,3 of the 6 features)
        num_particle_features_to_std = len(self.PARTICLE_STANDARDIZE_INDICES)
        self.particle_means = torch.zeros(num_particle_features_to_std, device=X.device)
        self.particle_stds = torch.ones(num_particle_features_to_std, device=X.device)

        for i, feat_idx in enumerate(self.PARTICLE_STANDARDIZE_INDICES):
            # Select the feature column for all particles
            feature_column = unstd_particle_features[:, :, feat_idx]
            # Apply mask to select only valid particle features
            valid_features = feature_column[particle_mask]
            if valid_features.numel() > 0:
                self.particle_means[i] = torch.mean(valid_features)
                self.particle_stds[i] = torch.std(valid_features)
            # If no valid features, mean remains 0, std remains 1 (rare case)

        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
            
        processed_met, unstd_particle_features, particle_mask = self._transform_features(X)
        
        # Standardize MET E_T_miss (index 0)
        std_met_e = (processed_met[:, 0:1] - self.met_means) / (self.met_stds + self.division_eps)
        # Other MET features (sin/cos phi) are not standardized
        final_met_features = torch.cat([std_met_e, processed_met[:, 1:]], dim=1) # (N, 3)

        # Standardize particle features
        std_particle_features_list = []
        for i, feat_idx in enumerate(self.PARTICLE_STANDARDIZE_INDICES):
            feature_column = unstd_particle_features[:, :, feat_idx]
            std_feature = (feature_column - self.particle_means[i]) / (self.particle_stds[i] + self.division_eps)
            std_particle_features_list.append(std_feature)
        
        # Concatenate standardized features with non-standardized ones (sin/cos phi)
        # Standardized features are (N, 18, 4), non-standardized are (N, 18, 2)
        standardized_block = torch.stack(std_particle_features_list, dim=-1) # (N, 18, 4)
        non_standardized_block = unstd_particle_features[:, :, len(self.PARTICLE_STANDARDIZE_INDICES):] # (N, 18, 2)
        
        processed_particles = torch.cat([standardized_block, non_standardized_block], dim=-1) # (N, 18, 6)

        # Apply mask: set features of non-valid particles to zero AFTER standardization
        # This ensures that padded inputs are actual zeros.
        # Mask shape (N, 18), needs to be (N, 18, 1) for broadcasting
        processed_particles = processed_particles * particle_mask.unsqueeze(-1).float()
        
        # Flatten particle features and concatenate with MET features
        # Output shape: (N, D_met + MAX_OBJECTS * D_particle_final)
        # D_met=3, MAX_OBJECTS=18, D_particle_final=6 -> 3 + 18*6 = 3 + 108 = 111
        flat_particles = processed_particles.view(X.shape[0], -1) # (N, 18*6)
        final_output = torch.cat([final_met_features, flat_particles], dim=1)
        
        return final_output

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim_input, slot_dim, iters, mlp_hidden_dim, epsilon=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.slot_dim = slot_dim
        self.epsilon = epsilon

        self.slots_init = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        
        self.norm_input  = nn.LayerNorm(dim_input)
        self.norm_slots  = nn.LayerNorm(slot_dim)
        self.norm_mlp_in = nn.LayerNorm(slot_dim) # Normalization before update MLP's input

        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(dim_input, slot_dim)
        self.to_v = nn.Linear(dim_input, slot_dim)

        # MLP for slot updates (instead of GRU for simplicity)
        self.update_mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, slot_dim)
        )

    def forward(self, inputs, particle_mask):
        # inputs: (batch, num_particles, dim_input)
        # particle_mask: (batch, num_particles), True for valid particles
        b, n_inputs, _ = inputs.shape
        
        slots = self.slots_init.expand(b, -1, -1)
        inputs_norm = self.norm_input(inputs)
        k = self.to_k(inputs_norm)  # (b, n_inputs, slot_dim)
        v = self.to_v(inputs_norm)  # (b, n_inputs, slot_dim)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)  # (b, num_slots, slot_dim)

            attn_logits = torch.einsum('bsd,bnd->bsn', q, k) / (self.slot_dim ** 0.5)
            
            # Mask attention to invalid particles
            # particle_mask is (b, n_inputs), need (b, 1, n_inputs) for broadcasting
            attn_logits.masked_fill_(~particle_mask.unsqueeze(1), -torch.finfo(attn_logits.dtype).max)
            
            attn = F.softmax(attn_logits, dim=-1) # (b, num_slots, n_inputs)
            
            # Weighted sum of values (updates for slots)
            updates = torch.einsum('bsn,bnd->bsd', attn, v) # (b, num_slots, slot_dim)
            
            # Slot update using MLP and residual connection
            slots = slots_prev + self.update_mlp(self.norm_mlp_in(updates))
            
        return slots

class FourTopTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_slots=8, slot_iters=3, 
                 transformer_nhead=4, transformer_ff_dim=256, num_transformer_layers=1,
                 dropout_rate=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_met_features = 3 # From preprocessor
        self.num_particle_features_in = 6 # From preprocessor (obj_id_T, E_T, pT_T, eta_T, sin_phi, cos_phi)
        self.max_objects = (input_dim - self.num_met_features) // self.num_particle_features_in

        # Initial MLP to project particle features to model_dim
        self.particle_projector = nn.Sequential(
            nn.Linear(self.num_particle_features_in, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim)
        )
        
        self.slot_attention = SlotAttention(
            num_slots=num_slots, 
            dim_input=model_dim, 
            slot_dim=model_dim, 
            iters=slot_iters, 
            mlp_hidden_dim=model_dim * 2 # typical expansion for MLPs
        )

        # Transformer Encoder for processing slot representations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=transformer_nhead, 
            dim_feedforward=transformer_ff_dim, 
            dropout=dropout_rate, 
            batch_first=True, # Important!
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Final classifier MLP
        # Input to classifier: flattened slots + MET features
        classifier_input_dim = num_slots * model_dim + self.num_met_features
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(model_dim, 1)
        )

    def forward(self, x):
        # x shape: (batch, input_dim from preprocessor = 111)
        met_features = x[:, :self.num_met_features] # (batch, 3)
        particle_data_flat = x[:, self.num_met_features:] # (batch, 18*6 = 108)
        
        # Reshape particle data to (batch, max_objects, num_particle_features)
        particle_features = particle_data_flat.view(-1, self.max_objects, self.num_particle_features_in) # (batch, 18, 6)

        # Create particle mask. Valid particles have non-zero (log-scaled) energy.
        # Energy is at index 1 of the 6 particle features (obj_id_T, E_T, pT_T, eta_T, sin_phi, cos_phi)
        # Using a small threshold due to potential floating point inaccuracies. 
        # The preprocessor ensures padded features are zero.
        particle_mask = (torch.abs(particle_features[:, :, 1]) > 1e-5) # (batch, 18)
        
        # Project particle features
        projected_particles = self.particle_projector(particle_features) # (batch, 18, model_dim)
        
        # Slot Attention
        # Pass particle_mask to ignore padding in attention
        slots = self.slot_attention(projected_particles, particle_mask) # (batch, num_slots, model_dim)
        
        # Process slots with Transformer Encoder
        # TransformerEncoderLayer expects src_key_padding_mask for positions to ignore.
        # Since slots are always present (num_slots is fixed), no padding mask is needed for slots themselves.
        processed_slots = self.transformer_encoder(slots) # (batch, num_slots, model_dim)
        
        # Flatten slots and concatenate with MET features
        slot_summary = processed_slots.flatten(start_dim=1) # (batch, num_slots * model_dim)
        combined_features = torch.cat([slot_summary, met_features], dim=1)
        
        # Classification
        logits = self.classifier(combined_features) # (batch, 1)
        return logits

def make_model(input_dim: int):
    # These hyperparameters are chosen to be modest for the given constraints
    model = FourTopTransformer(
        input_dim=input_dim,       # Should be 111 from preprocessor
        model_dim=64,              # Main dimension for embeddings, slots, transformer
        num_slots=8,               # Number of slots (e.g. for 4 tops, maybe 2 slots per top for products)
        slot_iters=3,              # Slot attention iterations
        transformer_nhead=4,       # Transformer heads for slot processing
        transformer_ff_dim=128,    # Transformer feed-forward dim
        num_transformer_layers=1,  # Number of transformer layers for slots
        dropout_rate=0.1
    )
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30 # Adjusted for potential training time constraints

def train_model(model, train_loader, val_loader, epochs):
    device = next(model.parameters()).device
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # A scheduler can be helpful, e.g., ReduceLROnPlateau or CosineAnnealingLR
    # For simplicity and to avoid unsupported "verbose" argument, keeping it simple.
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
        
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # if scheduler:
        #    scheduler.step(epoch_val_loss)

        # This print statement is for local debugging, can be removed or conditionalized in production.
        # print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
            
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

