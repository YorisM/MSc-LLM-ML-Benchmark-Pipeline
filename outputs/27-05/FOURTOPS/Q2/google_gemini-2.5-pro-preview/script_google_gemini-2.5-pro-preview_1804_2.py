
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
import math # For isfinite, just in case, though torch.isfinite is better

# 0. ---------- IMPORTS ----------
# (already provided: torch, numpy, nn, TensorDataset, DataLoader)
# Standard library imports are fine
from torch import optim
from torch.optim import lr_scheduler

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    #    Must implement:
    #   - fit(X: torch.Tensor, y: torch.Tensor) -> self
    #   - transform(X: torch.Tensor) -> torch.Tensor

    def __init__(self):
        self.met_means = None
        self.met_stds = None
        self.obj_means = None
        self.obj_stds = None
        self.eps = 1e-7 # For safe division by std

    def _calculate_kinematics(self, X):
        # X shape: (N, 92)
        # Global features
        E_T_miss = X[:, 0]
        phi_E_T_miss = X[:, 1]
        met_x = E_T_miss * torch.cos(phi_E_T_miss)
        met_y = E_T_miss * torch.sin(phi_E_T_miss)
        # met_features shape: (N, 2)
        met_features = torch.stack([met_x, met_y], dim=1)

        # Object features
        objects_flat = X[:, 2:] # Shape (N, 90)
        objects_reshaped = objects_flat.reshape(-1, 18, 5) # Shape (N, 18, 5)
                                                           # (obj_id, E, pT, eta, phi)
        
        obj_ids = objects_reshaped[:, :, 0] # (N, 18)
        E = objects_reshaped[:, :, 1]       # (N, 18)
        pT = objects_reshaped[:, :, 2]      # (N, 18)
        eta = objects_reshaped[:, :, 3]     # (N, 18)
        phi = objects_reshaped[:, :, 4]     # (N, 18)

        # Calculate px, py, pz
        # Need to handle pT=0 for padded objects to avoid nan from sinh(eta) if eta is inf/nan
        # However, if pT=0, then px, py, pz should be 0 regardless of eta/phi values.
        # Padded objects should have pT=0. If eta/phi are also 0/nan for padded, calculation is fine.
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        # Ensure pz is 0 if pT is 0, even if eta is large/NaN (e.g. from 0-momentum particles in padding)
        # A more robust way for pz: pz = torch.where(pT > self.eps, pT * torch.sinh(eta), torch.zeros_like(pT))
        # However, zero-padded means E, pT, eta, phi are all zero for padded objects. So sinh(0)=0.
        pz = pT * torch.sinh(eta)
        
        # Order for features: E, px, py, pz, obj_id
        # object_kinematics shape: (N, 18, 5)
        object_kinematics = torch.stack([E, px, py, pz, obj_ids], dim=-1)

        # Mask for actual particles (based on original Energy)
        # Padded objects have E=0. Small epsilon for float comparisons.
        is_particle_mask = (objects_reshaped[:, :, 1] > self.eps) # (N, 18)
        
        return met_features, object_kinematics, is_particle_mask

    def fit(self, X, y=None):
        met_features, object_kinematics, is_particle_mask = self._calculate_kinematics(X)
        
        # Calculate means and stds for MET features (already N, 2)
        self.met_means = met_features.mean(dim=0)
        self.met_stds = met_features.std(dim=0) + self.eps

        # Calculate means and stds for object features
        # Only consider actual particles for stats
        # valid_object_kinematics: (num_actual_particles, 5)
        valid_object_kinematics = object_kinematics[is_particle_mask]
        if valid_object_kinematics.shape[0] > 0:
            self.obj_means = valid_object_kinematics.mean(dim=0) # Shape (5,)
            self.obj_stds = valid_object_kinematics.std(dim=0) + self.eps  # Shape (5,)
        else: # Handle case with no valid particles in sample (e.g. small test batch)
            self.obj_means = torch.zeros(object_kinematics.shape[-1], device=X.device, dtype=X.dtype)
            self.obj_stds = torch.ones(object_kinematics.shape[-1], device=X.device, dtype=X.dtype)
            
        return self

    def transform(self, X):
        if self.met_means is None: # fit not called
            raise RuntimeError("Preprocessor must be fit before transform.")

        met_features, object_kinematics, is_particle_mask = self._calculate_kinematics(X)
        
        # Scale MET features
        scaled_met_features = (met_features - self.met_means) / self.met_stds
        
        # Scale object features
        # object_kinematics shape: (N, 18, 5)
        # self.obj_means/stds shape: (5,)
        scaled_object_kinematics = (object_kinematics - self.obj_means.unsqueeze(0).unsqueeze(0)) / \
                                   self.obj_stds.unsqueeze(0).unsqueeze(0)
        
        # Zero out features for padded particles *after* scaling
        # is_particle_mask shape (N, 18), need (N, 18, 1) for broadcasting
        scaled_object_kinematics = scaled_object_kinematics * is_particle_mask.unsqueeze(-1).float()
        
        # Flatten object features and concatenate with MET features
        # scaled_object_kinematics_flat shape: (N, 18*5=90)
        scaled_object_kinematics_flat = scaled_object_kinematics.reshape(X.shape[0], -1)
        
        # final_features shape: (N, 2+90=92)
        final_features = torch.cat([scaled_met_features, scaled_object_kinematics_flat], dim=1)
        
        # Replace any NaN/inf from division by zero if std was exactly zero (eps might not catch all cases for specific data)
        final_features = torch.nan_to_num(final_features, nan=0.0, posinf=0.0, neginf=0.0)

        return final_features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class LorentzNet(nn.Module):
    def __init__(self, input_dim_flat): # input_dim_flat is 92
        super().__init__()
        # These dimensions are based on the preprocessor output
        self.num_particles = 18
        self.particle_kinematic_dim = 4 # E, px, py, pz
        self.particle_scalar_feat_dim = 1 # obj_id_scaled
        self.met_feature_dim = 2 # METx, METy

        hidden_dim = 64 # Hidden dimension for MLPs in the equivariant block
        self.new_scalar_dim = 32 # Output dimension for updated scalar features per particle
        
        # MLP for scalar messages: input is (s_i, s_j, pi . pj)
        # s_i, s_j are 1-dim (obj_id_scaled). pi.pj is 1-dim. Total 1+1+1=3.
        self.mlp_scalar_msg = nn.Sequential(
            nn.Linear(self.particle_scalar_feat_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # MLP for updating scalar features: input is (s_i_old, aggregated_scalar_msg_i)
        # s_i_old is 1-dim. aggregated_scalar_msg_i is hidden_dim.
        self.mlp_scalar_update = nn.Sequential(
            nn.Linear(self.particle_scalar_feat_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.new_scalar_dim)
        )

        # Final MLP classifier
        # Input features: P_total (4), M_total_sq (1), S_total (new_scalar_dim), met_xy (2)
        final_mlp_input_dim = self.particle_kinematic_dim + 1 + self.new_scalar_dim + self.met_feature_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(final_mlp_input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def minkowski_dot(self, v1, v2):
        # v1, v2 shape: (..., 4) representing (E, px, py, pz)
        # Metric: diag(1, -1, -1, -1)
        return (v1[..., 0:1] * v2[..., 0:1] - 
                v1[..., 1:2] * v2[..., 1:2] - 
                v1[..., 2:3] * v2[..., 2:3] - 
                v1[..., 3:4] * v2[..., 3:4]) # Keep last dim for broadcasting

    def forward(self, x):
        N = x.shape[0]
        
        # Unflatten input from preprocessor
        met_xy = x[:, :self.met_feature_dim] # (N, 2)
        particles_flat = x[:, self.met_feature_dim:] # (N, 18 * (4+1))
        # particles shape: (N, 18, 5) where 5 is (E, px, py, pz, obj_id_scaled)
        particles = particles_flat.view(N, self.num_particles, 
                                      self.particle_kinematic_dim + self.particle_scalar_feat_dim)

        p_mu = particles[:, :, :self.particle_kinematic_dim]      # (N, 18, 4)
        s_feat = particles[:, :, self.particle_kinematic_dim:]   # (N, 18, 1)
        
        # Particle mask: padded particles have all-zero features (due to preprocessor step)
        # We can use the scaled Energy (first component of p_mu) to make the mask.
        # A very small epsilon because of floating point arithmetic.
        mask = (torch.abs(p_mu[:, :, 0:1]) > 1e-9).float() # (N, 18, 1)
        
        # --- Equivariant Message Passing Block --- 
        # Pairwise features for messages
        s_feat_i = s_feat.unsqueeze(2).expand(-1, -1, self.num_particles, -1) # (N, 18, 18, D_s)
        s_feat_j = s_feat.unsqueeze(1).expand(-1, self.num_particles, -1, -1) # (N, 18, 18, D_s)
        
        # Minkowski dot products: p_i . p_j (Lorentz invariant)
        # p_mu needs to be (N, 18, 1, 4) for i and (N, 1, 18, 4) for j for broadcasting minkowski_dot
        dot_prods = self.minkowski_dot(p_mu.unsqueeze(2), p_mu.unsqueeze(1)) # (N, 18, 18, 1)

        pairwise_msg_input = torch.cat([s_feat_i, s_feat_j, dot_prods], dim=-1) # (N, 18, 18, 2*D_s+1)
        
        # Pairwise mask: particle i and particle j must both be present
        pairwise_mask = mask.unsqueeze(2) * mask.unsqueeze(1) # (N, 18, 18, 1)
        
        scalar_messages = self.mlp_scalar_msg(pairwise_msg_input) # (N, 18, 18, hidden_dim)
        scalar_messages = scalar_messages * pairwise_mask # Zero out messages for padded pairs
        
        # Aggregate messages for each particle i (sum over j)
        aggregated_scalar_msg = scalar_messages.sum(dim=2) # (N, 18, hidden_dim)
        
        # Update scalar features
        update_scalar_input = torch.cat([s_feat, aggregated_scalar_msg], dim=-1) # (N, 18, D_s + hidden_dim)
        new_s_feat = self.mlp_scalar_update(update_scalar_input) # (N, 18, new_scalar_dim)
        new_s_feat = new_s_feat * mask # Ensure padded particles have zero updated features

        # --- Pooling --- 
        # Sum of 4-vectors (Lorentz equivariant)
        P_total = (p_mu * mask).sum(dim=1) # (N, 4)
        
        # Squared invariant mass of the summed system (Lorentz invariant)
        M_total_sq = self.minkowski_dot(P_total.unsqueeze(1), P_total.unsqueeze(1)).squeeze(1) # (N, 1)
        
        # Sum of updated scalar features (becomes an invariant global property)
        S_total = (new_s_feat * mask).sum(dim=1) # (N, new_scalar_dim)
        
        # Concatenate all global features for final classification MLP
        final_features = torch.cat([P_total, M_total_sq, S_total, met_xy], dim=1)
        
        return self.final_mlp(final_features)

def make_model(input_dim: int):
    # input_dim is the number of features per event after preprocessing (92)
    # The model internally reshapes this flat input.
    model = LorentzNet(input_dim_flat=input_dim)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 15

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Learning rate scheduler
    # ReduceLROnPlateau is suitable if validation loss is monitored per epoch
    # CosineAnnealingLR is simpler as it doesn't require manual steps with val_loss
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

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
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Update learning rate (for schedulers like CosineAnnealingLR)
        scheduler.step()

        # Print epoch stats (optional, but useful for monitoring)
        # print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")

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

