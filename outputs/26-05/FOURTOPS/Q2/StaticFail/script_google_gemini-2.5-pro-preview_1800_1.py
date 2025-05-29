
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
import torch.optim as optim

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        # Constants for feature indices
        self.feat_et_miss = 0
        self.feat_phi_et_miss = 1
        self.particle_feat_offset = 2
        self.num_particle_features = 5 # obj_id, E, pT, eta, phi (original)
                                        # obj_id, E, px, py, pz (transformed)
        self.max_objects = 18

    def _transform_features(self, X: torch.Tensor) -> torch.Tensor:
        X_transformed = X.clone()

        # Transform MET features
        ET_miss = X[:, self.feat_et_miss]
        phi_ET_miss = X[:, self.feat_phi_et_miss]
        METx = ET_miss * torch.cos(phi_ET_miss)
        METy = ET_miss * torch.sin(phi_ET_miss)
        X_transformed[:, self.feat_et_miss] = METx
        X_transformed[:, self.feat_phi_et_miss] = METy
        
        # Transform particle features
        for i in range(self.max_objects):
            base_idx = self.particle_feat_offset + i * self.num_particle_features
            # obj_id is at base_idx + 0, E is at base_idx + 1
            # pT, eta, phi are at base_idx + 2, 3, 4 respectively
            
            E_k = X[:, base_idx + 1]
            pT_k = X[:, base_idx + 2]
            eta_k = X[:, base_idx + 3]
            phi_k = X[:, base_idx + 4]

            # For padded particles, pT_k=0, eta_k=0, phi_k=0.
            # This results in px_k=0, py_k=0, pz_k=0.
            px_k = pT_k * torch.cos(phi_k)
            py_k = pT_k * torch.sin(phi_k)
            pz_k = pT_k * torch.sinh(eta_k)
            
            # Store E, px, py, pz in place of E, pT, eta, phi (obj_id remains)
            # Original: obj_id, E, pT, eta, phi
            # New:      obj_id, E, px, py, pz
            X_transformed[:, base_idx + 1] = E_k
            X_transformed[:, base_idx + 2] = px_k
            X_transformed[:, base_idx + 3] = py_k
            X_transformed[:, base_idx + 4] = pz_k
            
        return X_transformed

    def fit(self, X: torch.Tensor, y: torch.Tensor = None) -> 'MyPreprocessor':
        X_transformed = self._transform_features(X)
        self.mean = torch.mean(X_transformed, dim=0)
        self.std = torch.std(X_transformed, dim=0)
        # Add epsilon to std to prevent division by zero for constant features
        self.std = self.std + 1e-7 
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        
        # Create particle existence mask from original pT values
        # Mask is 1 if particle exists (pT != 0), 0 otherwise.
        particle_masks = []
        for i in range(self.max_objects):
            pT_k_idx = self.particle_feat_offset + i * self.num_particle_features + 2
            mask_k = (X[:, pT_k_idx] != 0.0).float()
            particle_masks.append(mask_k)
        mask_tensor = torch.stack(particle_masks, dim=1) # Shape (N, 18)

        X_cartesian = self._transform_features(X)
        X_normalized = (X_cartesian - self.mean) / self.std
        
        # Concatenate normalized features and mask
        # Output shape will be (N, 92 + 18) = (N, 110)
        return torch.cat((X_normalized, mask_tensor), dim=1)

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor() -> MyPreprocessor:
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class LorentzEquivariantNet(nn.Module):
    def __init__(self, input_dim_processed: int, num_objects: int = 18, particle_feature_dim: int = 5):
        super().__init__()
        self.num_objects = num_objects
        self.particle_feature_dim = particle_feature_dim # obj_id, E, px, py, pz
        self.raw_feature_dim = input_dim_processed - num_objects # Should be 92

        # Minkowski metric tensor diag(1, -1, -1, -1) for dot product calculation
        self.minkowski_metric = torch.tensor([1., -1., -1., -1.], dtype=torch.float32)

        # Per-particle MLP: processes (m_sq, sum_dot_prods, obj_id)
        # Input features per particle: 3 (m_sq_i, sum_dot_prods_i, obj_id_i)
        # Output: particle embedding (e.g., 32 dimensional)
        self.particle_mlp_hidden1 = 64
        self.particle_mlp_hidden2 = 64
        self.particle_embedding_dim = 32
        self.particle_mlp = nn.Sequential(
            nn.Linear(3, self.particle_mlp_hidden1),
            nn.ReLU(),
            nn.Linear(self.particle_mlp_hidden1, self.particle_mlp_hidden2),
            nn.ReLU(),
            nn.Linear(self.particle_mlp_hidden2, self.particle_embedding_dim)
        )

        # Final MLP: processes pooled particle embeddings and MET features
        # Pooled features: sum_pooled (particle_embedding_dim) + mean_pooled (particle_embedding_dim)
        # MET features: 2 (norm_METx, norm_METy)
        # Total input to final MLP: particle_embedding_dim * 2 + 2
        self.final_mlp_input_dim = self.particle_embedding_dim * 2 + 2
        self.final_mlp_hidden1 = 128
        self.final_mlp_hidden2 = 128
        self.final_mlp_hidden3 = 64

        self.output_mlp = nn.Sequential(
            nn.Linear(self.final_mlp_input_dim, self.final_mlp_hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.final_mlp_hidden1, self.final_mlp_hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.final_mlp_hidden2, self.final_mlp_hidden3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.final_mlp_hidden3, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_dim_processed = 110)
        # raw_features: (batch_size, 92), mask: (batch_size, 18)
        raw_features = x[:, :self.raw_feature_dim]
        mask = x[:, self.raw_feature_dim:].bool() # (N, 18)
        
        # MET features: (norm_METx, norm_METy)
        # Indices 0, 1 of raw_features
        met_features = raw_features[:, :2] # (N, 2)
        
        # Particle features
        # Shape (N, 18, 5), 5 features are (obj_id, E, px, py, pz)
        particle_features_flat = raw_features[:, 2:] # (N, 90)
        particle_features = particle_features_flat.reshape(-1, self.num_objects, self.particle_feature_dim)
        
        obj_ids = particle_features[:, :, 0] # (N, 18)
        # Four-vectors p_mu = (E, px, py, pz) correspond to indices 1,2,3,4 of particle_features
        # particle_features structure: [obj_id, E, px, py, pz]
        # So, four_vectors are particle_features[:, :, 1:]
        four_vectors = particle_features[:, :, 1:] # (N, 18, 4)

        if self.minkowski_metric.device != x.device:
            self.minkowski_metric = self.minkowski_metric.to(x.device)

        # Compute pairwise Minkowski dot products: p_i \cdot p_j
        # p_mu_minkowski has components (E, -px, -py, -pz)
        p_mu_minkowski = four_vectors * self.minkowski_metric.view(1, 1, 4)
        # dot_products[b, i, j] = sum_f four_vectors[b, i, f] * p_mu_minkowski[b, j, f]
        # This is p_i . p_j (standard definition)
        dot_products = torch.einsum('bif,bjf->bij', four_vectors, p_mu_minkowski) # (N, 18, 18)

        # Apply mask to dot products to zero out contributions from non-existent particles
        # mask_2d[b,i,j] = 1 if particle i and j exist, 0 otherwise
        mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1) # (N, 18, 18)
        dot_products = dot_products * mask_2d.float()

        # Per-particle features for particle_mlp
        # 1. Squared invariant mass m_i^2 = p_i \cdot p_i (diagonal of dot_products)
        m_sq = torch.diagonal(dot_products, dim1=-2, dim2=-1) # (N, 18)
        # m_sq already masked by dot_products mask_2d along diagonal

        # 2. Sum of dot products for each particle: sum_j (p_i \cdot p_j)
        # This is sum over rows or columns of masked dot_products matrix
        sum_dot_prods = torch.sum(dot_products, dim=2) # (N, 18), sum over j for fixed i

        # 3. Object IDs (already extracted as obj_ids)
        
        # Concatenate per-particle features: (m_sq, sum_dot_prods, obj_id)
        # Each is (N, 18), stack to (N, 18, 3)
        particle_combined_feats = torch.stack([m_sq, sum_dot_prods, obj_ids], dim=2)

        # Apply per-particle MLP
        particle_embeddings = self.particle_mlp(particle_combined_feats) # (N, 18, particle_embedding_dim)
        
        # Mask embeddings of non-existent particles before pooling
        particle_embeddings = particle_embeddings * mask.unsqueeze(-1).float()

        # Pooling over particles
        # Sum pooling
        sum_pooled_particles = torch.sum(particle_embeddings, dim=1) # (N, particle_embedding_dim)
        
        # Mean pooling (careful about division by zero for events with no particles)
        num_valid_particles = torch.sum(mask.float(), dim=1, keepdim=True).clamp(min=1) # (N, 1)
        mean_pooled_particles = sum_pooled_particles / num_valid_particles # (N, particle_embedding_dim)
        
        # Concatenate pooled features
        pooled_particles = torch.cat([sum_pooled_particles, mean_pooled_particles], dim=1) # (N, particle_embedding_dim * 2)

        # Concatenate pooled particle features with MET features
        final_features = torch.cat([pooled_particles, met_features], dim=1) # (N, particle_embedding_dim * 2 + 2)
        
        # Apply final MLP for classification
        output = self.output_mlp(final_features)
        return output.squeeze(-1) # (N,)

def make_model(input_dim: int):
    # input_dim here is the output dim of MyPreprocessor, which is 110
    model = LorentzEquivariantNet(input_dim_processed=input_dim)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25
BATCH_SIZE = 256 # Adjusted for typical dataset sizes and memory limits
LEARNING_RATE = 1e-3

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    # Scheduler to reduce learning rate if validation loss plateaus
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
                predicted = (outputs > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Step the scheduler based on the epoch, not validation loss for StepLR
        scheduler.step()
        
        # Print epoch stats (optional - usually for interactive use)
        # print(f"Epoch {epoch+1}/{epochs} - "
        #       f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
        #       f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} - "
        #       f"LR: {scheduler.get_last_lr()[0]:.2e}")
            
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

