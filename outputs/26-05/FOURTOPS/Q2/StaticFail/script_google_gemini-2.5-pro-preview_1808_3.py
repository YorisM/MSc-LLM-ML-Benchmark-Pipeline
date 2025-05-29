
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
import pickle # Already std-lib, but explicitly state for clarity
from sklearn.metrics import roc_auc_score

# 0. ---------- IMPORTS ----------
# (Already covered by above imports)

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    #    Must implement:
    #   - fit(X: torch.Tensor, y: torch.Tensor) -> self
    #   - transform(X: torch.Tensor) -> torch.Tensor

    def __init__(self, num_objects=18, epsilon=1e-7):
        self.num_objects = num_objects
        self.object_feature_dim = 5 # obj_id, E, pT, eta, phi
        self.epsilon = epsilon

        # For E_T_miss (will be converted to MET_x, MET_y, then scaled)
        self.met_means = None
        self.met_stds = None

        # For object features (E, px, py, pz, obj_id)
        # Order: E, px, py, pz, obj_id
        self.obj_means = None
        self.obj_stds = None

    def _calculate_cartesian(self, E, pT, eta, phi):
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)
        return E, px, py, pz

    def fit(self, X, y=None):
        # X shape: (N_samples, 92)
        # Extract MET related features (first 2 columns)
        E_T_miss = X[:, 0]
        phi_E_t_miss = X[:, 1]
        
        met_x = E_T_miss * torch.cos(phi_E_t_miss)
        met_y = E_T_miss * torch.sin(phi_E_t_miss)
        met_cartesian = torch.stack([met_x, met_y], dim=1) # (N_samples, 2)

        self.met_means = torch.mean(met_cartesian, dim=0)
        self.met_stds = torch.std(met_cartesian, dim=0) + self.epsilon

        # Extract object features
        # X[:, 2:] contains object data, 5 features per object
        # obj_id, E, pT, eta, phi
        obj_data_flat = X[:, 2:].reshape(-1, self.num_objects, self.object_feature_dim)
        # obj_data_flat shape: (N_samples, num_objects, 5)

        raw_obj_ids = obj_data_flat[..., 0]
        raw_E = obj_data_flat[..., 1]
        raw_pT = obj_data_flat[..., 2]
        raw_eta = obj_data_flat[..., 3]
        raw_phi = obj_data_flat[..., 4]

        # Create mask for actual particles (pT > 0)
        # Using a small threshold for pT to define a valid particle
        particle_mask = raw_pT > self.epsilon # (N_samples, num_objects)

        # Convert to Cartesian coordinates for relevant particles
        E_cart, px_cart, py_cart, pz_cart = self._calculate_cartesian(raw_E, raw_pT, raw_eta, raw_phi)
        
        # Features to scale: E, px, py, pz, obj_id
        # Shape: (N_samples, num_objects, 5 features for scaling)
        obj_features_to_scale = torch.stack([E_cart, px_cart, py_cart, pz_cart, raw_obj_ids], dim=-1)

        # Calculate mean and std only for actual particles
        # Expand mask for feature dimension: (N_samples, num_objects, 1)
        expanded_mask = particle_mask.unsqueeze(-1)
        
        # Sum features and count for actual particles
        # Mask out non-particles by multiplying with mask (0 for non-particles)
        masked_obj_features = obj_features_to_scale * expanded_mask
        
        num_actual_particles = expanded_mask.sum()
        if num_actual_particles == 0: # Handle cases with no actual particles in the training set (edge case)
             # Fallback: use 0 mean and 1 std (no scaling)
            self.obj_means = torch.zeros(obj_features_to_scale.shape[-1])
            self.obj_stds = torch.ones(obj_features_to_scale.shape[-1])
        else:
            self.obj_means = (masked_obj_features.sum(dim=(0,1))) / num_actual_particles
            # For std, need (x - mu)^2. Sum of squares approach used for numerical stability.
            # More direct: calculate std over the valid, masked values
            # Collect all valid particle features into a flat list for correct std calculation
            valid_features_list = []
            for i in range(X.shape[0]):
                for j in range(self.num_objects):
                    if particle_mask[i,j]:
                        valid_features_list.append(obj_features_to_scale[i,j,:])
            if not valid_features_list: # If no valid particles across entire dataset (unlikely)
                 self.obj_means = torch.zeros(obj_features_to_scale.shape[-1])
                 self.obj_stds = torch.ones(obj_features_to_scale.shape[-1])
            else:
                valid_features_tensor = torch.stack(valid_features_list, dim=0)
                self.obj_means = torch.mean(valid_features_tensor, dim=0)
                self.obj_stds = torch.std(valid_features_tensor, dim=0) + self.epsilon

        return self

    def transform(self, X):
        # Ensure fitted statistics are available
        if self.met_means is None or self.obj_means is None:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        # MET features
        E_T_miss = X[:, 0]
        phi_E_t_miss = X[:, 1]
        met_x = E_T_miss * torch.cos(phi_E_t_miss)
        met_y = E_T_miss * torch.sin(phi_E_t_miss)
        met_cartesian = torch.stack([met_x, met_y], dim=1) # (N_samples, 2)
        scaled_met = (met_cartesian - self.met_means.to(X.device)) / self.met_stds.to(X.device)

        # Object features
        obj_data_flat = X[:, 2:].reshape(-1, self.num_objects, self.object_feature_dim)
        raw_obj_ids = obj_data_flat[..., 0]
        raw_E = obj_data_flat[..., 1]
        raw_pT = obj_data_flat[..., 2]
        raw_eta = obj_data_flat[..., 3]
        raw_phi = obj_data_flat[..., 4]

        particle_mask = (raw_pT > self.epsilon).float() # (N_samples, num_objects)
        
        E_cart, px_cart, py_cart, pz_cart = self._calculate_cartesian(raw_E, raw_pT, raw_eta, raw_phi)
        # Correctly set features of non-particles to zero BEFORE scaling.
        # This ensures that padded values remain zero and do not affect scaling logic for valid particles.
        E_cart = E_cart * particle_mask
        px_cart = px_cart * particle_mask
        py_cart = py_cart * particle_mask
        pz_cart = pz_cart * particle_mask
        processed_obj_ids = raw_obj_ids * particle_mask # obj_id for padded particles becomes 0

        obj_features_to_scale = torch.stack([E_cart, px_cart, py_cart, pz_cart, processed_obj_ids], dim=-1)
        # (N_samples, num_objects, 5 features for scaling)

        # Scale features, ensuring broadcasting is correct
        # self.obj_means and self.obj_stds are (5,)
        scaled_obj_features = (obj_features_to_scale - self.obj_means.to(X.device).view(1,1,5)) / \
                                self.obj_stds.to(X.device).view(1,1,5)
        
        # For padded particles, their scaled features might not be zero due to (0-mean)/std.
        # We must ensure their features are zero *after* scaling, or that their mask is used effectively.
        # Here, we pass the mask separately, so it's fine if scaled padded features are non-zero.
        # The mask will be used in the model to ignore these.

        # Final structure: [met_xy (2), P1(4), mask1(1), S1(obj_id)(1), P2(4), mask2(1), S2(obj_id)(1), ...]
        # P_k are scaled E, px, py, pz. S_k is scaled obj_id.
        output_features = [scaled_met]
        for i in range(self.num_objects):
            # Scaled E, px, py, pz for object i
            output_features.append(scaled_obj_features[:, i, 0:4]) 
            # Mask for object i
            output_features.append(particle_mask[:, i].unsqueeze(-1)) 
            # Scaled obj_id for object i
            output_features.append(scaled_obj_features[:, i, 4].unsqueeze(-1)) 
        
        return torch.cat(output_features, dim=1) # (N_samples, 2 + num_objects * (4+1+1))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION -----------
class EquivariantLayer(nn.Module):
    def __init__(self, scalar_feature_dim, mlp_hidden_dim=64):
        super().__init__()
        self.scalar_feature_dim = scalar_feature_dim
        self.minkowski_factors = torch.tensor([1., -1., -1., -1.], dtype=torch.float32).view(1,1,1,4)

        # Edge MLP: processes (S_k, S_l, P_k dot P_l)
        # Input: 2*scalar_feature_dim (for S_k, S_l) + 1 (for dot product)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * scalar_feature_dim + 1, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim) # Output C_edge_out (mlp_hidden_dim)
        )

        # Node Scalar MLP: processes (S_k, aggregated_scalar_messages_k)
        # Input: scalar_feature_dim (S_k) + mlp_hidden_dim (aggr_msgs)
        self.node_scalar_mlp = nn.Sequential(
            nn.Linear(scalar_feature_dim + mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, scalar_feature_dim) # Output new S_k (scalar_feature_dim)
        )

        # Coefficient MLPs for vector update: P_new_k = c1*P_k + sum_l(c2_kl*P_l)
        # c1 depends on updated S_k (or S_k and aggr_msgs)
        self.coeff_Pk_mlp = nn.Sequential(
            nn.Linear(scalar_feature_dim + mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1) # scalar coefficient for P_k
        )
        # c2_kl depends on edge features (S_k, S_l, P_k dot P_l)
        self.coeff_Pl_mlp = nn.Sequential(
            nn.Linear(2 * scalar_feature_dim + 1, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1) # scalar coefficient for P_l in sum
        )

    def forward(self, P, S, mask):
        # P: (B, N, 4), S: (B, N, C_s), mask: (B, N, 1)
        B, N, C_s = S.shape
        minkowski_factors_dev = self.minkowski_factors.to(P.device)

        # Pairwise dot products P_k ⋅ P_l
        P_expanded_k = P.unsqueeze(2) # (B, N_k, 1, 4)
        P_expanded_l = P.unsqueeze(1) # (B, 1, N_l, 4)
        # Element-wise product -> (B, N_k, N_l, 4) for (E_k E_l, px_k px_l, ...)
        prods_components = P_expanded_k * P_expanded_l 
        dot_prods = (prods_components * minkowski_factors_dev).sum(dim=-1, keepdim=True) # (B, N_k, N_l, 1)

        # Edge features construction
        S_k_expanded = S.unsqueeze(2).expand(-1, -1, N, -1) # (B, N_k, N_l, C_s)
        S_l_expanded = S.unsqueeze(1).expand(-1, N, -1, -1) # (B, N_k, N_l, C_s)
        edge_features_in = torch.cat([S_k_expanded, S_l_expanded, dot_prods], dim=-1)
        
        edge_scalars = self.edge_mlp(edge_features_in) # (B, N_k, N_l, C_edge_out=mlp_hidden_dim)

        # Mask messages from non-existent particles (mask_l)
        # mask is (B,N,1). mask_l_expanded shape (B, N_k, N_l, 1)
        mask_l_expanded = mask.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars_masked = edge_scalars * mask_l_expanded
        
        # Aggregate scalar messages: sum over l for each k
        aggr_scalar_messages = edge_scalars_masked.sum(dim=2) # (B, N_k, C_edge_out)
        
        # Update scalar features S_new
        node_scalar_input = torch.cat([S, aggr_scalar_messages], dim=-1)
        S_new = self.node_scalar_mlp(node_scalar_input)

        # Update vector features P_new
        # P_new_k = c1 * P_k + sum_l(c2_kl * P_l)
        # c1 is coeff_Pk, depends on (S, aggr_scalar_messages) (Identical to node_scalar_input for MLP)
        coeff_Pk_val = self.coeff_Pk_mlp(node_scalar_input) # (B, N_k, 1)
        
        # c2_kl is coeff_Pl, depends on edge_features_in (S_k, S_l, P_k dot P_l)
        coeff_Pl_val = self.coeff_Pl_mlp(edge_features_in) # (B, N_k, N_l, 1)
        coeff_Pl_masked = coeff_Pl_val * mask_l_expanded # (B, N_k, N_l, 1)

        # sum_l (coeff_Pl_kl * P_l)
        # P is (B, N, 4). P_l means P_j. Need P_l where l is the index being summed over.
        # Einsum: P_l is indexed by 'm' (from 'bmf'), coeff_Pl_masked is 'bnm' (b:batch, n:k-index, m:l-index)
        # Result is 'bnf' (b:batch, n:k-index, f:feature-4_vector)
        aggr_vector_messages = torch.einsum('bnmf,bmf->bnf', coeff_Pl_masked, P) # (B, N_k, 4)

        P_new = coeff_Pk_val * P + aggr_vector_messages

        # Apply mask to outputs to ensure padded particles remain zero (or close to zero)
        S_out = S_new * mask
        P_out = P_new * mask 
        return P_out, S_out

class LorentzEquivariantNet(nn.Module):
    def __init__(self, input_dim, num_objects=18, initial_scalar_dim=1, 
                 model_scalar_dim=32, mlp_hidden_dim=64, num_equivariant_layers=3):
        super().__init__()
        self.num_objects = num_objects
        self.input_dim_per_object_processed = 4 + 1 + 1 # P(4), mask(1), S_scalar(1)
        self.minkowski_factors = torch.tensor([1., -1., -1., -1.], dtype=torch.float32).view(1,4)

        # Initial MLP to process obj_id and m^2 into model_scalar_dim
        # Input: obj_id (1) + m^2 (1) = 2 initial scalar features
        self.initial_scalar_mlp = nn.Sequential(
            nn.Linear(initial_scalar_dim + 1, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, model_scalar_dim)
        )

        self.equivariant_layers = nn.ModuleList(
            [EquivariantLayer(model_scalar_dim, mlp_hidden_dim) for _ in range(num_equivariant_layers)]
        )

        # Final MLP for classification
        # Input: pooled S_final (model_scalar_dim) + P_total_mass_sq (1) + MET_features (2)
        self.final_mlp = nn.Sequential(
            nn.Linear(model_scalar_dim + 1 + 2, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1) # Logit output
        )

    def forward(self, x_flat):
        # x_flat: (B, 2 + num_objects * (4+1+1))
        # Unflatten input
        met_features = x_flat[:, 0:2] # (B, 2)
        obj_features_flat = x_flat[:, 2:] # (B, num_objects * 6)
        obj_features = obj_features_flat.view(-1, self.num_objects, self.input_dim_per_object_processed)
        # obj_features: (B, N, 6) where 6 is (E,px,py,pz, mask, obj_id_scaled)

        P = obj_features[:, :, 0:4]    # (B, N, 4)
        mask = obj_features[:, :, 4:5]  # (B, N, 1)
        S_obj_id = obj_features[:, :, 5:6] # (B, N, 1) (scaled obj_id)

        # Compute initial scalar features: m^2
        # P_sq = (P^mu P_mu) -> (B, N, 1) for mass_sq for each particle
        minkowski_factors_dev = self.minkowski_factors.to(P.device).unsqueeze(1) # (1,1,4)
        mass_sq = (P * P * minkowski_factors_dev).sum(dim=-1, keepdim=True) # (B, N, 1)
        
        # Initial scalar features S_0 = MLP_initial(obj_id, m^2)
        S = self.initial_scalar_mlp(torch.cat([S_obj_id, mass_sq], dim=-1)) # (B, N, model_scalar_dim)
        S = S * mask # Ensure non-particles have zero scalar features

        # Apply equivariant layers
        for layer in self.equivariant_layers:
            P, S = layer(P, S, mask)

        # Global pooling
        # Masked average pooling for scalar features
        S_sum = (S * mask).sum(dim=1) # (B, model_scalar_dim)
        mask_sum = mask.sum(dim=1)    # (B, 1)
        S_pooled = S_sum / (mask_sum + 1e-7) # Add epsilon to avoid div by zero if no particles

        # Sum of 4-vectors for total momentum P_total, then P_total_mass_sq
        P_total = (P * mask).sum(dim=1) # (B, 4)
        P_total_mass_sq = (P_total * P_total * self.minkowski_factors.to(P.device)).sum(dim=-1, keepdim=True) # (B,1)

        # Concatenate pooled features and MET for final MLP
        final_mlp_input = torch.cat([S_pooled, P_total_mass_sq, met_features], dim=1)
        logits = self.final_mlp(final_mlp_input)
        return logits

def make_model(input_dim: int):
    # Parameters can be tuned. These are example values.
    # input_dim should be 2 + 18 * (4+1+1) = 110 from preprocessor
    # initial_scalar_dim from preprocessor is 1 (obj_id). Concatenated with m^2 (1), so MLP input is 2.
    model = LorentzEquivariantNet(input_dim=input_dim, 
                                  num_objects=18, 
                                  initial_scalar_dim=1, 
                                  model_scalar_dim=32, 
                                  mlp_hidden_dim=64, 
                                  num_equivariant_layers=3)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25 
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    # For AUC, we'd also track train_aucs, val_aucs, but problem asks for acc/loss only in return lists.
    # We'll print AUC during validation for monitoring.

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_labels = []
        all_val_preds_proba = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                preds_proba = torch.sigmoid(outputs)
                predicted = (preds_proba > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds_proba.extend(preds_proba.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        val_auc = roc_auc_score(np.array(all_val_labels), np.array(all_val_preds_proba))

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val AUC: {val_auc:.4f}")

    return model, train_losses, val_losses, train_accs, val_accs

# IMPORTANT: The problem states NOT to write code to run these functions.
# The following would be how one might use them (commented out):

# if __name__ == '__main__':
#     # Dummy data for testing (replace with actual X_train, Y_train etc.)
#     X_train_tensor = torch.randn(241657, 92)
#     Y_train_tensor = torch.randint(0, 2, (241657,))
#     X_val_tensor = torch.randn(30272, 92)
#     Y_val_tensor = torch.randint(0, 2, (30272,))

#     # Preprocessing
#     preprocessor = make_preprocessor()
#     X_train_processed = preprocessor.fit_transform(X_train_tensor)
#     X_val_processed = preprocessor.transform(X_val_tensor)
    
#     # Create DataLoaders
#     train_dataset = TensorDataset(X_train_processed, Y_train_tensor)
#     val_dataset = TensorDataset(X_val_processed, Y_val_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#     # Model creation
#     input_dim = X_train_processed.shape[1]
#     model = make_model(input_dim)

#     # Training
#     trained_model, train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, EPOCHS)
    
#     # Example of saving the preprocessor (if needed later)
#     # with open('preprocessor.pkl', 'wb') as f:
#     #     pickle.dump(preprocessor, f)

#     # Example of saving the model (if needed later)
#     # torch.save(trained_model.state_dict(), 'trained_model.pth')

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

