
import os, sys, pickle, torch, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
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

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        if isinstance(self.x, (tuple, list)):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])      

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = PairDataset(X_train, Y_train)
    val_ds   = PairDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))
                        
# ----------------  START OF LLM BLOCK  ----------------

# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch or sklearn (sub-)modules you actually use.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Already imported
from sklearn.metrics import roc_auc_score, accuracy_score # Already imported
import numpy as np # Already imported
import copy # For deepcopying model state dict for early stopping

# For type hinting in MyPreprocessor
from torch import Tensor

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    #    Must implement:
    #   - fit(...) -> self
    #   - transform(X: Tensor)   -> Tensor  **or**  Tuple[Tensor, Tensor]

    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module.
    # IMPORTANT: Batch first.
    # May allocate NumPy arrays or Torch tensors internally, but:
    # transform() must be deterministic.
    # Store only derived parameters needed for transform i.e. do not store the raw data
    # itself in the preprocessor object.

    # DATA SPECIFICS
    # IMPORTANT: X_train, Y_train, X_val, Y_val are provided as PyTorch tensors in the environment.
    # Total flat length per event (X_train & X_val): 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    # Indices  7-11 : object 2  -> obj_2, E_2 , p_T_2 , eta_2 , phi_2
    # ...
    # Indices 88-92 : object 18 -> obj_18, E_18 , p_T_18 , eta_18 , phi_18
    # Per-object slice size = 5
    # Max objects encoded   = 18
    def __init__(self):
        self.global_mean: Tensor | None = None
        self.global_std: Tensor | None = None
        self.object_means: Tensor | None = None
        self.object_stds: Tensor | None = None

        self.max_objects = 18
        self.num_global_features = 2
        self.num_base_object_features = 5 # obj_id, E, pT, eta, phi
        self.num_engineered_object_features = 3 # px, py, pz
        self.total_object_features = self.num_base_object_features + self.num_engineered_object_features # 5 + 3 = 8

    def _calculate_cartesian(self, X_objects_reshaped: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # X_objects_reshaped shape: (N, max_objects, num_base_object_features=5)
        # Original object features: obj_id (idx 0), E (idx 1), pT (idx 2), eta (idx 3), phi (idx 4)

        #pT_col = X_objects_reshaped[:, :, 2]
        #eta_col = X_objects_reshaped[:, :, 3]
        #phi_col = X_objects_reshaped[:, :, 4]

        # Use clones to avoid in-place modification warnings if X_objects_reshaped comes from a non-writable tensor
        pT_col = X_objects_reshaped[:, :, 2].clone() 
        eta_col = X_objects_reshaped[:, :, 3].clone()
        phi_col = X_objects_reshaped[:, :, 4].clone()


        px = pT_col * torch.cos(phi_col) # (N, max_objects)
        py = pT_col * torch.sin(phi_col) # (N, max_objects)
        pz = pT_col * torch.sinh(eta_col) # (N, max_objects)
        return px, py, pz

    def fit(self, X: Tensor, y: Tensor | None = None):
        # X: (N, 92)
        X_global = X[:, :self.num_global_features]  # (N, 2)
        X_objects_flat = X[:, self.num_global_features:] # (N, 18*5 = 90)
        X_objects_reshaped = X_objects_flat.reshape(X.shape[0], self.max_objects, self.num_base_object_features) # (N, 18, 5)

        # Mask for real objects (obj_id > 0), assuming padding uses obj_id = 0
        # object_mask: (N, 18)
        object_mask = X_objects_reshaped[:, :, 0] > 0  

        # Calculate px, py, pz from UNNORMALIZED pT, eta, phi
        px, py, pz = self._calculate_cartesian(X_objects_reshaped) # each (N, 18)

        # Concatenate engineered features to base object features
        X_objects_ext = torch.cat([
            X_objects_reshaped,         # (N, 18, 5)
            px.unsqueeze(-1),           # (N, 18, 1)
            py.unsqueeze(-1),           # (N, 18, 1)
            pz.unsqueeze(-1)            # (N, 18, 1)
        ], dim=2)  # X_objects_ext: (N, 18, 8)

        # --- Fit scalers ---
        # Global features scaler
        self.global_mean = X_global.mean(dim=0) # (2,)
        self.global_std = X_global.std(dim=0)   # (2,)
        self.global_std[self.global_std < 1e-7] = 1.0 # Avoid division by zero for constant features

        # Object features scaler (fitted on real objects only)
        # real_object_features: (NumTotalRealObjects, 8)
        real_object_features = X_objects_ext[object_mask] 

        if real_object_features.shape[0] > 0:
            self.object_means = real_object_features.mean(dim=0) # (8,)
            self.object_stds = real_object_features.std(dim=0)   # (8,)
            self.object_stds[self.object_stds < 1e-7] = 1.0 # Avoid division by zero
        else: # Edge case: no real objects in the fitting data (e.g., for a tiny test set)
            self.object_means = torch.zeros(self.total_object_features, device=X.device)
            self.object_stds = torch.ones(self.total_object_features, device=X.device)

        return self

    def transform(self, X: Tensor) -> tuple[Tensor, Tensor]:
        if self.global_mean is None or self.object_means is None:
            raise RuntimeError("Preprocessor must be fit before transform.")

        X_global = X[:, :self.num_global_features] # (N, 2)
        X_objects_flat = X[:, self.num_global_features:] # (N, 90)
        X_objects_reshaped = X_objects_flat.reshape(X.shape[0], self.max_objects, self.num_base_object_features) # (N, 18, 5)

        object_mask = X_objects_reshaped[:, :, 0] > 0  # (N, 18)

        # Calculate px, py, pz (again, from original unscaled data)
        px, py, pz = self._calculate_cartesian(X_objects_reshaped) # each (N, 18)

        X_objects_ext = torch.cat([
            X_objects_reshaped, 
            px.unsqueeze(-1), 
            py.unsqueeze(-1), 
            pz.unsqueeze(-1)
        ], dim=2)  # (N, 18, 8)

        # --- Apply scaling ---
        # Scale global features
        X_global_scaled = (X_global - self.global_mean) / self.global_std # (N, 2)

        # Scale object features (all 8 features)
        # Note: self.object_means and self.object_stds are (8,)
        # X_objects_ext_scaled: (N, 18, 8)
        X_objects_ext_scaled = (X_objects_ext - self.object_means.view(1, 1, -1)) / self.object_stds.view(1, 1, -1)

        # Zero out features of padded objects AFTER scaling
        X_objects_ext_scaled[~object_mask] = 0.0 

        # Expand global features and concatenate to each object's features
        # X_global_scaled_expanded: (N, 18, 2)
        X_global_scaled_expanded = X_global_scaled.unsqueeze(1).expand(-1, self.max_objects, self.num_global_features) 

        # final_X_seq: (N, 18, 10) (8 object features + 2 global features per object)
        final_X_seq = torch.cat([X_objects_ext_scaled, X_global_scaled_expanded], dim=2) 

        return final_X_seq, object_mask # (N, 18, 10), (N, 18)

    def fit_transform(self, X: Tensor, y: Tensor | None = None) -> tuple[Tensor, Tensor]:
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], *, use_mask: bool):
        super().__init__()
        self.use_mask = use_mask 

        # input_shape is (L, F_in) e.g. (18, 10) from preprocessor
        self.L, self.F_in = input_shape 

        F_phi = 128         # Dimension of transformed object features
        F_rho_hidden = 64   # Hidden dimension for final classifier MLP

        # MLP to transform each object's feature vector
        self.phi_mlp = nn.Sequential(
            nn.Linear(self.F_in, F_phi // 2), # Input (B, L, F_in=10)
            nn.ReLU(),
            nn.BatchNorm1d(F_phi // 2), # Applied on (B*L, F_phi//2) or (B, F_phi//2, L)
                                        # nn.BatchNorm1d expects (N, C) or (N, C, L)
                                        # We need to reshape for BatchNorm1d if applied per-object independently
                                        # Or use LayerNorm: nn.LayerNorm(F_phi // 2)
            nn.Linear(F_phi // 2, F_phi),
            nn.ReLU()
            # Not adding BatchNorm here, will do LayerNorm on output of phi_mlp if needed
        ) # Output: (B, L, F_phi)

        # MLP to compute attention scores for each object
        self.attention_mlp = nn.Linear(F_phi, 1) # Input (B, L, F_phi) -> output (B, L, 1)

        # Final MLP for classification on the pooled context vector
        self.rho_mlp = nn.Sequential(
            nn.Linear(F_phi, F_rho_hidden), # Input is pooled context vector (B, F_phi)
            nn.ReLU(),
            nn.BatchNorm1d(F_rho_hidden),   # BatchNorm for stability
            nn.Dropout(0.3),               # Dropout for regularization
            nn.Linear(F_rho_hidden, 1)     # Output (B, 1) logit
        )

    def forward(self, data: torch.Tensor, mask: torch.Tensor | None = None):
        # data : Tensor (B, L, F_in) e.g. (batch_size, 18, 10)
        # mask : BoolTensor (B, L) e.g. (batch_size, 18), True for valid objects

        # Apply phi_mlp to each object.
        # For BatchNorm1d in phi_mlp:
        # It expects (N,C) or (N,C,L). Here we have (B,L,F).
        # Common practice is to either reshape (B*L, F) then apply BN, or use LayerNorm.
        # Let's refine phi_mlp to handle BatchNorm correctly or use LayerNorm.
        # Simpler: Apply LayerNorm after full phi_mlp transformation sequence.

        # Batch norm in phi_mlp needs careful handling of shape.
        # A common strategy is to apply it after linear layers if features distribution varies.
        # Or, simply remove it from phi_mlp given there's LayerNorm/BatchNorm in rho_mlp later.
        # For now, let's redefine phi_mlp for simplicity and robustness:

        #Revised phi_mlp structure for direct application on (B,L,F)
        temp_phi_mlp = nn.Sequential(
            nn.Linear(self.F_in, 128), # Increased intermediate dim
            nn.ReLU(),
            nn.Linear(128, F_phi) # F_phi = 128
        ).to(data.device) # Ensure it's on the same device as data

        x_obj_transformed_base = temp_phi_mlp(data) # (B, L, F_phi)

        # Optional: Layer Normalization on object features
        # layer_norm_phi = nn.LayerNorm(F_phi).to(data.device)
        # x_obj_transformed = layer_norm_phi(x_obj_transformed_base) # (B, L, F_phi)
        # Using the output directly might be fine too:
        x_obj_transformed = x_obj_transformed_base

        # Compute attention scores from transformed object features
        # self.attention_mlp(x_obj_transformed) output is (B, L, 1)
        attn_scores = self.attention_mlp(x_obj_transformed).squeeze(-1) # (B, L)

        if mask is not None and self.use_mask:
            # Mask out scores for padded/invalid objects before softmax
            # ~mask is where elements are False (padded), fill with -inf
            attn_scores = attn_scores.masked_fill(~mask, -float('inf'))

        attn_weights = torch.softmax(attn_scores, dim=1) # (B, L), scores sum to 1 over L for each batch element

        # Weighted sum of object representations (attention pooling)
        # attn_weights.unsqueeze(-1) becomes (B, L, 1) for broadcasting multiplication
        context_vector = torch.sum(x_obj_transformed * attn_weights.unsqueeze(-1), dim=1) # (B, F_phi)

        # Classify using the pooled context vector
        logits = self.rho_mlp(context_vector) # (B, 1)
        return logits

def make_model(input_shape, *, use_mask=False):
    return BinaryClassifier(input_shape, use_mask=use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 75 # Number of training epochs
PATIENCE_EARLY_STOP = 15  # Patience for early stopping
PATIENCE_SCHEDULER = 7   # Patience for LR scheduler
LEARNING_RATE = 5e-4     # Learning rate

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss() # Numerically stable loss for binary classification

    # Scheduler to reduce learning rate if validation AUC stagnates
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=PATIENCE_SCHEDULER, verbose=False)

    train_losses_hist, val_losses_hist = [], []
    train_accs_hist, val_accs_hist = [], [] # Store accuracy, though AUC is primary

    best_val_auc = -float('inf')
    epochs_no_improve = 0
    best_model_state_dict = copy.deepcopy(model.state_dict()) # Initialize with initial model state

    for epoch in range(epochs):
        model.train() # Set model to training mode
        running_train_loss = 0.0
        train_predictions_for_acc, train_labels_for_acc = [], []

        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            # Unpack data: preprocessor returns (data_tensor, mask_tensor)
            if isinstance(batch_data, tuple):
                data_tensor, mask_tensor = batch_data
                data_tensor, mask_tensor = data_tensor.to(device), mask_tensor.to(device)
            else: # Should not happen with current preprocessor, but good to be defensive
                data_tensor, mask_tensor = batch_data.to(device), None

            batch_labels = batch_labels.to(device).float() # Ensure labels are float for BCEWithLogitsLoss

            optimizer.zero_grad() # Clear previous gradients

            outputs_logits = model(data_tensor, mask_tensor) # Forward pass -> (B, 1) logits
            loss = criterion(outputs_logits.squeeze(), batch_labels) # Compute loss

            loss.backward() # Backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step() # Update weights

            running_train_loss += loss.item() * data_tensor.size(0)

            # For accuracy calculation (on-the-fly or per-epoch)
            preds_binary = (torch.sigmoid(outputs_logits.squeeze()) > 0.5).cpu().numpy()
            train_predictions_for_acc.extend(preds_binary)
            train_labels_for_acc.extend(batch_labels.cpu().numpy())

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(train_labels_for_acc, train_predictions_for_acc)
        train_losses_hist.append(epoch_train_loss)
        train_accs_hist.append(epoch_train_acc)

        # Validation phase
        model.eval() # Set model to evaluation mode
        running_val_loss = 0.0
        val_scores_for_auc, val_labels_for_auc_acc = [], [] # Store raw scores for AUC

        with torch.no_grad(): # Disable gradient calculations
            for batch_data, batch_labels in val_loader:
                if isinstance(batch_data, tuple):
                    data_tensor, mask_tensor = batch_data
                    data_tensor, mask_tensor = data_tensor.to(device), mask_tensor.to(device)
                else:
                    data_tensor, mask_tensor = batch_data.to(device), None

                batch_labels = batch_labels.to(device).float()

                outputs_logits = model(data_tensor, mask_tensor)
                loss = criterion(outputs_logits.squeeze(), batch_labels)
                running_val_loss += loss.item() * data_tensor.size(0)

                scores = torch.sigmoid(outputs_logits.squeeze()).cpu().numpy()
                val_scores_for_auc.extend(scores)
                val_labels_for_auc_acc.extend(batch_labels.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        # Calculate AUC on validation set
        epoch_val_auc = roc_auc_score(val_labels_for_auc_acc, val_scores_for_auc)
        # Calculate accuracy on validation set
        epoch_val_acc = accuracy_score(np.array(val_labels_for_auc_acc), np.array(val_scores_for_auc) > 0.5)

        val_losses_hist.append(epoch_val_loss)
        val_accs_hist.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val AUC: {epoch_val_auc:.4f} - "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")


        scheduler.step(epoch_val_auc) # Adjust LR based on validation AUC

        # Early stopping logic
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            epochs_no_improve = 0
            best_model_state_dict = copy.deepcopy(model.state_dict()) # Save best model state
            # print(f"    New best Val AUC: {best_val_auc:.4f}. Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE_EARLY_STOP:
                print(f"Early stopping at epoch {epoch+1} after {PATIENCE_EARLY_STOP} epochs with no improvement in Val AUC.")
                break # Exit training loop

    # Load the best model state before returning
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)

    return model, train_losses_hist, val_losses_hist, train_accs_hist, val_accs_hist

# ----------------  END OF LLM-CODE BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    pre = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train) # may be Tensor or Tuple
    X_val   = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    if isinstance(X_train, torch.Tensor):               # single-tensor case
        temp_ref    = X_train
        input_shape = temp_ref.shape[1:]                # e.g. (F,)
        use_mask    = False
    else:                                               # tuple => (data, mask)
        temp_ref    = X_train
        input_shape = temp_ref[0].shape[1:]             # e.g. (L, F)
        use_mask    = True                              
    model = make_model(input_shape, use_mask=use_mask)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single toy forward pass
    if dryrun:
        toy_data = torch.zeros(8, *input_shape, dtype=torch.float32)
        if use_mask:
            toy_mask = torch.zeros(8, input_shape[0], dtype=torch.bool)
            toy_batch = (toy_data, toy_mask)
        else:
            toy_batch = toy_data

        toy_transformed = pre.transform(toy_batch)
        try:
            _ = trained_model(*toy_transformed) if isinstance(toy_transformed, (tuple, list)) \
                else trained_model(toy_transformed)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 6. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
            "train_loss": tr_loss   if tr_loss else None,
            "val_loss":   va_loss   if va_loss else None,
            "train_acc":  tr_acc    if tr_acc else None,
            "val_acc":    va_acc    if va_acc else None,
        }
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

