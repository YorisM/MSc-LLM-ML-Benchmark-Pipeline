
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
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score # roc_auc_score is already imported by harness

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_means = None
        self.global_stds = None
        # For obj_id, E, pT, eta (phi is handled by sin/cos)
        self.object_feature_means = None
        self.object_feature_stds = None
        self.num_object_features_to_scale = 4 # obj_id, E, pT, eta
        self.object_phi_feature_idx = 4

        # Constants for feature structure
        self.max_objects = 18
        self.num_kinematic_features_per_object = 5 # obj_id, E, pT, eta, phi
        self.num_global_features_raw = 2 # E_T_miss, phi_E_T_miss

        # Output feature dimensions
        # scaled E_T_miss, cos(phi_miss), sin(phi_miss)
        self.global_transformed_dim = 3 
        # scaled_obj_id, scaled_E, scaled_pT, scaled_eta, cos_phi, sin_phi
        self.object_transformed_dim = 6 
        # We'll make them both self.object_transformed_dim (6) for unified processing
        self.unified_feature_dim = self.object_transformed_dim


    def fit(self, X, y=None):
        # X is a torch.Tensor
        # Global features (E_T_miss, phi_E_t_miss)
        X_global_raw = X[:, :self.num_global_features_raw] # (N, 2)

        # E_T_miss is X_global_raw[:, 0]
        et_miss = X_global_raw[:, 0]
        self.global_means = torch.mean(et_miss, dim=0, keepdim=True) # Shape (1,)
        self.global_stds = torch.std(et_miss, dim=0, keepdim=True)   # Shape (1,)
        # Avoid division by zero/small std
        self.global_stds[self.global_stds < 1e-6] = 1.0 

        # Object features (18 objects * 5 features/object)
        X_objects_raw = X[:, self.num_global_features_raw:].reshape(
            -1, self.max_objects, self.num_kinematic_features_per_object
        ) # (N, 18, 5)

        # Mask for active objects (e.g., pT > a small epsilon, effectively >0 for zero-padding)
        # Using pT (index 2 in object features)
        object_mask = X_objects_raw[:, :, 2] > 1e-9 # (N, 18)

        self.object_feature_means = torch.zeros(self.num_object_features_to_scale, device=X.device)
        self.object_feature_stds = torch.ones(self.num_object_features_to_scale, device=X.device)

        for i in range(self.num_object_features_to_scale): # For obj_id, E, pT, eta
            # X_objects_raw shape: (N, 18, 5)
            # object_mask shape: (N, 18)
            # Valid values for current feature i are X_objects_raw[:, :, i] where object_mask is True
            feat_values = X_objects_raw[:, :, i][object_mask] # This selects elements, shape (num_valid_elements,)
            if feat_values.numel() > 0: # Check if there are any valid elements
                self.object_feature_means[i] = torch.mean(feat_values)
                self.object_feature_stds[i] = torch.std(feat_values)
            # Ensure std is not too small to avoid division by zero or large scaled values
            if self.object_feature_stds[i] < 1e-6:
                self.object_feature_stds[i] = 1.0

        return self

    def transform(self, X):
        N = X.shape[0]
        current_device = X.device

        # Global features Transform
        X_global_raw = X[:, :self.num_global_features_raw] # (N, 2)

        et_miss = X_global_raw[:, 0:1]      # (N, 1)
        phi_et_miss = X_global_raw[:, 1:2]  # (N, 1)

        scaled_et_miss = (et_miss - self.global_means.to(current_device)) / self.global_stds.to(current_device) # (N, 1)
        cos_phi_miss = torch.cos(phi_et_miss) # (N, 1)
        sin_phi_miss = torch.sin(phi_et_miss) # (N, 1)

        X_g_transformed = torch.cat([scaled_et_miss, cos_phi_miss, sin_phi_miss], dim=1) # (N, 3)

        # Pad global features to self.unified_feature_dim (6)
        padding_dims = self.unified_feature_dim - self.global_transformed_dim
        if padding_dims > 0:
            padding = torch.zeros(N, padding_dims, device=current_device)
            X_g_final = torch.cat([X_g_transformed, padding], dim=1) # (N, 6)
        else: # Should be self.global_transformed_dim == self.unified_feature_dim (if both 6)
              # or self.global_transformed_dim > self.unified_feature_dim (needs slicing)
              # For current setup (3 -> 6), padding_dims is 3.
            X_g_final = X_g_transformed[:, :self.unified_feature_dim] 

        X_g_final = X_g_final.unsqueeze(1) # (N, 1, 6) - treating global features as a single "object"

        # Object features Transform
        X_objects_raw = X[:, self.num_global_features_raw:].reshape(
            N, self.max_objects, self.num_kinematic_features_per_object
        ) # (N, 18, 5)

        scaled_object_parts = []
        for i in range(self.num_object_features_to_scale): # Scales obj_id, E, pT, eta
            feat_col = X_objects_raw[:, :, i:i+1] # (N, 18, 1)
            scaled_feat = (feat_col - self.object_feature_means[i].to(current_device)) / self.object_feature_stds[i].to(current_device)
            scaled_object_parts.append(scaled_feat)

        # Handle phi for objects (index 4)
        phi_obj = X_objects_raw[:, :, self.object_phi_feature_idx:self.object_phi_feature_idx+1] # (N, 18, 1)
        cos_phi_obj = torch.cos(phi_obj) # (N, 18, 1)
        sin_phi_obj = torch.sin(phi_obj) # (N, 18, 1)
        scaled_object_parts.extend([cos_phi_obj, sin_phi_obj]) # Appends two tensors of shape (N, 18, 1)

        X_o_transformed = torch.cat(scaled_object_parts, dim=2) # (N, 18, 6)

        # Create mask for actual objects (non-padded)
        # pT is at index 2 of the original 5 object features
        object_mask_o = X_objects_raw[:, :, 2] > 1e-9 # (N, 18)
        # Apply mask to zero out features of padded objects AFTER scaling and transformation
        X_o_transformed = X_o_transformed * object_mask_o.unsqueeze(-1).float() # (N, 18, 6)

        # Combine global "object" and actual objects for the sequence
        # X_seq: (N, 1 (global) + 18 (objects), 6 features) = (N, 19, 6)
        X_seq = torch.cat((X_g_final, X_o_transformed), dim=1) 

        # Create final mask corresponding to X_seq (N, 19)
        mask_g = torch.ones(N, 1, dtype=torch.bool, device=current_device) # Global "object" is always present
        mask_seq = torch.cat((mask_g, object_mask_o), dim=1) # (N, 19)

        return (X_seq, mask_seq) # Tuple: (data_tensor, mask_tensor)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], *, use_mask: bool):
        super().__init__()
        self.use_mask = use_mask # Expected to be True for this setup
        # input_shape is (L, F), e.g., (19, 6)
        # L = input_shape[0] # num_sequence_items, e.g., 19 (1 global + 18 objects)
        item_feature_dim = input_shape[1] # num_features_per_item, e.g., 6

        # phi_mlp: processes each item in the sequence (permutation equivariant)
        # Batch norm is applied to the feature dimension for each item.
        self.phi_mlp = nn.Sequential(
            nn.Linear(item_feature_dim, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # rho_mlp: processes aggregated features (permutation invariant)
        # Input dim is 64 (output dim of phi_mlp after aggregation)
        self.rho_mlp = nn.Sequential(
            nn.Linear(64, 128), # Input from aggregated features
            nn.ReLU(),
            nn.Dropout(0.35), # Adjusted dropout
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.35), # Adjusted dropout
            nn.Linear(64, 1)  # Output logit
        )

    def forward(self, data: torch.Tensor, mask: torch.Tensor | None = None):
        # data: (B, L, F_in), e.g. (B, 19, 6)
        # mask: (B, L), e.g. (B, 19)

        B, L, F_in = data.shape

        # Reshape for BatchNorm1d which expects (Batch, Features) or (Batch, Features, Length)
        # Here, we treat each item in the sequence as an independent sample for feature transformation.
        x_flat = data.reshape(B * L, F_in) # (B*L, F_in)
        phi_out_flat = self.phi_mlp(x_flat) # (B*L, F_phi_out), F_phi_out = 64

        # Reshape back to sequence structure: (B, L, F_phi_out)
        x_item_processed = phi_out_flat.reshape(B, L, -1) # (B, 19, 64)

        if self.use_mask and mask is not None:
            # Mask features of non-present items (padded objects) before aggregation
            # mask.unsqueeze(-1) expands mask from (B,L) to (B,L,1) for broadcasting
            x_item_processed = x_item_processed * mask.unsqueeze(-1).float() # (B, 19, 64)

        # Aggregation: Sum or Mean. Mean is often more stable if num_items varies.
        # Use mean aggregation, being careful with division by zero if no items are present (clamp protects this)
        num_actual_items = mask.sum(dim=1, keepdim=True).clamp(min=1).float() # (B, 1), count of True in mask per batch entry
        aggregated_features = torch.sum(x_item_processed, dim=1) / num_actual_items # (B, 64)

        logits = self.rho_mlp(aggregated_features) # (B, 1)
        return logits

def make_model(input_shape, *, use_mask=False):
    # This preprocessor always implies use_mask=True because it returns (X_seq, mask_seq)
    return BinaryClassifier(input_shape, use_mask=True)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 75 # Number of training epochs; early stopping should manage actual duration.

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    # Scheduler reduces learning rate if validation AUC doesn't improve
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=6, verbose=False)

    early_stopping_patience = 15 
    best_val_auc = 0.0
    epochs_no_improve = 0
    # Store the state_dict of the best model found so far
    best_model_state = model.state_dict() 

    history_train_loss, history_val_loss = [], []
    history_train_acc, history_val_acc = [], []

    print(f"Starting training on {device} for up to {epochs} epochs...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        current_train_loss = 0.0
        correct_train_predictions = 0
        total_train_samples_epoch = 0

        for batch_data in train_loader:
            # Data from loader: ((X_seq_batch, mask_batch), label_batch)
            (X_seq_b, mask_b), labels_b = batch_data
            # Move data to device and ensure correct types/shapes
            X_seq_b = X_seq_b.to(device)
            mask_b = mask_b.to(device)
            labels_b = labels_b.to(device).float().unsqueeze(1) # Target for BCEWithLogitsLoss

            optimizer.zero_grad()
            outputs_logits = model(X_seq_b, mask_b) # Model returns (B, 1) logits
            loss = criterion(outputs_logits, labels_b)
            loss.backward()
            optimizer.step()

            current_train_loss += loss.item() * X_seq_b.size(0) # Accumulate loss weighted by batch size
            # Calculate accuracy: convert logits to predictions (0 or 1)
            predicted_classes = (torch.sigmoid(outputs_logits) > 0.5).long()
            correct_train_predictions += (predicted_classes == labels_b.long()).sum().item()
            total_train_samples_epoch += X_seq_b.size(0)

        avg_epoch_train_loss = current_train_loss / total_train_samples_epoch
        avg_epoch_train_acc = correct_train_predictions / total_train_samples_epoch
        history_train_loss.append(avg_epoch_train_loss)
        history_train_acc.append(avg_epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        current_val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples_epoch = 0
        all_validation_labels = []
        all_validation_probs = [] # Probabilities for AUC calculation

        with torch.no_grad():
            for batch_data in val_loader:
                (X_seq_b, mask_b), labels_b_cpu = batch_data # labels_b_cpu are on CPU initially
                X_seq_b = X_seq_b.to(device)
                mask_b = mask_b.to(device)
                labels_b_dev = labels_b_cpu.to(device).float().unsqueeze(1) # For loss calculation

                outputs_logits = model(X_seq_b, mask_b)
                loss = criterion(outputs_logits, labels_b_dev)

                current_val_loss += loss.item() * X_seq_b.size(0)
                output_probs = torch.sigmoid(outputs_logits) # Probabilities for AUC and acc
                predicted_classes = (output_probs > 0.5).long()
                correct_val_predictions += (predicted_classes == labels_b_dev.long()).sum().item()
                total_val_samples_epoch += X_seq_b.size(0)

                # Collect all labels and probabilities for epoch-wide AUC
                all_validation_labels.extend(labels_b_cpu.numpy()) # Use original CPU labels
                all_validation_probs.extend(output_probs.cpu().numpy().flatten()) # Flatten probs

        avg_epoch_val_loss = current_val_loss / total_val_samples_epoch
        avg_epoch_val_acc = correct_val_predictions / total_val_samples_epoch
        history_val_loss.append(avg_epoch_val_loss)
        history_val_acc.append(avg_epoch_val_acc)

        # Calculate AUC for the entire validation set for this epoch
        epoch_val_auc = roc_auc_score(all_validation_labels, all_validation_probs)

        print(f"Epoch {epoch+1}/{epochs} :: "
              f"Train Loss: {avg_epoch_train_loss:.4f}, Train Acc: {avg_epoch_train_acc:.4f} :: "
              f"Val Loss: {avg_epoch_val_loss:.4f}, Val Acc: {avg_epoch_val_acc:.4f}, Val AUC: {epoch_val_auc:.4f}")

        scheduler.step(epoch_val_auc) # Step scheduler based on validation AUC

        # Early stopping logic
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            epochs_no_improve = 0
            best_model_state = model.state_dict() # Save current model state as best
            print(f"    Improved Val AUC to {best_val_auc:.4f}. Saving model state.")
        else:
            epochs_no_improve += 1
            print(f"    Val AUC ({epoch_val_auc:.4f}) did not improve for {epochs_no_improve} epoch(s). Best: {best_val_auc:.4f}")


        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping after {epoch+1} epochs. Best Val AUC: {best_val_auc:.4f}. Restoring best model.")
            model.load_state_dict(best_model_state) # Restore the best model state
            break

    # After loop, if not early stopped, ensure best model is loaded (could be last epoch wasn't the best)
    if not (epochs_no_improve >= early_stopping_patience):
        print("Training finished. Restoring best model state based on Val AUC.")
        model.load_state_dict(best_model_state)

    return model, history_train_loss, history_val_loss, history_train_acc, history_val_acc

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

