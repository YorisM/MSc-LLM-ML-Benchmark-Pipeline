
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
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F # For F.one_hot if used, not used in current version
from sklearn.metrics import roc_auc_score # For AUC calculation

# Using a small epsilon for numerical stability in divisions and log.
EPSILON = 1e-7

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
    # Total flat length per event (X_train & X_val): 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2-6  : object 1  ->  obj_id_1, E_1, p_T1, eta_1, phi_1 (assuming obj_id is the first feature)
    # ...
    # Indices 88-92 : object 18 -> obj_id_18, E_18 , p_T_18 , eta_18 , phi_18
    # Per-object slice size = 5
    # Max objects encoded   = 18

    # Feature indices within the raw 92-feature vector
    MET_IDX = 0
    MET_PHI_IDX = 1
    OBJ_START_IDX = 2
    FEATURES_PER_OBJECT = 5 # obj_id, E, pT, eta, phi
    MAX_OBJECTS = 18

    # Indices for features within each 5-feature object block
    OBJ_ID_IDX = 0
    E_IDX = 1
    PT_IDX = 2
    ETA_IDX = 3
    PHI_IDX = 4

    def __init__(self):
        self.fitted = False
        # For global features (MET)
        self.met_mean = torch.tensor(0.0)
        self.met_std = torch.tensor(1.0)

        # For object features
        # We will scale obj_id as a numerical feature.
        # We will log-transform E and pT, then scale. Eta will be scaled. Phi transformed to cos/sin.
        self.obj_feat_means = torch.zeros(4) # For obj_id, log(E), log(pT), eta
        self.obj_feat_stds = torch.ones(4)   # For obj_id, log(E), log(pT), eta

    def fit(self, X, y=None):
        # X is expected to be a PyTorch tensor (N, 92)

        # 1. Global features: E_T_miss
        met_magnitude = X[:, self.MET_IDX]
        self.met_mean = torch.mean(met_magnitude)
        self.met_std = torch.std(met_magnitude)
        if self.met_std < EPSILON: # Prevent division by zero or very small std
            self.met_std = torch.tensor(EPSILON)

        # 2. Object features
        # Reshape objects part to (N, MAX_OBJECTS, FEATURES_PER_OBJECT)
        objects_raw = X[:, self.OBJ_START_IDX:].reshape(
            -1, self.MAX_OBJECTS, self.FEATURES_PER_OBJECT
        ) # Shape: (N, 18, 5)

        # Create mask for actual (non-padded) objects. pT > 0 is a robust indicator.
        actual_object_mask = objects_raw[:, :, self.PT_IDX] > 0 # Shape: (N, 18)

        # Extract features from actual objects only for calculating statistics
        # This ensures that padding values (zeros) do not distort the statistics.
        obj_ids_actual = objects_raw[:, :, self.OBJ_ID_IDX][actual_object_mask]
        E_actual = objects_raw[:, :, self.E_IDX][actual_object_mask]
        pT_actual = objects_raw[:, :, self.PT_IDX][actual_object_mask]
        eta_actual = objects_raw[:, :, self.ETA_IDX][actual_object_mask]

        # Log transform E and pT. Add EPSILON to avoid log(0).
        # Log transformation helps in handling wide ranges of E/pT values typical in HEP.
        log_E_actual = torch.log(E_actual + EPSILON)
        log_pT_actual = torch.log(pT_actual + EPSILON)

        # Calculate means and stds for obj_id, log(E), log(pT), eta
        # obj_id is treated as a numerical feature and scaled. This is a simplification;
        # if obj_id were known to be categorical (like PDG ID), embedding would be better.
        if obj_ids_actual.numel() > 0: # Check if tensor is not empty
            features_to_scale = torch.stack([
                obj_ids_actual.float(), 
                log_E_actual,
                log_pT_actual,
                eta_actual
            ], dim=1) # Shape: (num_actual_objects, 4)

            self.obj_feat_means = torch.mean(features_to_scale, dim=0)
            self.obj_feat_stds = torch.std(features_to_scale, dim=0)
            # Prevent std from being zero
            self.obj_feat_stds[self.obj_feat_stds < EPSILON] = EPSILON
        else:
            # Fallback if no actual objects are found (e.g., fitting on empty or all-padding data)
            # Uses default initialized values (means=0, stds=1).
            # This shouldn't happen with the provided dataset but is a safeguard.
            # print("Warning: No actual objects found during preprocessor fitting. Using default scaling params.")
            pass # Already initialized

        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        N = X.shape[0]
        current_device = X.device # Ensure all tensors are on the same device

        # 1. Process global features
        met_magnitude = X[:, self.MET_IDX]
        met_phi = X[:, self.MET_PHI_IDX]

        # Standardize MET magnitude
        met_scaled = (met_magnitude - self.met_mean.to(current_device)) / self.met_std.to(current_device)
        # Transform phi (cyclical feature) to its Cartesian components
        met_phi_cos = torch.cos(met_phi)
        met_phi_sin = torch.sin(met_phi)

        # Shape : (N, 3) [met_scaled, met_phi_cos, met_phi_sin]
        global_features_processed = torch.stack([met_scaled, met_phi_cos, met_phi_sin], dim=1)

        # 2. Process object features
        objects_raw = X[:, self.OBJ_START_IDX:].reshape(
            N, self.MAX_OBJECTS, self.FEATURES_PER_OBJECT
        ) # Shape: (N, 18, 5)

        # Extract individual kinematic variables
        obj_ids = objects_raw[:, :, self.OBJ_ID_IDX].float() # (N, 18)
        E = objects_raw[:, :, self.E_IDX]         # (N, 18)
        pT = objects_raw[:, :, self.PT_IDX]        # (N, 18)
        eta = objects_raw[:, :, self.ETA_IDX]       # (N, 18)
        phi_obj = objects_raw[:, :, self.PHI_IDX]    # (N, 18)

        # Log transform E and pT
        log_E = torch.log(E + EPSILON)
        log_pT = torch.log(pT + EPSILON)

        # Apply scaling (standardization) using stored means and stds
        obj_ids_scaled = (obj_ids - self.obj_feat_means[0].to(current_device)) / self.obj_feat_stds[0].to(current_device)
        log_E_scaled = (log_E - self.obj_feat_means[1].to(current_device)) / self.obj_feat_stds[1].to(current_device)
        log_pT_scaled = (log_pT - self.obj_feat_means[2].to(current_device)) / self.obj_feat_stds[2].to(current_device)
        eta_scaled = (eta - self.obj_feat_means[3].to(current_device)) / self.obj_feat_stds[3].to(current_device)

        # Transform object phi to (cos(phi), sin(phi))
        obj_phi_cos = torch.cos(phi_obj)
        obj_phi_sin = torch.sin(phi_obj)

        # Stack processed object features. Resulting shape: (N, 18, 6)
        # Features: [obj_id_scaled, log_E_scaled, log_pT_scaled, eta_scaled, obj_phi_cos, obj_phi_sin]
        object_features_processed = torch.stack([
            obj_ids_scaled, log_E_scaled, log_pT_scaled, eta_scaled, obj_phi_cos, obj_phi_sin
        ], dim=-1)

        # 3. Create the sequence input for the model (X_seq)
        # Global features are repeated for each object/timestep and concatenated.
        # This allows the Transformer to access global information at each step if needed.
        expanded_global_features = global_features_processed.unsqueeze(1).repeat(1, self.MAX_OBJECTS, 1)

        # X_seq combines object-specific features and global event features.
        # Shape: (N, 18, 6+3=9)
        X_seq = torch.cat([object_features_processed, expanded_global_features], dim=-1)

        # 4. Create the padding mask
        # Mask is True for actual objects (pT > 0), False for padding.
        # This mask format (True for valid) is then inverted inside the model for PyTorch's Transformer.
        mask = (pT > 0) # Shape: (N, 18). True indicates a valid, non-padded object.

        return X_seq, mask # Tensor shapes: (N, 18, 9), (N, 18)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class ParticleTransformerClassifier(nn.Module):
    def __init__(self, input_features, model_dim, n_heads, num_encoder_layers, mlp_hidden_dim, dropout_rate):
        super().__init__()

        # Linear projection layer to project input features (9) to model_dim (e.g., 64)
        self.input_projection = nn.Linear(input_features, model_dim)

        # Transformer Encoder Layers
        # batch_first=True expects (Batch, Seq, Feature)
        # norm_first=True applies LayerNorm before self-attention and FFN, often more stable.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=mlp_hidden_dim, 
            dropout=dropout_rate,
            activation='relu', 
            batch_first=True, 
            norm_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        # Output MLP for classification
        # Takes pooled output from transformer (size model_dim) and produces a single logit.
        self.output_mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, 1) 
        )

    def forward(self, x_seq, src_padding_mask_valid):
        # x_seq: (Batch, SeqLen=18, Features=9)
        # src_padding_mask_valid: (Batch, SeqLen=18), True for valid elements.

        # PyTorch's TransformerEncoderLayer expects src_key_padding_mask where True means "ignore this position".
        # So, we invert the mask: True for padded elements, False for valid elements.
        transformer_padding_mask = ~src_padding_mask_valid # Shape: (Batch, 18)

        # Project input features to model_dim
        x_projected = self.input_projection(x_seq) # Shape: (Batch, 18, model_dim)

        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(x_projected, src_key_padding_mask=transformer_padding_mask)
        # Shape: (Batch, 18, model_dim)

        # Masked Average Pooling:
        # We want to average only over the valid (non-padded) elements in the sequence.
        # Expand src_padding_mask_valid to (B, 18, 1) to allow element-wise multiplication.
        expanded_src_padding_mask_valid = src_padding_mask_valid.unsqueeze(-1).float()

        # Zero out contributions from padded elements
        masked_transformer_output = transformer_output * expanded_src_padding_mask_valid

        # Summing valid elements along sequence length dimension (dim=1)
        summed_output = masked_transformer_output.sum(dim=1) # Shape: (B, model_dim)

        # Count of valid (non-padded) elements per batch entry
        # .clamp(min=EPSILON) to avoid division by zero if a sequence has no valid elements (unlikely here).
        valid_elements_count = src_padding_mask_valid.sum(dim=1, keepdim=True).float().clamp(min=EPSILON) # Shape: (B, 1)

        pooled_output = summed_output / valid_elements_count # Shape: (B, model_dim)

        # Pass pooled output through MLP for classification
        logits = self.output_mlp(pooled_output) # Shape: (B, 1)
        return logits

def make_model(input_shape, *, use_mask=False):
    # input_shape is (SeqLen, FeaturesPerElement), e.g. (18, 9)
    # use_mask indicates if forward will receive a mask. Our preprocessor provides one.

    seq_len, num_features = input_shape 

    # Hyperparameters for the Transformer model. These can be further tuned.
    model_dim = 64          # Internal dimension of the Transformer.
    n_heads = 4             # Number of attention heads. (model_dim must be divisible by n_heads)
    num_encoder_layers = 3  # Number of Transformer encoder layers.
    mlp_hidden_dim_transformer = 256 # Feedforward dimension in Transformer layers (often 2*model_dim or 4*model_dim)
    mlp_hidden_dim_final = 128 # Hidden dimension for the final MLP classifier.
    dropout_rate = 0.1      # Dropout rate for regularization.

    model = ParticleTransformerClassifier(
        input_features=num_features,
        model_dim=model_dim,
        n_heads=n_heads,
        num_encoder_layers=num_encoder_layers,
        mlp_hidden_dim=mlp_hidden_dim_transformer, # Using one mlp_hidden_dim for both for simplicity here
        dropout_rate=dropout_rate
    )
    # Adjust final MLP if different hidden dim desired
    model.output_mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_hidden_dim_final),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim_final, 1))

    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 100 # Max number of epochs. Early stopping usually finishes sooner.   
# BATCH_SIZE, LEARNING_RATE, etc., are standard hyperparameters. The harness might set some.
# Assuming BATCH_SIZE around 256-1024.
# LEARNING_RATE = 1e-3 is a common starting point for Adam.
# WEIGHT_DECAY = 1e-2 for AdamW helps with regularization.
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_EARLY_STOPPING_PATIENCE = 10 # For validation AUC
DEFAULT_LR_SCHEDULER_PATIENCE = 3   # For ReduceLROnPlateau

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function: BCEWithLogitsLoss is numerically stable for binary classification.
    criterion = nn.BCEWithLogitsLoss()
    # Optimizer: AdamW is Adam with decoupled weight decay.
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LEARNING_RATE, weight_decay=DEFAULT_WEIGHT_DECAY)
    # Learning rate scheduler: Reduces LR when validation AUC stagnates.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, 
                                                           patience=DEFAULT_LR_SCHEDULER_PATIENCE, verbose=False)

    train_losses, val_losses = [], []
    # The problem asks for train_acc, val_acc. We will return AUC scores for these lists
    # as AUC is the primary metric and more informative for imbalanced classes (though this dataset is balanced).
    train_metrics, val_metrics = [], [] # Storing AUCs here

    best_val_metric = 0.0 # For AUC, higher is better
    epochs_no_improve = 0
    best_model_state = model.state_dict() # Initialize with initial state

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss_train = 0.0
        all_train_labels_list = []
        all_train_preds_list = []

        for data in train_loader:
            # Data format from loader assumed to be (x_seq_batch, mask_batch, y_batch)
            x_seq, x_mask, labels = data
            x_seq, x_mask, labels_float = x_seq.to(device), x_mask.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(x_seq, x_mask) # Model outputs logits
            loss = criterion(outputs, labels_float)
            loss.backward()
            # Gradient clipping can stabilize training for RNNs/Transformers, optional
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss_train += loss.item() * x_seq.size(0)

            probs = torch.sigmoid(outputs.detach())
            all_train_preds_list.append(probs.cpu())
            all_train_labels_list.append(labels.cpu()) # Original int labels

        epoch_loss_train = running_loss_train / len(train_loader.dataset)
        train_losses.append(epoch_loss_train)

        train_preds_cat = torch.cat(all_train_preds_list).numpy().ravel()
        train_labels_cat = torch.cat(all_train_labels_list).numpy().ravel()
        epoch_train_metric = roc_auc_score(train_labels_cat, train_preds_cat)
        train_metrics.append(epoch_train_metric)

        # --- Validation Phase ---
        model.eval()
        running_loss_val = 0.0
        all_val_labels_list = []
        all_val_preds_list = []

        with torch.no_grad():
            for data in val_loader:
                x_seq, x_mask, labels = data
                x_seq, x_mask, labels_float = x_seq.to(device), x_mask.to(device), labels.to(device).float().unsqueeze(1)

                outputs = model(x_seq, x_mask) # Logits
                loss = criterion(outputs, labels_float)
                running_loss_val += loss.item() * x_seq.size(0)

                probs = torch.sigmoid(outputs.detach())
                all_val_preds_list.append(probs.cpu())
                all_val_labels_list.append(labels.cpu())

        epoch_loss_val = running_loss_val / len(val_loader.dataset)
        val_losses.append(epoch_loss_val)

        val_preds_cat = torch.cat(all_val_preds_list).numpy().ravel()
        val_labels_cat = torch.cat(all_val_labels_list).numpy().ravel()
        epoch_val_metric = roc_auc_score(val_labels_cat, val_preds_cat)
        val_metrics.append(epoch_val_metric)

        # print(f"Epoch {epoch+1}/{epochs} - "
        #       f"Train Loss: {epoch_loss_train:.4f}, Train AUC: {epoch_train_metric:.4f} - "
        #       f"Val Loss: {epoch_loss_val:.4f}, Val AUC: {epoch_val_metric:.4f} - "
        #       f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(epoch_val_metric) # Scheduler step on validation AUC

        if epoch_val_metric > best_val_metric:
            best_val_metric = epoch_val_metric
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            # print(f"New best validation AUC: {best_val_metric:.4f}. Model saved.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= DEFAULT_EARLY_STOPPING_PATIENCE:
            # print(f"Early stopping triggered after {epoch+1} epochs. Best Val AUC: {best_val_metric:.4f}")
            break

    if best_model_state is not None: # Ensure best_model_state was set
        model.load_state_dict(best_model_state)

    # The template asks for train_acc, val_acc. We are returning AUC scores in their place.
    return model, train_losses, val_losses, train_metrics, val_metrics

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

    # 4. *Dry-run safety check* – run a single toy forward pass
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

