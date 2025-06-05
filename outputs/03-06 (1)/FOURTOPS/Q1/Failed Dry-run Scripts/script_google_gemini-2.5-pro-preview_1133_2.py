
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
from torch.utils.data import Dataset, DataLoader
# <LLM: Import modules>
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score # For validation AUC reporting, as AUC is the primary metric
# pickle is not directly used but MyPreprocessor must be picklable. Python floats and torch tensors (on CPU) are.

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    #    Must implement:
    #   - fit(...) -> self
    #   - transform(X: Tensor)   -> Tensor  **or**  Tuple[Tensor, Tensor]

    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module. (Achieved by storing Python floats)
    # IMPORTANT: Batch first. (Input X is batch-first, output also respects this)
    # transform() must be deterministic. (Achieved)
    # Store only derived parameters needed for transform. (Achieved)

    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        self.eps = 1e-8 # Small epsilon to prevent division by zero in standardization

        # Parameters for global features (E_T_miss, phi_Et_miss)
        self.et_miss_mean = 0.0
        self.et_miss_std = 1.0

        # Parameters for object features (E, p_T, eta, phi)
        # For E and p_T, we will use log1p transform before scaling
        self.log_energy_mean = 0.0
        self.log_energy_std = 1.0
        self.log_pt_mean = 0.0
        self.log_pt_std = 1.0
        self.eta_mean = 0.0
        self.eta_std = 1.0
        # Phi features will be transformed using sin/cos, no stored parameters needed.

    def fit(self, X, y=None):
        # <LLM: Extract statistics or fit transformers>
        # All calculations are done on CPU tensors to store Python floats (which are picklable).
        X_cpu = X.cpu() 

        # Extract global features (first 2 columns)
        X_global = X_cpu[:, :2] # Shape: (N, 2)
        # Extract object features (remaining columns)
        # Each event can have up to 18 objects, each with 5 features (obj_id, E, p_T, eta, phi). Total 18*5 = 90 columns.
        X_objects_raw = X_cpu[:, 2:].reshape(X_cpu.shape[0], 18, 5) # Shape: (N, 18, 5)

        # Create a mask for valid (non-padded) objects.
        # Padded objects have obj_id = 0. Original obj_id is X_objects_raw[:, :, 0].
        object_ids_for_mask = X_objects_raw[:, :, 0]
        valid_object_mask = (object_ids_for_mask > 0) # Shape: (N, 18), boolean

        # Calculate statistics for E_T_miss (global feature)
        et_miss = X_global[:, 0]
        self.et_miss_mean = torch.mean(et_miss).item()
        self.et_miss_std = torch.std(et_miss).item() + self.eps

        # Extract features for valid objects only to calculate statistics
        # Object features are: X_objects_raw[:, :, 0] = obj_id (used for mask)
        #                      X_objects_raw[:, :, 1] = Energy (E)
        #                      X_objects_raw[:, :, 2] = Transverse Momentum (p_T)
        #                      X_objects_raw[:, :, 3] = Pseudorapidity (eta)
        #                      X_objects_raw[:, :, 4] = Azimuthal angle (phi)

        # Check if there are any valid objects in the dataset before calculations to avoid errors on empty tensors
        if torch.any(valid_object_mask):
            energies = X_objects_raw[:, :, 1][valid_object_mask]
            pts = X_objects_raw[:, :, 2][valid_object_mask]
            etas = X_objects_raw[:, :, 3][valid_object_mask]

            # Log transform for E and p_T (common practice for energy/momentum scales), then standardize
            log_energies = torch.log1p(energies) # log1p(x) = log(1+x) for numerical stability with x near 0
            self.log_energy_mean = torch.mean(log_energies).item()
            self.log_energy_std = torch.std(log_energies).item() + self.eps

            log_pts = torch.log1p(pts)
            self.log_pt_mean = torch.mean(log_pts).item()
            self.log_pt_std = torch.std(log_pts).item() + self.eps

            # Standardize eta
            self.eta_mean = torch.mean(etas).item()
            self.eta_std = torch.std(etas).item() + self.eps
        # If no valid objects are found (e.g., fitting on a batch of only padded events, unlikely), 
        # the default UCF_ACCESS_DENIED - Incorrect path.s (mean=0.0, std=1.0) will be used.

        return self

    def transform(self, X):
        # <LLM: Apply preprocessing logic, return torch.Tensor>
        # Ensure all transformations happen on CPU, output CPU tensors.
        # This makes preprocessor independent of device used in training loop.
        X_cpu = X.cpu() 
        N = X_cpu.shape[0] # Batch size

        X_global_raw = X_cpu[:, :2] # Shape: (N, 2)
        X_objects_raw = X_cpu[:, 2:].reshape(N, 18, 5) # Shape: (N, 18, 5)

        # --- Process global features ---
        # Standardize E_T_miss
        et_miss_scaled = (X_global_raw[:, 0] - self.et_miss_mean) / self.et_miss_std # Shape: (N,)
        # Transform phi_Et_miss using sin/cos (common for angular features)
        phi_et_miss = X_global_raw[:, 1] # Shape: (N,)
        cos_phi_et_miss = torch.cos(phi_et_miss) # Shape: (N,)
        sin_phi_et_miss = torch.sin(phi_et_miss) # Shape: (N,)
        # Concatenate processed global features
        processed_global_features = torch.stack(
            [et_miss_scaled, cos_phi_et_miss, sin_phi_et_miss], dim=1
        ) # Shape: (N, 3)

        # --- Process object features ---
        obj_ids = X_objects_raw[:, :, 0]    # Shape: (N, 18)
        energies = X_objects_raw[:, :, 1]   # Shape: (N, 18)
        pts = X_objects_raw[:, :, 2]        # Shape: (N, 18)
        etas = X_objects_raw[:, :, 3]       # Shape: (N, 18)
        phis = X_objects_raw[:, :, 4]       # Shape: (N, 18)

        # Create mask for valid objects (obj_id > 0 indicates a real particle, not padding)
        object_mask = (obj_ids > 0) # Shape: (N, 18), boolean

        # Initialize tensors for processed object features (filled with zeros)
        log_energies_scaled = torch.zeros_like(energies) # Shape: (N, 18)
        log_pts_scaled = torch.zeros_like(pts)           # Shape: (N, 18)
        etas_scaled = torch.zeros_like(etas)             # Shape: (N, 18)

        # Apply transformations only to valid objects using the mask
        if torch.any(object_mask): # Ensure there's at least one valid object in the batch
            current_energies = energies[object_mask] # Selects valid entries into a flat tensor
            current_pts = pts[object_mask]
            current_etas = etas[object_mask]

            # Apply log1p, then standardize
            log_energies_scaled[object_mask] = (torch.log1p(current_energies) - self.log_energy_mean) / self.log_energy_std
            log_pts_scaled[object_mask] = (torch.log1p(current_pts) - self.log_pt_mean) / self.log_pt_std
            etas_scaled[object_mask] = (current_etas - self.eta_mean) / self.eta_std

        # Transform object phi using sin/cos
        cos_phis = torch.cos(phis) # Shape: (N, 18)
        sin_phis = torch.sin(phis) # Shape: (N, 18)

        # Stack processed object features: E, pT, eta, cos(phi), sin(phi)
        # Resulting shape: (N, 18, 5 features)
        processed_object_features_stacked = torch.stack(
            [log_energies_scaled, log_pts_scaled, etas_scaled, cos_phis, sin_phis], dim=-1
        ) 

        # Ensure features of padded objects are strictly zero
        # This is important because cos(0)=1, sin(0)=0 from padded phi=0 values would otherwise be non-zero.
        float_mask_expanded = object_mask.unsqueeze(-1).float() # Shape: (N, 18, 1)
        processed_object_features = processed_object_features_stacked * float_mask_expanded # Zeroes out padded objects

        # Return tuple: ((object_features, global_features), object_mask)
        # All returned tensors are on CPU.
        return (processed_object_features, processed_global_features), object_mask

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# Custom Dataset class to work with PyTorch DataLoader
# It handles the specific ((obj_feats, glob_feats), mask) structure from MyPreprocessor
class CustomDataset(Dataset):
    def __init__(self, preprocessed_X_tuple, Y_labels):
        # preprocessed_X_tuple is ((obj_feats, glob_feats), mask)
        # Y_labels is the tensor of labels
        self.obj_feats = preprocessed_X_tuple[0][0]    # Tensor of shape (N, 18, 5)
        self.glob_feats = preprocessed_X_tuple[0][1]   # Tensor of shape (N, 3)
        self.mask = preprocessed_X_tuple[1]            # Tensor of shape (N, 18)
        self.Y = Y_labels                              # Tensor of shape (N,)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        # Returns a single sample: ((obj_feats_sample, glob_feats_sample), mask_sample), label_sample
        return ((self.obj_feats[idx], self.glob_feats[idx]), self.mask[idx]), self.Y[idx]

# 2. ---------- MODEL DEFINITION ----------
class ParticleTransformer(nn.Module):
    def __init__(self, obj_feat_dim, global_feat_dim, 
                 d_model, n_head, num_encoder_layers, 
                 dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model # Internal dimension of the model

        # Embedding layer for object features: projects 5 input features to d_model dimensions
        self.obj_embedder = nn.Linear(obj_feat_dim, d_model)

        # Standard PyTorch Transformer Encoder
        # TransformerEncoderLayer defines one layer of the encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,                # Internal dimension
            nhead=n_head,                   # Number of attention heads
            dim_feedforward=dim_feedforward,# Dimension of the feed-forward network within the layer
            dropout=dropout,                # Dropout rate
            batch_first=True,               # Crucial: Input format is (N, SequenceLength, FeatureDim)
            activation='relu'               # Activation function (can also be 'gelu')
        )
        # TransformerEncoder stacks multiple encoder_layer instances
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )

        # Output MLP for classification
        # Takes aggregated object representation + global features, outputs a single logit
        mlp_input_dim = d_model + global_feat_dim
        self.output_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, d_model * 2), # Hidden layer, size can be tuned
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),       # Another hidden layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)                  # Output layer producing a single logit
        )

    def forward(self, data_input):
        # data_input is ((obj_feats, glob_feats), object_mask)
        (obj_features, global_features), object_mask = data_input
        # obj_features: (N, MaxObjects, obj_feat_dim), e.g., (N, 18, 5)
        # global_features: (N, global_feat_dim), e.g., (N, 3)
        # object_mask: (N, MaxObjects), boolean, True for valid (non-padded) objects

        # 1. Embed object features: (N, 18, 5) -> (N, 18, d_model)
        obj_embedded = self.obj_embedder(obj_features)

        # 2. Apply Transformer encoder
        # src_key_padding_mask: (N, SequenceLength), True for PADDED elements.
        # Our object_mask is True for VALID elements, so we invert it.
        padding_mask = ~object_mask # Invert mask: (N, 18)
        transformer_output = self.transformer_encoder(obj_embedded, src_key_padding_mask=padding_mask) # (N, 18, d_model)

        # 3. Aggregate object representations using masked average pooling
        # Expand mask for broadcasting: (N, 18) -> (N, 18, 1)
        float_mask_expanded = object_mask.unsqueeze(-1).float()
        # Zero out contributions from padded objects before summing
        masked_transformer_output = transformer_output * float_mask_expanded

        # Sum over sequence length (dim=1): (N, d_model)
        summed_output = masked_transformer_output.sum(dim=1) 

        # Normalize by the number of valid objects to get mean; clamp num_valid_objects to min 1.0 to avoid division by zero
        num_valid_objects = object_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0) # (N, 1)
        mean_pooled_output = summed_output / num_valid_objects # (N, d_model) E.g. mean of features of actual particles

        # 4. Concatenate aggregated_object_features with global_features
        combined_features = torch.cat([mean_pooled_output, global_features], dim=1) # (N, d_model + global_feat_dim)

        # 5. Pass through output MLP to get logits
        logits = self.output_mlp(combined_features) # (N, 1)

        # Return logits (N,). BCEWithLogitsLoss expects (N,) or (N,1) logits and (N,) or (N,1) targets.
        return logits.squeeze(-1) 

def make_model(input_shape, *, use_mask=False):
    # <LLM: Write code to define a binary-classifier network>
    # input_shape is derived from preprocessor's output format FOR A SINGLE SAMPLE:
    # ((obj_shape_single, glob_shape_single), mask_shape_single)
    # Example: obj_shape_single=(18, 5), glob_shape_single=(3,), mask_shape_single=(18,)

    obj_shape_single = input_shape[0][0]    # e.g. (18, 5)
    glob_shape_single = input_shape[0][1]   # e.g. (3,)

    obj_feat_dim = obj_shape_single[1]      # Number of features per object, e.g., 5
    global_feat_dim = glob_shape_single[0]  # Number of global features, e.g., 3

    # Hyperparameters for the ParticleTransformer model
    # These are chosen based on common practices and can be tuned further for optimal performance.
    d_model = 128                 # Embedding dimension and internal model dimension
    n_head = 8                    # Number of attention heads in Transformer layers
    num_encoder_layers = 4        # Number of Transformer encoder layers (depth)
    dim_feedforward = d_model * 4 # Dimension of feedforward network in Transformer layers (often 2x or 4x d_model)
    dropout = 0.15                # Dropout rate for regularization

    model = ParticleTransformer(
        obj_feat_dim=obj_feat_dim,
        global_feat_dim=global_feat_dim,
        d_model=d_model,
        n_head=n_head,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 75 # <LLM: define the amount of training epochs>    
# Max number of epochs for training. Early stopping will likely intervene sooner.

def train_model(model, train_loader, val_loader, epochs):
    # <LLM: Write code to define training loop>
    # <LLM: Implement early stopping if possible>
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function: Binary Cross-Entropy with Logits (numerically stable)
    criterion = nn.BCEWithLogitsLoss() 
    # Optimizer: AdamW is often preferred for Transformer-based models due to better weight decay handling
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2) 

    # Learning rate scheduler: reduces learning rate if validation loss plateaus
    # "verbose" parameter is not used as per instructions.
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

    # Early stopping parameters
    early_stopping_patience = 15  # Number of epochs to wait for improvement before stopping
    epochs_no_improve = 0         # Counter for epochs without improvement
    best_val_loss = float('inf')  # Initialize best validation loss to infinity

    # Lists to store metrics per epoch
    train_losses, val_losses = [], []
    # The problem asks for train_acc, val_acc. Given AUC is the main metric,
    # we interpret this as "Area Under ROC Curve" for both training and validation sets.
    train_aucs, val_aucs = [], []

    print(f"Training on {device} for {epochs} epochs with early stopping (patience={early_stopping_patience}).")
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train() # Set model to training mode
        running_loss = 0.0
        train_epoch_labels_list = [] # Store all labels for epoch AUC calculation
        train_epoch_preds_list = []  # Store all predictions for epoch AUC calculation

        for batch_data, labels in train_loader:
            # Unpack batch_data and move to device
            (obj_feats, glob_feats), mask_feats = batch_data
            obj_feats, glob_feats = obj_feats.to(device), glob_feats.to(device)
            mask_feats = mask_feats.to(device)
            labels = labels.to(device).float() # BCEWithLogitsLoss expects float targets (Y_train is int64)

            model_input = ((obj_feats, glob_feats), mask_feats)

            optimizer.zero_grad() # Clear previous gradients
            outputs = model(model_input) # Forward pass: get logits
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backward pass: compute gradients
            optimizer.step() # Update model parameters

            running_loss += loss.item() * obj_feats.size(0) # Accumulate loss (weighted by actual batch size)

            # Store predictions (probabilities after sigmoid) and true labels for AUC calculation
            train_epoch_labels_list.append(labels.cpu().numpy())
            train_epoch_preds_list.append(torch.sigmoid(outputs).detach().cpu().numpy())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Calculate training AUC for the epoch
        train_epoch_labels_np = np.concatenate(train_epoch_labels_list)
        train_epoch_preds_np = np.concatenate(train_epoch_preds_list)
        epoch_train_auc = roc_auc_score(train_epoch_labels_np, train_epoch_preds_np)
        train_aucs.append(epoch_train_auc)

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        running_val_loss = 0.0
        val_epoch_labels_list = [] # Store all labels for validation epoch AUC
        val_epoch_preds_list = []  # Store all predictions for validation epoch AUC

        with torch.no_grad(): # Disable gradient calculations for validation
            for batch_data, labels in val_loader:
                (obj_feats, glob_feats), mask_feats = batch_data
                obj_feats, glob_feats = obj_feats.to(device), glob_feats.to(device)
                mask_feats = mask_feats.to(device)
                labels = labels.to(device).float()

                model_input = ((obj_feats, glob_feats), mask_feats)

                outputs = model(model_input) # Forward pass
                loss = criterion(outputs, labels) # Calculate loss
                running_val_loss += loss.item() * obj_feats.size(0) # Accumulate loss

                val_epoch_labels_list.append(labels.cpu().numpy())
                val_epoch_preds_list.append(torch.sigmoid(outputs).detach().cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Calculate validation AUC for the epoch
        val_epoch_labels_np = np.concatenate(val_epoch_labels_list)
        val_epoch_preds_np = np.concatenate(val_epoch_preds_list)
        epoch_val_auc = roc_auc_score(val_epoch_labels_np, val_epoch_preds_np)
        val_aucs.append(epoch_val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train AUC: {epoch_train_auc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val AUC: {epoch_val_auc:.4f}")

        # Update learning rate scheduler based on validation loss
        scheduler.step(epoch_val_loss)

        # Early stopping logic: check if validation loss has improved
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Optionally, save the best model state if the harness allows/requires it.
            # For this problem, the model is trained in-place and returned.
            # torch.save(model.state_dict(), 'best_model_checkpoint.pth') 
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs as validation loss did not improve.")
            break # Exit training loop

    # The problem asks for the trained model instance to be returned.
    # If a 'best_model_checkpoint.pth' was saved, one might load it here:
    # model.load_state_dict(torch.load('best_model_checkpoint.pth'))
    # However, returning the model at the point of early stopping is standard.

    return model, train_losses, val_losses, train_aucs, val_aucs

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.

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
    if isinstance(X_train, torch.Tensor):
        # still a “vanilla” tensor (no masking)
        input_shape = X_train.shape[1:]   # e.g. (F,)
        use_mask    = False
    else:
        # X_train = ((obj_feats, glob_feats), mask)
        obj_feats, glob_feats = X_train[0]
        # each of those is a tensor of shape (batch_size, …)
        # so for one sample:
        obj_shape_single  = obj_feats.shape[1:]   # e.g. (18, 5)
        glob_shape_single = glob_feats.shape[1:]  # e.g. (3,)
        input_shape = (obj_shape_single, glob_shape_single)
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

