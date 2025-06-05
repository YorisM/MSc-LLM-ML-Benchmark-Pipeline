
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
import math
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import copy # For saving best model

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
    # Indices  2-6  : object 1  ->  obj_id_1, E_1, p_T1, eta_1, phi_1  (assuming obj_id_1 is object type)
    # ...
    # Indices 88-92 : object 18 -> obj_id_18, E_18 , p_T_18 , eta_18 , phi_18
    # Per-object slice size = 5
    # Max objects encoded   = 18

    def __init__(self):
        self.et_miss_scalar = StandardScaler() # For log(E_T_miss)
        self.phi_miss_scalar = StandardScaler() # For sin/cos transformed phi_E_t_miss

        self.obj_type_encoder = None # Will be OneHotEncoder
        self.fitted_obj_types = None
        self.num_obj_type_features = 0

        self.obj_continuous_scalar = StandardScaler() # For log(E), log(pT), eta
        self.obj_phi_scalar = StandardScaler() # For sin/cos transformed object phi

        self.num_object_slots = 18
        self.features_per_object_raw = 5
        self.eps = 1e-6 # Small epsilon for log1p stability and pT threshold

    def fit(self, X, y=None):
        # X is a PyTorch tensor
        # Extract global features
        et_miss_raw = X[:, 0:1] # (N, 1)
        phi_et_miss_raw = X[:, 1:2] # (N, 1)

        # Fit E_T_miss scalar
        # Apply log1p transformation, ensure non-negative before log
        log_et_miss = torch.log1p(et_miss_raw.clamp(min=0.0))
        self.et_miss_scalar.fit(log_et_miss.cpu().numpy())

        # Fit phi_E_t_miss scalar (after sin/cos transformation)
        sincos_phi_miss = torch.cat([torch.cos(phi_et_miss_raw), torch.sin(phi_et_miss_raw)], dim=1) # (N, 2)
        self.phi_miss_scalar.fit(sincos_phi_miss.cpu().numpy())

        # Reshape object features
        N = X.shape[0]
        objects_flat = X[:, 2:<ctrl62>
        obj_pt = objects_raw[:, :, 2] # (N, 18)
        # Mask for actual objects (pT > eps, indicating presence)
        # Ensure mask is boolean
        object_presence_mask = (obj_pt > self.eps).bool() # (N, 18)

        # Fit object type encoder
        obj_types_raw = objects_raw[:, :, 0] # (N, 18)
        # Consider only types of present objects for fitting the encoder
        present_obj_types = obj_types_raw[object_presence_mask]
        if present_obj_types.numel() == 0: # No objects in the dataset or all have pT <= eps
            # This is an edge case. Assume no type features or handle as error.
            # For now, let's assume at least some objects are present overall.
            # If not, obj_type_encoder might remain None or be problematic.
            # A robust way: if no present_obj_types, create a dummy type to avoid error
            # For this challenge, assume training data is rich enough.
            self.fitted_obj_types = torch.empty(0, device=X.device, dtype=obj_types_raw.dtype) # Ensure it's a tensor
        else:
            self.fitted_obj_types = torch.unique(present_obj_types, sorted=True)

        # Initialize OneHotEncoder with categories learned from present types
        # Ensure self.fitted_obj_types is a list of 1D array for categories argument
        self.obj_type_encoder = OneHotEncoder(categories=[self.fitted_obj_types.cpu().numpy()], 
                                              handle_unknown='ignore', sparse_output=False)
        # Fit the encoder (mainly to validate categories and set up transform logic)
        # OneHotEncoder fit expects a 2D array [n_samples, n_features]
        # We fit it on the unique types themselves to establish the encoding.
        # A dummy fit call to establish transformation parameters based on categories
        if self.fitted_obj_types.numel() > 0:
             self.obj_type_encoder.fit(self.fitted_obj_types.cpu().numpy().reshape(-1, 1))
             self.num_obj_type_features = self.obj_type_encoder.transform(self.fitted_obj_types.cpu().numpy().reshape(-1,1)).shape[1]
        else: # No object types found
             # To prevent errors, fit OHE with an empty array or handle it as a special case
             # This will result in 0 features for OHE.
             self.obj_type_encoder.fit(np.empty((0,1))) # Fit with empty array if no types
             self.num_obj_type_features = 0


        # Fit object continuous feature scalar (logE, logpT, eta)
        obj_E_raw = objects_raw[:, :, 1] # (N, 18)
        obj_eta_raw = objects_raw[:, :, 3] # (N, 18)

        # Apply transformations only to present objects for fitting scalers
        log_E_present = torch.log1p(obj_E_raw[object_presence_mask].clamp(min=0.0))
        log_pT_present = torch.log1p(obj_pt[object_presence_mask].clamp(min=0.0)) # obj_pt already extracted
        eta_present = obj_eta_raw[object_presence_mask]

        # Stack them: (num_present_objects, 3)
        obj_continuous_to_scale = torch.stack([log_E_present, log_pT_present, eta_present], dim=-1)
        if obj_continuous_to_scale.numel() > 0:
            self.obj_continuous_scalar.fit(obj_continuous_to_scale.cpu().numpy())
        else: # Handle case with no present objects for continuous features
            # Fit with a dummy 2D array matching expected feature dimension
            self.obj_continuous_scalar.fit(np.empty((0,3)))


        # Fit object phi scalar (sin/cos transformation)
        obj_phi_raw = objects_raw[:, :, 4] # (N, 18)
        phi_present = obj_phi_raw[object_presence_mask]
        sincos_obj_phi_present = torch.stack([torch.cos(phi_present), torch.sin(phi_present)], dim=-1) # (num_present_objects, 2)
        if sincos_obj_phi_present.numel() > 0:
            self.obj_phi_scalar.fit(sincos_obj_phi_present.cpu().numpy())
        else: # Handle case with no present objects for phi features
            self.obj_phi_scalar.fit(np.empty((0,2)))

        return self

    def transform(self, X):
        # X is a PyTorch tensor
        N = X.shape[0]
        current_device = X.device
        dtype = X.dtype

        # Process global features
        et_miss_raw = X[:, 0:1]
        phi_et_miss_raw = X[:, 1:2]

        log_et_miss = torch.log1p(et_miss_raw.clamp(min=0.0))
        scaled_log_et_miss = torch.from_numpy(self.et_miss_scalar.transform(log_et_miss.cpu().numpy())).to(current_device, dtype=dtype) # (N, 1)

        sincos_phi_miss = torch.cat([torch.cos(phi_et_miss_raw), torch.sin(phi_et_miss_raw)], dim=1) # (N, 2)
        scaled_sincos_phi_miss = torch.from_numpy(self.phi_miss_scalar.transform(sincos_phi_miss.cpu().numpy())).to(current_device, dtype=dtype) # (N, 2)

        global_features_processed = torch.cat([scaled_log_et_miss, scaled_sincos_phi_miss], dim=1) # (N, 3)

        # Reshape and process object features
        objects_flat = X[:, 2:] 
        objects_raw = objects_flat.reshape(N, self.num_object_slots, self.features_per_object_raw) # (N, 18, 5)

        obj_pt = objects_raw[:, :, 2]
        object_presence_mask = (obj_pt > self.eps).bool() # (N, 18), True for present objects

        # Object types (One-Hot Encoding)
        obj_types_raw = objects_raw[:, :, 0] # (N, 18)
        # Reshape for OHE: (N * 18, 1)
        obj_types_flat_np = obj_types_raw.reshape(-1, 1).cpu().numpy()
        if self.fitted_obj_types.numel() > 0 :
            obj_types_one_hot_flat_np = self.obj_type_encoder.transform(obj_types_flat_np)
            obj_types_processed = torch.from_numpy(obj_types_one_hot_flat_np).to(current_device, dtype=dtype).reshape(N, self.num_object_slots, -1) # (N, 18, num_type_features)
        else: # No types were learned during fit
            obj_types_processed = torch.empty(N, self.num_object_slots, 0, device=current_device, dtype=dtype) # (N, 18, 0)


        # Object continuous features (logE, logpT, eta)
        obj_E_raw = objects_raw[:, :, 1]
        obj_eta_raw = objects_raw[:, :, 3]

        log_E = torch.log1p(obj_E_raw.clamp(min=0.0))
        log_pT = torch.log1p(obj_pt.clamp(min=0.0)) # obj_pt already extracted

        obj_continuous_engineered = torch.stack([log_E, log_pT, obj_eta_raw], dim=-1) # (N, 18, 3)
        obj_continuous_engineered_flat = obj_continuous_engineered.reshape(-1, 3)

        # Handle empty scaler case
        if self.obj_continuous_scalar.n_features_in_ > 0:
            scaled_obj_continuous_flat = torch.from_numpy(self.obj_continuous_scalar.transform(obj_continuous_engineered_flat.cpu().numpy())).to(current_device, dtype=dtype)
        else: # scaler wasn't fit on any features. Output zeros or identity based on feature count.
            scaled_obj_continuous_flat = torch.zeros_like(obj_continuous_engineered_flat)

        obj_continuous_processed = scaled_obj_continuous_flat.reshape(N, self.num_object_slots, 3) # (N, 18, 3)

        # Object phi features (sin/cos)
        obj_phi_raw = objects_raw[:, :, 4]
        sincos_obj_phi = torch.stack([torch.cos(obj_phi_raw), torch.sin(obj_phi_raw)], dim=-1) # (N, 18, 2)
        sincos_obj_phi_flat = sincos_obj_phi.reshape(-1, 2)

        if self.obj_phi_scalar.n_features_in_ > 0:
            scaled_sincos_obj_phi_flat = torch.from_numpy(self.obj_phi_scalar.transform(sincos_obj_phi_flat.cpu().numpy())).to(current_device, dtype=dtype)
        else:
            scaled_sincos_obj_phi_flat = torch.zeros_like(sincos_obj_phi_flat)

        obj_phi_processed = scaled_sincos_obj_phi_flat.reshape(N, self.num_object_slots, 2) # (N, 18, 2)

        # Concatenate all processed object features
        # (N, 18, num_type_features + 3 + 2)
        all_obj_features_processed = torch.cat([obj_types_processed, obj_continuous_processed, obj_phi_processed], dim=-1)

        # Apply presence mask (zero out features of non-present objects)
        # Mask needs to be unsqueezed to (N, 18, 1) for broadcasting
        all_obj_features_masked = all_obj_features_processed * object_presence_mask.unsqueeze(-1).to(dtype)

        # Combine global features with object features for sequence model
        # Expand global features to match sequence length: (N, 1, 3) -> (N, 18, 3)
        global_features_expanded = global_features_processed.unsqueeze(1).repeat(1, self.num_object_slots, 1)

        # Final sequence features: (N, 18, 3_global + num_type_features + 3_continuous + 2_phi)
        X_seq = torch.cat([global_features_expanded, all_obj_features_masked], dim=-1)

        return X_seq, object_presence_mask # (N, L, F_combined), (N, L)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20): # max_len set to num_object_slots
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x): # x: (batch_size, seq_len, d_model)
        # Ensure pe is sliced correctly if x.size(1) (seq_len) < max_len
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 2. ---------- MODEL DEFINITION ----------
class ParticleTransformer(nn.Module):
    def __init__(self, input_features_per_object, d_model=128, nhead=4, num_encoder_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_features_per_object, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=18) # Max objects = 18

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        self.d_model = d_model

    def forward(self, data):
        # Data is a tuple (X_seq, mask) from preprocessor
        x_seq, object_presence_mask = data # x_seq: (N, L, F), object_presence_mask: (N, L)

        # Project input features to d_model
        x = self.input_projection(x_seq) # (N, L, d_model)
        x = self.pos_encoder(x) # Add positional encoding

        # Transformer encoder expects padding_mask where True indicates a padded element
        # Our object_presence_mask is True for VALID elements. So invert.
        padding_mask = ~object_presence_mask # (N, L)

        transformer_output = self.transformer_encoder(x, src_key_padding_mask=padding_mask) # (N, L, d_model)

        # Aggregate features. Mean pooling over present objects.
        # Mask out padded elements before summing
        masked_transformer_output = transformer_output * object_presence_mask.unsqueeze(-1).to(x.dtype)

        summed_output = masked_transformer_output.sum(dim=1) # (N, d_model)

        # Count number of present objects for averaging, clamp to 1 to avoid div by zero if no objects
        num_present_objects = object_presence_mask.sum(dim=1).unsqueeze(-1).clamp(min=1).to(x.dtype) # (N, 1)

        pooled_output = summed_output / num_present_objects # (N, d_model)

        logits = self.output_mlp(pooled_output) # (N, 1)
        return logits

def make_model(input_shape, *, use_mask=False):
    # PARAMETERS
    # input_shape : tuple[int, ...]  – (seq_len, features_per_object_slot) if use_mask=True
    # use_mask     : bool             – True if forward will receive a mask

    # RETURNS
    # model : torch.nn.Module : Untrained binary-classifier network.

    if not use_mask:
        raise ValueError("This model implementation requires use_mask=True, as it processes sequence data with a mask.")

    # input_shape is (seq_len, features_per_object_slot)
    # Example: seq_len=18, features_per_object_slot = 3 (global) + num_ohe_types + 3 (E,pT,eta) + 2 (phi)
    features_per_object_slot = input_shape[1] 

    # Hyperparameters for the Transformer model
    d_model = 64 # Dimensionality of the model
    nhead = 4    # Number of attention heads
    num_encoder_layers = 2 # Number of Transformer encoder layers
    dim_feedforward = d_model * 4 # Hidden dimension in FFN

    model = ParticleTransformer(
        input_features_per_object=features_per_object_slot,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.15 # Adjusted dropout
    )
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 40 # Adjusted number of training epochs
# With early stopping, can be set higher. Data size seems to permit more epochs.

def train_model(model, train_loader, val_loader, epochs, preprocessor_instance): # Added preprocessor_instance
    # PARAMETERS
    # model : torch.nn.Module   
    # train_loader / val_loader yield either
    #   (data,  label)            # single tensor
    #   ((data_seq, mask), label)     # tensor + padding mask
    # epochs: int
    # preprocessor_instance: The fitted preprocessor object, used to get input_shape for model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    # AdamW is generally preferred for Transformers
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) 
    # Scheduler to reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=False)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    best_val_auc = 0.0
    best_model_state_dict = None
    patience_counter = 0
    early_stopping_patience = 7 # Stop after 7 epochs of no improvement in val_auc

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        correct_train_preds = 0
        total_train_samples = 0

        for data, labels in train_loader:
            # Assuming data from loader is X, preprocessor expects (X_seq, mask)
            # The preprocessor is applied outside, train_loader yields preprocessed data
            # So, data is already (X_seq, mask) here for the Transformer

            if isinstance(data, tuple): # (X_seq, mask)
                x_seq, mask = data
                x_seq, mask = x_seq.to(device), mask.to(device)
                model_input = (x_seq, mask)
            else: # Single tensor data
                model_input = data.to(device)

            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(model_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * labels.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).byte()
            correct_train_preds += (preds == labels).sum().item()
            total_train_samples += labels.size(0)

        epoch_train_loss = running_train_loss / total_train_samples
        epoch_train_acc = correct_train_preds / total_train_samples
        train_loss_list.append(epoch_train_loss)
        train_acc_list.append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val_preds = 0
        total_val_samples = 0
        all_val_labels = []
        all_val_probs = []

        with torch.no_grad():
            for data, labels in val_loader:
                if isinstance(data, tuple):
                    x_seq, mask = data
                    x_seq, mask = x_seq.to(device), mask.to(device)
                    model_input = (x_seq, mask)
                else:
                    model_input = data.to(device)

                labels = labels.to(device).float().unsqueeze(1)

                outputs = model(model_input)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * labels.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).byte()
                correct_val_preds += (preds == labels).sum().item()
                total_val_samples += labels.size(0)

                all_val_labels.append(labels.cpu())
                all_val_probs.append(probs.cpu())

        epoch_val_loss = running_val_loss / total_val_samples
        epoch_val_acc = correct_val_preds / total_val_samples
        val_loss_list.append(epoch_val_loss)
        val_acc_list.append(epoch_val_acc)

        all_val_labels_cat = torch.cat(all_val_labels).numpy()
        all_val_probs_cat = torch.cat(all_val_probs).numpy()
        epoch_val_auc = roc_auc_score(all_val_labels_cat, all_val_probs_cat)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val AUC: {epoch_val_auc:.4f}")

        scheduler.step(epoch_val_auc) # ReduceLROnPlateau based on validation AUC

        # Early stopping based on validation AUC
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            best_model_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"New best validation AUC: {best_val_auc:.4f}. Saving model.")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load best model state if early stopping occurred or training finished
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)
        print(f"Loaded best model with Val AUC: {best_val_auc:.4f}")

    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
# The harness will typically do:
# preprocessor = make_preprocessor()
# X_train_processed, train_mask = preprocessor.fit_transform(X_train)
# X_val_processed, val_mask = preprocessor.transform(X_val)
# model = make_model(input_shape=(X_train_processed.shape[1], X_train_processed.shape[2]), use_mask=True)
# train_dataset = TensorDataset(X_train_processed, train_mask, Y_train) # Custom dataset might be needed if preprocessor outputs other things
# val_dataset = TensorDataset(X_val_processed, val_mask, Y_val)
# ... and then call train_model.
# The provided template doesn't show Dataset creation, so I make train_loader/val_loader inside train_model or assume they're passed.
# Correcting: train_loader / val_loader are passed to train_model. My train_model signature matches this.
# The train_loader must yield ((x_seq, mask), label) tuples.

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

