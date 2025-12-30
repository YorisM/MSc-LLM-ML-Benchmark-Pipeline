
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
                        
DATASET = {
    "X_train": "./challenges/FOURTOPS/data/train/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/train/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/train/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/train/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv(DATASET["X_train"], dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv(DATASET["Y_train"], dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv(DATASET["X_val"], dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv(DATASET['Y_val'], dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class FourTopsDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import math

# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    # TIPS
    #   - When modifying data features or feature engineering: annotate tensor size as comments after 
    #   - each tensor operation to reduce dimension mismatches.

    # DATA SPECIFICS
    #    Total flat length per event (X_train & X_val): 92
    #    Index  0 :  missing-ET magnitude  (E_T_miss)
    #    Index  1 :  missing-ET azimuth    (phi_Et_miss)
    #    Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    #    Indices  7-11 : object 2  ->  obj_2, E_2 , p_T_2 , eta_2 , phi_2
    #    ...
    #    Indices 87-91 : object 18 ->  obj_18, E_18 , p_T_18 , eta_18 , phi_18
    #    Global features       = 2
    #    Per-object slice size = 5
    #    Max objects encoded   = 18

    def __init__(self):
        # Define and initialize any stateful components here
        self.global_mean = None
        self.global_std = None
        self.obj_mean = None
        self.obj_std = None
        self.obj_type_stats = None
        self.pad_value = 0.0

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,

            # NO custom collate callables allowed. Choose one: 
            "collate": None, # (or "ragged_xy" or "identity" - If loader_class is torch_geometric.loader:DataLoader, set "collate": None.)

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False, "num_workers": 2},
        }

    def fit(self, X, y=None):
        # Extract statistics for transform
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Global features: missing ET magnitude and angle
        global_features = X_np[:, :2]
        self.global_mean = np.mean(global_features, axis=0, keepdims=True)
        self.global_std = np.std(global_features, axis=0, keepdims=True)
        self.global_std = np.where(self.global_std == 0, 1.0, self.global_std)

        # Object features: reshape to (n_events * 18, 5)
        n_events = X_np.shape[0]
        obj_features = X_np[:, 2:].reshape(-1, 5)

        # Filter out padding (where object type is 0)
        mask = obj_features[:, 0] != self.pad_value
        valid_obj_features = obj_features[mask]

        # For object type (first column), collect statistics for one-hot encoding
        obj_types = valid_obj_features[:, 0]
        unique_types = np.unique(obj_types)
        self.obj_type_stats = {
            'unique_types': unique_types,
            'n_types': len(unique_types),
            'type_to_idx': {t: i for i, t in enumerate(unique_types)}
        }

        # For kinematic features (E, pT, eta, phi), compute robust statistics
        kinematic_features = valid_obj_features[:, 1:]
        self.obj_mean = np.mean(kinematic_features, axis=0, keepdims=True)
        self.obj_std = np.std(kinematic_features, axis=0, keepdims=True)
        self.obj_std = np.where(self.obj_std == 0, 1.0, self.obj_std)

        # Compute robust min/max for clipping
        self.kinematic_min = np.percentile(kinematic_features, 1, axis=0)
        self.kinematic_max = np.percentile(kinematic_features, 99, axis=0)

        return self

    def transform(self, X):
        # Apply pre-processing logic
        if torch.is_tensor(X):
            X_np = X.numpy()
            return_tensor = True
        else:
            X_np = X
            return_tensor = False

        n_events = X_np.shape[0]

        # Initialize transformed features
        # We'll keep global features and create enhanced object features
        # Original: 2 global + 18*5 = 92 features
        # New: 2 global (normalized) + 18*(1 + 4 + 4) = 2 + 18*9 = 164 features
        # Where per object: one-hot type (1), normalized kinematics (4), derived features (4)
        n_obj_features = 1 + 4 + 4  # type, kinematics, derived
        transformed = np.zeros((n_events, 2 + 18 * n_obj_features), dtype=np.float32)

        # Process global features
        global_features = X_np[:, :2]
        global_norm = (global_features - self.global_mean) / (self.global_std + 1e-8)
        transformed[:, :2] = global_norm  # [batch, 2]

        # Process object features
        for event_idx in range(n_events):
            obj_start = 2
            for obj_idx in range(18):
                obj_slice = slice(obj_start + obj_idx*5, obj_start + obj_idx*5 + 5)
                obj_features = X_np[event_idx, obj_slice]  # [5]

                # Check if object is padding
                if obj_features[0] == self.pad_value:
                    # Pad with zeros
                    feature_start = 2 + obj_idx * n_obj_features
                    transformed[event_idx, feature_start:feature_start + n_obj_features] = 0
                    continue

                # Object type: one-hot encoding
                obj_type = obj_features[0]
                type_idx = self.obj_type_stats['type_to_idx'].get(obj_type, 0)
                type_onehot = np.zeros(1, dtype=np.float32)
                type_onehot[0] = type_idx  # We'll use embedding in model instead of one-hot

                # Kinematic features: normalize
                kinematics = obj_features[1:]  # [4]
                # Clip extreme values
                kinematics = np.clip(kinematics, self.kinematic_min, self.kinematic_max)
                kinematics_norm = (kinematics - self.obj_mean) / (self.obj_std + 1e-8)  # [4]

                # Derived features
                pT, eta, phi = kinematics[1], kinematics[2], kinematics[3]
                derived = np.zeros(4, dtype=np.float32)
                derived[0] = pT * np.cos(phi)  # px
                derived[1] = pT * np.sin(phi)  # py
                derived[2] = pT * np.sinh(eta)  # pz (approximation)
                derived[3] = np.sqrt(pT**2 + derived[2]**2)  # p magnitude

                # Combine features
                feature_start = 2 + obj_idx * n_obj_features
                transformed[event_idx, feature_start] = type_onehot[0]  # type index for embedding
                transformed[event_idx, feature_start + 1:feature_start + 5] = kinematics_norm
                transformed[event_idx, feature_start + 5:feature_start + 9] = derived

        if return_tensor:
            return torch.from_numpy(transformed)
        return transformed

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Determine input dimensions from sample
        if isinstance(sample_object, torch.Tensor):
            input_dim = sample_object.shape[-1]
        else:
            # Handle other input formats if needed
            input_dim = 164  # From our preprocessing

        # Model hyperparameters
        self.hidden_dim = 512
        self.dropout_rate = 0.3
        self.num_objects = 18
        self.obj_feature_dim = 9  # type + kinematics + derived

        # Object type embedding
        self.type_embedding = nn.Embedding(50, 16)  # Assume up to 50 object types

        # After embedding: 16 (type) + 8 (kinematics+derived) = 24 per object
        obj_in_dim = 16 + 8

        # Object-level processing
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )

        # Attention mechanism for object aggregation
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Dropout(self.dropout_rate)
        )

        # Global feature processing
        self.global_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # Combined processing
        combined_dim = 64 + 32  # object aggregated + global
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_x):
        # batch_x shape: [batch_size, 164]
        batch_size = batch_x.shape[0]

        # Split into global and object features
        global_features = batch_x[:, :2]  # [batch_size, 2]
        obj_features = batch_x[:, 2:].reshape(batch_size, self.num_objects, -1)  # [batch_size, 18, 9]

        # Process objects
        obj_types = obj_features[:, :, 0].long()  # [batch_size, 18]
        obj_kinematics = obj_features[:, :, 1:]   # [batch_size, 18, 8]

        # Embed object types
        type_emb = self.type_embedding(obj_types)  # [batch_size, 18, 16]

        # Concatenate with kinematics
        obj_combined = torch.cat([type_emb, obj_kinematics], dim=-1)  # [batch_size, 18, 24]

        # Reshape for batch processing: [batch_size*18, 24]
        obj_flat = obj_combined.reshape(-1, obj_combined.shape[-1])

        # Encode each object
        obj_encoded = self.obj_encoder(obj_flat)  # [batch_size*18, 64]
        obj_encoded = obj_encoded.reshape(batch_size, self.num_objects, -1)  # [batch_size, 18, 64]

        # Attention weights
        attn_weights = self.attention(obj_encoded)  # [batch_size, 18, 1]
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum of objects
        obj_aggregated = torch.sum(obj_encoded * attn_weights, dim=1)  # [batch_size, 64]

        # Process global features
        global_encoded = self.global_encoder(global_features)  # [batch_size, 32]

        # Combine
        combined = torch.cat([obj_aggregated, global_encoded], dim=1)  # [batch_size, 96]

        # Final classification
        logits = self.classifier(combined)  # [batch_size, 1]

        return logits.squeeze(-1)  # [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 100   # Increased epochs with early stopping

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    #   - Must return: trained_model, train_loss, val_loss, train_acc, val_acc
    #   - Do NOT:
    #       - pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #       - batch = batch.to(device)
    #       - xb, yb = batch
    #       - for xb, yb in loader: ...

    # Canonical batch handling (use this inside every loop):
    # for batch in train_loader:
    #     view = normalise_batch(batch, device=device)
    #     xb, yb = view.batch_x, view.batch_y
    #     out = model(xb)

    # Write code to define training loop, use the code above
    # Implement early stopping if possible

    # Loss function with label smoothing for better calibration
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    # Training history
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    val_auc_history = []

    # Early stopping
    best_val_auc = 0.0
    best_model_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()

            # Forward pass
            logits = model(xb)
            loss = criterion(logits, yb.float())

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            train_loss += loss.item() * xb.shape[0]
            preds = (torch.sigmoid(logits) > 0.5).long()
            train_correct += (preds == yb).sum().item()
            train_total += xb.shape[0]

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                logits = model(xb)
                loss = criterion(logits, yb.float())

                val_loss += loss.item() * xb.shape[0]
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                val_correct += (preds == yb).sum().item()
                val_total += xb.shape[0]

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Calculate AUC
        val_auc = roc_auc_score(all_labels, all_probs)

        # Update learning rate
        scheduler.step()

        # Record history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        val_auc_history.append(val_auc)

        # Early stopping based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if epoch % 5 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')

        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch}')
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

# DO NOT execute the pipeline here – the harness will do that.

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:20]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre     = make_preprocessor().fit(X_train, Y_train)
    
    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec, require_torch_collate=False)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, (X_train, Y_train), pre, train=True)
    val_ds       = build_dataset(spec, (X_val,   Y_val),   pre, train=False)
    train_loader = build_dataloader(spec, train_ds, is_eval=False)
    val_loader   = build_dataloader(spec, val_ds,   is_eval=True)

    # Build model
    first_batch = next(iter(train_loader))
    view        = normalise_batch(first_batch, device=device)
    model       = make_model(view.batch_x).to(device)

    # Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        try:
            with torch.no_grad():
                view = normalise_batch(first_batch, device=device)
                out  = trained_model(view.batch_x)
                scores, kind = assert_binary_output(view, out)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e

    if not dryrun:
        # Persist artefacts
        base = base_from_argv0()
        persist_artefacts(base, SCRIPT_DIR, trained_model, pre, spec)

        # Save plots
        plot_train_val(tr_loss, va_loss, f"{base} Loss", os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        plot_train_val(tr_acc, va_acc, f"{base} Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))
        
        # Write JSON Summary
        summary = {
            "epochs": n_epochs      if n_epochs else None,
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

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

