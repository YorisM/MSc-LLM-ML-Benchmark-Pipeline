
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops

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
        X2 = pre.transform(X) if pre is not None else X
        if not torch.is_tensor(X2):
            X2 = torch.as_tensor(X2)
        self.X = X2.float()
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        self.y = y.long()
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

# -------------------------- START OF LLM BLOCK ------------------------------
# <start code template>
# ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from typing import Optional, Tuple

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# We'll use the default FourTopsDataset

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.kinematic_mean = None
        self.kinematic_std = None
        self.obj_id_min = None
        self.obj_id_max = None
        self.num_objects = 18
        self.obj_features = 5

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True if torch.cuda.is_available() else False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Global features: ETmiss and phi_ETmiss
        global_features = X_np[:, :2]
        self.global_mean = global_features.mean(axis=0, keepdims=True)
        self.global_std = global_features.std(axis=0, keepdims=True) + 1e-8

        # Object features: extract kinematic features (E, pT, eta, phi) for all objects
        kinematic_features = []
        obj_ids = []

        for i in range(self.num_objects):
            start_idx = 2 + i * self.obj_features
            obj_slice = X_np[:, start_idx:start_idx + self.obj_features]
            obj_ids.append(obj_slice[:, 0])  # First column is object ID
            kinematic_features.append(obj_slice[:, 1:])  # Last 4 columns are kinematic

        obj_ids = np.concatenate(obj_ids, axis=0)
        kinematic_features = np.concatenate(kinematic_features, axis=0)

        # Only compute stats on non-zero objects (obj_id != 0)
        mask = obj_ids != 0
        if mask.any():
            kinematic_features = kinematic_features[mask]
            self.kinematic_mean = kinematic_features.mean(axis=0, keepdims=True)
            self.kinematic_std = kinematic_features.std(axis=0, keepdims=True) + 1e-8
        else:
            self.kinematic_mean = np.zeros((1, 4))
            self.kinematic_std = np.ones((1, 4))

        # Store object ID range for potential one-hot encoding
        self.obj_id_min = int(obj_ids.min())
        self.obj_id_max = int(obj_ids.max())

        return self

    def transform(self, X):
        if torch.is_tensor(X):
            X_np = X.numpy()
            return_tensor = True
        else:
            X_np = X
            return_tensor = False

        batch_size = X_np.shape[0]

        # Normalize global features
        global_norm = (X_np[:, :2] - self.global_mean) / self.global_std

        # Process object features
        obj_features_list = []

        for i in range(self.num_objects):
            start_idx = 2 + i * self.obj_features
            obj_slice = X_np[:, start_idx:start_idx + self.obj_features]

            # Separate object ID and kinematics
            obj_ids = obj_slice[:, 0:1]  # Keep as is for now
            kinematics = obj_slice[:, 1:]

            # Normalize kinematics
            kinematics_norm = (kinematics - self.kinematic_mean) / self.kinematic_std

            # Apply mask based on object ID
            mask = (obj_ids != 0).astype(np.float32)
            kinematics_norm = kinematics_norm * mask

            # Concatenate back
            obj_processed = np.concatenate([obj_ids, kinematics_norm], axis=1)
            obj_features_list.append(obj_processed)

        # Stack all objects
        obj_features = np.concatenate(obj_features_list, axis=1)

        # Combine with normalized global features
        X_transformed = np.concatenate([global_norm, obj_features], axis=1)

        if return_tensor:
            return torch.from_numpy(X_transformed).float()
        return X_transformed

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class ParticleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        # x shape: [batch, 92]
        batch_size = x.shape[0]

        # Reshape to [batch, 19, hidden_dim] where 19 = 1 global + 18 objects
        global_features = x[:, :2]  # [batch, 2]
        global_proj = self.input_proj(global_features).unsqueeze(1)  # [batch, 1, hidden]

        # Process objects
        obj_features_list = []
        for i in range(18):
            start_idx = 2 + i * 5
            obj_feat = x[:, start_idx:start_idx + 5]  # [batch, 5]
            obj_proj = self.input_proj(obj_feat).unsqueeze(1)  # [batch, 1, hidden]
            obj_features_list.append(obj_proj)

        obj_features = torch.cat(obj_features_list, dim=1)  # [batch, 18, hidden]

        # Create mask for zero-padded objects (obj_id == 0)
        obj_ids = x[:, 2::5]  # Extract obj_id for each object
        padding_mask = (obj_ids == 0)  # [batch, 18], True for padded

        # Combine with CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, hidden]
        combined = torch.cat([cls_tokens, global_proj, obj_features], dim=1)  # [batch, 20, hidden]

        # Expand padding mask for CLS and global tokens (always not padded)
        cls_global_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_global_mask, padding_mask], dim=1)  # [batch, 20]

        # Transformer
        transformer_out = self.transformer(combined, src_key_padding_mask=full_mask)  # [batch, 20, hidden]

        # Extract CLS token output and global token output
        cls_output = transformer_out[:, 0, :]  # [batch, hidden]
        global_output = transformer_out[:, 1, :]  # [batch, hidden]

        # Pool object features (excluding CLS and global)
        obj_output = transformer_out[:, 2:, :]  # [batch, 18, hidden]
        obj_pooled = torch.mean(obj_output, dim=1)  # [batch, hidden]

        # Combine representations
        combined_features = torch.cat([cls_output, global_output, obj_pooled], dim=1)  # [batch, hidden*3]

        # Final output
        output = self.output(combined_features).squeeze(-1)  # [batch]
        return output

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[1]  # Should be 92

        # Main model
        self.model = ParticleTransformer(
            input_dim=5,  # Each object has 5 features
            hidden_dim=256,
            num_heads=8,
            num_layers=6,
            dropout=0.1
        )

        # Auxiliary fully connected network for global features
        self.global_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128)
        )

        # Feature engineering: compute invariant masses and deltaR
        self.feature_engineering = True

    def compute_pairwise_features(self, x):
        batch_size = x.shape[0]

        # Extract object features
        obj_features = []
        for i in range(18):
            start_idx = 2 + i * 5
            obj_feat = x[:, start_idx:start_idx + 5]  # [batch, 5]
            obj_features.append(obj_feat.unsqueeze(1))  # [batch, 1, 5]

        obj_tensor = torch.cat(obj_features, dim=1)  # [batch, 18, 5]

        # Extract object IDs and kinematics
        obj_ids = obj_tensor[:, :, 0]  # [batch, 18]
        E = obj_tensor[:, :, 1]  # [batch, 18]
        pT = obj_tensor[:, :, 2]  # [batch, 18]
        eta = obj_tensor[:, :, 3]  # [batch, 18]
        phi = obj_tensor[:, :, 4]  # [batch, 18]

        # Create mask for real objects
        mask = (obj_ids != 0).float()  # [batch, 18]

        # Compute invariant masses for all pairs
        batch_masses = []
        for b in range(batch_size):
            # Get indices of real objects
            real_idx = torch.where(mask[b] > 0.5)[0]
            if len(real_idx) < 2:
                batch_masses.append(torch.zeros(1, device=x.device))
                continue

            # Extract kinematics for real objects
            E_real = E[b, real_idx]
            pT_real = pT[b, real_idx]
            eta_real = eta[b, real_idx]
            phi_real = phi[b, real_idx]

            # Compute px, py, pz from pT, eta, phi
            px = pT_real * torch.cos(phi_real)
            py = pT_real * torch.sin(phi_real)
            pz = pT_real * torch.sinh(eta_real)

            # Compute invariant mass for all pairs
            n_real = len(real_idx)
            masses = []
            for i in range(n_real):
                for j in range(i+1, n_real):
                    # Sum four-momenta
                    E_sum = E_real[i] + E_real[j]
                    px_sum = px[i] + px[j]
                    py_sum = py[i] + py[j]
                    pz_sum = pz[i] + pz[j]

                    # Compute invariant mass
                    m2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
                    m = torch.sqrt(torch.clamp(m2, min=1e-8))
                    masses.append(m)

            if masses:
                batch_masses.append(torch.stack(masses).mean())
            else:
                batch_masses.append(torch.zeros(1, device=x.device))

        return torch.stack(batch_masses)  # [batch]

    def forward(self, batch_x):
        # batch_x shape: [batch, 92]

        # Extract global features
        global_features = batch_x[:, :2]  # [batch, 2]

        # Process through main model
        main_output = self.model(batch_x)  # [batch]

        # Process global features
        global_encoded = self.global_net(global_features)  # [batch, 128]
        global_output = torch.mean(global_encoded, dim=1)  # [batch]

        # Compute pairwise features if enabled
        if self.feature_engineering and self.training:  # Only in training for efficiency
            pairwise_features = self.compute_pairwise_features(batch_x)  # [batch]
            combined = main_output + 0.1 * global_output + 0.05 * pairwise_features
        else:
            combined = main_output + 0.1 * global_output

        return combined

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Loss function with label smoothing for better calibration
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-5
    )

    # Early stopping
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0
    best_model_state = None

    # Training history
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).float()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # L2 regularization
            l2_lambda = 1e-5
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Compute accuracy
            pred = torch.sigmoid(output) > 0.5
            correct = (pred.float() == target).sum().item()

            train_loss += loss.item() * data.size(0)
            train_correct += correct
            train_total += data.size(0)

            # Print progress
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device).float()
                output = model(data)
                loss = criterion(output, target)

                pred = torch.sigmoid(output) > 0.5
                correct = (pred.float() == target).sum().item()

                val_loss += loss.item() * data.size(0)
                val_correct += correct
                val_total += data.size(0)

        # Compute metrics
        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # Update learning rate
        scheduler.step()

        # Store history
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f'New best model saved with val loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

# <end code template>
# ---------------------------  END OF LLM-CODE BLOCK  ---------------------------

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

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane_fourtops(spec, first_batch)
    view = make_view_by_lane_fourtops(mode, first_batch, device)

    # Build model
    model = make_model(view.batch_x).to(device)

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
                mode = detect_and_assert_lane_fourtops(spec, first_batch)
                view = make_view_by_lane_fourtops(mode, first_batch, device)
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

