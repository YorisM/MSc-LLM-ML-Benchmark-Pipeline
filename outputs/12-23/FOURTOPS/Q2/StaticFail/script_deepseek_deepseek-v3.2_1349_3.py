
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
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

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------- IMPORTS ----------
# Additional imports for our model
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import OrderedDict

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_scaler = StandardScaler()
        self.kinematic_scaler = StandardScaler()
        self.obj_id_min = 0
        self.obj_id_max = 0

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True if torch.cuda.is_available() else False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        X_np = X.numpy()

        # Global features (first 2)
        global_features = X_np[:, :2]
        self.global_scaler.fit(global_features)

        # Object features: reshape to (N*18, 5) for scaler fitting
        obj_features = X_np[:, 2:].reshape(-1, 5)

        # Identify padding (obj_id == 0)
        non_padding_mask = obj_features[:, 0] != 0

        # Fit scaler only on non-padding kinematic features (E, pT, eta, phi)
        kinematic_features = obj_features[non_padding_mask, 1:]
        if len(kinematic_features) > 0:
            self.kinematic_scaler.fit(kinematic_features)

        # Store obj_id range for embedding
        obj_ids = obj_features[non_padding_mask, 0]
        self.obj_id_min = int(obj_ids.min())
        self.obj_id_max = int(obj_ids.max())

        return self

    def transform(self, X):
        X_np = X.numpy()
        batch_size = X_np.shape[0]

        # Transform global features
        global_features = self.global_scaler.transform(X_np[:, :2])  # (batch_size, 2)

        # Reshape object features
        obj_features = X_np[:, 2:].reshape(-1, 5)  # (batch_size*18, 5)

        # Identify padding
        obj_ids = obj_features[:, 0]
        non_padding_mask = obj_ids != 0

        # Transform kinematic features
        transformed_kinematic = np.zeros_like(obj_features[:, 1:])
        if len(self.kinematic_scaler.scale_) > 0:
            transformed_kinematic[non_padding_mask] = self.kinematic_scaler.transform(
                obj_features[non_padding_mask, 1:]
            )

        # Create mask (1 for real objects, 0 for padding)
        mask = non_padding_mask.astype(np.float32).reshape(batch_size, 18)

        # Normalize object IDs to start from 0 for embedding
        obj_ids_normalized = obj_ids.copy()
        obj_ids_normalized[non_padding_mask] = obj_ids[non_padding_mask] - self.obj_id_min

        # Reshape everything back
        obj_ids_reshaped = obj_ids_normalized.reshape(batch_size, 18, 1)
        kinematic_reshaped = transformed_kinematic.reshape(batch_size, 18, 4)

        # Concatenate object features: [obj_id, E, pT, eta, phi]
        obj_features_combined = np.concatenate(
            [obj_ids_reshaped, kinematic_reshaped], axis=2
        )

        # Flatten object features
        obj_features_flat = obj_features_combined.reshape(batch_size, -1)

        # Combine with global features
        X_transformed = np.concatenate([global_features, obj_features_flat, mask], axis=1)

        return torch.from_numpy(X_transformed.astype(np.float32))

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class ParticleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pairwise features computation
        self.pairwise_proj = nn.Linear(hidden_dim, hidden_dim // 2)

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def compute_pairwise_features(self, x, mask):
        batch_size, num_objects, hidden_dim = x.shape

        # Expand for pairwise computation
        x1 = x.unsqueeze(2).expand(-1, -1, num_objects, -1)  # (B, N, N, H)
        x2 = x.unsqueeze(1).expand(-1, num_objects, -1, -1)  # (B, N, N, H)

        # Compute delta eta, delta phi, delta R
        # Extract kinematic features from input (assuming last 4 are E, pT, eta, phi)
        # Note: This is simplified - in practice you'd need to extract these from input
        eta = x[:, :, -3]  # assuming eta is at position -3
        phi = x[:, :, -2]  # assuming phi is at position -2

        delta_eta = eta.unsqueeze(2) - eta.unsqueeze(1)  # (B, N, N)
        delta_phi = torch.atan2(torch.sin(phi.unsqueeze(2) - phi.unsqueeze(1)),
                              torch.cos(phi.unsqueeze(2) - phi.unsqueeze(1)))
        delta_r = torch.sqrt(delta_eta**2 + delta_phi**2 + 1e-8)

        # Combine with learned features
        pairwise_features = self.pairwise_proj(x1 * x2)  # (B, N, N, H/2)

        # Add deltaR as extra channel
        pairwise_features = torch.cat([
            pairwise_features,
            delta_r.unsqueeze(-1),
            delta_eta.unsqueeze(-1),
            delta_phi.unsqueeze(-1)
        ], dim=-1)

        # Mask invalid pairs
        mask2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
        pairwise_features = pairwise_features * mask2d.unsqueeze(-1)

        return pairwise_features

    def forward(self, x):
        # x shape: (batch_size, 2 + 18*5 + 18) = (batch_size, 110)
        batch_size = x.shape[0]

        # Split inputs
        global_features = x[:, :2]  # (B, 2)
        obj_features = x[:, 2:-18].reshape(batch_size, 18, 5)  # (B, 18, 5)
        mask = x[:, -18:].unsqueeze(-1)  # (B, 18, 1)

        # Project object features
        obj_embedded = self.input_proj(obj_features)  # (B, 18, H)

        # Apply mask (zero out padding)
        obj_embedded = obj_embedded * mask

        # Transformer with mask
        transformer_mask = mask.squeeze(-1) == 0  # (B, 18)
        obj_transformed = self.transformer(obj_embedded, src_key_padding_mask=transformer_mask)

        # Attention pooling
        attn_weights = self.attention_pool(obj_transformed)  # (B, 18, 1)
        attn_weights = attn_weights.masked_fill(transformer_mask.unsqueeze(-1), -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        aggregated = torch.sum(obj_transformed * attn_weights, dim=1)  # (B, H)

        # Compute pairwise features and pool
        pairwise = self.compute_pairwise_features(obj_transformed, mask.squeeze(-1))
        pairwise_pooled = torch.mean(pairwise, dim=(1, 2))  # (B, H/2 + 3)

        # Combine features
        combined = torch.cat([
            aggregated,
            pairwise_pooled,
            global_features
        ], dim=1)  # (B, H + H/2 + 3 + 2)

        # Classification
        output = self.classifier(combined)
        return output

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[0]

        # Calculate input dimension for transformer
        # 2 global + 18*5 object + 18 mask = 110
        if input_dim == 110:
            self.model = ParticleTransformer(input_dim=5, hidden_dim=256, num_layers=6, num_heads=8, dropout=0.2)
        else:
            # Fallback MLP if dimensions don't match
            self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            )

    def forward(self, batch_x):
        return self.model(batch_x)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    # Tracking metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_train_loss += loss.item() * batch_x.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predictions == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_loss = epoch_train_loss / total_train
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float().view(-1, 1)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                epoch_val_loss += loss.item() * batch_x.size(0)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (predictions == batch_y).sum().item()
                total_val += batch_y.size(0)

        val_loss = epoch_val_loss / total_val
        val_acc = correct_val / total_val

        # Update scheduler
        scheduler.step(val_loss)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return model, train_losses, val_losses, train_accs, val_accs

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

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


