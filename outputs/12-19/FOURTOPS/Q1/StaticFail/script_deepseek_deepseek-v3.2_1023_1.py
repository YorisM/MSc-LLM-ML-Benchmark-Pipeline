
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
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
import math

# ---------- IMPORTS ----------
# Additional imports beyond those in prefix
import torch.optim as optim
from typing import Tuple, List

#  -------- CUSTOM DATASET --------
class FourTopsDatasetEnhanced(torch.utils.data.Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
        self.train = train

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    # DATA SPECIFICS
    # Total flat length per event: 92
    # Index 0: E_T_miss magnitude
    # Index 1: phi_Et_miss
    # Indices 2-6: object 1 (obj_id, E, pT, eta, phi)
    # ...
    # Indices 87-91: object 18
    # Global features = 2, Per-object = 5, Max objects = 18

    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.obj_feat_mean = None
        self.obj_feat_std = None
        self.obj_id_mapping = None
        self.obj_id_counts = None

    def make_loader_cfg(self):
        return {
            "dataset_builder": "llm_script:FourTopsDatasetEnhanced",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 256,  # Reduced for memory efficiency
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True if torch.cuda.is_available() else False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512},
        }

    def _extract_obj_features(self, X):
        """Extract object features from flattened array"""
        batch_size = X.shape[0]
        # Reshape to [batch, 18, 5]
        objects = X[:, 2:].reshape(batch_size, 18, 5)
        return objects

    def _create_obj_id_mapping(self, objects):
        """Create mapping for object IDs"""
        # Flatten object IDs across all training events
        obj_ids = objects[:, :, 0].flatten()
        obj_ids = obj_ids[obj_ids != 0]  # Remove zero-padding

        # Count frequencies
        counts = Counter(obj_ids.numpy() if torch.is_tensor(obj_ids) else obj_ids)

        # Create mapping: frequent IDs get unique embedding, others share a common "rare" embedding
        frequent_ids = [id for id, cnt in counts.items() if cnt >= 100]  # At least 100 occurrences
        self.obj_id_mapping = {id: i+1 for i, id in enumerate(frequent_ids)}  # 0 reserved for padding
        self.obj_id_mapping[0] = 0  # Padding
        self.obj_id_counts = counts

        # Rare objects get index len(frequent_ids) + 1
        self.rare_obj_index = len(frequent_ids) + 1

    def fit(self, X, y=None):
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Global features normalization
        global_feats = X_np[:, :2]
        self.global_mean = global_feats.mean(axis=0)
        self.global_std = global_feats.std(axis=0) + 1e-8

        # Object features normalization
        objects = self._extract_obj_features(torch.from_numpy(X_np))

        # Create object ID mapping
        self._create_obj_id_mapping(objects)

        # Normalize kinematic features (E, pT, eta, phi) - skip object ID
        # Consider only non-zero-padded objects
        mask = objects[:, :, 0] != 0  # Shape: [batch, 18]
        kin_feats = objects[:, :, 1:][mask.unsqueeze(-1).expand(-1, -1, 4)].reshape(-1, 4)  # Shape: [n_objects*4]

        self.obj_feat_mean = kin_feats.mean(dim=0).numpy()
        self.obj_feat_std = kin_feats.std(dim=0).numpy() + 1e-8

        return self

    def _map_obj_id(self, obj_id):
        """Map object ID to embedding index"""
        if obj_id in self.obj_id_mapping:
            return self.obj_id_mapping[obj_id]
        else:
            return self.rare_obj_index

    def transform(self, X):
        X_tensor = X if torch.is_tensor(X) else torch.from_numpy(X)
        batch_size = X_tensor.shape[0]

        # Normalize global features
        global_norm = (X_tensor[:, :2] - torch.from_numpy(self.global_mean).to(X_tensor.device)) / \
                     torch.from_numpy(self.global_std).to(X_tensor.device)

        # Process objects
        objects = self._extract_obj_features(X_tensor)  # [batch, 18, 5]

        # Map object IDs
        obj_ids = objects[:, :, 0].long()
        mapped_ids = torch.zeros_like(obj_ids)
        for i in range(batch_size):
            for j in range(18):
                mapped_ids[i, j] = self._map_obj_id(obj_ids[i, j].item())

        # Normalize kinematic features
        kin_feats = objects[:, :, 1:]  # [batch, 18, 4]
        kin_norm = (kin_feats - torch.from_numpy(self.obj_feat_mean).to(X_tensor.device)) / \
                  torch.from_numpy(self.obj_feat_std).to(X_tensor.device)

        # Create mask for zero-padded objects
        mask = (obj_ids != 0).float()  # [batch, 18]

        # Replace original object ID with mapped ID
        processed_objects = torch.cat([
            mapped_ids.unsqueeze(-1).float(),  # [batch, 18, 1]
            kin_norm  # [batch, 18, 4]
        ], dim=-1)  # [batch, 18, 5]

        # Flatten
        processed_objects_flat = processed_objects.reshape(batch_size, -1)  # [batch, 90]
        mask_flat = mask.reshape(batch_size, -1)  # [batch, 18]

        # Concatenate everything: [global_norm, processed_objects_flat, mask_flat]
        # Output shape: [batch, 2 + 90 + 18 = 110]
        result = torch.cat([global_norm, processed_objects_flat, mask_flat], dim=1)

        return result.float()

# ---------- MODEL DEFINITION ----------
class AttentionPooling(nn.Module):
    """Attention-based pooling over objects"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x, mask=None):
        # x: [batch, num_objects, feat_dim]
        # mask: [batch, num_objects], 1 for real object, 0 for padded
        batch_size, num_objects, feat_dim = x.shape

        Q = self.query(x)  # [batch, num_objects, hidden_dim]
        K = self.key(x)    # [batch, num_objects, hidden_dim]
        V = self.value(x)  # [batch, num_objects, hidden_dim]

        # Attention scores
        scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # [batch, num_objects, num_objects]

        if mask is not None:
            # Apply mask to prevent attention to padded objects
            mask_expanded = mask.unsqueeze(1)  # [batch, 1, num_objects]
            scores = scores.masked_fill(mask_expanded == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # [batch, num_objects, num_objects]

        # Apply attention
        output = torch.matmul(attn_weights, V)  # [batch, num_objects, hidden_dim]

        # Pool over objects (mean pooling)
        if mask is not None:
            output = output * mask.unsqueeze(-1)  # Zero out padded
            pooled = output.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # [batch, hidden_dim]
        else:
            pooled = output.mean(dim=1)  # [batch, hidden_dim]

        return pooled

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Input shape after preprocessing: [batch, 110]
        # First 2: global features, next 90: 18 objects * 5 features, last 18: mask

        # Object processing branch
        self.obj_embedding = nn.Embedding(
            num_embeddings=50,  # Max 50 unique object types
            embedding_dim=16,
            padding_idx=0
        )

        # After embedding: [batch, 18, 16] for IDs + [batch, 18, 4] for kinematics = [batch, 18, 20]
        obj_feat_dim = 20

        # Attention pooling for objects
        self.attention_pool = AttentionPooling(obj_feat_dim, 64)

        # Global features processing
        self.global_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Combined processing
        combined_dim = 64 + 64  # global features + object features

        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
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
        # batch_x shape: [batch, 110]
        batch_size = batch_x.shape[0]

        # Split input
        global_feats = batch_x[:, :2]  # [batch, 2]
        obj_data = batch_x[:, 2:92]    # [batch, 90] = 18 * 5
        mask = batch_x[:, 92:]         # [batch, 18]

        # Reshape object data
        objects = obj_data.reshape(batch_size, 18, 5)  # [batch, 18, 5]

        # Separate object ID and kinematic features
        obj_ids = objects[:, :, 0].long()  # [batch, 18]
        obj_kin = objects[:, :, 1:]        # [batch, 18, 4]

        # Embed object IDs
        obj_emb = self.obj_embedding(obj_ids)  # [batch, 18, 16]

        # Combine embedded IDs with kinematic features
        obj_feats = torch.cat([obj_emb, obj_kin], dim=-1)  # [batch, 18, 20]

        # Attention pooling over objects
        obj_pooled = self.attention_pool(obj_feats, mask)  # [batch, 64]

        # Process global features
        global_processed = self.global_net(global_feats)  # [batch, 64]

        # Combine features
        combined = torch.cat([global_processed, obj_pooled], dim=1)  # [batch, 128]

        # Final classification
        output = self.combined_net(combined)  # [batch, 1]

        return output

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def compute_auc(model, data_loader, device):
    """Compute AUC score"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            probs = torch.sigmoid(outputs).squeeze()

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    model.train()
    return roc_auc_score(all_labels, all_preds)

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=False
    )

    # Loss function with label smoothing
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))

    # Early stopping
    best_auc = 0.0
    patience_counter = 0
    patience = 10

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

            # Compute accuracy
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float().unsqueeze(1)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)

        # Compute metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val

        # Compute AUC
        val_auc = compute_auc(model, val_loader, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step(val_auc)

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.

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


