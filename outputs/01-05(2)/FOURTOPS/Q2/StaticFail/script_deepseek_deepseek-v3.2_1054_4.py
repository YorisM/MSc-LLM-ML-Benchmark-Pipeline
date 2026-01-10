
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
# ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pairwise_feature_scaler = StandardScaler()
        self.feature_indices = None
        self.obj_feature_start = 2  # First object starts at index 2

    def compute_pairwise_features(self, X_batch):
        """
        Compute pairwise invariant mass and deltaR for all object pairs.
        Returns: Additional features to concatenate to each object
        """
        batch_size = X_batch.shape[0]
        n_objects = 18
        obj_features = 5

        # Reshape to get object-wise features
        objects = X_batch[:, 2:].reshape(batch_size, n_objects, obj_features)

        # Mask for real objects (obj_type != 0)
        obj_mask = objects[:, :, 0] != 0
        n_real_objects = obj_mask.sum(axis=1)

        # Prepare arrays for pairwise features per object
        max_pairs_per_object = n_objects - 1
        pairwise_features = torch.zeros(batch_size, n_objects, max_pairs_per_object * 2, device=X_batch.device)

        for b in range(batch_size):
            real_idx = torch.where(obj_mask[b])[0]
            n_real = len(real_idx)

            if n_real < 2:
                continue

            # Get real objects
            real_objs = objects[b, real_idx]
            obj_types = real_objs[:, 0]
            E = real_objs[:, 1] / 1000.0  # Convert MeV to GeV
            pT = real_objs[:, 2] / 1000.0
            eta = real_objs[:, 3]
            phi = real_objs[:, 4]

            # Compute 3-momentum components
            px = pT * torch.cos(phi)
            py = pT * torch.sin(phi)
            pz = pT * torch.sinh(eta)

            # Compute invariant mass for all pairs
            for i in range(n_real):
                for j in range(i+1, n_real):
                    # 4-vectors
                    p4_i = torch.stack([E[i], px[i], py[i], pz[i]])
                    p4_j = torch.stack([E[j], px[j], py[j], pz[j]])

                    # Invariant mass
                    m2 = (E[i] + E[j])**2 - torch.sum((p4_i[1:] + p4_j[1:])**2)
                    m = torch.sqrt(torch.clamp(m2, min=1e-6))

                    # DeltaR
                    deta = eta[i] - eta[j]
                    dphi = phi[i] - phi[j]
                    dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))  # Wrap to [-pi, pi]
                    deltaR = torch.sqrt(deta**2 + dphi**2)

                    # Store for both objects i and j
                    pair_idx = min(j-1, max_pairs_per_object-1)
                    pairwise_features[b, real_idx[i], pair_idx*2] = m
                    pairwise_features[b, real_idx[i], pair_idx*2+1] = deltaR

                    pair_idx = min(i, max_pairs_per_object-1)
                    pairwise_features[b, real_idx[j], pair_idx*2] = m
                    pairwise_features[b, real_idx[j], pair_idx*2+1] = deltaR

        # Flatten pairwise features (take mean and std per object)
        pairwise_agg = torch.zeros(batch_size, n_objects, 4, device=X_batch.device)
        for b in range(batch_size):
            for o in range(n_objects):
                if obj_mask[b, o]:
                    # Get valid pairs for this object
                    valid_pairs = pairwise_features[b, o]
                    valid_mask = valid_pairs != 0
                    if valid_mask.any():
                        valid_vals = valid_pairs[valid_mask].reshape(-1, 2)
                        pairwise_agg[b, o, 0] = valid_vals[:, 0].mean()  # mean m
                        pairwise_agg[b, o, 1] = valid_vals[:, 0].std()   # std m
                        pairwise_agg[b, o, 2] = valid_vals[:, 1].mean()  # mean deltaR
                        pairwise_agg[b, o, 3] = valid_vals[:, 1].std()   # std deltaR

        return pairwise_agg.reshape(batch_size, -1)  # Flatten to [batch, n_objects*4]

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
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        # Basic preprocessing
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Normalize kinematic features (skip object type identifiers)
        kinematic_features = []
        for i in range(2, 92, 5):  # Start from first object's E (index 2)
            kinematic_features.extend([i, i+1, i+2, i+3])  # E, pT, eta, phi

        # Include MET and phi_MET
        kinematic_features = [0, 1] + kinematic_features

        self.feature_indices = kinematic_features

        # Fit scaler on kinematic features only
        self.scaler.fit(X_np[:, kinematic_features])

        # Also fit a scaler for pairwise features (estimate from sample)
        sample_idx = np.random.choice(len(X_np), min(10000, len(X_np)), replace=False)
        X_sample = torch.from_numpy(X_np[sample_idx])
        pairwise_feats = self.compute_pairwise_features(X_sample)
        self.pairwise_feature_scaler.fit(pairwise_feats.cpu().numpy())

        return self

    def transform(self, X):
        if torch.is_tensor(X):
            X_np = X.cpu().numpy()
            return_tensor = True
        else:
            X_np = X
            return_tensor = False

        # Apply scaling to kinematic features
        X_scaled = X_np.copy()
        X_scaled[:, self.feature_indices] = self.scaler.transform(X_np[:, self.feature_indices])

        # Convert back to tensor for pairwise computation
        X_tensor = torch.from_numpy(X_scaled).float()

        # Compute pairwise features
        pairwise_features = self.compute_pairwise_features(X_tensor)

        # Scale pairwise features
        pairwise_scaled = self.pairwise_feature_scaler.transform(pairwise_features.cpu().numpy())
        pairwise_tensor = torch.from_numpy(pairwise_scaled).float()

        # Concatenate original features with pairwise features
        result = torch.cat([X_tensor, pairwise_tensor], dim=1)

        if not return_tensor:
            return result.numpy()
        return result

# ---------- MODEL ARCHITECTURE ----------
class ParticleAttentionLayer(nn.Module):
    """Multi-head attention layer for particle sequences"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, d_model]
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Input shape: original 92 features + 72 pairwise features = 164
        input_dim = 164

        # Object embedding layer (treat each object's 5 features separately)
        self.obj_embedding = nn.Sequential(
            nn.Linear(9, 64),  # 5 original + 4 pairwise per object
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )

        # Attention layers
        self.attention_layers = nn.ModuleList([
            ParticleAttentionLayer(32, 4, dropout=0.1) for _ in range(3)
        ])

        # Global feature processing (MET + aggregated object info)
        self.global_processor = nn.Sequential(
            nn.Linear(32 + 2, 128),  # 32 from objects + 2 MET features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

        # Pooling layers
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mean_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, batch_x):
        # batch_x: [batch, 164]
        batch_size = batch_x.shape[0]

        # Split features
        met = batch_x[:, :2]  # E_T_miss, phi_Et_miss
        objects = batch_x[:, 2:92].reshape(batch_size, 18, 5)  # 18 objects × 5 features
        pairwise = batch_x[:, 92:].reshape(batch_size, 18, 4)  # 18 objects × 4 pairwise features

        # Combine object features with pairwise features
        obj_combined = torch.cat([objects, pairwise], dim=2)  # [batch, 18, 9]

        # Embed each object
        obj_embedded = self.obj_embedding(obj_combined)  # [batch, 18, 32]

        # Create mask for zero-padded objects (obj_type == 0)
        mask = (objects[:, :, 0] == 0)  # [batch, 18]

        # Apply attention layers
        x = obj_embedded
        for attn_layer in self.attention_layers:
            x = attn_layer(x, mask)

        # Pool object representations
        x_pooled = x.mean(dim=1)  # [batch, 32]

        # Combine with MET features
        global_features = torch.cat([x_pooled, met], dim=1)  # [batch, 34]

        # Process global features
        processed = self.global_processor(global_features)  # [batch, 64]

        # Final classification
        logits = self.classifier(processed).squeeze(-1)  # [batch]

        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 60

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    patience = 15

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            predictions = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item()
                predictions = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)

                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        # Compute AUC (simple implementation for binary classification)
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_labels, all_preds)

        # Update learning rate based on validation AUC
        scheduler.step(val_auc)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Record metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

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

