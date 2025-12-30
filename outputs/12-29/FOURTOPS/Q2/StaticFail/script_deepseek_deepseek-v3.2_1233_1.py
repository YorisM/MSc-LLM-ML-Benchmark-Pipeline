
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
        x = self.X[idx]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x, self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

# -------------------------- START OF LLM BLOCK ------------------------------
# ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_scaler = StandardScaler()
        self.obj_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def _extract_features(self, X: torch.Tensor):
        """Extract and separate features from flat input tensor."""
        batch_size = X.shape[0]

        # Global features: E_T_miss, phi_Et_miss
        global_feats = X[:, :2]  # [batch, 2]

        # Object features: reshape to [batch, max_objects=18, 5]
        obj_feats = X[:, 2:].reshape(batch_size, 18, 5)  # [batch, 18, 5]

        # Mask: non-zero object identifier indicates real object
        mask = obj_feats[:, :, 0] != 0  # [batch, 18]

        # Separate object ID and kinematic features
        obj_ids = obj_feats[:, :, 0:1]  # [batch, 18, 1]
        kin_feats = obj_feats[:, :, 1:]  # [batch, 18, 4] - E, pT, eta, phi

        return global_feats, kin_feats, obj_ids, mask

    def _compute_pairwise_features(self, kin_feats: torch.Tensor, mask: torch.Tensor):
        """Compute invariant mass and deltaR for all object pairs."""
        batch_size, max_objs, _ = kin_feats.shape

        # Expand for pairwise computation
        # E, pT, eta, phi
        E = kin_feats[:, :, 0:1]  # [batch, 18, 1]
        pT = kin_feats[:, :, 1:2]  # [batch, 18, 1]
        eta = kin_feats[:, :, 2:3]  # [batch, 18, 1]
        phi = kin_feats[:, :, 3:4]  # [batch, 18, 1]

        # Compute 4-momentum components
        px = pT * torch.cos(phi)  # [batch, 18, 1]
        py = pT * torch.sin(phi)  # [batch, 18, 1]
        pz = pT * torch.sinh(eta)  # [batch, 18, 1]

        # Expand for pairwise operations
        E1 = E.unsqueeze(2)  # [batch, 18, 1, 1]
        E2 = E.unsqueeze(1)  # [batch, 1, 18, 1]

        px1 = px.unsqueeze(2)  # [batch, 18, 1, 1]
        px2 = px.unsqueeze(1)  # [batch, 1, 18, 1]
        py1 = py.unsqueeze(2)  # [batch, 18, 1, 1]
        py2 = py.unsqueeze(1)  # [batch, 1, 18, 1]
        pz1 = pz.unsqueeze(2)  # [batch, 18, 1, 1]
        pz2 = pz.unsqueeze(1)  # [batch, 1, 18, 1]

        # Compute invariant mass: m^2 = (E1+E2)^2 - (p1+p2)^2
        E_sum = E1 + E2  # [batch, 18, 18, 1]
        px_sum = px1 + px2  # [batch, 18, 18, 1]
        py_sum = py1 + py2  # [batch, 18, 18, 1]
        pz_sum = pz1 + pz2  # [batch, 18, 18, 1]

        m2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)  # [batch, 18, 18, 1]
        m2 = torch.clamp(m2, min=1e-6)  # Avoid negative due to numerical errors
        inv_mass = torch.sqrt(m2)  # [batch, 18, 18, 1]

        # Compute deltaR = sqrt((delta_eta)^2 + (delta_phi)^2)
        eta1 = eta.unsqueeze(2)  # [batch, 18, 1, 1]
        eta2 = eta.unsqueeze(1)  # [batch, 1, 18, 1]
        phi1 = phi.unsqueeze(2)  # [batch, 18, 1, 1]
        phi2 = phi.unsqueeze(1)  # [batch, 1, 18, 1]

        delta_eta = eta1 - eta2  # [batch, 18, 18, 1]
        delta_phi = phi1 - phi2  # [batch, 18, 18, 1]
        # Handle phi periodicity
        delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))

        deltaR = torch.sqrt(delta_eta**2 + delta_phi**2)  # [batch, 18, 18, 1]

        # Combine pairwise features
        pairwise = torch.cat([inv_mass, deltaR], dim=-1)  # [batch, 18, 18, 2]

        # Apply mask: zero out pairs involving padded objects
        mask1 = mask.unsqueeze(2).unsqueeze(-1)  # [batch, 18, 1, 1]
        mask2 = mask.unsqueeze(1).unsqueeze(-1)  # [batch, 1, 18, 1]
        pair_mask = mask1 & mask2  # [batch, 18, 18, 1]
        pairwise = pairwise * pair_mask.float()

        return pairwise

    def fit(self, X, y=None):
        X_np = X.numpy() if torch.is_tensor(X) else X
        batch_size = X_np.shape[0]

        # Extract features
        global_feats, kin_feats, obj_ids, mask = self._extract_features(
            torch.from_numpy(X_np).float()
        )

        # Fit global feature scaler
        self.global_scaler.fit(global_feats.numpy())

        # Fit kinematic feature scaler (only on real objects)
        kin_np = kin_feats.numpy()
        mask_np = mask.numpy()
        real_kin = kin_np[mask_np].reshape(-1, 4) if np.any(mask_np) else kin_np.reshape(-1, 4)
        self.obj_scaler.fit(real_kin)

        # Fit pairwise feature scaler
        pairwise = self._compute_pairwise_features(kin_feats, mask)
        pairwise_np = pairwise.numpy()
        pair_mask_np = mask.unsqueeze(1) & mask.unsqueeze(2)
        pair_mask_np = pair_mask_np.unsqueeze(-1)
        real_pairs = pairwise_np[pair_mask_np].reshape(-1, 2) if np.any(pair_mask_np) else pairwise_np.reshape(-1, 2)
        self.edge_scaler.fit(real_pairs)

        return self

    def transform(self, X):
        is_tensor = torch.is_tensor(X)
        X_tensor = X if is_tensor else torch.from_numpy(X).float()
        batch_size = X_tensor.shape[0]

        # Extract features
        global_feats, kin_feats, obj_ids, mask = self._extract_features(X_tensor)

        # Normalize global features
        global_norm = torch.from_numpy(
            self.global_scaler.transform(global_feats.numpy())
        ).float()

        # Normalize kinematic features
        kin_np = kin_feats.numpy()
        orig_shape = kin_np.shape
        kin_norm = self.obj_scaler.transform(kin_np.reshape(-1, 4)).reshape(orig_shape)
        kin_norm = torch.from_numpy(kin_norm).float()

        # Compute and normalize pairwise features
        pairwise = self._compute_pairwise_features(kin_feats, mask)
        pair_np = pairwise.numpy()
        orig_pair_shape = pair_np.shape
        pair_norm = self.edge_scaler.transform(pair_np.reshape(-1, 2)).reshape(orig_pair_shape)
        pair_norm = torch.from_numpy(pair_norm).float()

        # Combine object ID with normalized kinematic features
        obj_feats_norm = torch.cat([obj_ids, kin_norm], dim=-1)  # [batch, 18, 5]

        # Return as dictionary for easier access in model
        return {
            'global': global_norm,
            'objects': obj_feats_norm,
            'edges': pair_norm,
            'mask': mask
        }

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class AttentionPooling(nn.Module):
    """Attention-based pooling for graph nodes."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask):
        # x: [batch, nodes, hidden], mask: [batch, nodes]
        attn_scores = self.attn(x)  # [batch, nodes, 1]
        attn_scores = attn_scores.squeeze(-1)  # [batch, nodes]

        # Mask out padded nodes
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch, nodes]

        # Weighted sum
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [batch, hidden]
        return pooled

class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block."""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with mask
        attn_mask = None
        if mask is not None:
            # Create attention mask: padded positions cannot attend or be attended to
            attn_mask = ~mask.unsqueeze(1).expand(-1, mask.size(1), -1)  # [batch, seq_len, seq_len]
            attn_mask = attn_mask.repeat(1, self.self_attn.num_heads, 1, 1)  # [batch, heads, seq_len, seq_len]
            attn_mask = attn_mask.flatten(0, 1)  # [batch*heads, seq_len, seq_len]

        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed forward
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Sample object is a dictionary from preprocessor
        self.obj_embedding = nn.Embedding(100, 32)  # Assume max 100 object types
        obj_feat_dim = 32 + 4  # Embedding + normalized kinematic features

        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Transformer for object features
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_feat_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(256, num_heads=8, dropout=0.1)
            for _ in range(4)
        ])

        # Attention pooling
        self.pooling = AttentionPooling(256)

        # Global feature processing
        self.global_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Combine object and global features
        combined_dim = 256 + 128

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, batch_x):
        # Unpack batch dictionary
        global_feats = batch_x['global']  # [batch, 2]
        obj_feats = batch_x['objects']  # [batch, 18, 5]
        edge_feats = batch_x['edges']  # [batch, 18, 18, 2]
        mask = batch_x['mask']  # [batch, 18]

        batch_size, num_nodes = mask.shape

        # Process object features
        obj_ids = obj_feats[:, :, 0].long()  # [batch, 18]
        kin_feats = obj_feats[:, :, 1:]  # [batch, 18, 4]

        # Embed object IDs
        obj_emb = self.obj_embedding(obj_ids)  # [batch, 18, 32]

        # Combine with kinematic features
        obj_combined = torch.cat([obj_emb, kin_feats], dim=-1)  # [batch, 18, 36]

        # Initial encoding
        x = self.obj_encoder(obj_combined)  # [batch, 18, 256]

        # Process edge features for transformer
        edge_encoded = self.edge_encoder(edge_feats)  # [batch, 18, 18, 64]

        # Use edge features in transformer (simplified: mean over neighbors)
        # In practice, you'd use proper edge attention
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)

        # Attention pooling over objects
        obj_pooled = self.pooling(x, mask)  # [batch, 256]

        # Process global features
        global_encoded = self.global_encoder(global_feats)  # [batch, 128]

        # Combine features
        combined = torch.cat([obj_pooled, global_encoded], dim=-1)  # [batch, 384]

        # Final classification
        logits = self.classifier(combined)  # [batch, 1]

        return logits.squeeze(-1)  # [batch]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Early stopping
    best_val_auc = 0
    patience = 10
    patience_counter = 0

    # Metrics storage
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()

            # Forward pass
            logits = model(xb)
            loss = criterion(logits, yb.float())

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate metrics
            total_train_loss += loss.item()
            probs = torch.sigmoid(logits)
            train_preds.append(probs.detach().cpu())
            train_labels.append(yb.cpu())

        # Validation
        model.eval()
        total_val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                logits = model(xb)
                loss = criterion(logits, yb.float())

                total_val_loss += loss.item()
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu())
                val_labels.append(yb.cpu())

        # Compute metrics
        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)

        # Compute AUC using sklearn (available in environment)
        from sklearn.metrics import roc_auc_score, accuracy_score

        train_auc = roc_auc_score(train_labels.numpy(), train_preds.numpy())
        val_auc = roc_auc_score(val_labels.numpy(), val_preds.numpy())

        # Compute accuracy at 0.5 threshold
        train_acc = accuracy_score(train_labels.numpy(), (train_preds > 0.5).numpy())
        val_acc = accuracy_score(val_labels.numpy(), (val_preds > 0.5).numpy())

        # Average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Update scheduler
        scheduler.step(val_auc)

        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
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

