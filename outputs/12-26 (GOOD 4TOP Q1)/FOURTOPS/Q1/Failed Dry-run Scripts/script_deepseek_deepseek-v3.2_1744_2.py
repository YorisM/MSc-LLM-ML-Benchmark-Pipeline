
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

# -------------------------- START OF LLM BLOCK ------------------------------
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import roc_auc_score
import torch
import numpy as np

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.kinematic_mean = None
        self.kinematic_std = None
        self.obj_id_embedding_dim = 8

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate": "ragged_xy",
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def _process_event(self, event):
        # event shape: [92]
        # Split into global and object features
        global_features = event[:2]  # [2]
        object_features = event[2:].reshape(-1, 5)  # [18, 5]

        # Create mask for real objects (obj_id != 0)
        obj_ids = object_features[:, 0]  # [18]
        mask = obj_ids != 0  # [18]

        # Normalize global features
        global_norm = (global_features - self.global_mean) / (self.global_std + 1e-8)

        # Process kinematic features for real objects only
        kinematic_features = object_features[:, 1:]  # [18, 4]
        kinematic_norm = torch.zeros_like(kinematic_features)
        kinematic_norm[mask] = (kinematic_features[mask] - self.kinematic_mean) / (self.kinematic_std + 1e-8)

        # Combine features
        obj_features = torch.cat([
            obj_ids.unsqueeze(1),  # [18, 1]
            kinematic_norm,        # [18, 4]
        ], dim=1)  # [18, 5]

        return {
            'global': global_norm,
            'objects': obj_features,
            'mask': mask,
            'obj_ids': obj_ids
        }

    def fit(self, X, y=None):
        # Convert to torch tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        # Calculate statistics for global features
        global_features = X[:, :2]
        self.global_mean = global_features.mean(dim=0)
        self.global_std = global_features.std(dim=0)

        # Calculate statistics for kinematic features (only real objects)
        object_features = X[:, 2:].reshape(-1, 5)
        obj_ids = object_features[:, 0]
        mask = obj_ids != 0
        kinematic_features = object_features[mask, 1:]
        self.kinematic_mean = kinematic_features.mean(dim=0)
        self.kinematic_std = kinematic_features.std(dim=0)

        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        processed = []
        for i in range(X.shape[0]):
            processed.append(self._process_event(X[i]))

        return processed

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, mask=None):
        # x: [batch, seq_len, dim]
        attn_out, _ = self.attn(x, x, x, key_padding_mask=~mask if mask is not None else None)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Object ID embedding
        self.obj_embedding = nn.Embedding(11, 8)  # 0-10 object IDs

        # Object feature processing
        obj_feat_dim = 8 + 4  # embedding + kinematic features
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_feat_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.LayerNorm(128)
        )

        # Attention blocks for object interactions
        self.attn_blocks = nn.ModuleList([
            AttentionBlock(128) for _ in range(2)
        ])

        # Global feature processing
        self.global_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.LayerNorm(64)
        )

        # Aggregation and classification
        self.aggregate = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, batch_x):
        # Process batch_x which is a list of dicts (from ragged_xy collate)
        batch_size = len(batch_x)

        # Get max number of objects in batch
        max_objs = max(item['objects'].shape[0] for item in batch_x)

        # Prepare batched tensors
        obj_features_list = []
        global_features_list = []
        masks = []

        for item in batch_x:
            n_objs = item['objects'].shape[0]

            # Object embeddings
            obj_ids = item['obj_ids'].long()  # [n_objs]
            kinematic = item['objects'][:, 1:]  # [n_objs, 4]
            obj_emb = self.obj_embedding(obj_ids)  # [n_objs, 8]
            obj_feat = torch.cat([obj_emb, kinematic], dim=1)  # [n_objs, 12]
            encoded = self.obj_encoder(obj_feat)  # [n_objs, 128]

            # Pad to max_objs
            pad_len = max_objs - n_objs
            if pad_len > 0:
                encoded = F.pad(encoded, (0, 0, 0, pad_len), 'constant', 0)  # [max_objs, 128]
                mask = F.pad(item['mask'], (0, pad_len), 'constant', 0)  # [max_objs]
            else:
                mask = item['mask']  # [max_objs]

            obj_features_list.append(encoded)
            masks.append(mask)
            global_features_list.append(item['global'])

        # Stack batched tensors
        obj_features = torch.stack(obj_features_list)  # [batch, max_objs, 128]
        mask = torch.stack(masks).bool()  # [batch, max_objs]
        global_features = torch.stack(global_features_list)  # [batch, 2]

        # Apply attention blocks
        for attn_block in self.attn_blocks:
            obj_features = attn_block(obj_features, mask)

        # Aggregate object features (masked mean)
        obj_features = obj_features * mask.unsqueeze(-1)  # [batch, max_objs, 128]
        obj_agg = obj_features.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 128]

        # Process global features
        global_encoded = self.global_encoder(global_features)  # [batch, 64]

        # Combine and classify
        combined = torch.cat([obj_agg, global_encoded], dim=1)  # [batch, 192]
        logits = self.aggregate(combined)  # [batch, 1]

        return logits.squeeze(-1)  # [batch]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_auc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb.float()).sum().item()
            total += yb.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation phase
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb.float())

                total_loss += loss.item() * yb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == yb.float()).sum().item()
                total += yb.size(0)

                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_targets.extend(yb.cpu().numpy())

        val_loss = total_loss / total
        val_acc = correct / total
        val_auc = roc_auc_score(all_targets, all_preds)

        # Update scheduler
        scheduler.step()

        # Early stopping based on AUC
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
              f"Val AUC: {val_auc:.4f}")

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

