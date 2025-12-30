
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

# ---------- IMPORTS ----------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# Not needed - using default FourTopsDataset

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.kinematic_mean = None
        self.kinematic_std = None
        self.obj_type_mean = None
        self.obj_type_std = None

    def make_loader_cfg(self):
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
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X

        # Global features (2)
        self.global_mean = X_np[:, :2].mean(axis=0)  # [2]
        self.global_std = X_np[:, :2].std(axis=0) + 1e-8  # [2]

        # Object features (18 objects × 5 features)
        obj_features = X_np[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]

        # Kinematic features (E, pT, η, φ) - normalize only non-zero objects
        kinematic = obj_features[:, :, 1:5].reshape(-1, 4)  # [N*18, 4]
        mask = (kinematic[:, 0] != 0)  # Non-zero objects
        self.kinematic_mean = kinematic[mask].mean(axis=0)  # [4]
        self.kinematic_std = kinematic[mask].std(axis=0) + 1e-8  # [4]

        # Object type (obj_id) - normalize
        obj_type = obj_features[:, :, 0].reshape(-1, 1)  # [N*18, 1]
        mask_type = (obj_type != 0)
        self.obj_type_mean = obj_type[mask_type].mean() if np.any(mask_type) else 0.0
        self.obj_type_std = obj_type[mask_type].std() + 1e-8 if np.any(mask_type) else 1.0

        return self

    def transform(self, X):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        X_out = X_np.copy()

        # Normalize global features
        X_out[:, :2] = (X_out[:, :2] - self.global_mean) / self.global_std

        # Reshape object features
        obj_features = X_out[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]

        # Normalize object type (obj_id)
        mask_type = (obj_features[:, :, 0] != 0)
        obj_features[:, :, 0][mask_type] = (
            (obj_features[:, :, 0][mask_type] - self.obj_type_mean) / self.obj_type_std
        )

        # Normalize kinematic features
        kinematic = obj_features[:, :, 1:5]  # [N, 18, 4]
        mask_kin = (kinematic[:, :, 0] != 0)  # [N, 18]

        for i in range(4):
            kinematic[:, :, i][mask_kin] = (
                (kinematic[:, :, i][mask_kin] - self.kinematic_mean[i]) / self.kinematic_std[i]
            )

        obj_features[:, :, 1:5] = kinematic
        X_out[:, 2:] = obj_features.reshape(-1, 90)

        return X_out

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class ObjectAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x, mask=None):
        # x: [batch, num_objects, hidden]
        Q = self.query(x)  # [batch, num_objects, hidden]
        K = self.key(x)    # [batch, num_objects, hidden]
        V = self.value(x)  # [batch, num_objects, hidden]

        attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [batch, num_objects, num_objects]

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, V)  # [batch, num_objects, hidden]
        return out

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = 92

        # Object processing
        obj_hidden = 64
        self.obj_embedding = nn.Sequential(
            nn.Linear(5, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, obj_hidden),
            nn.BatchNorm1d(obj_hidden),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Object attention
        self.obj_attention = ObjectAttention(obj_hidden)

        # Global feature processing
        global_hidden = 32
        self.global_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, global_hidden),
            nn.BatchNorm1d(global_hidden),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Combined processing
        combined_dim = obj_hidden + global_hidden
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, batch_x):
        # batch_x: [batch, 92]
        batch_size = batch_x.shape[0]

        # Split features
        global_feat = batch_x[:, :2]  # [batch, 2]
        obj_feat = batch_x[:, 2:].view(batch_size, 18, 5)  # [batch, 18, 5]

        # Create mask for zero-padded objects
        mask = (obj_feat[:, :, 1] != 0).float()  # [batch, 18] - using E as indicator

        # Process objects
        obj_flat = obj_feat.view(-1, 5)  # [batch*18, 5]
        obj_emb = self.obj_embedding(obj_flat)  # [batch*18, obj_hidden]
        obj_emb = obj_emb.view(batch_size, 18, -1)  # [batch, 18, obj_hidden]

        # Apply attention
        obj_attn = self.obj_attention(obj_emb, mask)  # [batch, 18, obj_hidden]

        # Aggregate objects (weighted by mask)
        obj_weighted = obj_attn * mask.unsqueeze(-1)  # [batch, 18, obj_hidden]
        obj_agg = torch.sum(obj_weighted, dim=1) / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)  # [batch, obj_hidden]

        # Process global features
        global_emb = self.global_net(global_feat)  # [batch, global_hidden]

        # Combine features
        combined = torch.cat([obj_agg, global_emb], dim=1)  # [batch, obj_hidden+global_hidden]

        # Final classification
        output = self.combined_net(combined)  # [batch, 1]
        return output.squeeze(-1)  # [batch]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_aucs, val_aucs = [], []

    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())

        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_preds_np = np.array(train_preds)
        train_targets_np = np.array(train_targets)
        train_acc = ((train_preds_np > 0.5) == train_targets_np).mean()
        train_auc = roc_auc_score(train_targets_np, train_preds_np)

        # Validation phase
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                val_targets.extend(batch_y.detach().cpu().numpy())

        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_preds_np = np.array(val_preds)
        val_targets_np = np.array(val_targets)
        val_acc = ((val_preds_np > 0.5) == val_targets_np).mean()
        val_auc = roc_auc_score(val_targets_np, val_preds_np)

        # Update scheduler
        scheduler.step(val_auc)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

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
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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


