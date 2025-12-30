
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

```python
# -------------------------- START OF LLM BLOCK ------------------------------
# <start code template>
# ---------- IMPORTS ----------
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_indices = []
        self.object_indices = []
        self.valid_feature_mask = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 2048,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 4096},
        }

    def fit(self, X, y=None):
        X_np = X.numpy()
        # Identify valid features (non-zero columns)
        valid_cols = np.std(X_np, axis=0) > 1e-8
        self.valid_feature_mask = valid_cols

        # Fit scaler only on valid features
        X_valid = X_np[:, valid_cols]
        self.scaler.fit(X_valid)

        # Store feature engineering indices
        self._setup_feature_indices()
        return self

    def _setup_feature_indices(self):
        # Global features: ETmiss and phi_ETmiss
        self.global_features = [0, 1]

        # Object features: group by particle properties
        for obj_idx in range(18):
            start_idx = 2 + obj_idx * 5
            # obj_id, E, pT, eta, phi
            self.object_indices.append({
                'id': start_idx,
                'energy': start_idx + 1,
                'pt': start_idx + 2,
                'eta': start_idx + 3,
                'phi': start_idx + 4
            })

    def transform(self, X):
        X_np = X.numpy()

        # Apply scaling to valid features
        X_valid = X_np[:, self.valid_feature_mask]
        X_scaled = self.scaler.transform(X_valid)

        # Reconstruct full tensor with scaled valid features
        X_result = np.zeros_like(X_np)
        X_result[:, self.valid_feature_mask] = X_scaled

        # Convert back to torch tensor
        return torch.from_numpy(X_result.astype(np.float32))

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: [batch_size, num_objects, dim]
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Input dimension: 92 features per event
        input_dim = 92

        # Feature engineering dimensions
        self.object_dim = 18  # max objects
        self.feats_per_obj = 5

        # Projection layer for object-centric processing
        self.object_proj = nn.Sequential(
            nn.Linear(self.feats_per_obj, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU()
        )

        # Transformer for object interactions
        self.transformer = TransformerBlock(32, heads=4, dropout=0.1)

        # Global features processing
        self.global_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.GELU()
        )

        # High-level feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(32 * (self.object_dim + 1), 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Output layers with skip connections
        self.output_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
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
        # batch_x shape: [batch_size, 92]
        batch_size = batch_x.shape[0]

        # Extract global features (ETmiss, phi_ETmiss)
        global_feats = batch_x[:, :2]  # [batch_size, 2]
        global_processed = self.global_net(global_feats)  # [batch_size, 32]

        # Process objects
        objects = batch_x[:, 2:]  # [batch_size, 90]
        objects = objects.view(batch_size, self.object_dim, self.feats_per_obj)  # [batch_size, 18, 5]

        # Project each object
        objects_processed = self.object_proj(objects.view(-1, self.feats_per_obj))  # [batch_size*18, 32]
        objects_processed = objects_processed.view(batch_size, self.object_dim, 32)  # [batch_size, 18, 32]

        # Apply transformer
        objects_transformed = self.transformer(objects_processed)  # [batch_size, 18, 32]

        # Pool object features
        obj_max = torch.max(objects_transformed, dim=1)[0]  # [batch_size, 32]
        obj_mean = torch.mean(objects_transformed, dim=1)  # [batch_size, 32]
        obj_std = torch.std(objects_transformed, dim=1)  # [batch_size, 32]

        # Combine all features
        combined = torch.cat([
            global_processed,
            obj_max,
            obj_mean,
            obj_std,
            objects_transformed.view(batch_size, -1)  # Flatten all objects
        ], dim=1)  # [batch_size, 32*(1+1+1+1+18) = 32*22 = 704]

        # Extract high-level features

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

