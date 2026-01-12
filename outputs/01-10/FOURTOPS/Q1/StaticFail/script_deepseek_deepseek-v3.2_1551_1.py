
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, to_python
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops, dryrun_finite_check_fourtops

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_means = None
        self.feature_stds = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 1024}
        }

    def fit(self, X, y=None):
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Store original feature statistics for physics-aware normalization
        self.feature_means = X_np.mean(axis=0, keepdims=True)
        self.feature_stds = X_np.std(axis=0, keepdims=True)

        # Replace zero stds with 1 to avoid division by zero
        self.feature_stds[self.feature_stds == 0] = 1.0

        # Physics-aware processing: normalize differently for different feature types
        # For E, pT: log(1 + x) transformation then normalize
        # For angles: keep as is, just normalize
        # For object ids: one-hot encode
        return self

    def transform(self, X):
        if torch.is_tensor(X):
            X_np = X.numpy()
        else:
            X_np = X.copy()

        # Apply robust normalization
        X_norm = (X_np - self.feature_means) / self.feature_stds

        # Additional feature engineering
        # 1. Create object multiplicity feature (non-zero objects)
        # Each object has 5 features: obj_id, E, pT, eta, phi
        n_objects = 18
        obj_feat_len = 5
        event_length = X_norm.shape[1]

        # Reshape to [batch, n_objects, obj_feat_len]
        X_reshaped = X_norm.reshape(-1, n_objects, obj_feat_len)

        # Create mask for real objects (obj_id != 0)
        obj_ids = X_reshaped[..., 0]
        mask = (obj_ids != 0).astype(np.float32)

        # Count non-zero objects per event
        n_objects_per_event = mask.sum(axis=1, keepdims=True)  # [batch, 1]

        # 2. Compute aggregate statistics for each event
        # For energy and pT
        energies = X_reshaped[..., 1] * mask  # [batch, n_objects]
        pts = X_reshaped[..., 2] * mask  # [batch, n_objects]

        sum_E = energies.sum(axis=1, keepdims=True)  # [batch, 1]
        sum_pT = pts.sum(axis=1, keepdims=True)  # [batch, 1]
        mean_pT = sum_pT / (n_objects_per_event + 1e-8)  # [batch, 1]
        max_pT = np.max(pts, axis=1, keepdims=True)  # [batch, 1]

        # 3. Missing ET significance feature
        missing_et = X_norm[:, 0:1]  # E_T_miss
        missing_et_phi = X_norm[:, 1:2]  # phi_Et_miss

        # 4. Create interaction features between global and object features
        # This helps capture correlations

        # Combine all engineered features
        engineered_features = np.concatenate([
            n_objects_per_event,
            sum_E,
            sum_pT,
            mean_pT,
            max_pT,
            missing_et,
            missing_et_phi
        ], axis=1)

        # Combine original normalized features with engineered features
        X_final = np.concatenate([X_norm, engineered_features], axis=1)

        return torch.from_numpy(X_final.astype(np.float32))

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[1]

        # Architecture optimized for particle physics data
        self.bn_input = nn.BatchNorm1d(input_dim)

        # Main processing blocks with residual connections
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.block3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Final classification layers
        self.fc1 = nn.Linear(128, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn_fc2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input shape: [batch_size, features]
        x = self.bn_input(x)  # [batch_size, features]

        # Process through blocks with skip connections
        x1 = self.block1(x)  # [batch_size, 512]
        x2 = self.block2(x1)  # [batch_size, 256]
        x3 = self.block3(x2)  # [batch_size, 128]

        # Apply attention
        attn_weights = self.attention(x3)  # [batch_size, 1]
        x_attn = x3 * attn_weights  # [batch_size, 128]

        # Final classification
        x = F.relu(self.bn_fc1(self.fc1(x_attn)))  # [batch_size, 64]
        x = F.relu(self.bn_fc2(self.fc2(x)))  # [batch_size, 32]
        x = self.fc3(x)  # [batch_size, 1]

        return x.squeeze(1)  # [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = F.binary_cross_entropy_with_logits(outputs, batch_y.float())
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = F.binary_cross_entropy_with_logits(outputs, batch_y.float())
                epoch_val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)

        val_loss = epoch_val_loss / len(val_loader)
        val_acc = correct_val / total_val

        # Update learning rate
        scheduler.step()

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            # Load best model
            model.load_state_dict(best_model_state)
            break

    return model, train_losses, val_losses, train_accs, val_accs

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    X_fit, Y_fit = X_train, Y_train
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:200]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre = make_preprocessor().fit(X_fit, Y_fit)
    
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
    n_epochs = 10 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        try:
            dryrun_finite_check_fourtops(trained_model, spec, val_loader, device, batches=10)
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
        summary = to_python(summary)
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

