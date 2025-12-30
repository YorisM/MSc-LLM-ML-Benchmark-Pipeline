
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
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from scipy import stats

# ---------- IMPORTS ----------
import warnings
warnings.filterwarnings('ignore')

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_scaler = StandardScaler()
        self.energy_scaler = StandardScaler()
        self.pt_scaler = StandardScaler()
        self.eta_scaler = StandardScaler()
        self.phi_scaler = StandardScaler()
        self.obj_id_min = 0
        self.obj_id_max = 0
        self.quantile_transformers = {}

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

    def _extract_object_features(self, X):
        # Reshape to (batch, 18 objects, 5 features)
        batch_size = X.shape[0]
        objects = X[:, 2:].reshape(batch_size, 18, 5)  # [batch, 18, 5]
        return objects

    def fit(self, X, y=None):
        # Extract global features
        global_features = X[:, :2].numpy()  # [n_samples, 2]
        self.global_scaler.fit(global_features)

        # Extract object features
        objects = self._extract_object_features(X)  # [n_samples, 18, 5]

        # Flatten object features for scaling
        obj_ids = objects[:, :, 0].flatten().numpy()  # [n_samples*18]
        energies = objects[:, :, 1].flatten().numpy()  # [n_samples*18]
        pts = objects[:, :, 2].flatten().numpy()  # [n_samples*18]
        etas = objects[:, :, 3].flatten().numpy()  # [n_samples*18]
        phis = objects[:, :, 4].flatten().numpy()  # [n_samples*18]

        # Filter out padding (where energy == 0)
        mask = energies > 1e-6
        energies_nonzero = energies[mask]
        pts_nonzero = pts[mask]
        etas_nonzero = etas[mask]
        phis_nonzero = phis[mask]

        # Fit scalers on non-zero values
        self.energy_scaler.fit(energies_nonzero.reshape(-1, 1))
        self.pt_scaler.fit(pts_nonzero.reshape(-1, 1))
        self.eta_scaler.fit(etas_nonzero.reshape(-1, 1))
        self.phi_scaler.fit(phis_nonzero.reshape(-1, 1))

        # Get object ID range
        self.obj_id_min = int(obj_ids.min())
        self.obj_id_max = int(obj_ids.max())

        # Create quantile transformers for important features
        for feature_name, data in [('energy', energies_nonzero),
                                   ('pt', pts_nonzero),
                                   ('eta', etas_nonzero)]:
            qt = QuantileTransformer(n_quantiles=1000, 
                                   output_distribution='normal',
                                   random_state=42)
            qt.fit(data.reshape(-1, 1))
            self.quantile_transformers[feature_name] = qt

        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X

        batch_size = X_np.shape[0]
        X_transformed = np.zeros((batch_size, 131), dtype=np.float32)  # Increased feature dimension

        # 1. Process global features (2)
        global_feats = X_np[:, :2]
        global_feats_scaled = self.global_scaler.transform(global_feats)
        X_transformed[:, :2] = global_feats_scaled

        # 2. Process object features
        objects = X_np[:, 2:].reshape(batch_size, 18, 5)

        # Start index for engineered features
        feat_idx = 2

        for i in range(batch_size):
            obj_count = 0
            for j in range(18):
                obj_id = objects[i, j, 0]
                energy = objects[i, j, 1]
                pt = objects[i, j, 2]
                eta = objects[i, j, 3]
                phi = objects[i, j, 4]

                # Skip padded objects (energy ≈ 0)
                if energy < 1e-6:
                    continue

                obj_count += 1

                # Scale continuous features
                energy_scaled = self.energy_scaler.transform([[energy]])[0, 0]
                pt_scaled = self.pt_scaler.transform([[pt]])[0, 0]
                eta_scaled = self.eta_scaler.transform([[eta]])[0, 0]
                phi_scaled = self.phi_scaler.transform([[phi]])[0, 0]

                # Apply quantile transformation for important features
                if 'energy' in self.quantile_transformers:
                    energy_qt = self.quantile_transformers['energy'].transform([[energy]])[0, 0]
                else:
                    energy_qt = energy_scaled

                if 'pt' in self.quantile_transformers:
                    pt_qt = self.quantile_transformers['pt'].transform([[pt]])[0, 0]
                else:
                    pt_qt = pt_scaled

                if 'eta' in self.quantile_transformers:
                    eta_qt = self.quantile_transformers['eta'].transform([[eta]])[0, 0]
                else:
                    eta_qt = eta_scaled

                # Feature engineering
                # 1. Object type features (one-hot like)
                X_transformed[i, feat_idx + int(obj_id)] = 1.0

                # 2. Kinematic features per object type
                type_base = feat_idx + 20  # Offset after object types
                type_idx = min(int(obj_id), 4)  # Group rare types
                X_transformed[i, type_base + type_idx*4] = energy_qt
                X_transformed[i, type_base + type_idx*4 + 1] = pt_qt
                X_transformed[i, type_base + type_idx*4 + 2] = eta_qt
                X_transformed[i, type_base + type_idx*4 + 3] = phi_scaled

            # 3. Aggregated features
            agg_base = feat_idx + 40  # Offset after type-specific features

            # Filter non-padded objects
            mask = objects[i, :, 1] > 1e-6
            if np.sum(mask) > 0:
                non_padded = objects[i, mask]

                # Basic aggregates
                X_transformed[i, agg_base] = np.sum(non_padded[:, 1])  # total energy
                X_transformed[i, agg_base + 1] = np.sum(non_padded[:, 2])  # total pT
                X_transformed[i, agg_base + 2] = np.max(non_padded[:, 2])  # max pT
                X_transformed[i, agg_base + 3] = np.mean(non_padded[:, 2])  # mean pT
                X_transformed[i, agg_base + 4] = np.std(non_padded[:, 2])  # pT std

                # Object count features
                X_transformed[i, agg_base + 5] = np.sum(mask)  # total objects
                for k in range(1, 7):  # Count specific object types
                    type_mask = non_padded[:, 0] == k
                    X_transformed[i, agg_base + 5 + k] = np.sum(type_mask)

                # Energy ratios
                if X_transformed[i, agg_base] > 0:
                    X_transformed[i, agg_base + 12] = X_transformed[i, agg_base + 1] / X_transformed[i, agg_base]

                # Angular features
                if len(non_padded) > 1:
                    X_transformed[i, agg_base + 13] = np.std(non_padded[:, 3])  # eta spread
                    X_transformed[i, agg_base + 14] = np.std(non_padded[:, 4])  # phi spread

                    # Delta R between highest pT objects
                    sorted_idx = np.argsort(-non_padded[:, 2])
                    if len(sorted_idx) >= 2:
                        eta1, phi1 = non_padded[sorted_idx[0], 3], non_padded[sorted_idx[0], 4]
                        eta2, phi2 = non_padded[sorted_idx[1], 3], non_padded[sorted_idx[1], 4]
                        delta_phi = abs(phi1 - phi2)
                        delta_phi = min(delta_phi, 2*np.pi - delta_phi)
                        delta_eta = abs(eta1 - eta2)
                        X_transformed[i, agg_base + 15] = np.sqrt(delta_eta**2 + delta_phi**2)

        # Apply additional transformations to aggregated features
        for col in range(agg_base, agg_base + 16):
            col_data = X_transformed[:, col]
            if np.std(col_data) > 1e-6:
                # Log transform for positive skewed features
                if col in [agg_base, agg_base + 1, agg_base + 2]:
                    positive_mask = col_data > 0
                    if np.sum(positive_mask) > 0:
                        col_data[positive_mask] = np.log1p(col_data[positive_mask])
                # Standardize
                mean_val = np.mean(col_data)
                std_val = np.std(col_data)
                if std_val > 1e-6:
                    X_transformed[:, col] = (col_data - mean_val) / std_val

        return torch.from_numpy(X_transformed)

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[1]

        # Feature compression layer
        self.feature_compression = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Multiple attention heads for different feature groups
        self.attention1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=0)
        )

        # Main processing blocks
        self.block1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.block2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.block3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.block4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Output layers with residual connections
        self.output1 = nn.Linear(128, 64)
        self.output2 = nn.Linear(64, 32)
        self.output3 = nn.Linear(32, 1)

        # Batch norms for residual
        self.bn_res1 = nn.BatchNorm1d(64)
        self.bn_res2 = nn.BatchNorm1d(32)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_x):
        # Input: [batch_size, 131]

        # Feature compression
        x = self.feature_compression(batch_x)  # [batch_size, 128]

        # Apply attention
        attn_weights = self.attention1(x)  # [batch_size, 1]
        x = x * attn_weights  # [batch_size, 128]

        # Process through blocks with skip connections
        x1 = self.block1(x)  # [batch_size, 256]
        x2 = self.block2(x1)  # [batch_size, 512]
        x3 = self.block3(x2)  # [batch_size, 256]
        x4 = self.block4(x3 + x1[:, :256])  # [batch_size, 128]

        # Output layers with residual
        out = F.relu(self.output1(x4))  # [batch_size, 64]
        out = self.bn_res1(out)
        out = F.relu(self.output2(out))  # [batch_size, 32]
        out = self.bn_res2(out)
        out = self.output3(out)  # [batch_size, 1]

        return out.squeeze(-1)  # [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 80

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function with class weighting
    pos_weight = torch.tensor([1.2]).to(device)  # Slightly upweight signal
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Training metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Early stopping
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Calculate accuracy
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == batch_y.bool()).sum().item()
            train_total += batch_y.size(0)
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == batch_y.bool()).sum().item()
                val_total += batch_y.size(0)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Update scheduler
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
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

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


