
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
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class MyPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.pca = None
        self.kinematic_features = None
        self.global_scaler = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        X_np = X.numpy()

        # Global features scaling
        global_features = X_np[:, :2]
        self.global_scaler = RobustScaler()
        self.global_scaler.fit(global_features)

        # Process object features
        num_objects = 18
        obj_features = []

        for obj_idx in range(num_objects):
            start_idx = 2 + obj_idx * 5
            obj_slice = X_np[:, start_idx:start_idx+5]

            # Only use non-zero padded objects for fitting
            mask = obj_slice[:, 0] != 0  # obj_id != 0

            if np.sum(mask) > 0:
                # Scale kinematic features (E, pT, eta, phi)
                kinematic = obj_slice[mask, 1:]
                scaler = RobustScaler()
                scaler.fit(kinematic)
                self.scalers[obj_idx] = scaler

        # Create enhanced features
        enhanced_features = self._create_enhanced_features(X_np, fit_mode=True)

        # Fit PCA on enhanced features
        self.pca = PCA(n_components=32)
        self.pca.fit(enhanced_features)

        # Store kinematic features info
        self.kinematic_features = {
            'num_objects': num_objects,
            'global_dim': 2,
            'obj_dim': 5
        }

        return self

    def _create_enhanced_features(self, X, fit_mode=False):
        """Create physics-inspired enhanced features"""
        batch_size = X.shape[0]
        num_objects = 18
        features = []

        # Global features
        global_feats = X[:, :2]  # E_T_miss, phi_Et_miss

        # Object-based features
        obj_vectors = []
        pt_values = []
        eta_values = []
        phi_values = []

        for obj_idx in range(num_objects):
            start_idx = 2 + obj_idx * 5
            obj_data = X[:, start_idx:start_idx+5]

            # Filter zero-padded objects
            mask = obj_data[:, 0] != 0
            valid_indices = np.where(mask)[0]

            if len(valid_indices) > 0:
                # Basic kinematic features
                pt = obj_data[mask, 2]
                eta = obj_data[mask, 3]
                phi = obj_data[mask, 4]

                # Calculate deltaR between objects and missing ET
                delta_phi = np.abs(phi - global_feats[mask, 1])
                delta_phi = np.minimum(delta_phi, 2*np.pi - delta_phi)
                delta_eta = eta
                delta_r = np.sqrt(delta_eta**2 + delta_phi**2)

                # Store for aggregation
                pt_values.append(pt)
                eta_values.append(eta)
                phi_values.append(phi)

                # Create object vector with enhanced features
                obj_vec = np.zeros((batch_size, 8))
                obj_vec[mask, 0] = obj_data[mask, 0]  # obj_id
                obj_vec[mask, 1] = obj_data[mask, 1]  # E
                obj_vec[mask, 2] = pt  # pT
                obj_vec[mask, 3] = eta  # eta
                obj_vec[mask, 4] = phi  # phi
                obj_vec[mask, 5] = delta_phi
                obj_vec[mask, 6] = delta_eta
                obj_vec[mask, 7] = delta_r

                obj_vectors.append(obj_vec)

        # Aggregate features
        if pt_values:
            all_pt = np.concatenate(pt_values)
            all_eta = np.concatenate(eta_values)
            all_phi = np.concatenate(phi_values)

            # Event-level aggregates
            agg_features = np.zeros((batch_size, 10))

            # HT-like feature (scalar sum of pT)
            for i in range(batch_size):
                event_pt = []
                for pt_arr in pt_values:
                    if i < len(pt_arr):
                        event_pt.append(pt_arr[i])
                if event_pt:
                    agg_features[i, 0] = np.sum(event_pt)
                    agg_features[i, 1] = np.max(event_pt) if event_pt else 0
                    agg_features[i, 2] = np.mean(event_pt) if event_pt else 0
                    agg_features[i, 3] = np.std(event_pt) if len(event_pt) > 1 else 0

            # Angular spreads
            for i in range(batch_size):
                event_eta = []
                event_phi = []
                for j, (eta_arr, phi_arr) in enumerate(zip(eta_values, phi_values)):
                    if i < len(eta_arr):
                        event_eta.append(eta_arr[i])
                        event_phi.append(phi_arr[i])

                if event_eta:
                    agg_features[i, 4] = np.max(event_eta) - np.min(event_eta) if event_eta else 0
                    agg_features[i, 5] = np.std(event_eta) if len(event_eta) > 1 else 0
                    agg_features[i, 6] = np.std(event_phi) if len(event_phi) > 1 else 0

                    # Centrality-like feature
                    agg_features[i, 7] = np.mean(np.abs(event_eta)) if event_eta else 0

                    # Sphericity-like feature
                    if len(event_phi) >= 2:
                        phi_diff = np.max(event_phi) - np.min(event_phi)
                        agg_features[i, 8] = phi_diff / (2*np.pi) if phi_diff > 0 else 0

            # Object multiplicity
            obj_counts = np.sum(X[:, 2::5] != 0, axis=1)
            agg_features[:, 9] = obj_counts

            # Combine all features
            if obj_vectors:
                obj_features = np.stack(obj_vectors, axis=1)  # [batch, num_valid_objs, 8]
                obj_features_flat = obj_features.reshape(batch_size, -1)

                # Pad or truncate to fixed size
                max_obj_features = 18 * 8  # max_objects * features_per_obj
                if obj_features_flat.shape[1] < max_obj_features:
                    padding = max_obj_features - obj_features_flat.shape[1]
                    obj_features_flat = np.pad(obj_features_flat, 
                                             ((0, 0), (0, padding)), 
                                             mode='constant')
                else:
                    obj_features_flat = obj_features_flat[:, :max_obj_features]

                features = np.concatenate([
                    global_feats,
                    agg_features,
                    obj_features_flat
                ], axis=1)
            else:
                features = np.concatenate([
                    global_feats,
                    agg_features,
                    np.zeros((batch_size, 18 * 8))
                ], axis=1)
        else:
            features = np.concatenate([
                global_feats,
                np.zeros((batch_size, 10 + 18 * 8))
            ], axis=1)

        return features

    def transform(self, X):
        X_np = X.numpy()

        # Create enhanced features
        enhanced_features = self._create_enhanced_features(X_np, fit_mode=False)

        # Apply PCA
        if self.pca is not None:
            pca_features = self.pca.transform(enhanced_features)
        else:
            pca_features = enhanced_features

        # Convert back to torch tensor
        return torch.from_numpy(pca_features.astype(np.float32))

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[1]

        self.batch_norm0 = nn.BatchNorm1d(input_dim)

        # Enhanced architecture with residual connections
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )

        self.block2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)
        )

        self.block3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )

        self.block4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=0)
        )

        # Output layers
        self.fc_out = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # Residual connections
        self.residual1 = nn.Linear(input_dim, 256) if input_dim != 256 else nn.Identity()
        self.residual2 = nn.Linear(256, 512) if 256 != 512 else nn.Identity()
        self.residual3 = nn.Linear(512, 256) if 512 != 256 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_x):
        # Initial batch norm
        x = self.batch_norm0(batch_x)  # [batch, input_dim]

        # Block 1 with residual
        identity = self.residual1(x)
        x = self.block1(x) + identity
        x = F.leaky_relu(x, 0.1)

        # Block 2 with residual
        identity = self.residual2(x)
        x = self.block2(x) + identity
        x = F.leaky_relu(x, 0.1)

        # Block 3 with residual
        identity = self.residual3(x)
        x = self.block3(x) + identity
        x = F.leaky_relu(x, 0.1)

        # Block 4
        x = self.block4(x)

        # Attention
        attn_weights = self.attention(x)  # [batch, 1]
        x = x * attn_weights

        # Output
        x = self.fc_out(x)

        return x

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 200

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function with label smoothing
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        verbose=False
    )

    # Gradient clipping
    grad_clip_value = 1.0

    # Early stopping
    best_val_auc = 0
    patience_counter = 0
    patience = 25

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Calculate accuracy
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float().unsqueeze(1)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                total_val_loss += loss.item()

                # Calculate accuracy
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

                # Store for AUC calculation
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_labels, all_preds)

        # Update learning rate
        scheduler.step(val_auc)

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Train Acc: {train_accuracy:.4f}, '
                  f'Val Acc: {val_accuracy:.4f}, '
                  f'Val AUC: {val_auc:.4f}')

        # Early stopping check
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))

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


