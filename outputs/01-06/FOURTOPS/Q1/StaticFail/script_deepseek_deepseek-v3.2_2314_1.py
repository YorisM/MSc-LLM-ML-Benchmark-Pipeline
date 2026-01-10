
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, GATConv
import warnings
warnings.filterwarnings('ignore')

# -------------------------- START OF LLM BLOCK ------------------------------
# ---------- IMPORTS ----------
# Already imported above

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_scaler = StandardScaler()
        self.obj_scaler = StandardScaler()
        self.obj_id_encoder = QuantileTransformer(n_quantiles=100, output_distribution='normal')
        self.mask_value = -999.0

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
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Process global features
        global_feats = X_np[:, :2]
        self.global_scaler.fit(global_feats)

        # Process object features
        obj_feats_list = []
        for i in range(18):
            start_idx = 2 + i * 5
            end_idx = start_idx + 5
            obj_slice = X_np[:, start_idx:end_idx]
            mask = obj_slice[:, 0] != 0  # obj_id != 0 indicates real object
            real_objs = obj_slice[mask]
            if len(real_objs) > 0:
                obj_feats_list.append(real_objs[:, 1:])  # Exclude obj_id

        if obj_feats_list:
            all_obj_feats = np.vstack(obj_feats_list)
            self.obj_scaler.fit(all_obj_feats)

            # Fit obj_id encoder
            all_obj_ids = np.concatenate([X_np[:, 2 + i*5][X_np[:, 2 + i*5] != 0] 
                                        for i in range(18)])
            if len(all_obj_ids) > 0:
                self.obj_id_encoder.fit(all_obj_ids.reshape(-1, 1))

        return self

    def transform(self, X):
        if torch.is_tensor(X):
            X_np = X.numpy()
        else:
            X_np = X.copy()

        # Normalize global features
        X_np[:, :2] = self.global_scaler.transform(X_np[:, :2])

        # Process each object
        obj_masks = []
        for i in range(18):
            start_idx = 2 + i * 5
            obj_id_idx = start_idx
            feat_start = start_idx + 1
            feat_end = start_idx + 5

            # Create mask: obj_id != 0 indicates real object
            mask = X_np[:, obj_id_idx] != 0
            obj_masks.append(mask)

            # Encode obj_id
            obj_ids = X_np[:, obj_id_idx].reshape(-1, 1)
            if hasattr(self.obj_id_encoder, 'n_quantiles_'):
                encoded_ids = self.obj_id_encoder.transform(obj_ids)
                X_np[:, obj_id_idx] = encoded_ids.flatten()

            # Normalize kinematic features for real objects
            if mask.any():
                feats = X_np[:, feat_start:feat_end]
                feats_normalized = feats.copy()
                feats_normalized[mask] = self.obj_scaler.transform(feats[mask])
                X_np[:, feat_start:feat_end] = feats_normalized

        # Create additional features
        processed = []
        for event_idx in range(X_np.shape[0]):
            event = X_np[event_idx]
            features = []

            # Global features
            features.extend(event[:2])  # [2]

            # Collect all real objects
            obj_features = []
            for i in range(18):
                start_idx = 2 + i * 5
                if event[start_idx] != self.mask_value and event[start_idx] != 0:
                    obj_feat = event[start_idx:start_idx+5]  # [5]
                    obj_features.append(obj_feat)

            if obj_features:
                obj_array = np.array(obj_features)  # [N_obj, 5]

                # Object counts and statistics
                features.append(len(obj_features))  # [1]

                # Mean of kinematic features
                features.extend(np.mean(obj_array[:, 1:], axis=0))  # [4]

                # Std of kinematic features
                features.extend(np.std(obj_array[:, 1:], axis=0))  # [4]

                # Max of kinematic features
                features.extend(np.max(obj_array[:, 1:], axis=0))  # [4]

                # Min of kinematic features  
                features.extend(np.min(obj_array[:, 1:], axis=0))  # [4]

                # Sum of pT
                features.append(np.sum(obj_array[:, 2]))  # [1]

                # HT-like feature (scalar sum of pT)
                features.append(np.sum(obj_array[:, 2]))  # [1]

                # Missing ET significance
                if np.sum(obj_array[:, 2]) > 0:
                    features.append(event[0] / np.sum(obj_array[:, 2]))  # [1]
                else:
                    features.append(0.0)
            else:
                # Pad with zeros if no objects
                features.extend([0] * 22)  # 1 + 4*4 + 2 + 1 = 22

            processed.append(features)

        # Ensure consistent size
        max_len = max(len(f) for f in processed)
        for i in range(len(processed)):
            if len(processed[i]) < max_len:
                processed[i].extend([0.0] * (max_len - len(processed[i])))

        return torch.tensor(np.array(processed), dtype=torch.float32)

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[-1] if len(sample_object.shape) > 1 else sample_object.shape[0]

        # Feature processing layers
        self.bn_input = nn.BatchNorm1d(input_dim)

        # Main network with skip connections
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)

        # Skip connection from fc2 to output
        self.skip_fc = nn.Linear(256, 64)

        # Output layer
        self.output = nn.Linear(128, 1)  # 64 + 64 from skip

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_x):
        # Input shape: [batch_size, features]
        x = self.bn_input(batch_x)  # [B, F]

        # First block
        x1 = F.relu(self.bn1(self.fc1(x)))  # [B, 512]
        x1 = self.dropout1(x1)

        # Second block
        x2 = F.relu(self.bn2(self.fc2(x1)))  # [B, 256]
        x2_skip = self.skip_fc(x2)  # [B, 64]
        x2 = self.dropout2(x2)

        # Third block
        x3 = F.relu(self.bn3(self.fc3(x2)))  # [B, 128]
        x3 = self.dropout3(x3)

        # Fourth block
        x4 = F.relu(self.bn4(self.fc4(x3)))  # [B, 64]

        # Combine with skip connection
        combined = torch.cat([x4, x2_skip], dim=1)  # [B, 128]

        # Output
        out = self.output(combined)  # [B, 1]
        return out.squeeze(-1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 100

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss with label smoothing
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )

    # Training history
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    max_patience = 15

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.float().to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == batch_y.bool()).sum().item()
            train_total += batch_X.size(0)

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.float().to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == batch_y.bool()).sum().item()
                val_total += batch_X.size(0)

                # Store for AUC calculation
                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        if len(all_probs) > 0:
            val_auc = roc_auc_score(all_labels, all_probs)
        else:
            val_auc = 0.5

        # Update learning rate based on AUC
        scheduler.step(val_auc)

        # Early stopping based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Store history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                  f'Val AUC: {val_auc:.4f}')

        # Early stopping
        if patience_counter >= max_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

