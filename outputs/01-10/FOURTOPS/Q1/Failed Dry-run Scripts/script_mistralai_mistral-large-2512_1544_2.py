
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

# ---------- IMPORTS ----------
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler_etmiss = RobustScaler()
        self.scaler_phi = StandardScaler()
        self.scaler_obj = RobustScaler()
        self.scaler_energy = RobustScaler()
        self.scaler_pt = RobustScaler()
        self.scaler_eta = StandardScaler()
        self.obj_ids = None
        self.max_objects = 18
        self.global_features = 2
        self.per_object_features = 5

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": None,

            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False,
                                "batch_size": 1024}
        }

    def fit(self, X, y=None):
        # Extract global features (E_T_miss and phi_Et_miss)
        et_miss = X[:, 0:1].numpy()
        phi_miss = X[:, 1:2].numpy()

        # Scale global features
        self.scaler_etmiss.fit(et_miss)
        self.scaler_phi.fit(phi_miss)

        # Extract per-object features
        obj_features = []
        energy_features = []
        pt_features = []
        eta_features = []
        phi_features = []

        for i in range(self.max_objects):
            start_idx = self.global_features + i * self.per_object_features
            end_idx = start_idx + self.per_object_features
            obj_slice = X[:, start_idx:end_idx]

            # Object ID (categorical)
            obj_ids = obj_slice[:, 0:1].numpy()
            if i == 0:
                self.obj_ids = np.unique(obj_ids)

            # Energy, pT, eta, phi
            energy_features.append(obj_slice[:, 1:2].numpy())
            pt_features.append(obj_slice[:, 2:3].numpy())
            eta_features.append(obj_slice[:, 3:4].numpy())
            phi_features.append(obj_slice[:, 4:5].numpy())

        # Stack and scale per-object features
        energy_stack = np.concatenate(energy_features, axis=1)
        pt_stack = np.concatenate(pt_features, axis=1)
        eta_stack = np.concatenate(eta_features, axis=1)
        phi_stack = np.concatenate(phi_features, axis=1)

        self.scaler_energy.fit(energy_stack)
        self.scaler_pt.fit(pt_stack)
        self.scaler_eta.fit(eta_stack)

        return self

    def transform(self, X):
        # Create output array
        X_transformed = np.zeros_like(X.numpy())

        # Transform global features
        X_transformed[:, 0:1] = self.scaler_etmiss.transform(X[:, 0:1].numpy())
        X_transformed[:, 1:2] = self.scaler_phi.transform(X[:, 1:2].numpy())

        # Transform per-object features
        for i in range(self.max_objects):
            start_idx = self.global_features + i * self.per_object_features
            end_idx = start_idx + self.per_object_features

            # Object ID (leave as is)
            X_transformed[:, start_idx] = X[:, start_idx].numpy()

            # Energy, pT, eta, phi
            obj_slice = X[:, start_idx+1:end_idx].numpy()
            energy = obj_slice[:, 0:1]
            pt = obj_slice[:, 1:2]
            eta = obj_slice[:, 2:3]
            phi = obj_slice[:, 3:4]

            X_transformed[:, start_idx+1] = self.scaler_energy.transform(energy).ravel()
            X_transformed[:, start_idx+2] = self.scaler_pt.transform(pt).ravel()
            X_transformed[:, start_idx+3] = self.scaler_eta.transform(eta).ravel()
            X_transformed[:, start_idx+4] = phi.ravel()  # phi is already in [-pi, pi]

        # Create object masks (1 if object exists, 0 otherwise)
        obj_masks = np.zeros((X.shape[0], self.max_objects))
        for i in range(self.max_objects):
            start_idx = self.global_features + i * self.per_object_features
            obj_masks[:, i] = (X[:, start_idx] != 0).numpy()

        # Add object masks as additional features
        obj_masks = obj_masks.astype(np.float32)
        X_transformed = np.concatenate([X_transformed, obj_masks], axis=1)

        return X_transformed

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Determine input size
        input_size = sample_object.shape[1]
        self.input_size = input_size

        # Feature extraction layers
        self.global_features = 2
        self.max_objects = 18
        self.per_object_features = 5
        self.obj_mask_size = self.max_objects

        # Calculate feature sizes
        global_feature_size = 64
        object_feature_size = 128
        combined_feature_size = 256

        # Global feature network
        self.global_net = nn.Sequential(
            nn.Linear(self.global_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, global_feature_size),
            nn.BatchNorm1d(global_feature_size),
            nn.ReLU()
        )

        # Object feature network
        self.object_net = nn.Sequential(
            nn.Linear(self.per_object_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, object_feature_size),
            nn.BatchNorm1d(object_feature_size),
            nn.ReLU()
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(object_feature_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Combined network
        self.combined_net = nn.Sequential(
            nn.Linear(global_feature_size + object_feature_size, combined_feature_size),
            nn.BatchNorm1d(combined_feature_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(combined_feature_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x: [B, F] where F = 92 (original) + 18 (object masks)

        # Extract global features
        global_feats = batch_x[:, :2]  # [B, 2]
        global_out = self.global_net(global_feats)  # [B, 64]

        # Extract object features and masks
        obj_masks = batch_x[:, -self.max_objects:]  # [B, 18]
        obj_features = []

        for i in range(self.max_objects):
            start_idx = 2 + i * self.per_object_features
            end_idx = start_idx + self.per_object_features
            obj_feat = batch_x[:, start_idx:end_idx]  # [B, 5]
            obj_features.append(obj_feat)

        # Process each object
        obj_outputs = []
        for i in range(self.max_objects):
            obj_feat = obj_features[i]  # [B, 5]
            obj_out = self.object_net(obj_feat)  # [B, 128]
            obj_outputs.append(obj_out)

        # Stack object outputs
        obj_outputs = torch.stack(obj_outputs, dim=1)  # [B, 18, 128]

        # Apply attention
        attention_weights = self.attention(obj_outputs)  # [B, 18, 1]
        attention_weights = attention_weights.squeeze(-1)  # [B, 18]

        # Apply object masks to attention weights
        attention_weights = attention_weights * obj_masks
        attention_weights = F.softmax(attention_weights, dim=1)  # [B, 18]

        # Weighted sum of object features
        attention_weights = attention_weights.unsqueeze(-1)  # [B, 18, 1]
        weighted_obj = torch.sum(obj_outputs * attention_weights, dim=1)  # [B, 128]

        # Combine global and object features
        combined = torch.cat([global_out, weighted_obj], dim=1)  # [B, 192]

        # Final classification
        logits = self.combined_net(combined)  # [B, 1]

        return logits.squeeze(-1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_state = None
    patience = 5
    patience_counter = 0

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.float())
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_train_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            epoch_train_correct += (preds == batch_y.float()).sum().item()
            epoch_train_total += batch_x.size(0)

        # Calculate training metrics
        epoch_train_loss /= epoch_train_total
        epoch_train_acc = epoch_train_correct / epoch_train_total
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.float())

                epoch_val_loss += loss.item() * batch_x.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                epoch_val_correct += (preds == batch_y.float()).sum().item()
                epoch_val_total += batch_x.size(0)

                val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        # Calculate validation metrics
        epoch_val_loss /= epoch_val_total
        epoch_val_acc = epoch_val_correct / epoch_val_total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        # Calculate AUC
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

        # Update scheduler
        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping based on AUC
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss, val_loss, train_acc, val_acc

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

