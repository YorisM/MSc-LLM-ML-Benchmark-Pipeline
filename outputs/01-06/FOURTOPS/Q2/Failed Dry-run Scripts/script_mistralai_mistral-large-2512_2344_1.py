
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

# ---------- IMPORTS ----------
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.obj_ids = [0, 1, 2, 3, 4, 5, 6, 21, 22]  # Common object IDs in particle physics
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
                               "batch_size": 512}
        }

    def fit(self, X, y=None):
        # Extract global features and object features separately
        global_features = X[:, :2]  # [N, 2]
        object_features = X[:, 2:]  # [N, 90]

        # Reshape object features to [N, 18, 5]
        object_features = object_features.reshape(-1, self.max_objects, self.per_object_features)

        # Filter out zero-padded objects (obj_id == 0)
        valid_mask = object_features[:, :, 0] != 0  # [N, 18]

        # For each event, get the number of valid objects
        num_valid_objects = valid_mask.sum(dim=1)  # [N]

        # Create a mask for valid objects across all events
        # We'll use this to extract only valid object features for scaling
        flat_valid_mask = valid_mask.flatten()  # [N*18]
        flat_object_features = object_features.reshape(-1, self.per_object_features)  # [N*18, 5]

        # Extract only valid object features
        valid_object_features = flat_object_features[flat_valid_mask]  # [M, 5] where M is total valid objects

        # Scale global features
        self.scaler.fit(global_features)
        global_features_scaled = self.scaler.transform(global_features)

        # Scale object features (only kinematic features, not obj_id)
        obj_kinematic_features = valid_object_features[:, 1:]  # [M, 4]
        self.obj_scaler = RobustScaler()
        self.obj_scaler.fit(obj_kinematic_features)

        return self

    def transform(self, X):
        # Extract global features
        global_features = X[:, :2]  # [N, 2]
        global_features_scaled = self.scaler.transform(global_features)

        # Extract and reshape object features
        object_features = X[:, 2:]  # [N, 90]
        object_features = object_features.reshape(-1, self.max_objects, self.per_object_features)  # [N, 18, 5]

        # Create mask for valid objects
        valid_mask = object_features[:, :, 0] != 0  # [N, 18]

        # Scale kinematic features of valid objects
        flat_object_features = object_features.reshape(-1, self.per_object_features)  # [N*18, 5]
        flat_valid_mask = valid_mask.flatten()  # [N*18]

        # Extract valid object kinematic features
        valid_object_features = flat_object_features[flat_valid_mask]  # [M, 5]
        if len(valid_object_features) > 0:
            obj_kinematic_features = valid_object_features[:, 1:]  # [M, 4]
            obj_kinematic_features_scaled = self.obj_scaler.transform(obj_kinematic_features)
            valid_object_features[:, 1:] = obj_kinematic_features_scaled

        # Reconstruct object features array
        flat_object_features[flat_valid_mask] = valid_object_features
        object_features = flat_object_features.reshape(-1, self.max_objects, self.per_object_features)  # [N, 18, 5]

        # Combine global and object features
        processed_X = torch.zeros_like(X)
        processed_X[:, :2] = torch.tensor(global_features_scaled, dtype=torch.float32)
        processed_X[:, 2:] = object_features.reshape(-1, 90)

        # Add pairwise features
        pairwise_features = self._compute_pairwise_features(object_features, valid_mask)
        processed_X = torch.cat([processed_X, pairwise_features], dim=1)  # [N, 92 + 18*17/2]

        return processed_X

    def _compute_pairwise_features(self, object_features, valid_mask):
        # object_features: [N, 18, 5]
        # valid_mask: [N, 18]
        N = object_features.shape[0]
        max_objects = object_features.shape[1]

        # Initialize pairwise features tensor
        pairwise_feature_list = []

        for i in range(max_objects):
            for j in range(i+1, max_objects):
                # Create mask for events where both objects i and j are valid
                mask_ij = valid_mask[:, i] & valid_mask[:, j]  # [N]

                # Initialize features with zeros
                delta_eta = torch.zeros(N, dtype=torch.float32)
                delta_phi = torch.zeros(N, dtype=torch.float32)
                delta_r = torch.zeros(N, dtype=torch.float32)
                inv_mass = torch.zeros(N, dtype=torch.float32)

                if mask_ij.any():
                    # Extract features for objects i and j
                    obj_i = object_features[mask_ij, i]  # [M, 5]
                    obj_j = object_features[mask_ij, j]  # [M, 5]

                    # Compute delta eta and delta phi
                    eta_i = obj_i[:, 3]
                    eta_j = obj_j[:, 3]
                    phi_i = obj_i[:, 4]
                    phi_j = obj_j[:, 4]

                    delta_eta_ij = eta_i - eta_j
                    delta_phi_ij = torch.abs(phi_i - phi_j)
                    delta_phi_ij = torch.min(delta_phi_ij, 2*torch.pi - delta_phi_ij)

                    # Compute delta R
                    delta_r_ij = torch.sqrt(delta_eta_ij**2 + delta_phi_ij**2)

                    # Compute invariant mass
                    E_i = obj_i[:, 1]
                    E_j = obj_j[:, 1]
                    px_i = obj_i[:, 2] * torch.cos(phi_i)
                    py_i = obj_i[:, 2] * torch.sin(phi_i)
                    pz_i = obj_i[:, 2] * torch.sinh(eta_i)
                    px_j = obj_j[:, 2] * torch.cos(phi_j)
                    py_j = obj_j[:, 2] * torch.sin(phi_j)
                    pz_j = obj_j[:, 2] * torch.sinh(eta_j)

                    E_tot = E_i + E_j
                    px_tot = px_i + px_j
                    py_tot = py_i + py_j
                    pz_tot = pz_i + pz_j

                    inv_mass_ij = torch.sqrt(E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2))

                    # Assign to output tensors
                    delta_eta[mask_ij] = delta_eta_ij
                    delta_phi[mask_ij] = delta_phi_ij
                    delta_r[mask_ij] = delta_r_ij
                    inv_mass[mask_ij] = inv_mass_ij

                # Stack features
                pairwise_features = torch.stack([delta_eta, delta_phi, delta_r, inv_mass], dim=1)  # [N, 4]
                pairwise_feature_list.append(pairwise_features)

        # Concatenate all pairwise features
        pairwise_features = torch.cat(pairwise_feature_list, dim=1)  # [N, 4 * (18*17/2)]

        return pairwise_features

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Determine input size from sample
        input_size = sample_object.shape[1]

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x: [B, F]
        features = self.feature_extractor(batch_x)  # [B, 128]
        logits = self.classifier(features)  # [B, 1]
        return logits.squeeze(1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    best_auc = 0.0
    best_model_state = None
    patience_counter = 0
    max_patience = 5

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_targets = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_total += batch_x.size(0)

            # Calculate accuracy
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds.float() == batch_y).sum().item()

            # Store predictions and targets for AUC calculation
            all_train_preds.append(torch.sigmoid(outputs).detach().cpu())
            all_train_targets.append(batch_y.detach().cpu())

        # Calculate training metrics
        train_loss /= train_total
        train_acc = train_correct / train_total
        all_train_preds = torch.cat(all_train_preds)
        all_train_targets = torch.cat(all_train_targets)
        train_auc = roc_auc_score(all_train_targets.numpy(), all_train_preds.numpy())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.float().to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                val_total += batch_x.size(0)

                # Calculate accuracy
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds.float() == batch_y).sum().item()

                # Store predictions and targets for AUC calculation
                all_val_preds.append(torch.sigmoid(outputs).detach().cpu())
                all_val_targets.append(batch_y.detach().cpu())

        # Calculate validation metrics
        val_loss /= val_total
        val_acc = val_correct / val_total
        all_val_preds = torch.cat(all_val_preds)
        all_val_targets = torch.cat(all_val_targets)
        val_auc = roc_auc_score(all_val_targets.numpy(), all_val_preds.numpy())

        # Update learning rate scheduler
        scheduler.step(val_auc)

        # Store history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping based on validation AUC
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model state
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

