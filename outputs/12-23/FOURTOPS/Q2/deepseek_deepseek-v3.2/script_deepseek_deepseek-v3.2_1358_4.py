
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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math

# -------- CUSTOM DATASET --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_scaler = StandardScaler()
        self.obj_scaler = StandardScaler()
        self.deltaR_scaler = StandardScaler()
        self.mass_scaler = StandardScaler()
        self.pairwise_scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Extract statistics for transform
        X_np = X.numpy()
        N = X_np.shape[0]

        # Extract global features
        global_features = X_np[:, :2]
        self.global_scaler.fit(global_features)

        # Extract object features
        obj_features = X_np[:, 2:].reshape(N, 18, 5)
        obj_ids = obj_features[:, :, 0]
        obj_kinematics = obj_features[:, :, 1:].reshape(-1, 4)
        valid_mask = obj_ids.flatten() != 0
        self.obj_scaler.fit(obj_kinematics[valid_mask])

        # Compute pairwise features for statistics
        deltaR_list = []
        mass_list = []
        pairwise_list = []

        for i in range(N):
            valid_idx = np.where(obj_ids[i] != 0)[0]
            if len(valid_idx) < 2:
                continue

            eta = obj_kinematics[i*18:(i+1)*18, 2][valid_idx]
            phi = obj_kinematics[i*18:(i+1)*18, 3][valid_idx]
            pt = obj_kinematics[i*18:(i+1)*18, 1][valid_idx]
            energy = obj_kinematics[i*18:(i+1)*18, 0][valid_idx]

            # Compute deltaR and invariant mass for all pairs
            for j in range(len(valid_idx)):
                for k in range(j+1, len(valid_idx)):
                    delta_eta = eta[j] - eta[k]
                    delta_phi = phi[j] - phi[k]
                    while delta_phi > math.pi:
                        delta_phi -= 2*math.pi
                    while delta_phi < -math.pi:
                        delta_phi += 2*math.pi

                    deltaR = math.sqrt(delta_eta**2 + delta_phi**2)
                    deltaR_list.append(deltaR)

                    # Compute invariant mass m_ij = sqrt(2*pT_i*pT_j*(cosh(Δη) - cos(Δφ)))
                    # For massless particles approximation
                    m = math.sqrt(2*pt[j]*pt[k]*(math.cosh(delta_eta) - math.cos(delta_phi)))
                    mass_list.append(m)

                    # Additional pairwise features
                    pairwise_list.append([pt[j]/pt[k] if pt[k] > 0 else 0, 
                                         energy[j]/energy[k] if energy[k] > 0 else 0,
                                         abs(eta[j] - eta[k]),
                                         abs(phi[j] - phi[k])])

        if deltaR_list:
            self.deltaR_scaler.fit(np.array(deltaR_list).reshape(-1, 1))
        if mass_list:
            self.mass_scaler.fit(np.array(mass_list).reshape(-1, 1))
        if pairwise_list:
            self.pairwise_scaler.fit(np.array(pairwise_list))

        return self

    def transform(self, X):
        X_np = X.numpy()
        N = X_np.shape[0]
        output_features = []

        # Process each event
        for i in range(N):
            event_features = []

            # 1. Normalized global features
            global_feat = X_np[i, :2].reshape(1, -1)
            global_feat_norm = self.global_scaler.transform(global_feat).flatten()
            event_features.extend(global_feat_norm)

            # 2. Extract and normalize object features
            obj_data = X_np[i, 2:].reshape(18, 5)
            obj_ids = obj_data[:, 0]
            obj_kinematics = obj_data[:, 1:]

            # Normalize kinematics
            obj_kinematics_norm = self.obj_scaler.transform(obj_kinematics)

            # Count valid objects
            valid_mask = obj_ids != 0
            num_valid = np.sum(valid_mask)

            # 3. Object-level aggregated features
            if num_valid > 0:
                valid_kinematics = obj_kinematics_norm[valid_mask]
                # Mean and std of normalized kinematics
                mean_features = np.mean(valid_kinematics, axis=0)
                std_features = np.std(valid_kinematics, axis=0)
                event_features.extend(mean_features)
                event_features.extend(std_features)

                # Original kinematics for physics features
                original_kinematics = obj_kinematics[valid_mask]
                pt = original_kinematics[:, 1]
                eta = original_kinematics[:, 2]
                phi = original_kinematics[:, 3]
                energy = original_kinematics[:, 0]

                # Basic statistics
                event_features.append(np.sum(pt))
                event_features.append(np.max(pt) if len(pt) > 0 else 0)
                event_features.append(np.min(pt) if len(pt) > 0 else 0)
                event_features.append(np.std(pt) if len(pt) > 0 else 0)

                # 4. Pairwise features
                if num_valid >= 2:
                    deltaR_values = []
                    mass_values = []
                    pairwise_values = []

                    for j in range(num_valid):
                        for k in range(j+1, num_valid):
                            delta_eta = eta[j] - eta[k]
                            delta_phi = phi[j] - phi[k]
                            # Handle phi periodicity
                            while delta_phi > math.pi:
                                delta_phi -= 2*math.pi
                            while delta_phi < -math.pi:
                                delta_phi += 2*math.pi

                            deltaR = math.sqrt(delta_eta**2 + delta_phi**2)
                            deltaR_values.append(deltaR)

                            # Invariant mass
                            m = math.sqrt(2*pt[j]*pt[k]*(math.cosh(delta_eta) - math.cos(delta_phi)))
                            mass_values.append(m)

                            # Additional pairwise features
                            pt_ratio = pt[j]/pt[k] if pt[k] > 0 else 0
                            energy_ratio = energy[j]/energy[k] if energy[k] > 0 else 0
                            pairwise_values.extend([pt_ratio, energy_ratio, abs(delta_eta), abs(delta_phi)])

                    if deltaR_values:
                        deltaR_norm = self.deltaR_scaler.transform(np.array(deltaR_values).reshape(-1, 1))
                        event_features.extend([np.mean(deltaR_norm), np.std(deltaR_norm)])

                        mass_norm = self.mass_scaler.transform(np.array(mass_values).reshape(-1, 1))
                        event_features.extend([np.mean(mass_norm), np.std(mass_norm)])

                        # Top N pairwise features
                        if pairwise_values:
                            pairwise_arr = np.array(pairwise_values).reshape(-1, 4)
                            pairwise_norm = self.pairwise_scaler.transform(pairwise_arr)
                            pairwise_flat = pairwise_norm.flatten()
                            # Take first 20 features or all if less
                            event_features.extend(pairwise_flat[:20])
                else:
                    # Pad with zeros if insufficient objects
                    event_features.extend([0, 0, 0, 0] + [0]*20)
            else:
                # Pad with zeros if no valid objects
                event_features.extend([0]*4 + [0]*4 + [0]*4 + [0]*24)

            output_features.append(event_features)

        # Pad to uniform length
        max_len = max(len(f) for f in output_features)
        padded_features = []
        for f in output_features:
            if len(f) < max_len:
                f.extend([0] * (max_len - len(f)))
            padded_features.append(f)

        return torch.tensor(padded_features, dtype=torch.float32)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[-1]

        # Enhanced architecture with residual connections
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
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
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

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
        # batch_x shape: [batch_size, feature_dim]
        return self.layers(batch_x)  # Output: [batch_size, 1]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_X.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).float().view(-1, 1)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_X.size(0)

                all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        # Calculate metrics
        epoch_train_loss = train_loss / train_total
        epoch_val_loss = val_loss / val_total
        epoch_train_acc = train_correct / train_total
        epoch_val_acc = val_correct / val_total

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(all_labels, all_probs)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
        print(f'  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val AUC: {auc_score:.4f}')

        # Learning rate scheduling
        scheduler.step(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

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


