
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
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.obj_ids = [0, 1, 2, 3, 4, 5, 6, 21, 22]  # Common object IDs in particle physics
        self.n_objects = 18
        self.obj_feature_size = 5
        self.global_features = 2

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
        # Extract global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :2].numpy()  # [N, 2]
        self.scaler.fit(global_features)
        return self

    def transform(self, X):
        # Convert to numpy for sklearn
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Scale global features
        global_features = X_np[:, :2]  # [N, 2]
        global_features_scaled = self.scaler.transform(global_features)

        # Process object features
        object_features = []
        for i in range(self.n_objects):
            start_idx = self.global_features + i * self.obj_feature_size
            end_idx = start_idx + self.obj_feature_size
            obj_slice = X_np[:, start_idx:end_idx]  # [N, 5]

            # Extract kinematic features (E, p_T, eta, phi) for non-zero objects
            obj_id = obj_slice[:, 0]  # [N]
            kinematic_features = obj_slice[:, 1:]  # [N, 4]

            # Create mask for valid objects (non-zero obj_id)
            valid_mask = (obj_id != 0).astype(np.float32)
            valid_mask = valid_mask.reshape(-1, 1)  # [N, 1]

            # Normalize kinematic features
            kinematic_features = np.where(valid_mask > 0, kinematic_features, 0)

            # Calculate derived features
            pt = kinematic_features[:, 1]  # p_T
            eta = kinematic_features[:, 2]  # eta
            phi = kinematic_features[:, 3]  # phi

            # Transverse mass for objects
            mt = np.sqrt(2 * pt * global_features[:, 0] * (1 - np.cos(phi - global_features[:, 1])))
            mt = np.where(valid_mask[:, 0] > 0, mt, 0).reshape(-1, 1)

            # Rapidity
            rapidity = 0.5 * np.log((kinematic_features[:, 0] + pt) / (kinematic_features[:, 0] - pt))
            rapidity = np.where(valid_mask[:, 0] > 0, rapidity, 0).reshape(-1, 1)

            # Combine all features
            derived_features = np.hstack([
                kinematic_features,
                mt,
                rapidity,
                valid_mask
            ])  # [N, 6]

            object_features.append(derived_features)

        # Stack all features
        object_features = np.stack(object_features, axis=1)  # [N, 18, 6]
        global_features_scaled = global_features_scaled.reshape(-1, 2)  # [N, 2]

        # Flatten object features and concatenate with global features
        object_features_flat = object_features.reshape(object_features.shape[0], -1)  # [N, 108]
        processed_features = np.hstack([global_features_scaled, object_features_flat])  # [N, 110]

        return torch.from_numpy(processed_features).float()

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

        # Attention mechanism for object features
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, batch_x):
        # batch_x: [B, F] where F = 110

        # Feature extraction
        features = self.feature_extractor(batch_x)  # [B, 128]

        # Attention over object features (assuming 18 objects with 6 features each)
        # Reshape to separate object features
        obj_features = features[:, 2:].reshape(-1, 18, 6)  # [B, 18, 6]
        global_features = features[:, :2]  # [B, 2]

        # Apply attention
        attention_weights = self.attention(obj_features)  # [B, 18, 1]
        attended_features = torch.sum(attention_weights * obj_features, dim=1)  # [B, 6]

        # Combine with global features
        combined = torch.cat([global_features, attended_features], dim=1)  # [B, 8]

        # Classification
        logits = self.classifier(combined)  # [B, 1]

        return logits.squeeze(1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Training metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_auc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == batch_y.float()).sum().item()
            total += batch_y.size(0)

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(correct / total)

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.float())
                val_running_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == batch_y.float()).sum().item()
                val_total += batch_y.size(0)

                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_loss.append(val_running_loss / len(val_loader))
        val_acc.append(val_correct / val_total)

        # Calculate AUC
        auc_score = roc_auc_score(all_labels, all_preds)
        scheduler.step(auc_score)

        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f} - '
              f'Val Loss: {val_loss[-1]:.4f}, Val Acc: {val_acc[-1]:.4f} - '
              f'AUC: {auc_score:.4f}')

        # Early stopping and model checkpointing
        if auc_score > best_auc:
            best_auc = auc_score
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered")
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

