
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
        self.obj_ids = None
        self.n_objects = 18
        self.n_features_per_obj = 5
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
                                "batch_size": 1024}
        }

    def fit(self, X, y=None):
        # Extract global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :self.global_features].numpy()
        self.scaler.fit(global_features)

        # Identify unique object IDs and their frequency
        obj_ids = set()
        for i in range(self.n_objects):
            start_idx = self.global_features + i * self.n_features_per_obj
            obj_ids.update(X[:, start_idx].unique().numpy())
        self.obj_ids = sorted(obj_ids)

        return self

    def transform(self, X):
        # Convert to numpy for sklearn
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Scale global features
        global_features = X_np[:, :self.global_features]
        global_features_scaled = self.scaler.transform(global_features)

        # Process object features
        processed_objects = []
        for i in range(self.n_objects):
            start_idx = self.global_features + i * self.n_features_per_obj
            obj_id = X_np[:, start_idx:start_idx+1]
            kinematics = X_np[:, start_idx+1:start_idx+5]

            # Create mask for valid objects (non-zero obj_id)
            valid_mask = (obj_id != 0).astype(np.float32)

            # Scale kinematic features
            kinematics_scaled = self.scaler.transform(kinematics) if i == 0 else kinematics
            if i == 0:
                self.kinematic_scaler = RobustScaler().fit(kinematics)
                kinematics_scaled = self.kinematic_scaler.transform(kinematics)
            else:
                kinematics_scaled = self.kinematic_scaler.transform(kinematics)

            # Combine with mask
            processed_obj = np.concatenate([
                obj_id,
                kinematics_scaled,
                valid_mask
            ], axis=1)
            processed_objects.append(processed_obj)

        # Stack all features
        processed_objects = np.stack(processed_objects, axis=1)  # [N, 18, 6]
        processed_global = np.concatenate([
            global_features_scaled,
            np.zeros((global_features_scaled.shape[0], 1))  # dummy for consistency
        ], axis=1)  # [N, 3]

        # Combine global and object features
        final_features = np.concatenate([
            processed_global,
            processed_objects.reshape(processed_objects.shape[0], -1)
        ], axis=1)  # [N, 3 + 18*6 = 111]

        return torch.from_numpy(final_features).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class ObjectLevelFeatures(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.obj_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, 18, 6]
        B, N, _ = x.shape
        x = x.view(B * N, -1)  # [B*18, 6]
        x = self.obj_encoder(x)  # [B*18, hidden_dim]
        x = x.view(B, N, -1)  # [B, 18, hidden_dim]
        return x

class GlobalFeatures(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.global_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, 3]
        return self.global_encoder(x)  # [B, hidden_dim]

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, x):
        # x: [B, N, hidden_dim]
        attention_weights = self.attention(x)  # [B, N, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        pooled = torch.sum(attention_weights * x, dim=1)  # [B, hidden_dim]
        return pooled

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Determine input dimensions from sample
        if len(sample_object.shape) == 2:
            # Dense batch mode
            B, total_features = sample_object.shape
            global_features = 3
            obj_features = (total_features - global_features) // 18
        else:
            raise ValueError("Unsupported input shape")

        self.global_encoder = GlobalFeatures(global_features, 64)
        self.obj_encoder = ObjectLevelFeatures(obj_features, 128)
        self.attention = AttentionPooling(128)

        self.classifier = nn.Sequential(
            nn.Linear(64 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, batch_x):
        # batch_x: [B, 111]
        B = batch_x.shape[0]

        # Split global and object features
        global_feats = batch_x[:, :3]  # [B, 3]
        obj_feats = batch_x[:, 3:].view(B, 18, -1)  # [B, 18, 6]

        # Process features
        global_encoded = self.global_encoder(global_feats)  # [B, 64]
        obj_encoded = self.obj_encoder(obj_feats)  # [B, 18, 128]
        obj_pooled = self.attention(obj_encoded)  # [B, 128]

        # Combine features
        combined = torch.cat([global_encoded, obj_pooled], dim=1)  # [B, 192]

        # Classify
        logits = self.classifier(combined)  # [B, 1]
        return logits.squeeze(1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    best_model = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(batch_y.cpu().numpy())

        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_targets, train_preds)
        train_acc = np.mean((np.array(train_preds) > 0.5) == np.array(train_targets))

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y.float())
                val_loss += loss.item()
                preds = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_acc = np.mean((np.array(val_preds) > 0.5) == np.array(val_targets))

        # Update learning rate
        scheduler.step(val_auc)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print("Early stopping triggered")
                break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_losses, val_losses, train_accs, val_accs

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

