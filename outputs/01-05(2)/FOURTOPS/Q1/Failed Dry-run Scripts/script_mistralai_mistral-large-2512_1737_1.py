
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
        self.obj_feature_size = 5
        self.global_feature_size = 2
        self.total_features = self.global_feature_size + self.n_objects * self.obj_feature_size

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
        global_features = X[:, :2].numpy()  # [N, 2]

        # Extract object features (obj_id, E, p_T, eta, phi)
        object_features = []
        for i in range(self.n_objects):
            start_idx = 2 + i * self.obj_feature_size
            end_idx = start_idx + self.obj_feature_size
            obj_feats = X[:, start_idx:end_idx]  # [N, 5]
            object_features.append(obj_feats.numpy())

        # Stack all features for scaling
        all_features = np.hstack([global_features] + object_features)  # [N, 2 + 18*5]

        # Fit scaler on all features
        self.scaler.fit(all_features)

        # Store object IDs for reference (though we'll scale them too)
        self.obj_ids = np.unique(X[:, 2::5].numpy())
        return self

    def transform(self, X):
        # Convert to numpy if not already
        if torch.is_tensor(X):
            X_np = X.numpy()
        else:
            X_np = X

        # Extract global features
        global_features = X_np[:, :2]  # [N, 2]

        # Extract object features
        object_features = []
        for i in range(self.n_objects):
            start_idx = 2 + i * self.obj_feature_size
            end_idx = start_idx + self.obj_feature_size
            obj_feats = X_np[:, start_idx:end_idx]  # [N, 5]
            object_features.append(obj_feats)

        # Stack all features
        all_features = np.hstack([global_features] + object_features)  # [N, 92]

        # Transform features
        all_features_scaled = self.scaler.transform(all_features)

        # Reconstruct the original structure
        output = np.zeros_like(X_np)
        output[:, :2] = all_features_scaled[:, :2]  # Global features

        for i in range(self.n_objects):
            start_idx = 2 + i * self.obj_feature_size
            end_idx = start_idx + self.obj_feature_size
            output[:, start_idx:end_idx] = all_features_scaled[:, 2 + i*5:2 + (i+1)*5]

        return torch.from_numpy(output).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Determine input size from sample
        input_size = sample_object.shape[1]  # [F]

        # Global features branch (first 2 features)
        self.global_branch = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Object features branch (remaining features)
        # Each object has 5 features, 18 objects
        self.obj_branch = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2)
        )

        # Attention mechanism for objects
        self.attention = nn.Sequential(
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Softmax(dim=1)
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(16 + 16, 64),  # global + object features
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, batch_x):
        # batch_x: [B, 92]

        # Extract global features (first 2 features)
        global_feats = batch_x[:, :2]  # [B, 2]
        global_out = self.global_branch(global_feats)  # [B, 16]

        # Extract object features (18 objects, 5 features each)
        obj_feats = batch_x[:, 2:].reshape(-1, 18, 5)  # [B, 18, 5]

        # Process each object through shared network
        obj_out = self.obj_branch(obj_feats)  # [B, 18, 16]

        # Apply attention to objects
        attention_weights = self.attention(obj_out)  # [B, 18, 1]
        attended_obj = torch.sum(attention_weights * obj_out, dim=1)  # [B, 16]

        # Combine global and object features
        combined = torch.cat([global_out, attended_obj], dim=1)  # [B, 32]

        # Final classification
        logits = self.classifier(combined)  # [B, 1]

        return logits.squeeze(1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=False)

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
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = F.binary_cross_entropy_with_logits(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == batch_y.float()).sum().item()
            total += batch_y.size(0)

        train_loss_epoch = running_loss / len(train_loader)
        train_acc_epoch = correct / total
        train_loss.append(train_loss_epoch)
        train_acc.append(train_acc_epoch)

        # Validation
        model.eval()
        val_loss_epoch = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = F.binary_cross_entropy_with_logits(outputs, batch_y.float())
                val_loss_epoch += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == batch_y.float()).sum().item()
                val_total += batch_y.size(0)

                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_loss_epoch /= len(val_loader)
        val_acc_epoch = val_correct / val_total
        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)

        # Calculate AUC
        auc_score = roc_auc_score(all_labels, all_preds)

        # Update scheduler
        scheduler.step(auc_score)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}, "
              f"Train Acc: {train_acc_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}, AUC: {auc_score:.4f}")

        # Early stopping based on AUC
        if auc_score > best_auc:
            best_auc = auc_score
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model state
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

