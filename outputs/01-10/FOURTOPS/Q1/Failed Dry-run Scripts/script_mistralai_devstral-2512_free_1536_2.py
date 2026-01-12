
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
from sklearn.preprocessing import StandardScaler
from torch.nn import Linear, ReLU, Dropout, BatchNorm1d, Sequential, Sigmoid
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn.functional as F

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.mask = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 1024}
        }

    def fit(self, X, y=None):
        # Create mask for non-zero objects (first feature of each object is obj_id)
        obj_mask = (X[:, 2::5] != 0).float()  # [N, 18]
        self.mask = obj_mask

        # Flatten all object features for scaling (excluding global features)
        obj_features = X[:, 2:].reshape(X.shape[0], -1)  # [N, 18*5]
        self.scaler.fit(obj_features)
        return self

    def transform(self, X):
        # Apply scaling to object features
        obj_features = X[:, 2:].reshape(X.shape[0], -1)  # [N, 18*5]
        scaled_features = self.scaler.transform(obj_features)

        # Reshape back and combine with global features
        X_scaled = X.clone()
        X_scaled[:, 2:] = scaled_features.reshape(X.shape[0], -1)

        # Create mask for non-padded objects
        obj_mask = (X[:, 2::5] != 0).float()  # [N, 18]
        return X_scaled, obj_mask

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        x, mask = sample_object
        num_objects = 18
        obj_feature_dim = 5

        # Global features processing
        self.global_mlp = Sequential(
            Linear(2, 32),
            ReLU(),
            BatchNorm1d(32),
            Dropout(0.2)
        )

        # Object features processing
        self.obj_mlp = Sequential(
            Linear(obj_feature_dim, 64),
            ReLU(),
            BatchNorm1d(64),
            Dropout(0.3),
            Linear(64, 32),
            ReLU(),
            BatchNorm1d(32),
            Dropout(0.3)
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            Linear(32 + 32, 64),
            ReLU(),
            Linear(64, 1),
            nn.Tanh()
        )

        # Final classifier
        self.classifier = Sequential(
            Linear(32 + 32, 128),
            ReLU(),
            BatchNorm1d(128),
            Dropout(0.4),
            Linear(128, 64),
            ReLU(),
            BatchNorm1d(64),
            Dropout(0.4),
            Linear(64, 1)
        )

    def forward(self, batch_x):
        x, mask = batch_x
        batch_size = x.shape[0]

        # Process global features [B, 2]
        global_feat = self.global_mlp(x[:, :2])  # [B, 32]

        # Process object features [B, 18, 5]
        obj_feats = x[:, 2:].view(batch_size, 18, 5)  # [B, 18, 5]
        obj_feats = self.obj_mlp(obj_feats)  # [B, 18, 32]

        # Apply mask to zero out padded objects
        obj_mask = mask.unsqueeze(-1)  # [B, 18, 1]
        obj_feats = obj_feats * obj_mask

        # Compute attention weights
        global_expanded = global_feat.unsqueeze(1).expand(-1, 18, -1)  # [B, 18, 32]
        combined = torch.cat([obj_feats, global_expanded], dim=-1)  # [B, 18, 64]
        attention_weights = self.attention(combined).squeeze(-1)  # [B, 18]
        attention_weights = F.softmax(attention_weights, dim=1)  # [B, 18]

        # Weighted sum of object features
        weighted_obj = torch.sum(attention_weights.unsqueeze(-1) * obj_feats, dim=1)  # [B, 32]

        # Combine with global features
        combined_features = torch.cat([global_feat, weighted_obj], dim=1)  # [B, 64]

        # Final classification
        logits = self.classifier(combined_features).squeeze(-1)  # [B]
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)
    criterion = BCEWithLogitsLoss()

    best_val_auc = 0
    best_model = None

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            x, y = batch
            x, mask = x
            x, y = x.to(device), y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            logits = model((x, mask))
            loss = criterion(logits, y.float())

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = (logits > 0).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, mask = x
                x, y = x.to(device), y.to(device)
                mask = mask.to(device)

                logits = model((x, mask))
                loss = criterion(logits, y.float())

                val_loss += loss.item() * y.size(0)
                preds = (logits > 0).float()
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

                all_preds.extend(logits.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_labels, all_preds)

        # Early stopping and model saving
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model.state_dict()
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

