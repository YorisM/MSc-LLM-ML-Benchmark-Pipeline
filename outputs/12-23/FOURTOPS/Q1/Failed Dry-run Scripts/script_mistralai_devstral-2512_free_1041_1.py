
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

# ---------- IMPORTS ----------
from sklearn.preprocessing import StandardScaler
from torch.nn import Linear, ReLU, Dropout, BatchNorm1d, Sequential, Sigmoid
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Create mask for non-zero objects (first feature of each object is obj_id)
        obj_mask = (X[:, 2::5] != 0).float()  # Shape: [n_events, 18]
        self.mask = obj_mask

        # Flatten all object features for scaling (excluding global features)
        obj_features = X[:, 2:].reshape(X.shape[0], -1)  # Shape: [n_events, 90]
        self.scaler.fit(obj_features)
        return self

    def transform(self, X):
        # Apply scaling to object features
        obj_features = X[:, 2:].reshape(X.shape[0], -1)  # Shape: [n_events, 90]
        scaled_features = self.scaler.transform(obj_features)
        X[:, 2:] = scaled_features.reshape(X.shape[0], -1)

        # Add mask as additional features
        obj_mask = (X[:, 2::5] != 0).float()  # Shape: [n_events, 18]
        X = torch.cat([X, obj_mask], dim=1)  # Shape: [n_events, 110]
        return X

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[1]  # Should be 110 after preprocessing

        # Feature extraction for global features
        self.global_net = Sequential(
            Linear(2, 16),
            ReLU(),
            BatchNorm1d(16),
            Dropout(0.2)
        )

        # Feature extraction for object features
        self.obj_net = Sequential(
            Linear(5, 32),
            ReLU(),
            BatchNorm1d(32),
            Dropout(0.2)
        )

        # Attention mechanism
        self.attention = Sequential(
            Linear(32, 16),
            ReLU(),
            Linear(16, 1),
            Sigmoid()
        )

        # Final classifier
        self.classifier = Sequential(
            Linear(16 + 32 + 18, 128),  # 16 (global) + 32 (obj) + 18 (mask)
            ReLU(),
            BatchNorm1d(128),
            Dropout(0.3),
            Linear(128, 64),
            ReLU(),
            BatchNorm1d(64),
            Dropout(0.3),
            Linear(64, 1),
            Sigmoid()
        )

    def forward(self, batch_x):
        # Split features
        global_feat = batch_x[:, :2]  # Shape: [batch, 2]
        obj_feats = batch_x[:, 2:102].reshape(-1, 18, 5)  # Shape: [batch, 18, 5]
        mask = batch_x[:, 102:]  # Shape: [batch, 18]

        # Process global features
        global_out = self.global_net(global_feat)  # Shape: [batch, 16]

        # Process object features
        obj_out = self.obj_net(obj_feats)  # Shape: [batch, 18, 32]

        # Apply attention
        attention_weights = self.attention(obj_out)  # Shape: [batch, 18, 1]
        weighted_obj = (obj_out * attention_weights).sum(dim=1)  # Shape: [batch, 32]

        # Combine features
        combined = torch.cat([global_out, weighted_obj, mask], dim=1)  # Shape: [batch, 66]

        # Final classification
        return self.classifier(combined).squeeze(1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    best_val_auc = 0
    patience = 10
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = (outputs > 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = (outputs > 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_labels, all_preds)

        # Update scheduler
        scheduler.step(val_auc)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping and model saving
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(best_model)

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


