
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
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Create mask for non-zero objects (first feature of each object is obj_id)
        obj_ids = X[:, 2::5]  # Shape: [n_events, 18]
        self.mask = (obj_ids != 0).float()  # Shape: [n_events, 18]

        # Flatten all object features for scaling (excluding global features)
        obj_features = X[:, 2:].reshape(-1, 5)  # Shape: [n_events*18, 5]
        valid_mask = (obj_features[:, 0] != 0)  # Mask for valid objects
        valid_features = obj_features[valid_mask, 1:]  # Shape: [n_valid, 4]

        if len(valid_features) > 0:
            self.scaler.fit(valid_features)
        return self

    def transform(self, X):
        # Apply scaling to object features
        obj_features = X[:, 2:].reshape(-1, 5)  # Shape: [n_events*18, 5]
        valid_mask = (obj_features[:, 0] != 0)
        scaled_features = obj_features.clone()

        if hasattr(self.scaler, 'mean_'):
            valid_features = obj_features[valid_mask, 1:]
            scaled_valid = self.scaler.transform(valid_features)
            scaled_features[valid_mask, 1:] = torch.from_numpy(scaled_valid).float()

        # Reshape back
        scaled_obj_features = scaled_features.reshape(X.shape[0], -1)  # Shape: [n_events, 90]

        # Combine with global features (first 2 features)
        global_features = X[:, :2]
        processed = torch.cat([global_features, scaled_obj_features], dim=1)  # Shape: [n_events, 92]

        return processed

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        self.global_dim = 2
        self.obj_dim = 4  # E, pT, eta, phi (excluding obj_id)
        self.max_objects = 18

        # Global feature processing
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Object feature processing
        self.obj_encoder = nn.Sequential(
            nn.Linear(self.obj_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(16 + 16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, batch_x):
        # batch_x shape: [batch_size, 92]
        batch_size = batch_x.shape[0]

        # Process global features (first 2 features)
        global_feats = batch_x[:, :2]  # Shape: [batch_size, 2]
        global_encoded = self.global_encoder(global_feats)  # Shape: [batch_size, 16]

        # Process object features (remaining 90 features)
        obj_feats = batch_x[:, 2:].reshape(batch_size, self.max_objects, 5)  # Shape: [batch_size, 18, 5]
        obj_ids = obj_feats[:, :, 0]  # Shape: [batch_size, 18]
        obj_values = obj_feats[:, :, 1:]  # Shape: [batch_size, 18, 4]

        # Mask for valid objects
        mask = (obj_ids != 0).float()  # Shape: [batch_size, 18]
        mask = mask.unsqueeze(-1)  # Shape: [batch_size, 18, 1]

        # Encode objects
        obj_encoded = self.obj_encoder(obj_values)  # Shape: [batch_size, 18, 16]

        # Apply attention
        attention_scores = self.attention(obj_encoded).squeeze(-1)  # Shape: [batch_size, 18]
        attention_scores = attention_scores.masked_fill(mask.squeeze(-1) == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)  # Shape: [batch_size, 18]
        attention_weights = attention_weights.unsqueeze(1)  # Shape: [batch_size, 1, 18]

        # Weighted sum of object features
        context = torch.bmm(attention_weights, obj_encoded).squeeze(1)  # Shape: [batch_size, 16]

        # Combine global and context features
        combined = torch.cat([global_encoded, context], dim=1)  # Shape: [batch_size, 32]

        # Final classification
        logits = self.classifier(combined)  # Shape: [batch_size, 1]
        return logits.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=False)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val

        # Update scheduler
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                model.load_state_dict(best_model)
                break

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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


