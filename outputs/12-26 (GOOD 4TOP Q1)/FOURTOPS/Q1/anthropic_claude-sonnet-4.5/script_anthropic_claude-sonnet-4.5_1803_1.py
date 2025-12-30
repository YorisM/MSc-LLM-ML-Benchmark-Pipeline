
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
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

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

# ---------- IMPORTS ----------
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import copy

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.obj_mean = None
        self.obj_std = None

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
        # X shape: [N, 92]
        # Extract global features (first 2 columns)
        global_feats = X[:, :2]  # [N, 2]

        # Extract object features (remaining 90 columns, reshape to [N, 18, 5])
        obj_feats = X[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]

        # Create mask for valid objects (non-padding)
        obj_mask = (obj_feats.abs().sum(dim=2) > 0).unsqueeze(2)  # [N, 18, 1]

        # Compute statistics for global features
        self.global_mean = global_feats.mean(dim=0)  # [2]
        self.global_std = global_feats.std(dim=0) + 1e-8  # [2]

        # Compute statistics for object features (only on valid objects)
        valid_obj_feats = obj_feats[obj_mask.squeeze(2)]  # [M, 5]
        if valid_obj_feats.shape[0] > 0:
            self.obj_mean = valid_obj_feats.mean(dim=0)  # [5]
            self.obj_std = valid_obj_feats.std(dim=0) + 1e-8  # [5]
        else:
            self.obj_mean = torch.zeros(5)
            self.obj_std = torch.ones(5)

        return self

    def transform(self, X):
        # X shape: [N, 92]
        X_out = X.clone()

        # Normalize global features
        X_out[:, :2] = (X_out[:, :2] - self.global_mean) / self.global_std

        # Normalize object features
        obj_feats = X_out[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]
        obj_mask = (obj_feats.abs().sum(dim=2, keepdim=True) > 0)  # [N, 18, 1]

        # Apply normalization only to non-padding entries
        obj_feats_norm = (obj_feats - self.obj_mean) / self.obj_std
        obj_feats = torch.where(obj_mask, obj_feats_norm, torch.zeros_like(obj_feats))

        X_out[:, 2:] = obj_feats.reshape(-1, 90)

        return X_out

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Per-object encoder (processes each object independently)
        self.obj_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
        )

        # Attention mechanism for object aggregation
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, batch_x):
        # batch_x shape: [B, 92]
        batch_size = batch_x.shape[0]

        # Extract global features
        global_feats = batch_x[:, :2]  # [B, 2]

        # Extract and reshape object features
        obj_feats = batch_x[:, 2:].reshape(batch_size, 18, 5)  # [B, 18, 5]

        # Create mask for valid objects (non-padding)
        obj_mask = (obj_feats.abs().sum(dim=2) > 0)  # [B, 18]

        # Encode each object
        obj_encoded = self.obj_encoder(obj_feats)  # [B, 18, 64]

        # Compute attention scores and weights
        attention_scores = self.attention(obj_encoded).squeeze(-1)  # [B, 18]

        # Mask out padding objects
        attention_scores = attention_scores.masked_fill(~obj_mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, 18]

        # Aggregate objects using attention weights
        obj_aggregated = (obj_encoded * attention_weights.unsqueeze(-1)).sum(dim=1)  # [B, 64]

        # Encode global features
        global_encoded = self.global_encoder(global_feats)  # [B, 64]

        # Concatenate and classify
        combined = torch.cat([obj_aggregated, global_encoded], dim=1)  # [B, 128]
        logits = self.classifier(combined).squeeze(-1)  # [B]

        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Tracking metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Early stopping
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    patience = 7

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            # Forward pass
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.float())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item() * xb.shape[0]
            preds = (torch.sigmoid(outputs) > 0.5).long()
            train_correct += (preds == yb).sum().item()
            train_total += xb.shape[0]

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                # Forward pass
                outputs = model(xb)
                loss = criterion(outputs, yb.float())

                # Track metrics
                val_loss += loss.item() * xb.shape[0]
                preds = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (preds == yb).sum().item()
                val_total += xb.shape[0]

                # Store predictions for AUC computation
                all_val_preds.append(torch.sigmoid(outputs).cpu())
                all_val_labels.append(yb.cpu())

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Compute AUC
        all_val_preds = torch.cat(all_val_preds).numpy()
        all_val_labels = torch.cat(all_val_labels).numpy()
        val_auc = roc_auc_score(all_val_labels, all_val_preds)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

        # Learning rate scheduling based on validation AUC
        scheduler.step(val_auc)

        # Early stopping based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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

