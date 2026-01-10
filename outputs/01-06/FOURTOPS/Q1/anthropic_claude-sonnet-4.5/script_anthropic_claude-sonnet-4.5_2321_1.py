
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
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        X_np = X.cpu().numpy() if torch.is_tensor(X) else X

        # Global features: indices 0-1 (MET magnitude and phi)
        global_features = X_np[:, :2]
        self.global_mean = np.mean(global_features, axis=0)
        self.global_std = np.std(global_features, axis=0) + 1e-8

        # Object features: indices 2-91 (18 objects × 5 features)
        obj_features = X_np[:, 2:].reshape(-1, 18, 5)
        mask = obj_features[:, :, 0] != 0  # Valid objects have non-zero obj_id

        self.obj_mean = np.zeros(5)
        self.obj_std = np.zeros(5)

        for i in range(5):
            valid_values = obj_features[:, :, i][mask]
            if len(valid_values) > 0:
                self.obj_mean[i] = np.mean(valid_values)
                self.obj_std[i] = np.std(valid_values) + 1e-8
            else:
                self.obj_mean[i] = 0.0
                self.obj_std[i] = 1.0

        return self

    def transform(self, X):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)

        X_out = X.clone()

        # Normalize global features
        global_mean_t = torch.tensor(self.global_mean, dtype=X.dtype, device=X.device)
        global_std_t = torch.tensor(self.global_std, dtype=X.dtype, device=X.device)
        X_out[:, :2] = (X[:, :2] - global_mean_t) / global_std_t

        # Get mask before normalization
        obj_features = X[:, 2:].reshape(-1, 18, 5)
        mask = (obj_features[:, :, 0] != 0).unsqueeze(-1).float()

        # Normalize object features
        obj_mean_t = torch.tensor(self.obj_mean, dtype=X.dtype, device=X.device)
        obj_std_t = torch.tensor(self.obj_std, dtype=X.dtype, device=X.device)

        obj_normalized = (obj_features - obj_mean_t) / obj_std_t
        obj_normalized = obj_normalized * mask  # Zero out padding

        X_out[:, 2:] = obj_normalized.reshape(-1, 90)

        return X_out

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        self.obj_feat_dim = 4  # E, pT, eta, phi (excluding obj_id)
        self.hidden_dim = 128

        # Object encoder - processes each object independently
        self.obj_encoder = nn.Sequential(
            nn.Linear(self.obj_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Attention mechanism for object aggregation
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Global feature encoder (MET)
        self.global_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x: [B, 92]
        batch_size = batch_x.shape[0]

        # Extract global features: [B, 2]
        global_feat = batch_x[:, :2]

        # Extract object features: [B, 18, 5]
        obj_features = batch_x[:, 2:].reshape(batch_size, 18, 5)

        # Compute mask - check if energy is non-zero: [B, 18]
        mask = torch.abs(obj_features[:, :, 1]) > 1e-6

        # Use kinematic features only (skip obj_id): [B, 18, 4]
        obj_features = obj_features[:, :, 1:]

        # Encode each object: [B*18, 4] -> [B*18, hidden_dim] -> [B, 18, hidden_dim]
        obj_flat = obj_features.reshape(batch_size * 18, self.obj_feat_dim)
        obj_encoded = self.obj_encoder(obj_flat)
        obj_encoded = obj_encoded.reshape(batch_size, 18, self.hidden_dim)

        # Compute attention scores: [B*18, hidden_dim] -> [B*18, 1] -> [B, 18]
        attention_logits = self.attention(obj_encoded.reshape(batch_size * 18, self.hidden_dim))
        attention_logits = attention_logits.reshape(batch_size, 18)

        # Mask out invalid objects: [B, 18]
        attention_logits = attention_logits.masked_fill(~mask, -1e9)
        attention_weights = F.softmax(attention_logits, dim=1)

        # Aggregate objects: [B, hidden_dim]
        obj_aggregated = torch.sum(obj_encoded * attention_weights.unsqueeze(-1), dim=1)

        # Encode global features: [B, 64]
        global_encoded = self.global_encoder(global_feat)

        # Concatenate and classify: [B, hidden_dim + 64] -> [B, 1] -> [B]
        combined = torch.cat([obj_aggregated, global_encoded], dim=1)
        output = self.classifier(combined)

        return output.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 15

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_loss = float('inf')
    best_model_state = None
    patience = 7
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
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

