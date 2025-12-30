
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
        self.obj_scalers = [RobustScaler() for _ in range(18)]
        self.global_features = [0, 1]
        self.per_object_features = list(range(2, 92))
        self.n_objects = 18
        self.features_per_object = 5

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True if torch.cuda.is_available() else False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Fit global features scaler
        global_data = X[:, self.global_features]
        self.scaler.fit(global_data)

        # Fit per-object scalers
        for i in range(self.n_objects):
            start_idx = 2 + i * self.features_per_object
            end_idx = start_idx + self.features_per_object
            obj_data = X[:, start_idx:end_idx]
            # Only scale kinematic features (skip object ID)
            self.obj_scalers[i].fit(obj_data[:, 1:])
        return self

    def transform(self, X):
        # Transform global features
        global_data = X[:, self.global_features]
        global_scaled = self.scaler.transform(global_data)
        X_transformed = X.clone()
        X_transformed[:, self.global_features] = torch.tensor(global_scaled, dtype=torch.float32)

        # Transform per-object features
        for i in range(self.n_objects):
            start_idx = 2 + i * self.features_per_object
            end_idx = start_idx + self.features_per_object
            obj_data = X[:, start_idx:end_idx]

            # Keep object ID as is
            obj_id = obj_data[:, 0:1]
            kinematic_data = obj_data[:, 1:]

            # Scale kinematic features
            kinematic_scaled = self.obj_scalers[i].transform(kinematic_data)
            kinematic_scaled = torch.tensor(kinematic_scaled, dtype=torch.float32)

            # Recombine
            obj_transformed = torch.cat([obj_id, kinematic_scaled], dim=1)
            X_transformed[:, start_idx:end_idx] = obj_transformed

        # Feature engineering: add object count and sum pT
        obj_ids = X_transformed[:, 2::5]  # Get all object IDs
        obj_mask = (obj_ids > 0).float()  # Mask for valid objects

        # Count objects per event
        n_objects = obj_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]

        # Sum pT of all objects
        pT_all = X_transformed[:, 4::5]  # Get all pT values
        sum_pT = (pT_all * obj_mask).sum(dim=1, keepdim=True)  # [batch_size, 1]

        # Add engineered features
        engineered_features = torch.cat([n_objects, sum_pT], dim=1)  # [batch_size, 2]
        X_transformed = torch.cat([X_transformed, engineered_features], dim=1)  # [batch_size, 94]

        return X_transformed

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_size = sample_object.shape[1]  # Should be 94 after preprocessing

        # Object-level processing
        self.obj_embed = nn.Sequential(
            nn.Linear(5, 32),  # Each object has 5 features (ID + 4 kinematic)
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Global processing
        self.global_mlp = nn.Sequential(
            nn.Linear(2, 32),  # 2 global features (E_T_miss, phi_Et_miss)
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Engineered features processing
        self.engineered_mlp = nn.Sequential(
            nn.Linear(2, 32),  # 2 engineered features (n_objects, sum_pT)
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 64 + 64, 256),  # Combined features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, batch_x):
        # batch_x shape: [batch_size, 94]

        # Extract different feature types
        global_feats = batch_x[:, :2]  # [batch_size, 2]
        engineered_feats = batch_x[:, -2:]  # [batch_size, 2]
        obj_feats = batch_x[:, 2:-2]  # [batch_size, 90] (18 objects * 5 features)

        # Process global features
        global_processed = self.global_mlp(global_feats)  # [batch_size, 64]

        # Process engineered features
        engineered_processed = self.engineered_mlp(engineered_feats)  # [batch_size, 64]

        # Process object features
        obj_feats = obj_feats.view(-1, 18, 5)  # [batch_size, 18, 5]
        obj_processed = self.obj_embed(obj_feats)  # [batch_size, 18, 64]

        # Attention over objects
        attention_weights = self.attention(obj_processed)  # [batch_size, 18, 1]
        obj_attended = (obj_processed * attention_weights).sum(dim=1)  # [batch_size, 64]

        # Combine all features
        combined = torch.cat([global_processed, engineered_processed, obj_attended], dim=1)  # [batch_size, 192]

        # Final classification
        output = self.classifier(combined)  # [batch_size, 1]
        return output

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.BCELoss()

    best_auc = 0.0
    best_model = None
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

        # Training loop
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_y).sum().item()
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
                batch_x = batch_x.to(device)
                batch_y = batch_y.float().to(device).unsqueeze(1)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_loss.append(val_running_loss / len(val_loader))
        val_acc.append(val_correct / val_total)

        # Calculate AUC
        auc_score = roc_auc_score(all_labels, all_preds)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, "
              f"Train Acc: {train_acc[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}, AUC: {auc_score:.4f}")

        # Early stopping based on AUC
        scheduler.step(auc_score)

        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_loss, val_loss, train_acc, val_acc

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


