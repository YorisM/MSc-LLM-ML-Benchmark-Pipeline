
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
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.obj_scaler = StandardScaler()
        self.fitted = False

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
        # Separate global and object features
        global_features = X[:, :2]  # E_T_miss, phi_Et_miss
        obj_features = X[:, 2:].reshape(-1, 18, 5)[:, :, 1:]  # Remove obj_id, keep E, pT, eta, phi

        # Flatten for scaling
        global_flat = global_features.reshape(-1, 2)
        obj_flat = obj_features.reshape(-1, 4)

        # Fit scalers
        self.global_scaler.fit(global_flat)
        self.obj_scaler.fit(obj_flat)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted")

        # Process global features
        global_features = X[:, :2]
        global_scaled = self.global_scaler.transform(global_features)

        # Process object features
        obj_features = X[:, 2:].reshape(-1, 18, 5)
        obj_ids = obj_features[:, :, 0]  # Keep object IDs
        obj_kinematics = obj_features[:, :, 1:]  # E, pT, eta, phi

        # Scale kinematics
        obj_flat = obj_kinematics.reshape(-1, 4)
        obj_scaled = self.obj_scaler.transform(obj_flat).reshape(-1, 18, 4)

        # Combine back
        processed = torch.zeros_like(X)
        processed[:, :2] = torch.from_numpy(global_scaled).float()
        for i in range(18):
            start_idx = 2 + i * 5
            processed[:, start_idx] = obj_ids[:, i]
            processed[:, start_idx+1:start_idx+5] = torch.from_numpy(obj_scaled[:, i, :]).float()

        return processed

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        batch_size, num_features = sample_object.shape
        self.num_objects = 18
        self.global_dim = 2
        self.obj_dim = 4  # E, pT, eta, phi (after removing obj_id)

        # Object embedding
        self.obj_embed = nn.Linear(self.obj_dim, 64)

        # Transformer layers for object interactions
        self.transformer1 = TransformerConv(64, 64, heads=4, concat=False)
        self.transformer2 = TransformerConv(64, 64, heads=4, concat=False)

        # Global feature processing
        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Attention for global-object fusion
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute_pairwise_features(self, obj_features):
        # obj_features: [batch_size, num_objects, obj_dim]
        batch_size = obj_features.size(0)

        # Compute invariant mass and delta R for all pairs
        pairwise = []
        for i in range(self.num_objects):
            for j in range(i+1, self.num_objects):
                # Get features for objects i and j
                obj_i = obj_features[:, i, :]  # [batch_size, 4]
                obj_j = obj_features[:, j, :]  # [batch_size, 4]

                # Compute invariant mass (simplified approximation)
                E_i, pT_i, eta_i, phi_i = obj_i[:, 0], obj_i[:, 1], obj_i[:, 2], obj_i[:, 3]
                E_j, pT_j, eta_j, phi_j = obj_j[:, 0], obj_j[:, 1], obj_j[:, 2], obj_j[:, 3]

                # Approximate invariant mass (using transverse components)
                m_ij = torch.sqrt(2 * pT_i * pT_j * (torch.cosh(eta_i - eta_j) - torch.cos(phi_i - phi_j)))

                # Compute delta R
                deta = eta_i - eta_j
                dphi = torch.abs(phi_i - phi_j)
                dphi = torch.min(dphi, 2*math.pi - dphi)
                dR = torch.sqrt(deta**2 + dphi**2)

                pairwise.append(torch.stack([m_ij, dR], dim=1))

        # Stack all pairwise features
        if pairwise:
            pairwise_features = torch.stack(pairwise, dim=1)  # [batch_size, num_pairs, 2]
            return pairwise_features
        else:
            return torch.zeros(batch_size, 0, 2, device=obj_features.device)

    def forward(self, batch_x):
        # batch_x shape: [batch_size, 92]
        batch_size = batch_x.size(0)

        # Extract global features (E_T_miss, phi_Et_miss)
        global_feat = batch_x[:, :2]  # [batch_size, 2]

        # Extract object features (remove obj_id, keep E, pT, eta, phi)
        obj_features = batch_x[:, 2:].reshape(batch_size, 18, 5)[:, :, 1:]  # [batch_size, 18, 4]

        # Compute pairwise features
        pairwise_feat = self.compute_pairwise_features(obj_features)  # [batch_size, num_pairs, 2]

        # Process object features
        obj_embedded = F.relu(self.obj_embed(obj_features))  # [batch_size, 18, 64]

        # Transformer layers
        x = F.relu(self.transformer1(obj_embedded, obj_embedded))
        x = F.relu(self.transformer2(x, x))

        # Global pooling of object features
        obj_global = global_mean_pool(x, batch=torch.zeros(batch_size, dtype=torch.long, device=batch_x.device))  # [batch_size, 64]

        # Process global features
        global_processed = self.global_mlp(global_feat)  # [batch_size, 32]

        # Combine features
        combined = torch.cat([obj_global, global_processed], dim=1)

        # Final classification
        logits = self.classifier(combined).squeeze(1)
        return torch.sigmoid(logits)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

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
            preds = (outputs > 0.5).float()
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_x.size(0)

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
                preds = (outputs > 0.5).float()
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_x.size(0)

        # Calculate metrics
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val

        # Store history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model)
                break

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


