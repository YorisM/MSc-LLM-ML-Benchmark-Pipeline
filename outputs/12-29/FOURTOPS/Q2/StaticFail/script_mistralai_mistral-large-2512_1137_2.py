
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
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.obj_ids = [0, 1, 2, 3, 4, 5, 6, 21]  # Common object IDs in particle physics
        self.max_objects = 18
        self.global_features = 2
        self.per_object_features = 5

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
        # Extract global features (E_T_miss, phi_Et_miss) and object features
        global_features = X[:, :2].numpy()  # [N, 2]
        self.scaler.fit(global_features)
        return self

    def transform(self, X):
        # Split into global and object features
        global_features = X[:, :2]  # [N, 2]
        object_features = X[:, 2:]  # [N, 90]

        # Scale global features
        global_features = torch.tensor(self.scaler.transform(global_features.numpy()), dtype=torch.float32)

        # Reshape object features into [N, 18, 5]
        object_features = object_features.view(-1, self.max_objects, self.per_object_features)  # [N, 18, 5]

        # Extract kinematic features (E, pT, eta, phi) and object IDs
        obj_ids = object_features[:, :, 0].long()  # [N, 18]
        kinematics = object_features[:, :, 1:]  # [N, 18, 4]

        # Create mask for valid objects (non-zero pT)
        mask = (kinematics[:, :, 1] > 0)  # [N, 18]

        # Compute pairwise features for valid objects
        N = X.shape[0]
        pairwise_features = []

        for i in range(N):
            # Get valid objects for this event
            valid_mask = mask[i]  # [18]
            valid_kinematics = kinematics[i][valid_mask]  # [num_valid, 4]
            num_valid = valid_kinematics.shape[0]

            if num_valid < 2:
                # Not enough objects for pairwise features
                pairwise_features.append(torch.zeros(1, 2))
                continue

            # Compute pairwise invariant mass and deltaR
            E = valid_kinematics[:, 0]  # [num_valid]
            px = valid_kinematics[:, 1] * torch.cos(valid_kinematics[:, 3])  # [num_valid]
            py = valid_kinematics[:, 1] * torch.sin(valid_kinematics[:, 3])  # [num_valid]
            pz = valid_kinematics[:, 1] * torch.sinh(valid_kinematics[:, 2])  # [num_valid]

            # Compute all pairwise combinations
            idx = torch.combinations(torch.arange(num_valid), 2)
            E_i, E_j = E[idx[:, 0]], E[idx[:, 1]]
            px_i, px_j = px[idx[:, 0]], px[idx[:, 1]]
            py_i, py_j = py[idx[:, 0]], py[idx[:, 1]]
            pz_i, pz_j = pz[idx[:, 0]], pz[idx[:, 1]]
            eta_i, eta_j = valid_kinematics[idx[:, 0], 2], valid_kinematics[idx[:, 1], 2]
            phi_i, phi_j = valid_kinematics[idx[:, 0], 3], valid_kinematics[idx[:, 1], 3]

            # Invariant mass: m_ij = sqrt((E_i + E_j)^2 - (px_i + px_j)^2 - (py_i + py_j)^2 - (pz_i + pz_j)^2)
            m_ij = torch.sqrt(
                (E_i + E_j)**2 -
                (px_i + px_j)**2 -
                (py_i + py_j)**2 -
                (pz_i + pz_j)**2
            )  # [num_pairs]

            # DeltaR: sqrt((eta_i - eta_j)^2 + (phi_i - phi_j)^2)
            deltaR = torch.sqrt(
                (eta_i - eta_j)**2 +
                (phi_i - phi_j)**2
            )  # [num_pairs]

            # Take mean and max of pairwise features
            pairwise_mean = torch.stack([m_ij.mean(), deltaR.mean()])
            pairwise_max = torch.stack([m_ij.max(), deltaR.max()])
            pairwise_features.append(torch.cat([pairwise_mean, pairwise_max]).unsqueeze(0))

        pairwise_features = torch.cat(pairwise_features, dim=0)  # [N, 4]

        # Combine all features
        combined_features = torch.cat([
            global_features,
            pairwise_features,
            kinematics.mean(dim=1),  # [N, 4] mean kinematics
            kinematics.max(dim=1)[0],  # [N, 4] max kinematics
            mask.sum(dim=1, keepdim=True).float()  # [N, 1] number of valid objects
        ], dim=1)  # [N, 2 + 4 + 4 + 4 + 1 = 15]

        return combined_features

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[1]

        # Transformer encoder
        self.encoder_layer = TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=4)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=128)

        # Input projection
        self.input_proj = nn.Linear(input_dim, 128)

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, batch_x):
        # batch_x: [batch_size, input_dim]
        x = self.input_proj(batch_x)  # [batch_size, 128]
        x = self.pos_encoder(x.unsqueeze(1))  # [batch_size, 1, 128]

        # Transformer expects [batch_size, seq_len, d_model]
        x = self.transformer_encoder(x)  # [batch_size, 1, 128]

        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, 128]

        # Output
        logits = self.fc(x).squeeze(-1)  # [batch_size]
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model = None
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y.float()

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(torch.sigmoid(logits).detach().cpu())
            train_targets.append(yb.detach().cpu())

        train_loss /= len(train_loader)
        train_preds = torch.cat(train_preds).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_auc = roc_auc_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y.float()
                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item()
                val_preds.append(torch.sigmoid(logits).detach().cpu())
                val_targets.append(yb.detach().cpu())

        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_auc = roc_auc_score(val_targets, val_preds)

        # Update scheduler
        scheduler.step(val_auc)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

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

    return model, train_losses, val_losses, train_aucs, val_aucs

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

