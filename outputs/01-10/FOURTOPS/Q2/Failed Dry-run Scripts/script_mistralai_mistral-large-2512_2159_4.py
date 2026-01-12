
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.spatial.distance import pdist, squareform
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.obj_scaler = RobustScaler()
        self.etmiss_scaler = RobustScaler()
        self.phi_scaler = RobustScaler()
        self.max_objects = 18
        self.obj_feature_size = 5
        self.global_feature_size = 2

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
                                "batch_size": 512}
        }

    def fit(self, X, y=None):
        # Extract global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :2].numpy()
        self.etmiss_scaler.fit(global_features[:, :1])
        self.phi_scaler.fit(global_features[:, 1:2])

        # Extract object features (obj_id, E, p_T, eta, phi)
        obj_features = []
        for i in range(self.max_objects):
            start_idx = 2 + i * self.obj_feature_size
            end_idx = start_idx + self.obj_feature_size
            obj_slice = X[:, start_idx:end_idx]
            # Only consider non-zero padded objects
            mask = obj_slice[:, 0] != 0
            if mask.sum() > 0:
                obj_features.append(obj_slice[mask, 1:])  # Exclude obj_id

        if obj_features:
            obj_features = np.concatenate(obj_features, axis=0)
            self.obj_scaler.fit(obj_features)

        return self

    def transform(self, X):
        # Create output array
        X_transformed = np.zeros_like(X.numpy())

        # Transform global features
        global_features = X[:, :2].numpy()
        X_transformed[:, 0] = self.etmiss_scaler.transform(global_features[:, :1]).ravel()
        X_transformed[:, 1] = self.phi_scaler.transform(global_features[:, 1:2]).ravel()

        # Transform object features
        for i in range(self.max_objects):
            start_idx = 2 + i * self.obj_feature_size
            end_idx = start_idx + self.obj_feature_size
            obj_slice = X[:, start_idx:end_idx].numpy()

            # Get mask for non-zero objects
            mask = obj_slice[:, 0] != 0
            if mask.sum() > 0:
                # Transform kinematic features (E, p_T, eta, phi)
                obj_features = obj_slice[mask, 1:]
                transformed_features = self.obj_scaler.transform(obj_features)
                obj_slice[mask, 1:] = transformed_features

                # Add object ID back (unchanged)
                obj_slice[mask, 0] = obj_slice[mask, 0]

            X_transformed[:, start_idx:end_idx] = obj_slice

        # Add pairwise features
        n_events = X.shape[0]
        pairwise_features = np.zeros((n_events, self.max_objects, self.max_objects, 2))

        for event_idx in range(n_events):
            event = X_transformed[event_idx]
            objects = []

            # Collect non-zero objects
            for i in range(self.max_objects):
                start_idx = 2 + i * self.obj_feature_size
                obj_id = event[start_idx]
                if obj_id != 0:
                    # Get kinematic features (E, p_T, eta, phi)
                    kinematics = event[start_idx+1:start_idx+5]
                    objects.append(kinematics)

            n_objects = len(objects)
            if n_objects < 2:
                continue

            # Convert to numpy array for vectorized operations
            objects = np.array(objects)  # [n_objects, 4]

            # Calculate pairwise invariant mass and delta R
            for i in range(n_objects):
                for j in range(i+1, n_objects):
                    E_i, pt_i, eta_i, phi_i = objects[i]
                    E_j, pt_j, eta_j, phi_j = objects[j]

                    # Calculate invariant mass m_ij
                    p_i = np.array([pt_i * np.cos(phi_i),
                                   pt_i * np.sin(phi_i),
                                   pt_i * np.sinh(eta_i)])
                    p_j = np.array([pt_j * np.cos(phi_j),
                                   pt_j * np.sin(phi_j),
                                   pt_j * np.sinh(eta_j)])

                    total_p = p_i + p_j
                    total_E = E_i + E_j
                    m_ij = np.sqrt(total_E**2 - np.sum(total_p**2))

                    # Calculate delta R
                    delta_eta = eta_i - eta_j
                    delta_phi = phi_i - phi_j
                    delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

                    # Store features
                    pairwise_features[event_idx, i, j, 0] = m_ij
                    pairwise_features[event_idx, i, j, 1] = delta_R
                    pairwise_features[event_idx, j, i, 0] = m_ij
                    pairwise_features[event_idx, j, i, 1] = delta_R

        # Flatten pairwise features and concatenate with original features
        pairwise_flat = pairwise_features.reshape(n_events, -1)  # [n_events, max_objects*max_objects*2]

        # Create final feature array
        final_features = np.zeros((n_events, X_transformed.shape[1] + pairwise_flat.shape[1]))
        final_features[:, :X_transformed.shape[1]] = X_transformed
        final_features[:, X_transformed.shape[1]:] = pairwise_flat

        return final_features

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Calculate input dimension
        input_dim = sample_object.shape[1]

        # Object embedding
        self.obj_embed = nn.Linear(input_dim, 128)

        # Transformer layers
        self.attention1 = AttentionBlock(128, 4, 0.1)
        self.attention2 = AttentionBlock(128, 4, 0.1)

        # Global features processing
        self.global_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, batch_x):
        # batch_x: [batch_size, num_features]

        # Get number of objects (18) and features per object
        num_objects = 18
        features_per_object = batch_x.shape[1] // num_objects

        # Reshape to [batch_size, num_objects, features_per_object]
        x = batch_x.reshape(-1, num_objects, features_per_object)

        # Embed objects
        x = self.obj_embed(x)  # [batch_size, num_objects, 128]

        # Apply attention
        x = self.attention1(x)
        x = self.attention2(x)

        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, 128]

        # Process global features
        x = self.global_mlp(x)  # [batch_size, 32]

        # Classifier
        logits = self.classifier(x)  # [batch_size, 1]

        return logits.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    best_auc = 0.0
    best_model_state = None
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.append(batch_y.detach().cpu().numpy())

        # Calculate training metrics
        train_loss /= len(train_loader)
        train_preds = np.concatenate(train_preds)
        train_targets = np.concatenate(train_targets)
        train_auc = roc_auc_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.float())

                val_loss += loss.item()
                val_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
                val_targets.append(batch_y.detach().cpu().numpy())

        val_loss /= len(val_loader)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

        # Update scheduler
        scheduler.step(val_auc)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Calculate final training accuracy (not really needed for AUC)
    train_acc = (train_preds > 0.5).mean()
    val_acc = (val_preds > 0.5).mean()

    return model, train_losses, val_losses, train_aucs, val_aucs

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

