
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.obj_scalers = [RobustScaler() for _ in range(18)]
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
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Fit global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :self.global_feature_size]
        self.scaler.fit(global_features)

        # Fit per-object features
        for i in range(self.max_objects):
            start = self.global_feature_size + i * self.obj_feature_size
            end = start + self.obj_feature_size
            obj_features = X[:, start:end]
            # Only fit on non-zero objects (not padding)
            mask = obj_features[:, 0] != 0  # obj_id != 0
            if mask.sum() > 0:
                self.obj_scalers[i].fit(obj_features[mask, 1:])  # Skip obj_id

        return self

    def transform(self, X):
        # Transform global features
        global_features = X[:, :self.global_feature_size]
        global_features = self.scaler.transform(global_features)
        X_transformed = np.zeros_like(X)
        X_transformed[:, :self.global_feature_size] = global_features

        # Transform per-object features
        for i in range(self.max_objects):
            start = self.global_feature_size + i * self.obj_feature_size
            end = start + self.obj_feature_size
            obj_features = X[:, start:end]
            mask = obj_features[:, 0] != 0  # obj_id != 0

            if mask.sum() > 0:
                # Scale kinematic features (skip obj_id)
                obj_kinematic = obj_features[mask, 1:]
                obj_kinematic = self.obj_scalers[i].transform(obj_kinematic)
                obj_features[mask, 1:] = obj_kinematic

            X_transformed[:, start:end] = obj_features

        # Add pairwise features (deltaR and invariant mass)
        n_events = X_transformed.shape[0]
        pairwise_features = np.zeros((n_events, self.max_objects, self.max_objects, 2))

        for i in range(self.max_objects):
            for j in range(i+1, self.max_objects):
                # Get object features
                start_i = self.global_feature_size + i * self.obj_feature_size
                start_j = self.global_feature_size + j * self.obj_feature_size
                obj_i = X_transformed[:, start_i+1:start_i+5]  # E, pT, eta, phi
                obj_j = X_transformed[:, start_j+1:start_j+5]

                # Skip if either object is padding
                mask_i = X[:, start_i] != 0
                mask_j = X[:, start_j] != 0
                mask = mask_i & mask_j

                if mask.sum() == 0:
                    continue

                # Calculate deltaR
                delta_eta = obj_i[mask, 2] - obj_j[mask, 2]
                delta_phi = np.abs(obj_i[mask, 3] - obj_j[mask, 3])
                delta_phi = np.minimum(delta_phi, 2*np.pi - delta_phi)
                deltaR = np.sqrt(delta_eta**2 + delta_phi**2)

                # Calculate invariant mass (approximate)
                E_i = obj_i[mask, 0]
                E_j = obj_j[mask, 0]
                px_i = obj_i[mask, 1] * np.cos(obj_i[mask, 3])
                px_j = obj_j[mask, 1] * np.cos(obj_j[mask, 3])
                py_i = obj_i[mask, 1] * np.sin(obj_i[mask, 3])
                py_j = obj_j[mask, 1] * np.sin(obj_j[mask, 3])
                pz_i = obj_i[mask, 1] * np.sinh(obj_i[mask, 2])
                pz_j = obj_j[mask, 1] * np.sinh(obj_j[mask, 2])

                m_squared = (E_i + E_j)**2 - (px_i + px_j)**2 - (py_i + py_j)**2 - (pz_i + pz_j)**2
                m_squared = np.maximum(m_squared, 0)  # Avoid negative values from numerical errors
                m_ij = np.sqrt(m_squared)

                # Store features
                pairwise_features[mask, i, j, 0] = deltaR
                pairwise_features[mask, i, j, 1] = m_ij
                pairwise_features[mask, j, i, 0] = deltaR
                pairwise_features[mask, j, i, 1] = m_ij

        # Flatten pairwise features and concatenate with original features
        pairwise_flat = pairwise_features.reshape(n_events, -1)
        X_final = np.concatenate([X_transformed, pairwise_flat], axis=1)

        return X_final

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
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1)]
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_size = sample_object.shape[1]

        # Calculate number of objects (18) and features per object
        self.n_objects = 18
        self.obj_feature_size = 5
        self.global_feature_size = 2
        self.pairwise_feature_size = 2  # deltaR and invariant mass

        # Calculate original per-object features (after scaling)
        original_obj_features = self.obj_feature_size - 1  # excluding obj_id

        # Total features per object after adding pairwise features
        # Each object gets its original features + pairwise features with all other objects
        # For 18 objects, each object has 17 pairwise relationships
        self.per_object_features = original_obj_features + (self.n_objects - 1) * self.pairwise_feature_size

        # Transformer parameters
        self.d_model = 128
        self.nhead = 8
        self.num_layers = 4
        self.dim_feedforward = 512
        self.dropout = 0.1

        # Embedding layer for object features
        self.obj_embed = nn.Linear(self.per_object_features, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=self.num_layers)

        # Global features processing
        self.global_embed = nn.Linear(self.global_feature_size, self.d_model)

        # Output layers
        self.fc1 = nn.Linear(self.d_model * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout_layer = nn.Dropout(0.3)

    def forward(self, batch_x):
        # batch_x shape: [batch_size, input_size]
        batch_size = batch_x.size(0)

        # Extract global features (first 2 features)
        global_features = batch_x[:, :self.global_feature_size]  # [batch_size, 2]

        # Extract and reshape object features
        # Original object features (after scaling) are in the first part
        # Pairwise features are appended at the end
        obj_features = batch_x[:, self.global_feature_size:]  # [batch_size, n_objects * per_object_features]

        # Reshape to [batch_size, n_objects, per_object_features]
        obj_features = obj_features.view(batch_size, self.n_objects, self.per_object_features)

        # Embed object features
        obj_embedded = self.obj_embed(obj_features)  # [batch_size, n_objects, d_model]

        # Add positional encoding
        obj_embedded = self.pos_encoder(obj_embedded)

        # Transformer encoder
        transformer_out = self.transformer_encoder(obj_embedded)  # [batch_size, n_objects, d_model]

        # Global average pooling over objects
        obj_pooled = transformer_out.mean(dim=1)  # [batch_size, d_model]

        # Embed global features
        global_embedded = self.global_embed(global_features)  # [batch_size, d_model]

        # Concatenate global and object features
        combined = torch.cat([global_embedded, obj_pooled], dim=1)  # [batch_size, d_model * 2]

        # Fully connected layers
        x = F.relu(self.fc1(combined))
        x = self.dropout_layer(x)
        x = torch.sigmoid(self.fc2(x))  # [batch_size, 1]

        return x.squeeze(1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

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
            loss = F.binary_cross_entropy(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())

        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_targets, train_preds)
        train_losses.append(train_loss)
        train_aucs.append(train_auc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = F.binary_cross_entropy(outputs, batch_y.float())

                val_loss += loss.item()
                val_preds.extend(outputs.detach().cpu().numpy())
                val_targets.extend(batch_y.detach().cpu().numpy())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        # Update learning rate
        scheduler.step(val_auc)

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

    return model, train_losses, val_losses, train_aucs, val_aucs

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


