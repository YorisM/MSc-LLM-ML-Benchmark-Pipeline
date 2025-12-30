
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
        self.obj_ids = [4, 5, 6, 24, 23, 22, 21, 1, 2, 3, 11, 13, 15, 12, 14, 16, 0, 0]  # Common object IDs in LHC data
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
        # Extract global features (E_T_miss and phi_Et_miss)
        global_features = X[:, :2].numpy()
        self.scaler.fit(global_features)
        return self

    def transform(self, X):
        # Convert to numpy for scikit-learn
        X_np = X.numpy()

        # Scale global features
        global_features = X_np[:, :2]
        global_features_scaled = self.scaler.transform(global_features)

        # Process per-object features
        object_features = []
        for i in range(self.max_objects):
            start_idx = self.global_features + i * self.per_object_features
            end_idx = start_idx + self.per_object_features
            obj_slice = X_np[:, start_idx:end_idx]

            # Extract kinematic features (E, pT, eta, phi)
            kinematic_features = obj_slice[:, 1:5]  # Exclude obj_id

            # Create pairwise features for each object
            obj_id = obj_slice[:, 0:1]
            obj_features = np.concatenate([obj_id, kinematic_features], axis=1)
            object_features.append(obj_features)

        # Stack all object features
        object_features = np.stack(object_features, axis=1)  # [batch, 18, 5]

        # Create pairwise features (invariant mass and deltaR)
        batch_size = object_features.shape[0]
        pairwise_features = []

        for b in range(batch_size):
            event_objects = object_features[b]
            valid_objects = event_objects[event_objects[:, 0] != 0]  # Filter zero-padded objects

            n_objects = valid_objects.shape[0]
            if n_objects < 2:
                # If less than 2 objects, create dummy features
                pairwise_feat = np.zeros((self.max_objects * (self.max_objects - 1) // 2, 2))
            else:
                # Calculate pairwise features
                m_ij = []
                deltaR_ij = []

                for i in range(n_objects):
                    for j in range(i + 1, n_objects):
                        E_i, pt_i, eta_i, phi_i = valid_objects[i, 1:5]
                        E_j, pt_j, eta_j, phi_j = valid_objects[j, 1:5]

                        # Invariant mass (simplified)
                        m_ij_val = math.sqrt(2 * pt_i * pt_j * (math.cosh(eta_i - eta_j) - math.cos(phi_i - phi_j)))
                        m_ij.append(m_ij_val)

                        # Delta R
                        delta_eta = eta_i - eta_j
                        delta_phi = phi_i - phi_j
                        deltaR = math.sqrt(delta_eta**2 + delta_phi**2)
                        deltaR_ij.append(deltaR)

                pairwise_feat = np.column_stack([m_ij, deltaR_ij])

                # Pad to maximum possible pairs (18*17/2 = 153)
                if pairwise_feat.shape[0] < 153:
                    padding = np.zeros((153 - pairwise_feat.shape[0], 2))
                    pairwise_feat = np.vstack([pairwise_feat, padding])

            pairwise_features.append(pairwise_feat)

        pairwise_features = np.stack(pairwise_features)  # [batch, 153, 2]

        # Combine all features
        processed_features = np.concatenate([
            global_features_scaled,
            object_features.reshape(batch_size, -1),  # Flatten object features
            pairwise_features.reshape(batch_size, -1)  # Flatten pairwise features
        ], axis=1)

        return torch.from_numpy(processed_features).float()

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

        # Determine input dimension from sample
        if isinstance(sample_object, torch.Tensor):
            input_dim = sample_object.shape[1]
        else:
            input_dim = sample_object[0].shape[1] if isinstance(sample_object, list) else sample_object.shape[1]

        # Feature extraction layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        # Transformer layers
        encoder_layers = TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=3)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(128)

        # Output layers
        self.fc_out = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, batch_x):
        # Handle different input types
        if isinstance(batch_x, list):
            x = batch_x[0]
        else:
            x = batch_x

        # Feature extraction
        x = F.relu(self.fc1(x))  # [batch, 512]
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # [batch, 256]
        x = self.dropout(x)
        x = F.relu(self.fc3(x))  # [batch, 128]

        # Reshape for transformer
        x = x.unsqueeze(1)  # [batch, 1, 128]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = x.transpose(0, 1)  # [1, batch, 128] for transformer
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # [batch, 1, 128]

        # Output
        x = x.squeeze(1)  # [batch, 128]
        x = self.fc_out(x)  # [batch, 1]

        return x.squeeze(-1)  # [batch]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_model = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == yb.float()).sum().item()
            total += yb.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                outputs = model(xb)
                loss = criterion(outputs, yb.float())
                val_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == yb.float()).sum().item()
                val_total += yb.size(0)

                all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_labels, all_probs)
        scheduler.step(val_auc)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')

        # Early stopping based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
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

