
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
        self.obj_ids = [0, 1, 2, 3, 4, 5, 6, 21, 22]  # Common particle IDs in LHC events
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
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        # Extract global features (E_T_miss, phi_Et_miss) and object features
        global_features = X[:, :2].numpy()  # [N, 2]
        object_features = X[:, 2:].reshape(-1, self.max_objects, self.per_object_features)  # [N, 18, 5]

        # Flatten object features for scaling
        flat_object_features = object_features.reshape(-1, self.per_object_features)  # [N*18, 5]

        # Scale global features
        self.scaler.fit(global_features)

        return self

    def transform(self, X):
        # Convert to numpy for preprocessing
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Extract global features
        global_features = X_np[:, :2]  # [N, 2]
        scaled_global = self.scaler.transform(global_features)

        # Extract object features
        object_features = X_np[:, 2:].reshape(-1, self.max_objects, self.per_object_features)  # [N, 18, 5]

        # Create new features: pairwise invariant mass and deltaR
        new_features = []
        for event in object_features:
            valid_objects = event[event[:, 0] != 0]  # Filter out padding (obj_id=0 is padding)
            n_valid = valid_objects.shape[0]
            if n_valid < 2:
                # Not enough objects to compute pairwise features
                pairwise_features = np.zeros((self.max_objects * (self.max_objects - 1) // 2, 2))
            else:
                # Compute pairwise features
                energies = valid_objects[:, 1]  # E
                pt = valid_objects[:, 2]  # p_T
                eta = valid_objects[:, 3]  # eta
                phi = valid_objects[:, 4]  # phi

                # Compute pairwise invariant mass
                m_ij = []
                deltaR = []
                for i in range(n_valid):
                    for j in range(i+1, n_valid):
                        # Invariant mass: m_ij = sqrt((E_i + E_j)^2 - (p_i + p_j)^2)
                        E_i, E_j = energies[i], energies[j]
                        p_i = np.array([pt[i]*np.cos(phi[i]), pt[i]*np.sin(phi[i]), pt[i]*np.sinh(eta[i])])
                        p_j = np.array([pt[j]*np.cos(phi[j]), pt[j]*np.sin(phi[j]), pt[j]*np.sinh(eta[j])])
                        p_sum = p_i + p_j
                        m_squared = (E_i + E_j)**2 - np.sum(p_sum**2)
                        m_ij.append(max(0, m_squared)**0.5)  # Ensure non-negative

                        # Delta R: sqrt((eta_i - eta_j)^2 + (phi_i - phi_j)^2)
                        delta_eta = eta[i] - eta[j]
                        delta_phi = np.minimum(np.abs(phi[i] - phi[j]), 2*np.pi - np.abs(phi[i] - phi[j]))
                        deltaR.append((delta_eta**2 + delta_phi**2)**0.5)

                pairwise_features = np.column_stack([m_ij, deltaR])

                # Pad to maximum possible pairs (18*17/2 = 153)
                if len(m_ij) < 153:
                    padding = np.zeros((153 - len(m_ij), 2))
                    pairwise_features = np.vstack([pairwise_features, padding])

            new_features.append(pairwise_features.flatten())

        # Stack all features
        new_features = np.stack(new_features)  # [N, 153*2]
        processed_features = np.hstack([scaled_global, new_features])  # [N, 2 + 306]

        # Convert back to torch tensor
        return torch.from_numpy(processed_features).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
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
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1)]
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[1]  # Should be 308 (2 global + 306 pairwise)

        # Transformer parameters
        self.d_model = 128
        self.nhead = 8
        self.num_layers = 4
        self.dim_feedforward = 512
        self.dropout = 0.1

        # Input projection
        self.input_proj = nn.Linear(input_dim, self.d_model)

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
        self.transformer = TransformerEncoder(encoder_layers, num_layers=self.num_layers)

        # Output layers
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x: [batch_size, input_dim]
        batch_size = batch_x.shape[0]

        # Project to d_model
        x = self.input_proj(batch_x)  # [batch_size, d_model]

        # Add sequence dimension (treat each feature as a token)
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer expects [batch_size, seq_len, d_model]
        # We'll treat each feature as a separate token in a sequence
        x = x.expand(-1, 10, -1)  # Expand to 10 tokens for transformer

        # Transformer
        x = self.transformer(x)  # [batch_size, seq_len, d_model]

        # Take the first token's output for classification
        x = x[:, 0, :]  # [batch_size, d_model]

        # Classifier
        logits = self.classifier(x)  # [batch_size, 1]

        return logits.squeeze(-1)  # [batch_size]

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

    best_val_auc = 0.0
    best_model_state = None
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

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss_epoch = running_loss / len(train_loader)
        train_acc_epoch = correct / total
        train_loss.append(train_loss_epoch)
        train_acc.append(train_acc_epoch)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_running_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(torch.sigmoid(outputs).cpu().numpy())

        val_loss_epoch = val_running_loss / len(val_loader)
        val_acc_epoch = val_correct / val_total
        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_labels, all_probs)

        # Update scheduler
        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}, "
              f"Train Acc: {train_acc_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
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

    return model, train_loss, val_loss, train_acc, val_acc

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

