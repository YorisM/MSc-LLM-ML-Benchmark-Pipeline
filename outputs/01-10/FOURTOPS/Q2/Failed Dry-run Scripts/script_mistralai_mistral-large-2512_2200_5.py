
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
        self.obj_ids = [0, 1, 2, 3, 4, 5, 6, 21, 22]  # Common object IDs in particle physics
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

            "eval_overrides": {"shuffle": False,
                               "batch_size": 512}
        }

    def fit(self, X, y=None):
        # Extract global features (E_T_miss, phi_Et_miss)
        global_feats = X[:, :2].numpy()  # [N, 2]
        self.scaler.fit(global_feats)
        return self

    def transform(self, X):
        # Convert to numpy for preprocessing
        X_np = X.numpy()

        # Scale global features
        global_feats = X_np[:, :2]  # [N, 2]
        global_feats_scaled = self.scaler.transform(global_feats)

        # Process object features
        num_objects = (X_np.shape[1] - 2) // 5
        object_features = []

        for i in range(num_objects):
            start_idx = 2 + i * 5
            end_idx = start_idx + 5
            obj_slice = X_np[:, start_idx:end_idx]  # [N, 5]

            # Extract kinematic features (E, pT, eta, phi) and object ID
            obj_id = obj_slice[:, 0]  # [N]
            kinematics = obj_slice[:, 1:]  # [N, 4]

            # Create one-hot encoding for object IDs
            obj_onehot = np.zeros((len(obj_id), len(self.obj_ids)), dtype=np.float32)
            for j, oid in enumerate(self.obj_ids):
                obj_onehot[:, j] = (obj_id == oid).astype(np.float32)

            # Combine kinematics with one-hot encoding
            obj_processed = np.concatenate([kinematics, obj_onehot], axis=1)  # [N, 4 + len(obj_ids)]
            object_features.append(obj_processed)

        # Stack all objects
        object_features = np.stack(object_features, axis=1)  # [N, 18, 4 + len(obj_ids)]

        # Create mask for zero-padded objects
        mask = (X_np[:, 2::5] != 0).astype(np.float32)  # [N, 18]
        mask = np.expand_dims(mask, axis=-1)  # [N, 18, 1]

        # Combine global features with object features
        global_feats_scaled = np.expand_dims(global_feats_scaled, axis=1)  # [N, 1, 2]
        combined = np.concatenate([global_feats_scaled, object_features], axis=1)  # [N, 19, 4 + len(obj_ids) + 2]

        # Add mask as additional feature
        combined = np.concatenate([combined, mask], axis=-1)  # [N, 19, 4 + len(obj_ids) + 3]

        # Compute pairwise features for the first 8 objects (to limit computational cost)
        pairwise_features = []
        for i in range(min(8, num_objects)):
            for j in range(i+1, min(8, num_objects)):
                # Get kinematics for objects i and j
                obj_i = X_np[:, 2 + i*5 + 1:2 + i*5 + 5]  # [N, 4]
                obj_j = X_np[:, 2 + j*5 + 1:2 + j*5 + 5]  # [N, 4]

                # Compute delta R
                delta_eta = obj_i[:, 2] - obj_j[:, 2]  # eta_i - eta_j
                delta_phi = obj_i[:, 3] - obj_j[:, 3]  # phi_i - phi_j
                delta_R = np.sqrt(delta_eta**2 + delta_phi**2)  # [N]

                # Compute invariant mass (approximate)
                E_i = obj_i[:, 0]
                E_j = obj_j[:, 0]
                px_i = obj_i[:, 1] * np.cos(obj_i[:, 3])
                py_i = obj_i[:, 1] * np.sin(obj_i[:, 3])
                pz_i = obj_i[:, 1] * np.sinh(obj_i[:, 2])
                px_j = obj_j[:, 1] * np.cos(obj_j[:, 3])
                py_j = obj_j[:, 1] * np.sin(obj_j[:, 3])
                pz_j = obj_j[:, 1] * np.sinh(obj_j[:, 2])

                E_sum = E_i + E_j
                px_sum = px_i + px_j
                py_sum = py_i + py_j
                pz_sum = pz_i + pz_j
                inv_mass = np.sqrt(E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2))  # [N]

                pairwise_features.append(np.stack([delta_R, inv_mass], axis=1))  # [N, 2]

        # Stack pairwise features
        if pairwise_features:
            pairwise_features = np.concatenate(pairwise_features, axis=1)  # [N, 2 * C(8,2)]
        else:
            pairwise_features = np.zeros((X_np.shape[0], 0), dtype=np.float32)

        # Combine all features
        final_features = np.concatenate([
            global_feats_scaled.squeeze(1),  # [N, 2]
            np.mean(object_features, axis=1),  # [N, 4 + len(obj_ids)]
            np.max(object_features, axis=1),  # [N, 4 + len(obj_ids)]
            np.sum(object_features, axis=1),  # [N, 4 + len(obj_ids)]
            pairwise_features  # [N, 2 * C(8,2)]
        ], axis=1)

        return final_features.astype(np.float32)

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

        # Determine input feature dimension from sample
        input_dim = sample_object.shape[1]

        # Transformer parameters
        d_model = 128
        nhead = 8
        num_layers = 4
        dim_feedforward = 512
        dropout = 0.1

        # Embedding layer
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, batch_x):
        # batch_x: [B, F]

        # Add sequence dimension for transformer
        x = batch_x.unsqueeze(1)  # [B, 1, F]

        # Embedding
        x = self.embedding(x)  # [B, 1, d_model]

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer_encoder(x)  # [B, 1, d_model]

        # Attention pooling
        attn_weights = self.attention(x)  # [B, 1, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        x = torch.sum(attn_weights * x, dim=1)  # [B, d_model]

        # Classifier
        logits = self.classifier(x)  # [B, 1]
        return logits.squeeze(1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
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
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == batch_y.float()).sum().item()
            total += batch_y.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.float())
                val_running_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == batch_y.float()).sum().item()
                val_total += batch_y.size(0)

                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_labels, all_probs)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, Val AUC: {val_auc:.4f}')

        # Early stopping based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        scheduler.step(val_auc)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss, val_loss, train_acc, val_acc

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

