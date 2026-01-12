
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
        self.max_objects = 18
        self.obj_feature_size = 5
        self.global_feature_size = 2
        self.pairwise_features = True

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
        global_features = X[:, :2].numpy()  # [N, 2]
        self.scaler.fit(global_features)
        return self

    def transform(self, X):
        # Convert to numpy for preprocessing
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Scale global features
        global_features = X_np[:, :2]  # [N, 2]
        global_features = self.scaler.transform(global_features)

        # Process object features
        object_features = []
        for i in range(self.max_objects):
            start_idx = 2 + i * self.obj_feature_size
            end_idx = start_idx + self.obj_feature_size
            obj_slice = X_np[:, start_idx:end_idx]  # [N, 5]

            # Convert object ID to one-hot (5 possible objects: 0=padding, 1-4=physics objects)
            obj_ids = obj_slice[:, 0].astype(int)
            one_hot = np.zeros((obj_ids.shape[0], 5))
            one_hot[np.arange(obj_ids.shape[0]), obj_ids] = 1

            # Keep kinematic features (E, pT, eta, phi)
            kinematics = obj_slice[:, 1:]  # [N, 4]

            # Combine features
            obj_features = np.concatenate([one_hot, kinematics], axis=1)  # [N, 9]
            object_features.append(obj_features)

        # Stack all objects
        object_features = np.stack(object_features, axis=1)  # [N, 18, 9]

        # Add pairwise features if enabled
        if self.pairwise_features:
            pairwise_features = self._compute_pairwise_features(object_features)
            object_features = np.concatenate([object_features, pairwise_features], axis=2)  # [N, 18, 9+2]

        # Combine global and object features
        processed = np.concatenate([
            global_features,
            object_features.reshape(object_features.shape[0], -1)  # Flatten objects
        ], axis=1)

        return torch.from_numpy(processed).float()

    def _compute_pairwise_features(self, object_features):
        # object_features shape: [N, 18, 9]
        N = object_features.shape[0]
        num_objects = object_features.shape[1]

        # Extract kinematic features (E, pT, eta, phi)
        kinematics = object_features[:, :, 5:9]  # [N, 18, 4]

        # Compute pairwise features for all objects
        pairwise_mass = np.zeros((N, num_objects, num_objects))
        pairwise_dR = np.zeros((N, num_objects, num_objects))

        for i in range(num_objects):
            for j in range(num_objects):
                if i == j:
                    continue

                # Get 4-vectors for objects i and j
                E_i, pt_i, eta_i, phi_i = kinematics[:, i, :].T
                E_j, pt_j, eta_j, phi_j = kinematics[:, j, :].T

                # Compute invariant mass
                px_i = pt_i * np.cos(phi_i)
                py_i = pt_i * np.sin(phi_i)
                pz_i = pt_i * np.sinh(eta_i)

                px_j = pt_j * np.cos(phi_j)
                py_j = pt_j * np.sin(phi_j)
                pz_j = pt_j * np.sinh(eta_j)

                E_sum = E_i + E_j
                px_sum = px_i + px_j
                py_sum = py_i + py_j
                pz_sum = pz_i + pz_j

                mass_sq = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
                mass_sq = np.clip(mass_sq, 0, None)  # Avoid numerical issues
                pairwise_mass[:, i, j] = np.sqrt(mass_sq)

                # Compute delta R
                d_eta = eta_i - eta_j
                d_phi = phi_i - phi_j
                d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
                pairwise_dR[:, i, j] = np.sqrt(d_eta**2 + d_phi**2)

        # For each object, compute mean pairwise features with other objects
        mean_pairwise_mass = np.zeros((N, num_objects, 1))
        mean_pairwise_dR = np.zeros((N, num_objects, 1))

        for i in range(num_objects):
            mask = np.ones(num_objects, dtype=bool)
            mask[i] = False
            mean_pairwise_mass[:, i, 0] = np.mean(pairwise_mass[:, i, mask], axis=1)
            mean_pairwise_dR[:, i, 0] = np.mean(pairwise_dR[:, i, mask], axis=1)

        # Combine features
        pairwise_features = np.concatenate([mean_pairwise_mass, mean_pairwise_dR], axis=2)  # [N, 18, 2]
        return pairwise_features

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Calculate input features
        self.max_objects = 18
        self.obj_feature_size = 9  # 5 one-hot + 4 kinematics
        self.pairwise_feature_size = 2
        self.global_feature_size = 2

        # Check if pairwise features are present
        if sample_object.shape[1] > (self.global_feature_size + self.max_objects * self.obj_feature_size):
            self.obj_feature_size += self.pairwise_feature_size

        # Transformer parameters
        self.d_model = 128
        self.nhead = 8
        self.num_layers = 4
        self.dim_feedforward = 512
        self.dropout = 0.1

        # Input embedding
        self.global_embed = nn.Linear(self.global_feature_size, self.d_model)
        self.obj_embed = nn.Linear(self.obj_feature_size, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, self.num_layers)

        # Output layers
        self.attention = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x shape: [B, F] where F = global_features + 18*obj_features
        B = batch_x.shape[0]

        # Split global and object features
        global_features = batch_x[:, :self.global_feature_size]  # [B, 2]
        obj_features = batch_x[:, self.global_feature_size:]  # [B, 18*obj_feature_size]
        obj_features = obj_features.reshape(B, self.max_objects, -1)  # [B, 18, obj_feature_size]

        # Embed global features
        global_embedded = self.global_embed(global_features)  # [B, d_model]
        global_embedded = global_embedded.unsqueeze(1)  # [B, 1, d_model]

        # Embed object features
        obj_embedded = self.obj_embed(obj_features)  # [B, 18, d_model]

        # Combine features
        combined = torch.cat([global_embedded, obj_embedded], dim=1)  # [B, 19, d_model]

        # Add positional encoding
        combined = self.pos_encoder(combined)

        # Transformer
        transformer_out = self.transformer(combined)  # [B, 19, d_model]

        # Global feature is at position 0
        global_out = transformer_out[:, 0, :]  # [B, d_model]

        # Attention over object features
        attention_weights = self.attention(transformer_out[:, 1:, :])  # [B, 18, 1]
        attention_weights = attention_weights.squeeze(-1)  # [B, 18]
        attention_weights = F.softmax(attention_weights, dim=1)  # [B, 18]

        # Weighted sum of object features
        obj_out = torch.sum(transformer_out[:, 1:, :] * attention_weights.unsqueeze(-1), dim=1)  # [B, d_model]

        # Combine global and object features
        combined_out = global_out + obj_out  # [B, d_model]

        # Classifier
        logits = self.classifier(combined_out)  # [B, 1]
        return logits.squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [B, seq_len, d_model]
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).long()
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
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
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.float())

                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

                all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(all_labels, all_probs)
        scheduler.step(auc_score)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, AUC: {auc_score:.4f}')

        # Early stopping
        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print(f'Early stopping at epoch {epoch+1}')
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

