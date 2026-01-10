
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
        self.max_objects = 18
        self.global_features = 2
        self.per_object_features = 5
        self.object_feature_indices = list(range(self.global_features, self.global_features + self.max_objects * self.per_object_features))

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
        # Extract only the kinematic features (skip object IDs)
        kinematic_features = X[:, [0,1] + [i for i in range(2, X.shape[1]) if (i-2) % 5 != 0]]
        self.scaler.fit(kinematic_features)
        return self

    def transform(self, X):
        # Create a copy to avoid modifying the original
        X_transformed = X.clone()

        # Extract kinematic features (skip object IDs)
        kinematic_indices = [0,1] + [i for i in range(2, X.shape[1]) if (i-2) % 5 != 0]
        kinematic_features = X_transformed[:, kinematic_indices]

        # Scale kinematic features
        kinematic_features_np = kinematic_features.numpy()
        kinematic_features_scaled = self.scaler.transform(kinematic_features_np)
        X_transformed[:, kinematic_indices] = torch.from_numpy(kinematic_features_scaled)

        # Add pairwise features
        batch_size = X_transformed.shape[0]
        pairwise_features = []

        for b in range(batch_size):
            event = X_transformed[b]
            objects = []

            # Extract objects (skip global features and object IDs)
            for i in range(self.max_objects):
                start_idx = self.global_features + i * self.per_object_features
                if start_idx + 4 >= event.shape[0]:
                    break
                # Skip object ID (first feature in each object block)
                obj_features = event[start_idx+1:start_idx+5]  # E, pT, eta, phi
                objects.append(obj_features)

            # Compute pairwise features
            n_objects = len(objects)
            pairwise_feature_list = []

            for i in range(n_objects):
                for j in range(i+1, n_objects):
                    obj_i = objects[i]
                    obj_j = objects[j]

                    # Invariant mass m_ij = sqrt((E_i + E_j)^2 - |p_i + p_j|^2)
                    E_i, pT_i, eta_i, phi_i = obj_i
                    E_j, pT_j, eta_j, phi_j = obj_j

                    # Convert to 4-vectors
                    px_i = pT_i * math.cos(phi_i)
                    py_i = pT_i * math.sin(phi_i)
                    pz_i = pT_i * math.sinh(eta_i)

                    px_j = pT_j * math.cos(phi_j)
                    py_j = pT_j * math.sin(phi_j)
                    pz_j = pT_j * math.sinh(eta_j)

                    # Total 4-vector
                    E_tot = E_i + E_j
                    px_tot = px_i + px_j
                    py_tot = py_i + py_j
                    pz_tot = pz_i + pz_j

                    # Invariant mass squared
                    m2 = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
                    m_ij = math.sqrt(max(0, m2))  # Ensure non-negative

                    # Angular distance ΔR_ij = sqrt(Δη^2 + Δφ^2)
                    delta_eta = eta_i - eta_j
                    delta_phi = (phi_i - phi_j + math.pi) % (2 * math.pi) - math.pi  # Handle periodicity
                    delta_R = math.sqrt(delta_eta**2 + delta_phi**2)

                    pairwise_feature_list.extend([m_ij, delta_R])

            # Pad pairwise features to fixed size (max possible pairs: 18*17/2 = 153)
            max_pairs = 153
            n_pairs = len(pairwise_feature_list) // 2
            if n_pairs < max_pairs:
                pairwise_feature_list.extend([0.0] * (2 * (max_pairs - n_pairs)))
            pairwise_features.append(pairwise_feature_list[:2*max_pairs])

        # Convert to tensor and concatenate with original features
        pairwise_features_tensor = torch.tensor(pairwise_features, dtype=torch.float32)
        X_transformed = torch.cat([X_transformed, pairwise_features_tensor], dim=1)

        return X_transformed

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
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1)]
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[1]

        # Calculate feature dimensions
        self.global_features = 2
        self.per_object_features = 4  # E, pT, eta, phi (after removing object IDs)
        self.max_objects = 18
        self.pairwise_features = 153 * 2  # m_ij and ΔR_ij for each pair

        # Feature dimensions
        object_feature_dim = 64
        pairwise_feature_dim = 32
        global_feature_dim = 32

        # Object feature embedding
        self.object_embed = nn.Sequential(
            nn.Linear(self.per_object_features, object_feature_dim),
            nn.ReLU(),
            nn.LayerNorm(object_feature_dim)
        )

        # Pairwise feature embedding
        self.pairwise_embed = nn.Sequential(
            nn.Linear(2, pairwise_feature_dim),  # m_ij and ΔR_ij
            nn.ReLU(),
            nn.LayerNorm(pairwise_feature_dim)
        )

        # Global feature embedding
        self.global_embed = nn.Sequential(
            nn.Linear(self.global_features, global_feature_dim),
            nn.ReLU(),
            nn.LayerNorm(global_feature_dim)
        )

        # Transformer for object features
        self.pos_encoder = PositionalEncoding(object_feature_dim)
        encoder_layers = TransformerEncoderLayer(
            d_model=object_feature_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=3)

        # Attention for pairwise features
        self.pairwise_attention = nn.MultiheadAttention(
            embed_dim=pairwise_feature_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Final classifier
        combined_dim = object_feature_dim + pairwise_feature_dim + global_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x shape: [batch_size, total_features]
        batch_size = batch_x.shape[0]

        # Extract features
        global_features = batch_x[:, :self.global_features]  # [batch_size, 2]

        # Extract object features (skip object IDs)
        object_features = []
        for i in range(self.max_objects):
            start_idx = self.global_features + i * 5 + 1  # skip object ID
            end_idx = start_idx + 4
            if end_idx > batch_x.shape[1]:
                break
            obj_feat = batch_x[:, start_idx:end_idx]  # [batch_size, 4]
            object_features.append(obj_feat)

        # Pad object features to max_objects
        if len(object_features) < self.max_objects:
            padding = torch.zeros(
                batch_size,
                self.max_objects - len(object_features),
                4,
                device=batch_x.device
            )
            object_features = torch.cat([torch.stack(object_features, dim=1), padding], dim=1)
        else:
            object_features = torch.stack(object_features, dim=1)  # [batch_size, 18, 4]

        # Extract pairwise features
        pairwise_start = self.global_features + self.max_objects * 5
        pairwise_features = batch_x[:, pairwise_start:pairwise_start + self.pairwise_features]
        pairwise_features = pairwise_features.view(batch_size, -1, 2)  # [batch_size, 153, 2]

        # Embed features
        embedded_objects = self.object_embed(object_features)  # [batch_size, 18, 64]
        embedded_objects = self.pos_encoder(embedded_objects)

        embedded_pairwise = self.pairwise_embed(pairwise_features)  # [batch_size, 153, 32]

        embedded_global = self.global_embed(global_features)  # [batch_size, 32]
        embedded_global = embedded_global.unsqueeze(1)  # [batch_size, 1, 32]

        # Process object features with transformer
        transformer_out = self.transformer(embedded_objects)  # [batch_size, 18, 64]
        object_mean = transformer_out.mean(dim=1)  # [batch_size, 64]

        # Process pairwise features with attention
        attn_out, _ = self.pairwise_attention(
            embedded_pairwise, embedded_pairwise, embedded_pairwise
        )
        pairwise_mean = attn_out.mean(dim=1)  # [batch_size, 32]

        # Combine features
        combined = torch.cat([object_mean, pairwise_mean, embedded_global.squeeze(1)], dim=1)  # [batch_size, 64+32+32]

        # Classify
        logits = self.classifier(combined)  # [batch_size, 1]
        return logits.squeeze(1)  # [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=False)

    best_val_auc = 0.0
    best_model_state = None

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = F.binary_cross_entropy_with_logits(outputs, batch_y.float())
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == batch_y.float()).sum().item()
            total += batch_y.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = F.binary_cross_entropy_with_logits(outputs, batch_y.float())
                val_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == batch_y.float()).sum().item()
                total += batch_y.size(0)

                all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_labels, all_probs)

        # Update learning rate
        scheduler.step(val_auc)

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

def make_preprocessor():
    return MyPreprocessor()

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

