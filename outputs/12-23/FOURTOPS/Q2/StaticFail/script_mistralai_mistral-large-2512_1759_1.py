
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
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
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
            "pin_memory": True if torch.cuda.is_available() else False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Fit global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :self.global_feature_size]  # [N, 2]
        self.scaler.fit(global_features)

        # Fit per-object features
        for i in range(self.max_objects):
            start = self.global_feature_size + i * self.obj_feature_size
            end = start + self.obj_feature_size
            obj_features = X[:, start:end]  # [N, 5]
            # Only scale kinematic features (skip obj_id)
            kinematic_features = obj_features[:, 1:]  # [N, 4]
            self.obj_scalers[i].fit(kinematic_features)

        return self

    def transform(self, X):
        # Transform global features
        global_features = X[:, :self.global_feature_size]  # [N, 2]
        global_features = self.scaler.transform(global_features)
        X_transformed = np.zeros_like(X)
        X_transformed[:, :self.global_feature_size] = global_features

        # Transform per-object features
        for i in range(self.max_objects):
            start = self.global_feature_size + i * self.obj_feature_size
            end = start + self.obj_feature_size
            obj_features = X[:, start:end]  # [N, 5]
            obj_id = obj_features[:, 0:1]  # [N, 1]
            kinematic_features = obj_features[:, 1:]  # [N, 4]
            kinematic_features = self.obj_scalers[i].transform(kinematic_features)
            X_transformed[:, start:end] = np.concatenate([obj_id, kinematic_features], axis=1)

        # Feature engineering: add pairwise features
        N = X_transformed.shape[0]
        pairwise_features = np.zeros((N, self.max_objects, self.max_objects, 2))  # [N, 18, 18, 2]

        for i in range(self.max_objects):
            for j in range(i+1, self.max_objects):
                # Get object features
                start_i = self.global_feature_size + i * self.obj_feature_size
                start_j = self.global_feature_size + j * self.obj_feature_size
                eta_i = X_transformed[:, start_i + 3]  # [N]
                phi_i = X_transformed[:, start_i + 4]  # [N]
                pt_i = X_transformed[:, start_i + 2]   # [N]
                eta_j = X_transformed[:, start_j + 3]  # [N]
                phi_j = X_transformed[:, start_j + 4]  # [N]
                pt_j = X_transformed[:, start_j + 2]   # [N]

                # Calculate deltaR
                delta_eta = eta_i - eta_j
                delta_phi = phi_i - phi_j
                delta_phi = np.mod(delta_phi + np.pi, 2 * np.pi) - np.pi  # wrap to [-pi, pi]
                deltaR = np.sqrt(delta_eta**2 + delta_phi**2)  # [N]

                # Calculate invariant mass approximation (m_ij^2 = 2 pT_i pT_j (cosh(delta_eta) - cos(delta_phi)))
                cosh_delta_eta = np.cosh(delta_eta)
                cos_delta_phi = np.cos(delta_phi)
                m_ij_squared = 2 * pt_i * pt_j * (cosh_delta_eta - cos_delta_phi)
                m_ij = np.sqrt(np.maximum(m_ij_squared, 0))  # [N]

                pairwise_features[:, i, j, 0] = deltaR
                pairwise_features[:, j, i, 0] = deltaR
                pairwise_features[:, i, j, 1] = m_ij
                pairwise_features[:, j, i, 1] = m_ij

        # Flatten pairwise features and concatenate
        pairwise_flat = pairwise_features.reshape(N, -1)  # [N, 18*18*2]
        X_transformed = np.concatenate([X_transformed, pairwise_flat], axis=1)

        return X_transformed

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class ParticleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Object embedding
        self.obj_embedding = nn.Linear(5, hidden_dim)  # obj_id + 4 kinematic features

        # Global features embedding
        self.global_embedding = nn.Linear(2, hidden_dim)

        # Pairwise features embedding
        self.pairwise_embedding = nn.Linear(2, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, batch_x):
        # batch_x shape: [batch_size, 92 + 18*18*2] = [batch_size, 730]
        batch_size = batch_x.size(0)

        # Extract global features
        global_features = batch_x[:, :2]  # [batch_size, 2]
        global_embed = self.global_embedding(global_features)  # [batch_size, hidden_dim]

        # Extract object features (18 objects, 5 features each)
        obj_features = []
        for i in range(18):
            start = 2 + i * 5
            end = start + 5
            obj_feat = batch_x[:, start:end]  # [batch_size, 5]
            obj_embed = self.obj_embedding(obj_feat)  # [batch_size, hidden_dim]
            obj_features.append(obj_embed)
        obj_features = torch.stack(obj_features, dim=1)  # [batch_size, 18, hidden_dim]

        # Extract pairwise features (18*18*2)
        pairwise_start = 2 + 18*5
        pairwise_features = batch_x[:, pairwise_start:]  # [batch_size, 648]
        pairwise_features = pairwise_features.reshape(batch_size, 18, 18, 2)  # [batch_size, 18, 18, 2]

        # Create pairwise embeddings
        pairwise_embed = self.pairwise_embedding(pairwise_features)  # [batch_size, 18, 18, hidden_dim]

        # Combine object features with pairwise information
        # For each object, aggregate information from all other objects
        obj_features_expanded = obj_features.unsqueeze(2)  # [batch_size, 18, 1, hidden_dim]
        pairwise_contrib = pairwise_embed.mean(dim=2)  # [batch_size, 18, hidden_dim]
        obj_features = obj_features + pairwise_contrib  # [batch_size, 18, hidden_dim]

        # Add global context to each object
        global_expanded = global_embed.unsqueeze(1).expand(-1, 18, -1)  # [batch_size, 18, hidden_dim]
        obj_features = obj_features + global_expanded  # [batch_size, 18, hidden_dim]

        # Transformer processing
        transformer_out = self.transformer(obj_features)  # [batch_size, 18, hidden_dim]

        # Global pooling
        pooled = transformer_out.mean(dim=1)  # [batch_size, hidden_dim]

        # Classification
        logits = self.classifier(pooled)  # [batch_size, 1]
        return logits.squeeze(-1)

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Calculate input dimension after preprocessing
        # Original: 92 features
        # After adding pairwise: 92 + 18*18*2 = 730
        self.model = ParticleTransformer(input_dim=730)

    def forward(self, batch_x):
        return self.model(batch_x)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
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
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            train_targets.append(batch_y.detach().cpu().numpy())

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
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()

                logits = model(batch_x)
                loss = F.binary_cross_entropy_with_logits(logits, batch_y)

                val_loss += loss.item()
                val_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
                val_targets.append(batch_y.detach().cpu().numpy())

        val_loss /= len(val_loader)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

        # Update learning rate
        scheduler.step(val_auc)

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

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

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


