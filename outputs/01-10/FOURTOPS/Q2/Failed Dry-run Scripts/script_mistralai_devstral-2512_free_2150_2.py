
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
import math
from sklearn.preprocessing import StandardScaler
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool

# ---------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
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

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.obj_scaler = StandardScaler()
        self.max_objects = 18
        self.global_features = 2
        self.obj_features = 5

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
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
        # Separate global and object features
        global_feats = X[:, :2]  # E_T_miss, phi_Et_miss
        obj_feats = X[:, 2:].reshape(-1, self.max_objects * self.obj_features)

        # Fit scalers
        self.global_scaler.fit(global_feats)
        self.obj_scaler.fit(obj_feats)
        return self

    def transform(self, X):
        # Apply scaling
        global_feats = X[:, :2]
        obj_feats = X[:, 2:].reshape(-1, self.max_objects * self.obj_features)

        global_scaled = self.global_scaler.transform(global_feats)
        obj_scaled = self.obj_scaler.transform(obj_feats)

        # Reshape back
        X_scaled = np.zeros_like(X)
        X_scaled[:, :2] = global_scaled
        X_scaled[:, 2:] = obj_scaled.reshape(-1, self.max_objects * self.obj_features)
        return X_scaled

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        if isinstance(sample_object, torch.Tensor):
            # Lane A: Dense tensor
            self.input_dim = sample_object.shape[1]
            self.lane = 'dense'
        else:
            # Lane B: PyG Data object
            self.input_dim = sample_object.x.shape[1]
            self.lane = 'graph'

        # Global features processing
        self.global_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Object features processing
        if self.lane == 'dense':
            # For dense input, we'll process objects separately
            self.obj_mlp = nn.Sequential(
                nn.Linear(5, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )

            # Transformer for object interactions
            encoder_layer = TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=64)
            self.transformer = TransformerEncoder(encoder_layer, num_layers=3)

            # Final classifier
            self.classifier = nn.Sequential(
                nn.Linear(16 + 16, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        else:
            # For graph input
            self.node_encoder = nn.Sequential(
                nn.Linear(5, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )

            # Graph convolution layers
            from torch_geometric.nn import GATConv
            self.conv1 = GATConv(16, 32, heads=4)
            self.conv2 = GATConv(32 * 4, 32, heads=4)
            self.conv3 = GATConv(32 * 4, 16, heads=1)

            # Global pooling and classifier
            self.classifier = nn.Sequential(
                nn.Linear(16 + 16, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

    def forward(self, batch_x):
        if self.lane == 'dense':
            # Separate global and object features
            global_feats = batch_x[:, :2]  # [B, 2]
            obj_feats = batch_x[:, 2:].view(-1, 18, 5)  # [B, 18, 5]

            # Process global features
            global_encoded = self.global_mlp(global_feats)  # [B, 16]

            # Process object features
            obj_encoded = self.obj_mlp(obj_feats)  # [B, 18, 16]

            # Create pairwise features
            B, N, D = obj_encoded.shape
            obj_encoded = obj_encoded.view(B * N, D)

            # Compute invariant mass and delta R for all pairs
            # First get all object properties
            E = obj_feats[:, :, 1].view(B * N)  # Energy
            pT = obj_feats[:, :, 2].view(B * N)  # Transverse momentum
            eta = obj_feats[:, :, 3].view(B * N)  # Pseudorapidity
            phi = obj_feats[:, :, 4].view(B * N)  # Azimuth

            # Compute pairwise features
            pairwise_feats = []
            for i in range(N):
                for j in range(i+1, N):
                    # Invariant mass
                    E_i = E[i::N]
                    E_j = E[j::N]
                    pT_i = pT[i::N]
                    pT_j = pT[j::N]
                    # Approximate invariant mass (simplified)
                    m_ij = torch.sqrt(2 * pT_i * pT_j * (torch.cosh(eta[i::N] - eta[j::N]) - torch.cos(phi[i::N] - phi[j::N])))

                    # Delta R
                    delta_eta = eta[i::N] - eta[j::N]
                    delta_phi = phi[i::N] - phi[j::N]
                    delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))  # Handle angle wrapping
                    delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

                    pairwise_feats.append(m_ij.unsqueeze(1))
                    pairwise_feats.append(delta_R.unsqueeze(1))

            if pairwise_feats:
                pairwise_feats = torch.cat(pairwise_feats, dim=1)  # [B, num_pairs*2]
                # Add pairwise features to object features
                obj_encoded = torch.cat([obj_encoded, pairwise_feats], dim=1)

            # Reshape for transformer
            obj_encoded = obj_encoded.view(B, N, -1)

            # Transformer processing
            transformer_out = self.transformer(obj_encoded)  # [B, N, D]
            obj_pooled = transformer_out.mean(dim=1)  # [B, D]

            # Combine with global features
            combined = torch.cat([global_encoded, obj_pooled], dim=1)
            out = self.classifier(combined)
        else:
            # Graph processing
            x = batch_x.x  # [N, 5]
            edge_index = batch_x.edge_index
            batch = batch_x.batch

            # Process node features
            x = self.node_encoder(x)  # [N, 16]

            # Graph convolutions
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)

            # Global pooling
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_global = torch.cat([x_mean, x_max], dim=1)  # [B, 32]

            # Process global features if available
            if hasattr(batch_x, 'global_feats'):
                global_feats = self.global_mlp(batch_x.global_feats)  # [B, 16]
                x_global = torch.cat([x_global, global_feats], dim=1)

            # Classifier
            out = self.classifier(x_global)

        return out.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_val_auc = 0
    best_model = None

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                X, y = batch
                X = X.to(device)
                y = y.to(device)
            else:
                # PyG batch
                X = batch.to(device)
                y = batch.y.to(device)

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y.float().view(-1, 1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            pred = (out > 0).float()
            correct += (pred.view(-1) == y.float()).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    X, y = batch
                    X = X.to(device)
                    y = y.to(device)
                else:
                    X = batch.to(device)
                    y = batch.y.to(device)

                out = model(X)
                loss = criterion(out, y.float().view(-1, 1))

                total_loss += loss.item() * y.size(0)
                pred = (out > 0).float()
                correct += (pred.view(-1) == y.float()).sum().item()
                total += y.size(0)

                all_preds.extend(out.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss = total_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
            scheduler.step(val_auc)
        except:
            val_auc = 0

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model.state_dict()
            torch.save(best_model, 'best_model.pth')

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

