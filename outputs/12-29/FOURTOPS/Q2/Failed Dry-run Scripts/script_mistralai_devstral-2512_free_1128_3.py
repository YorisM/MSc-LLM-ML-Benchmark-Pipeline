
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
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import TransformerConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
import math

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
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Separate global and object features
        global_feats = X[:, :self.global_features]
        obj_feats = X[:, self.global_features:].reshape(-1, self.max_objects * self.obj_features)

        # Fit scalers
        self.global_scaler.fit(global_feats)
        self.obj_scaler.fit(obj_feats)
        return self

    def transform(self, X):
        # Apply scaling
        global_feats = self.global_scaler.transform(X[:, :self.global_features])
        obj_feats = self.obj_scaler.transform(X[:, self.global_features:].reshape(-1, self.max_objects * self.obj_features))

        # Reshape back
        obj_feats = obj_feats.reshape(X.shape[0], -1)
        X_transformed = np.concatenate([global_feats, obj_feats], axis=1)
        return torch.from_numpy(X_transformed).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        if isinstance(sample_object, torch.Tensor):
            self.global_dim = 2
            self.obj_dim = 5
            self.max_objects = 18
        else:  # PyG Data/Batch
            self.global_dim = sample_object.x.shape[1] - 5  # Assuming first 2 are global
            self.obj_dim = 5
            self.max_objects = sample_object.num_nodes

        # Feature engineering layers
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Object feature processing
        self.obj_encoder = nn.Sequential(
            nn.Linear(self.obj_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Pairwise feature computation
        self.pairwise_mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Transformer layers
        self.transformer_conv1 = TransformerConv(32, 32, heads=4, dropout=0.1)
        self.transformer_conv2 = TransformerConv(32, 32, heads=4, dropout=0.1)

        # Global aggregation
        self.global_agg = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def compute_pairwise_features(self, obj_feats, obj_mask):
        # obj_feats: [batch_size, num_objects, obj_dim]
        # obj_mask: [batch_size, num_objects] (1 for real objects, 0 for padding)

        batch_size, num_objects, _ = obj_feats.shape
        device = obj_feats.device

        # Compute invariant mass and delta R for all pairs
        pairwise_feats = []

        # Get energy and momentum components
        E = obj_feats[:, :, 1].unsqueeze(2)  # [batch, obj, 1]
        pT = obj_feats[:, :, 2].unsqueeze(2)
        eta = obj_feats[:, :, 3].unsqueeze(2)
        phi = obj_feats[:, :, 4].unsqueeze(2)

        # Compute all pairs
        for i in range(num_objects):
            for j in range(i+1, num_objects):
                # Only consider pairs where both objects are real
                mask = obj_mask[:, i] * obj_mask[:, j]
                if mask.sum() == 0:
                    continue

                # Invariant mass: m = sqrt(2 * pT_i * pT_j * (cosh(eta_i - eta_j) - cos(phi_i - phi_j)))
                eta_diff = eta[:, i] - eta[:, j]
                phi_diff = phi[:, i] - phi[:, j]
                cosh_eta = torch.cosh(eta_diff)
                cos_phi = torch.cos(phi_diff)
                inv_mass = torch.sqrt(2 * pT[:, i] * pT[:, j] * (cosh_eta - cos_phi))

                # Delta R
                delta_eta = eta[:, i] - eta[:, j]
                delta_phi = phi_diff
                delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

                # Combine features
                pair_feat = torch.cat([
                    inv_mass.unsqueeze(1),
                    delta_R.unsqueeze(1),
                    (E[:, i] + E[:, j]).unsqueeze(1),
                    (pT[:, i] + pT[:, j]).unsqueeze(1)
                ], dim=1)

                pairwise_feats.append(pair_feat * mask.unsqueeze(1))

        if not pairwise_feats:
            return torch.zeros(batch_size, 0, 4).to(device)

        pairwise_feats = torch.cat(pairwise_feats, dim=1)
        return pairwise_feats

    def forward(self, batch_x):
        if isinstance(batch_x, torch.Tensor):
            # Flat tensor case - reshape to separate global and object features
            global_feats = batch_x[:, :2]  # [batch, 2]
            obj_feats = batch_x[:, 2:].reshape(batch_x.size(0), 18, 5)  # [batch, 18, 5]

            # Create object mask (non-zero padding)
            obj_mask = (obj_feats[:, :, 0] != 0).float()  # [batch, 18]

            # Encode features
            global_encoded = self.global_encoder(global_feats)  # [batch, 16]

            # Process object features
            obj_encoded = self.obj_encoder(obj_feats)  # [batch, 18, 16]

            # Compute pairwise features
            pairwise_feats = self.compute_pairwise_features(obj_feats, obj_mask)  # [batch, num_pairs, 4]

            if pairwise_feats.size(1) > 0:
                pairwise_encoded = self.pairwise_mlp(pairwise_feats)  # [batch, num_pairs, 32]
                pairwise_encoded = pairwise_encoded.mean(dim=1)  # [batch, 32]
            else:
                pairwise_encoded = torch.zeros(batch_x.size(0), 32).to(batch_x.device)

            # Combine features
            combined = torch.cat([global_encoded, pairwise_encoded], dim=1)

            # Final prediction
            logits = self.global_agg(combined).squeeze(1)
            return logits

        else:  # PyG Data/Batch case
            global_feats = batch_x.x[:, :2]
            obj_feats = batch_x.x[:, 2:]

            # For PyG, we need to handle the graph structure
            # This is a simplified version - in practice you'd want proper graph construction
            obj_encoded = self.obj_encoder(obj_feats)

            # Create edge indices for all possible pairs
            num_nodes = batch_x.num_nodes
            edge_index = []
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    edge_index.append([i, j])
                    edge_index.append([j, i])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(batch_x.x.device)

            # Apply transformer convolutions
            x = self.transformer_conv1(obj_encoded, edge_index)
            x = F.relu(x)
            x = self.transformer_conv2(x, edge_index)
            x = F.relu(x)

            # Global pooling
            x = global_mean_pool(x, batch_x.batch)

            # Combine with global features
            global_encoded = self.global_encoder(global_feats)
            combined = torch.cat([global_encoded, x], dim=1)

            # Final prediction
            logits = self.global_agg(combined).squeeze(1)
            return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0
    best_model = None

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.float().unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * yb.size(0)
            preds = (torch.sigmoid(out) > 0.5).float()
            correct_train += (preds.squeeze(1) == yb).sum().item()
            total_train += yb.size(0)

        # Validation
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                out = model(xb)
                loss = criterion(out, yb.float().unsqueeze(1))

                epoch_val_loss += loss.item() * yb.size(0)
                preds = (torch.sigmoid(out) > 0.5).float()
                correct_val += (preds.squeeze(1) == yb).sum().item()
                total_val += yb.size(0)

                val_probs.extend(torch.sigmoid(out).squeeze(1).cpu().numpy())
                val_labels.extend(yb.cpu().numpy())

        # Calculate metrics
        train_loss = epoch_train_loss / total_train
        val_loss = epoch_val_loss / total_val
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except:
            val_auc = 0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping and model saving
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model.state_dict()
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

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

