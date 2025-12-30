
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
        global_feats = X[:, :2]  # E_T_miss, phi_Et_miss
        obj_feats = X[:, 2:].reshape(-1, self.max_objects * self.obj_features)

        # Fit scalers
        self.global_scaler.fit(global_feats)
        self.obj_scaler.fit(obj_feats)
        return self

    def transform(self, X):
        # Apply scaling
        global_feats = self.global_scaler.transform(X[:, :2])
        obj_feats = self.obj_scaler.transform(X[:, 2:].reshape(-1, self.max_objects * self.obj_features))

        # Reshape back
        obj_feats = obj_feats.reshape(X.shape[0], -1)
        X_transformed = np.concatenate([global_feats, obj_feats], axis=1)
        return X_transformed

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        if isinstance(sample_object, torch.Tensor):
            # Flat tensor case
            self.global_dim = 2
            self.obj_dim = 5
            self.max_objects = 18
        else:
            # PyG Data case
            self.global_dim = sample_object.x.shape[1] - 5  # Assuming first 2 are global
            self.obj_dim = 5
            self.max_objects = sample_object.num_nodes

        # Global feature processing
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Object feature processing
        self.obj_encoder = nn.Sequential(
            nn.Linear(self.obj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Pairwise features
        self.pairwise_encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Transformer for object interactions
        self.transformer_conv1 = TransformerConv(32, 32, heads=4)
        self.transformer_conv2 = TransformerConv(32, 32, heads=4)

        # Global attention
        self.global_attention = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute_pairwise_features(self, obj_feats, obj_mask):
        # obj_feats: [batch_size * max_objects, 32]
        # obj_mask: [batch_size * max_objects] (1 for real objects, 0 for padding)

        batch_size = obj_mask.shape[0] // self.max_objects
        obj_feats = obj_feats.view(batch_size, self.max_objects, -1)
        obj_mask = obj_mask.view(batch_size, self.max_objects)

        # Compute pairwise features
        pairwise_feats = []
        for i in range(self.max_objects):
            for j in range(i+1, self.max_objects):
                # Get features for objects i and j
                feat_i = obj_feats[:, i, :]  # [batch_size, 32]
                feat_j = obj_feats[:, j, :]  # [batch_size, 32]

                # Compute invariant mass (approximate)
                E_i = feat_i[:, 0]  # Using first feature as proxy for energy
                E_j = feat_j[:, 0]
                pT_i = feat_i[:, 1]
                pT_j = feat_j[:, 1]
                eta_i = feat_i[:, 2]
                eta_j = feat_j[:, 2]
                phi_i = feat_i[:, 3]
                phi_j = feat_j[:, 3]

                # Approximate invariant mass
                m_ij = torch.sqrt(2 * pT_i * pT_j * (torch.cosh(eta_i - eta_j) - torch.cos(phi_i - phi_j)))

                # Angular distance
                delta_eta = eta_i - eta_j
                delta_phi = phi_i - phi_j
                delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

                # Combine features
                pairwise_feat = torch.cat([
                    feat_i,
                    feat_j,
                    m_ij.unsqueeze(1),
                    delta_R.unsqueeze(1)
                ], dim=1)

                pairwise_feats.append(pairwise_feat)

        # Stack all pairwise features
        if pairwise_feats:
            pairwise_feats = torch.stack(pairwise_feats, dim=1)  # [batch_size, num_pairs, 66]
            pairwise_feats = self.pairwise_encoder(pairwise_feats)
            pairwise_feats = torch.mean(pairwise_feats, dim=1)  # [batch_size, 16]
        else:
            pairwise_feats = torch.zeros(batch_size, 16, device=obj_feats.device)

        return pairwise_feats

    def forward(self, batch_x):
        if isinstance(batch_x, torch.Tensor):
            # Flat tensor case - convert to PyG format
            batch_size = batch_x.shape[0]
            global_feats = batch_x[:, :2]
            obj_feats = batch_x[:, 2:].view(batch_size, self.max_objects, self.obj_dim)

            # Create mask for real objects (non-zero padding)
            obj_mask = (obj_feats[:, :, 0] != 0).float()  # Using first feature as indicator

            # Encode features
            global_encoded = self.global_encoder(global_feats)
            obj_encoded = self.obj_encoder(obj_feats.view(-1, self.obj_dim)).view(batch_size, self.max_objects, -1)

            # Compute pairwise features
            pairwise_feats = self.compute_pairwise_features(obj_encoded.view(-1, 32), obj_mask.view(-1))

            # Create PyG data object
            edge_index = []
            for i in range(batch_size):
                # Create complete graph for each batch item
                nodes = torch.arange(self.max_objects, device=batch_x.device)
                edges = torch.combinations(nodes, 2).t()
                edge_index.append(edges + i * self.max_objects)

            if edge_index:
                edge_index = torch.cat(edge_index, dim=1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=batch_x.device)

            data = Data(
                x=obj_encoded.view(-1, 32),
                edge_index=edge_index,
                batch=torch.arange(batch_size, device=batch_x.device).repeat_interleave(self.max_objects)
            )
        else:
            # Already PyG format
            data = batch_x
            global_encoded = self.global_encoder(data.global_feats)
            obj_encoded = self.obj_encoder(data.x)
            pairwise_feats = self.compute_pairwise_features(obj_encoded, data.obj_mask)

        # Transformer layers
        x = self.transformer_conv1(obj_encoded, data.edge_index)
        x = F.relu(x)
        x = self.transformer_conv2(x, data.edge_index)
        x = F.relu(x)

        # Global pooling
        x = global_mean_pool(x, data.batch)

        # Combine with global features and pairwise features
        combined = torch.cat([x, global_encoded, pairwise_feats], dim=1)

        # Attention mechanism
        attention_weights = torch.softmax(self.global_attention(combined), dim=0)
        attended = combined * attention_weights

        # Classifier
        out = self.classifier(attended)
        return out.squeeze(1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
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
        epoch_train_acc = 0
        train_samples = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.float())

            # Compute accuracy
            preds = (torch.sigmoid(out) > 0.5).float()
            acc = (preds == yb).float().mean()

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * yb.size(0)
            epoch_train_acc += acc.item() * yb.size(0)
            train_samples += yb.size(0)

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        val_samples = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                out = model(xb)
                loss = criterion(out, yb.float())

                # Compute accuracy
                preds = (torch.sigmoid(out) > 0.5).float()
                acc = (preds == yb).float().mean()

                epoch_val_loss += loss.item() * yb.size(0)
                epoch_val_acc += acc.item() * yb.size(0)
                val_samples += yb.size(0)

                val_preds.append(torch.sigmoid(out).cpu())
                val_targets.append(yb.cpu())

        # Calculate metrics
        train_loss = epoch_train_loss / train_samples
        train_acc = epoch_train_acc / train_samples
        val_loss = epoch_val_loss / val_samples
        val_acc = epoch_val_acc / val_samples

        # Calculate AUC
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_auc = roc_auc_score(val_targets, val_preds)

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

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_losses, val_losses, train_accs, val_accs

# Helper function for AUC calculation
from sklearn.metrics import roc_auc_score

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

