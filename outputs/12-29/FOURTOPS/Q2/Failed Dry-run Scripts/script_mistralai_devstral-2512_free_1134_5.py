
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
import math
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Batch
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.max_objects = 18
        self.object_slice = 5

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Extract global features (first 2 columns)
        global_features = X[:, :2].numpy()
        self.global_scaler.fit(global_features)

        # Extract object features (remaining columns)
        # Reshape to (n_samples, n_objects, 5)
        object_features = X[:, 2:].numpy()
        object_features = object_features.reshape(-1, self.max_objects * self.object_slice)

        # Only scale non-zero object features (where obj_id != 0)
        mask = object_features.reshape(-1, self.object_slice)[:, 0] != 0
        valid_features = object_features.reshape(-1, self.object_slice)[mask]
        if len(valid_features) > 0:
            self.scaler.fit(valid_features)
        return self

    def transform(self, X):
        # Scale global features
        global_features = X[:, :2]
        global_features = torch.from_numpy(self.global_scaler.transform(global_features.numpy())).float()

        # Scale object features
        object_features = X[:, 2:].reshape(-1, self.max_objects, self.object_slice)
        n_samples = object_features.shape[0]

        # Create mask for valid objects
        obj_mask = object_features[:, :, 0] != 0  # [n_samples, 18]

        # Flatten and scale valid features
        flat_objects = object_features.reshape(-1, self.object_slice)
        valid_mask = flat_objects[:, 0] != 0
        scaled_objects = torch.zeros_like(flat_objects)

        if valid_mask.any():
            valid_features = flat_objects[valid_mask].numpy()
            scaled_valid = torch.from_numpy(self.scaler.transform(valid_features)).float()
            scaled_objects[valid_mask] = scaled_valid

        # Reshape back
        scaled_objects = scaled_objects.reshape(n_samples, self.max_objects, self.object_slice)

        # Create edge indices for pairwise features
        edge_indices = []
        edge_attrs = []
        for i in range(n_samples):
            # Get valid object indices for this sample
            valid_idx = torch.where(obj_mask[i])[0]
            n_valid = len(valid_idx)

            # Create complete graph for valid objects
            if n_valid > 1:
                # Create all pairs
                rows, cols = [], []
                for j in range(n_valid):
                    for k in range(j+1, n_valid):
                        rows.append(valid_idx[j].item())
                        cols.append(valid_idx[k].item())
                        rows.append(valid_idx[k].item())
                        cols.append(valid_idx[j].item())

                edge_index = torch.tensor([rows, cols], dtype=torch.long)
                edge_indices.append(edge_index)

                # Compute pairwise features
                obj_j = scaled_objects[i, valid_idx[j]]
                obj_k = scaled_objects[i, valid_idx[k]]

                # Invariant mass (simplified approximation)
                E_j, pt_j, eta_j, phi_j = obj_j[1], obj_j[2], obj_j[3], obj_j[4]
                E_k, pt_k, eta_k, phi_k = obj_k[1], obj_k[2], obj_k[3], obj_k[4]

                # Approximate invariant mass (using transverse components)
                m_ij = torch.sqrt(2 * pt_j * pt_k * (torch.cosh(eta_j - eta_k) - torch.cos(phi_j - phi_k)))

                # Angular distance
                delta_eta = eta_j - eta_k
                delta_phi = phi_j - phi_k
                delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

                edge_attr = torch.stack([m_ij, delta_R], dim=0)
                edge_attrs.append(edge_attr)

            else:
                # No edges for single object
                edge_indices.append(torch.empty((2, 0), dtype=torch.long))
                edge_attrs.append(torch.empty((0, 2)))

        # Create list of Data objects
        data_list = []
        for i in range(n_samples):
            x = scaled_objects[i]  # [18, 5]
            edge_index = edge_indices[i]  # [2, num_edges]
            edge_attr = edge_attrs[i]  # [num_edges, 2]
            y = None  # Will be added by dataset

            # Add global features as node features for the first node
            if obj_mask[i].any():
                first_valid = torch.where(obj_mask[i])[0][0]
                x[first_valid, 1:3] = global_features[i]  # Add MET features

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        x, edge_index, edge_attr = sample_object.x, sample_object.edge_index, sample_object.edge_attr
        num_node_features = x.shape[1]
        num_edge_features = edge_attr.shape[1] if edge_attr.numel() > 0 else 2

        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(num_node_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Edge embedding
        self.edge_embed = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU()
        )

        # Transformer layers
        self.transformer_conv1 = TransformerConv(in_channels=32,
                                               out_channels=64,
                                               heads=4,
                                               edge_dim=32,
                                               dropout=0.1)
        self.transformer_conv2 = TransformerConv(in_channels=64,
                                               out_channels=64,
                                               heads=4,
                                               edge_dim=32,
                                               dropout=0.1)

        # Global attention
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, batch_x):
        # batch_x is a Batch object from PyG
        x = batch_x.x
        edge_index = batch_x.edge_index
        edge_attr = batch_x.edge_attr
        batch = batch_x.batch

        # Embed nodes
        x = self.node_embed(x)

        # Embed edges if they exist
        if edge_attr.numel() > 0:
            edge_attr = self.edge_embed(edge_attr)
        else:
            edge_attr = None

        # Transformer layers
        x = self.transformer_conv1(x, edge_index, edge_attr)
        x = nn.functional.relu(x)
        x = self.transformer_conv2(x, edge_index, edge_attr)
        x = nn.functional.relu(x)

        # Attention pooling
        attention_weights = self.attention(x)
        attention_weights = torch.softmax(attention_weights, dim=0)
        x = x * attention_weights
        x = global_mean_pool(x, batch)

        # Classifier
        out = self.classifier(x)
        return out.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

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
            yb = yb.float()

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * yb.size(0)
            preds = (torch.sigmoid(out) > 0.5).float()
            correct_train += (preds == yb).sum().item()
            total_train += yb.size(0)

        # Validation
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                yb = yb.float()

                out = model(xb)
                loss = criterion(out, yb)

                epoch_val_loss += loss.item() * yb.size(0)
                preds = (torch.sigmoid(out) > 0.5).float()
                correct_val += (preds == yb).sum().item()
                total_val += yb.size(0)

                all_preds.extend(torch.sigmoid(out).cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        # Calculate metrics
        train_loss = epoch_train_loss / total_train
        val_loss = epoch_val_loss / total_val
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
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

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')

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

