
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
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.obj_scaler = StandardScaler()
        self.fitted = False

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
            "eval_overrides": {"shuffle": False, "batch_size": 256}
        }

    def fit(self, X, y=None):
        # Separate global and object features
        global_feats = X[:, :2]  # E_T_miss, phi_Et_miss
        obj_feats = X[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]

        # Flatten object features for scaling (excluding obj_id which is categorical)
        obj_feats_flat = obj_feats[:, :, 1:].reshape(-1, 4)  # [N*18, 4]

        # Fit scalers
        self.global_scaler.fit(global_feats)
        self.obj_scaler.fit(obj_feats_flat)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted")

        # Apply scaling
        global_feats = self.global_scaler.transform(X[:, :2].numpy())
        obj_feats = X[:, 2:].reshape(-1, 18, 5).numpy()

        # Scale object features (keep obj_id as is)
        obj_feats[:, :, 1:] = self.obj_scaler.transform(obj_feats[:, :, 1:].reshape(-1, 4)).reshape(-1, 18, 4)

        # Convert to PyG Data objects
        data_list = []
        for i in range(X.shape[0]):
            # Get non-zero objects (where obj_id != 0)
            obj_mask = obj_feats[i, :, 0] != 0
            num_objects = obj_mask.sum()

            if num_objects == 0:
                # Handle empty events (shouldn't happen but just in case)
                x = torch.zeros((1, 4), dtype=torch.float32)
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                # Object features (E, pT, eta, phi)
                x = torch.tensor(obj_feats[i, obj_mask, 1:], dtype=torch.float32)  # [num_objects, 4]

                # Create complete graph edges
                edges = []
                for src in range(num_objects):
                    for dst in range(num_objects):
                        if src != dst:
                            edges.append([src, dst])
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]

            # Global features
            global_feat = torch.tensor(global_feats[i], dtype=torch.float32)  # [2]

            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                global_feat=global_feat,
                y=torch.tensor([0], dtype=torch.long)  # placeholder, will be set in dataset
            )
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Extract dimensions from sample
        if isinstance(sample_object, Data):
            # PyG lane
            self.input_dim = sample_object.x.shape[1]  # 4 (E, pT, eta, phi)
            self.global_dim = sample_object.global_feat.shape[0]  # 2
            self.use_pyg = True
        else:
            # Dense lane (fallback)
            self.input_dim = sample_object.shape[1]
            self.use_pyg = False

        # Particle feature embedding
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Transformer layers for particle interactions
        self.transformer1 = TransformerConv(32, 32, heads=4, concat=False)
        self.transformer2 = TransformerConv(32, 32, heads=4, concat=False)

        # Global feature processing
        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        # Attention for global-particle interaction
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(32 + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, batch_x):
        if self.use_pyg:
            # PyG lane
            x = batch_x.x
            edge_index = batch_x.edge_index
            global_feat = batch_x.global_feat
            batch = batch_x.batch

            # Embed particle features
            x = self.embedding(x)  # [N, 32]

            # Transformer layers
            x = F.relu(self.transformer1(x, edge_index))
            x = F.relu(self.transformer2(x, edge_index))

            # Global pooling
            x_pool = global_mean_pool(x, batch)  # [B, 32]

            # Process global features
            global_feat = self.global_mlp(global_feat)  # [B, 8]

            # Combine features
            combined = torch.cat([x_pool, global_feat], dim=1)

            # Final classification
            out = self.classifier(combined).squeeze(-1)
        else:
            # Dense lane fallback
            x = self.embedding(batch_x)
            out = self.classifier(x).squeeze(-1)

        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_model = None

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            if isinstance(batch, Data):
                # PyG lane
                batch = batch.to(device)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch.y.float())
            else:
                # Dense lane
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.float())

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.y.size(0)

            # Calculate accuracy
            preds = (torch.sigmoid(outputs) > 0.5).float()
            if isinstance(batch, Data):
                correct += (preds == batch.y.float()).sum().item()
                total += batch.y.size(0)
            else:
                correct += (preds == y_batch.float()).sum().item()
                total += y_batch.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, Data):
                    batch = batch.to(device)
                    outputs = model(batch)
                    loss = criterion(outputs, batch.y.float())
                else:
                    X_batch, y_batch = batch
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch.float())

                val_loss += loss.item() * batch.y.size(0)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                if isinstance(batch, Data):
                    correct += (preds == batch.y.float()).sum().item()
                    total += batch.y.size(0)
                else:
                    correct += (preds == y_batch.float()).sum().item()
                    total += y_batch.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        # Update scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    # Load best model
    model.load_state_dict(best_model)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

