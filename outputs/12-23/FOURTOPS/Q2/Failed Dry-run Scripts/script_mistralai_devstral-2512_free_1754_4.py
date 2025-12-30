
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
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.loader import DataLoader

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.obj_scaler = StandardScaler()

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
        # Separate global and object features
        global_features = X[:, :2]  # E_T_miss, phi_Et_miss
        obj_features = X[:, 2:].reshape(-1, 18, 5)[:, :, 1:]  # Remove obj_id, keep E, pT, eta, phi

        # Flatten for scaling
        global_flat = global_features.reshape(-1, 2)
        obj_flat = obj_features.reshape(-1, 4)

        # Fit scalers
        self.global_scaler.fit(global_flat)
        self.obj_scaler.fit(obj_flat)
        return self

    def transform(self, X):
        # Apply scaling
        global_features = X[:, :2]
        obj_features = X[:, 2:].reshape(-1, 18, 5)

        # Scale global features
        global_scaled = self.global_scaler.transform(global_features)

        # Extract and scale object features (remove obj_id)
        obj_ids = obj_features[:, :, 0]  # Keep obj_ids for masking
        obj_kinematics = obj_features[:, :, 1:]
        obj_scaled = self.obj_scaler.transform(obj_kinematics.reshape(-1, 4)).reshape(-1, 18, 4)

        # Combine back
        obj_features = torch.cat([
            obj_ids.unsqueeze(-1),
            torch.from_numpy(obj_scaled).float()
        ], dim=-1).reshape(X.shape[0], -1)

        # Create PyG Data objects
        data_list = []
        for i in range(X.shape[0]):
            # Get non-zero objects (mask)
            obj_mask = obj_ids[i] != 0
            num_objects = obj_mask.sum().item()

            if num_objects == 0:
                # Handle empty events (shouldn't happen but just in case)
                x = torch.zeros(1, 4)
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                # Object features (E, pT, eta, phi)
                x = torch.from_numpy(obj_scaled[i][obj_mask])

                # Create complete graph
                edge_index = torch.combinations(torch.arange(num_objects), r=2).t()
                if edge_index.numel() == 0:
                    edge_index = torch.empty((2, 0), dtype=torch.long)

            # Global features
            u = torch.from_numpy(global_scaled[i])

            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                u=u,
                y=torch.tensor([0], dtype=torch.float)  # Dummy, will be replaced in dataset
            )
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        num_obj_features = sample_object.x.shape[1]  # Should be 4 (E, pT, eta, phi)
        num_global_features = sample_object.u.shape[0]  # Should be 2

        # Transformer layers for object interactions
        self.transformer1 = TransformerConv(
            in_channels=num_obj_features,
            out_channels=64,
            heads=4,
            concat=True,
            beta=True
        )
        self.transformer2 = TransformerConv(
            in_channels=64 * 4,
            out_channels=64,
            heads=4,
            concat=True,
            beta=True
        )

        # Global interaction
        self.global_interaction = nn.Sequential(
            nn.Linear(num_global_features + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x, edge_index, u = data.x, data.edge_index, data.u

        # Transformer layers
        x1 = F.relu(self.transformer1(x, edge_index))
        x2 = F.relu(self.transformer2(x1, edge_index))

        # Global pooling
        x_global = global_mean_pool(x2, torch.zeros(x2.size(0), dtype=torch.long, device=x2.device))

        # Combine with global features
        combined = torch.cat([u, x_global], dim=-1)
        global_feat = self.global_interaction(combined)

        # Final classification
        out = self.classifier(torch.cat([x_global, global_feat], dim=-1))
        return torch.sigmoid(out)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch).squeeze()
            targets = batch.y.float()

            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs
            preds = (outputs > 0.5).float()
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                outputs = model(batch).squeeze()
                targets = batch.y.float()

                loss = criterion(outputs, targets)
                val_loss += loss.item() * batch.num_graphs
                preds = (outputs > 0.5).float()
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        # Calculate metrics
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val

        # Store history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Early stopping and learning rate scheduling
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                model.load_state_dict(best_model)
                break

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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


