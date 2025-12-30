
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
        self.max_objects = 18
        self.global_features = 2
        self.obj_features = 5

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
        global_feats = X[:, :2]  # [n_events, 2]
        obj_feats = X[:, 2:].reshape(-1, self.max_objects * self.obj_features)  # [n_events, 90]

        # Fit scalers
        self.global_scaler.fit(global_feats)
        self.obj_scaler.fit(obj_feats)
        return self

    def transform(self, X):
        # Apply scaling
        global_feats = self.global_scaler.transform(X[:, :2])  # [n_events, 2]
        obj_feats = self.obj_scaler.transform(X[:, 2:].reshape(-1, self.max_objects * self.obj_features))  # [n_events, 90]

        # Reshape back
        X_scaled = torch.cat([
            torch.from_numpy(global_feats).float(),
            torch.from_numpy(obj_feats).float().reshape(-1, self.max_objects * self.obj_features)
        ], dim=1)

        return X_scaled

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        batch_size = sample_object.shape[0]
        total_features = sample_object.shape[1]

        # Global features (first 2) and object features (remaining)
        self.global_features = 2
        self.max_objects = 18
        self.obj_features = 5

        # Process global features
        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Process object features with GNN
        self.obj_embed = nn.Linear(self.obj_features, 32)
        self.conv1 = TransformerConv(32, 64, heads=4)
        self.conv2 = TransformerConv(64 * 4, 64, heads=4)
        self.conv3 = TransformerConv(64 * 4, 32, heads=4)

        # Combine features
        self.combined_mlp = nn.Sequential(
            nn.Linear(16 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, batch_x):
        # Separate global and object features
        global_feats = batch_x[:, :2]  # [batch_size, 2]
        obj_feats = batch_x[:, 2:].reshape(-1, self.max_objects, self.obj_features)  # [batch_size, 18, 5]

        # Process global features
        global_out = self.global_mlp(global_feats)  # [batch_size, 16]

        # Create graph data
        batch_list = []
        for i in range(obj_feats.shape[0]):
            # Get non-zero objects (mask)
            obj_mask = (obj_feats[i, :, 0] != 0).float()  # [18]
            valid_indices = torch.where(obj_mask == 1)[0]

            if len(valid_indices) == 0:
                # If no objects, use zero features
                x = torch.zeros(1, self.obj_features, device=obj_feats.device)
                edge_index = torch.tensor([[0], [0]], device=obj_feats.device)
            else:
                x = obj_feats[i, valid_indices, 1:]  # [n_objects, 4] (skip obj_id)
                # Create complete graph
                n_nodes = x.shape[0]
                edge_index = torch.combinations(torch.arange(n_nodes, device=obj_feats.device), r=2).t()
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

            # Add to batch
            batch_list.append(Data(x=x, edge_index=edge_index))

        # Process batch with GNN
        if len(batch_list) == 0:
            obj_out = torch.zeros(batch_x.shape[0], 32, device=batch_x.device)
        else:
            batch = DataLoader(batch_list, batch_size=len(batch_list)).__iter__().next().to(batch_x.device)
            x = self.obj_embed(batch.x)

            x = F.relu(self.conv1(x, batch.edge_index))
            x = F.relu(self.conv2(x, batch.edge_index))
            x = F.relu(self.conv3(x, batch.edge_index))

            obj_out = global_mean_pool(x, batch.batch)  # [batch_size, 32]

        # Combine features
        combined = torch.cat([global_out, obj_out], dim=1)
        out = self.combined_mlp(combined)  # [batch_size, 1]

        return torch.sigmoid(out)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    best_val_auc = 0
    patience = 10
    patience_counter = 0

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
            batch = batch.to(device)
            optimizer.zero_grad()

            # For standard DataLoader
            if isinstance(batch, torch.Tensor):
                x, y = batch[:, :-1], batch[:, -1]
            else:
                x, y = batch.x, batch.y

            y = y.float().unsqueeze(1)
            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            pred = (outputs > 0.5).float()
            correct += (pred == y).sum().item()
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
                batch = batch.to(device)

                if isinstance(batch, torch.Tensor):
                    x, y = batch[:, :-1], batch[:, -1]
                else:
                    x, y = batch.x, batch.y

                y = y.float().unsqueeze(1)
                outputs = model(x)
                loss = criterion(outputs, y)

                total_loss += loss.item() * y.size(0)
                pred = (outputs > 0.5).float()
                correct += (pred == y).sum().item()
                total += y.size(0)

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss = total_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
        except:
            val_auc = 0

        # Early stopping and learning rate scheduling
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
    model.load_state_dict(best_model)

    return model, train_losses, val_losses, train_accs, val_accs

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


