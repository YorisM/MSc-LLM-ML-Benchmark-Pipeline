
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data/train"
TAG      = "REDVID_10-50_linear_frac0.05"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def split_X_y(evt):
    X = np.column_stack([
        evt["hit_r"].astype(np.float32),
        evt["hit_theta"].astype(np.float32),
        evt["hit_z"].astype(np.float32),
        evt["layer_id"].astype(np.float32)
    ])
    y = evt["track_id"].astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, labels = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, labels)

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_encoder = None
        self.n_layers = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Collect all hits for scaling
        all_hits = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_hits[:, :3])  # Scale r, theta, z

        # Encode layer_id as one-hot
        unique_layers = np.unique(all_hits[:, 3])
        self.n_layers = len(unique_layers)
        self.layer_encoder = {layer: idx for idx, layer in enumerate(unique_layers)}
        return self

    def transform(self, X):
        # X: [N_hits, 4] (r, theta, z, layer_id)
        X = X.numpy() if isinstance(X, torch.Tensor) else X

        # Scale spatial coordinates
        X_scaled = self.scaler.transform(X[:, :3])

        # One-hot encode layer_id
        layer_onehot = np.zeros((X.shape[0], self.n_layers))
        layer_indices = np.array([self.layer_encoder[l] for l in X[:, 3]])
        layer_onehot[np.arange(X.shape[0]), layer_indices] = 1

        # Combine features: [r_scaled, theta_scaled, z_scaled, layer_onehot]
        X_transformed = np.hstack([X_scaled, layer_onehot])

        return torch.from_numpy(X_transformed).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # example_batch_x is a list of tensors, each [N_hits, F]
        # Get feature dimension from first event
        self.feat_dim = example_batch_x[0].shape[1]

        # Graph convolution layers
        self.conv1 = GCNConv(self.feat_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Output layer
        self.out = nn.Linear(32, 1)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(32)

    def forward(self, batch_x):
        # batch_x is list of [N_hits, F] tensors
        batch_list = []
        for x in batch_x:
            # Create edge indices for complete graph (all hits connected)
            n_hits = x.shape[0]
            edge_index = torch.combinations(torch.arange(n_hits), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index)

            # Graph convolutions
            x = F.relu(self.conv1(data.x, data.edge_index))
            x = F.relu(self.conv2(x, data.edge_index))
            x = self.conv3(x, data.edge_index)
            x = self.layer_norm(x)

            # Attention weights
            attn_weights = torch.softmax(self.attention(x), dim=0)
            x = x * attn_weights

            # Global pooling to get track features
            track_features = global_mean_pool(x, torch.zeros(n_hits, dtype=torch.long, device=x.device))

            # Predict cluster assignment
            logits = self.out(x)
            preds = torch.argmax(logits, dim=1) + 1  # +1 to avoid 0 (noise)

            batch_list.append(preds)

        return batch_list

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    best_val_loss = float('inf')
    best_model = None
    patience = 5
    no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        count = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            # Convert to list of tensors for our model
            xb_list = [x.to(device) for x in xb]
            yb_list = [y.to(device) for y in yb]

            optimizer.zero_grad()

            # Forward pass
            out_list = model(xb_list)

            # Calculate loss and accuracy
            loss = 0
            acc = 0
            for out, y in zip(out_list, yb_list):
                # Convert predictions to match y shape
                out = out.unsqueeze(1).expand(-1, y.max().item() + 1)
                loss += criterion(out, y - 1)  # -1 to make noise class 0
                acc += (out.argmax(dim=1) + 1 == y).float().mean()

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_acc += acc.item()
            count += 1

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                xb_list = [x.to(device) for x in xb]
                yb_list = [y.to(device) for y in yb]

                out_list = model(xb_list)

                loss = 0
                acc = 0
                for out, y in zip(out_list, yb_list):
                    out = out.unsqueeze(1).expand(-1, y.max().item() + 1)
                    loss += criterion(out, y - 1)
                    acc += (out.argmax(dim=1) + 1 == y).float().mean()

                epoch_val_loss += loss.item()
                epoch_val_acc += acc.item()
                val_count += 1

        # Average metrics
        train_loss.append(epoch_train_loss / count)
        train_acc.append(epoch_train_acc / count)
        val_loss.append(epoch_val_loss / val_count)
        val_acc.append(epoch_val_acc / val_count)

        # Learning rate scheduling
        scheduler.step(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_loss, val_loss, train_acc, val_acc

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    Xs = [split_X_y(evt)[0] for evt in raw_train]
    pre = make_preprocessor().fit(Xs)

    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, raw_train, pre, train=True)
    val_ds       = build_dataset(spec, raw_val,   pre, train=False)
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
                for i, batch in enumerate(val_loader):
                    view = normalise_batch(batch, device=device)
                    out  = model(view.batch_x)
                    assert_label_output(view.batch_x, out, allow_noise_label=True)
                    if i >= 4: # loop over 4 batches
                        break
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

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

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

