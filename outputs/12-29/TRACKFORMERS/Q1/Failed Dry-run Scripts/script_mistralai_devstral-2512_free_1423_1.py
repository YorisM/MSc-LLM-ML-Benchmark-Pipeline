
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
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import hdbscan

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_means = None
        self.layer_stds = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": "ragged_xy",
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Concatenate all events for global scaling
        all_hits = torch.cat(Xs, dim=0).numpy()
        self.scaler.fit(all_hits[:, :3])  # Scale r, theta, z

        # Compute per-layer statistics
        layer_stats = {}
        for X in Xs:
            layers = X[:, 3].numpy()
            for i, layer in enumerate(layers):
                if layer not in layer_stats:
                    layer_stats[layer] = []
                layer_stats[layer].append(X[i, :3].numpy())

        self.layer_means = {}
        self.layer_stds = {}
        for layer, hits in layer_stats.items():
            hits = np.array(hits)
            self.layer_means[layer] = np.mean(hits, axis=0)
            self.layer_stds[layer] = np.std(hits, axis=0)

        return self

    def transform(self, X):
        # Scale features
        X_np = X.numpy()
        X_np[:, :3] = self.scaler.transform(X_np[:, :3])

        # Layer-wise normalization
        for i in range(len(X)):
            layer = X_np[i, 3]
            if layer in self.layer_means:
                X_np[i, :3] = (X_np[i, :3] - self.layer_means[layer]) / (self.layer_stds[layer] + 1e-8)

        return torch.from_numpy(X_np).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Graph-based architecture
        self.conv1 = GCNConv(4, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)  # Predict cluster assignment

        # Positional encoding
        self.pos_encoder = nn.Linear(4, 32)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def build_graph(self, X):
        # X: [N_hits, 4] tensor
        N = X.size(0)
        pos = X[:, :3]  # Use spatial coordinates for edge weights

        # Compute pairwise distances
        dist = torch.cdist(pos, pos)
        sigma = 0.1 * torch.median(dist)
        adj = torch.exp(-dist / sigma)

        # Create edge index
        edge_index = torch.nonzero(adj > 0.1).t()
        edge_weight = adj[edge_index[0], edge_index[1]]

        return Data(x=X, edge_index=edge_index, edge_weight=edge_weight)

    def forward(self, batch_x):
        # batch_x is list of tensors [N_hits_i, 4]
        all_preds = []
        for X in batch_x:
            # Build graph for this event
            data = self.build_graph(X).to(device)

            # Graph convolution layers
            x = F.relu(self.conv1(data.x, data.edge_index, data.edge_weight))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, data.edge_index, data.edge_weight))
            x = self.dropout(x)
            x = F.relu(self.conv3(x, data.edge_index, data.edge_weight))

            # Global context
            global_ctx = global_mean_pool(x, torch.zeros(data.num_nodes, dtype=torch.long, device=device))

            # Combine local and global features
            x = torch.cat([x, global_ctx.expand(x.size(0), -1)], dim=1)

            # Attention mechanism
            x, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            x = x.squeeze(0)

            # Final prediction
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.fc3(x).squeeze(-1)

            # Convert to cluster assignments
            preds = torch.argmax(x, dim=0) + 1  # +1 to avoid 0 (noise)
            all_preds.append(preds)

        return all_preds

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            Xs, ys = view.batch_x, view.batch_y

            optimizer.zero_grad()

            # Process each event in batch
            loss = 0
            for X, y in zip(Xs, ys):
                # Build graph and forward pass
                data = model.build_graph(X).to(device)
                out = model.conv1(data.x, data.edge_index, data.edge_weight)
                out = F.relu(out)
                out = model.conv2(out, data.edge_index, data.edge_weight)
                out = F.relu(out)
                out = model.conv3(out, data.edge_index, data.edge_weight)

                # Global pooling and final layers
                global_ctx = global_mean_pool(out, torch.zeros(data.num_nodes, dtype=torch.long, device=device))
                out = torch.cat([out, global_ctx.expand(out.size(0), -1)], dim=1)
                out, _ = model.attention(out.unsqueeze(0), out.unsqueeze(0), out.unsqueeze(0))
                out = out.squeeze(0)
                out = F.relu(model.fc1(out))
                out = model.fc2(out)
                out = model.fc3(out).squeeze(-1)

                # Compute loss
                loss += criterion(out, y.to(device))

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            # Compute accuracy
            with torch.no_grad():
                preds = model(Xs)
                for pred, y in zip(preds, ys):
                    correct_train += (pred == y.to(device)).sum().item()
                    total_train += y.size(0)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                Xs, ys = view.batch_x, view.batch_y

                loss = 0
                for X, y in zip(Xs, ys):
                    data = model.build_graph(X).to(device)
                    out = model.conv1(data.x, data.edge_index, data.edge_weight)
                    out = F.relu(out)
                    out = model.conv2(out, data.edge_index, data.edge_weight)
                    out = F.relu(out)
                    out = model.conv3(out, data.edge_index, data.edge_weight)

                    global_ctx = global_mean_pool(out, torch.zeros(data.num_nodes, dtype=torch.long, device=device))
                    out = torch.cat([out, global_ctx.expand(out.size(0), -1)], dim=1)
                    out, _ = model.attention(out.unsqueeze(0), out.unsqueeze(0), out.unsqueeze(0))
                    out = out.squeeze(0)
                    out = F.relu(model.fc1(out))
                    out = model.fc2(out)
                    out = model.fc3(out).squeeze(-1)

                    loss += criterion(out, y.to(device))

                epoch_val_loss += loss.item()

                preds = model(Xs)
                for pred, y in zip(preds, ys):
                    correct_val += (pred == y.to(device)).sum().item()
                    total_val += y.size(0)

        # Update learning rate
        scheduler.step(epoch_val_loss)

        # Store metrics
        train_losses.append(epoch_train_loss / len(train_loader))
        val_losses.append(epoch_val_loss / len(val_loader))
        train_accs.append(correct_train / total_train)
        val_accs.append(correct_val / total_val)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
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

