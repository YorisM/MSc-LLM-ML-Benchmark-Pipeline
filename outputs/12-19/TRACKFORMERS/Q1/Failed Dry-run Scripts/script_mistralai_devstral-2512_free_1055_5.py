
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, torch, torch_geometric
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

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
import pickle

# -------- (OPTIONAL) CUSTOM DATASET  --------
def make_dataset(events, pre, train: bool, **kwargs):
    return EventDataset(events, pre, train=train)

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:make_dataset",
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,

            "collate": "identity",
            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Stack all events for fitting
        all_X = torch.cat(Xs, dim=0).numpy()
        self.scaler.fit(all_X[:, :3])  # Scale r, theta, z
        self.layer_scaler.fit(all_X[:, 3:4])  # Scale layer_id separately
        return self

    def transform(self, X):
        # X: [N_hits, 4]
        X_np = X.numpy()
        X_np[:, :3] = self.scaler.transform(X_np[:, :3])
        X_np[:, 3:4] = self.layer_scaler.transform(X_np[:, 3:4])
        return torch.from_numpy(X_np).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # example_batch_x is a list of tensors, take first to get feature dim
        in_features = example_batch_x[0].shape[1] if isinstance(example_batch_x, list) else example_batch_x.shape[1]

        # Graph layers
        self.conv1 = GCNConv(in_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)

        # Global pooling and classification
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Predict track ID offset

        # Edge construction parameters
        self.r_threshold = 0.1
        self.z_threshold = 0.2
        self.layer_threshold = 1.0

    def build_graph(self, X):
        # X: [N_hits, 4] (r, theta, z, layer_id)
        N = X.shape[0]
        edge_index = []

        # Connect hits in adjacent layers
        for i in range(N):
            for j in range(i+1, N):
                # Skip if same hit or far apart in layers
                if abs(X[i, 3] - X[j, 3]) > self.layer_threshold:
                    continue

                # Spatial proximity
                r_diff = abs(X[i, 0] - X[j, 0])
                z_diff = abs(X[i, 2] - X[j, 2])

                if r_diff < self.r_threshold and z_diff < self.z_threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        if not edge_index:
            # Fallback: connect all hits if no edges found
            for i in range(N):
                for j in range(N):
                    if i != j:
                        edge_index.append([i, j])

        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def forward(self, batch_x):
        # batch_x is list of tensors [N_hits_i, 4]
        if isinstance(batch_x, list):
            # Process each event separately
            all_preds = []
            for x in batch_x:
                x = x.to(device)
                edge_index = self.build_graph(x).to(device)
                data = Data(x=x, edge_index=edge_index)

                # Graph convolution
                h = F.relu(self.conv1(data.x, data.edge_index))
                h = F.relu(self.conv2(h, data.edge_index))
                h = F.relu(self.conv3(h, data.edge_index))

                # Global pooling
                h = global_mean_pool(h, torch.zeros(data.num_nodes, dtype=torch.long, device=device))

                # Classification
                h = F.relu(self.fc1(h))
                h = F.relu(self.fc2(h))
                pred_offset = self.fc3(h)  # [1]

                # Convert to track IDs (simple approach)
                pred_ids = (pred_offset * 100).long()  # Scale to get reasonable ID range
                pred_ids = pred_ids.repeat(data.num_nodes, 1).squeeze()

                # Ensure positive IDs (0 is noise)
                pred_ids = torch.clamp(pred_ids, min=1)
                all_preds.append(pred_ids)

            return all_preds
        else:
            # Handle single tensor case
            edge_index = self.build_graph(batch_x).to(device)
            data = Data(x=batch_x, edge_index=edge_index)

            h = F.relu(self.conv1(data.x, data.edge_index))
            h = F.relu(self.conv2(h, data.edge_index))
            h = F.relu(self.conv3(h, data.edge_index))

            h = global_mean_pool(h, torch.zeros(data.num_nodes, dtype=torch.long, device=device))
            pred_offset = self.fc3(h)
            pred_ids = (pred_offset * 100).long()
            pred_ids = pred_ids.repeat(data.num_nodes, 1).squeeze()
            pred_ids = torch.clamp(pred_ids, min=1)
            return pred_ids

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.MSELoss()  # Using MSE for track ID prediction

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

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
            optimizer.zero_grad()

            # Handle different batch formats
            if isinstance(batch, list):
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]
            else:
                Xs = batch.x.to(device)
                ys = batch.y.to(device)

            # Forward pass
            if isinstance(Xs, list):
                preds = model(Xs)
                loss = 0
                acc = 0
                for pred, y in zip(preds, ys):
                    # Convert track IDs to float for MSE
                    y_float = y.float().unsqueeze(1)
                    pred_float = pred.float().unsqueeze(1)
                    loss += criterion(pred_float, y_float)
                    acc += (pred == y).float().mean()
                loss /= len(Xs)
                acc /= len(Xs)
            else:
                pred = model(Xs)
                y_float = ys.float().unsqueeze(1)
                pred_float = pred.float().unsqueeze(1)
                loss = criterion(pred_float, y_float)
                acc = (pred == ys).float().mean()

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * len(ys) if isinstance(ys, list) else loss.item() * ys.shape[0]
            epoch_train_acc += acc.item() * len(ys) if isinstance(ys, list) else acc.item() * ys.shape[0]
            train_samples += len(ys) if isinstance(ys, list) else ys.shape[0]

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, list):
                    Xs, ys = batch
                    Xs = [x.to(device) for x in Xs]
                    ys = [y.to(device) for y in ys]
                else:
                    Xs = batch.x.to(device)
                    ys = batch.y.to(device)

                if isinstance(Xs, list):
                    preds = model(Xs)
                    loss = 0
                    acc = 0
                    for pred, y in zip(preds, ys):
                        y_float = y.float().unsqueeze(1)
                        pred_float = pred.float().unsqueeze(1)
                        loss += criterion(pred_float, y_float)
                        acc += (pred == y).float().mean()
                    loss /= len(Xs)
                    acc /= len(Xs)
                else:
                    pred = model(Xs)
                    y_float = ys.float().unsqueeze(1)
                    pred_float = pred.float().unsqueeze(1)
                    loss = criterion(pred_float, y_float)
                    acc = (pred == ys).float().mean()

                epoch_val_loss += loss.item() * len(ys) if isinstance(ys, list) else loss.item() * ys.shape[0]
                epoch_val_acc += acc.item() * len(ys) if isinstance(ys, list) else acc.item() * ys.shape[0]
                val_samples += len(ys) if isinstance(ys, list) else ys.shape[0]

        # Calculate epoch metrics
        train_loss = epoch_train_loss / train_samples
        val_loss = epoch_val_loss / val_samples
        train_acc = epoch_train_acc / train_samples
        val_acc = epoch_val_acc / val_samples

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

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
                break

    # Load best model
    model.load_state_dict(best_model)

    return model, train_losses, val_losses, train_accs, val_accs

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

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

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

