
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
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler
import pickle

# -------- (OPTIONAL) CUSTOM DATASET  --------
# Not implementing custom dataset, using default EventDataset

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,

            "collate": "ragged_xy",
            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, 4]
        all_features = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_features[:, :3])  # Scale r, theta, z
        self.layer_scaler.fit(all_features[:, 3:4])  # Scale layer_id separately
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, 4]
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
        # example_batch_x is a list of tensors, each [N_hits, 4]
        # We'll use the first event to get feature dimension
        in_features = example_batch_x[0].shape[1]

        # Graph convolution layers
        self.conv1 = GCNConv(in_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)

        # Global pooling and classification head
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Predict track ID as regression then cluster

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, batch_x):
        # batch_x is a list of tensors [N_hits_i, 4]
        # Convert to PyG batch format
        data_list = []
        for i, x in enumerate(batch_x):
            # Create edge indices based on spatial proximity
            # Simple k-NN graph construction (k=10)
            dist = torch.cdist(x[:, :3], x[:, :3])  # [N, N]
            knn_indices = torch.topk(dist, k=10, largest=False, dim=1).indices  # [N, 10]

            # Create edges (undirected)
            src = torch.arange(x.size(0), device=x.device).repeat_interleave(10)
            dst = knn_indices.flatten()

            # Remove self-loops
            mask = src != dst
            src = src[mask]
            dst = dst[mask]

            edge_index = torch.stack([src, dst], dim=0)

            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)

        # Graph convolutions
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, batch.edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, batch.edge_index))

        # Global pooling to get graph-level features
        graph_emb = global_mean_pool(x, batch.batch)

        # Classification head
        x = F.relu(self.fc1(graph_emb))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Convert regression output to cluster assignments
        # We'll use DBSCAN-like clustering on the predicted values
        predictions = []
        start_idx = 0
        for i, num_hits in enumerate(batch.ptr[1:] - batch.ptr[:-1]):
            end_idx = start_idx + num_hits
            event_pred = x[start_idx:end_idx]

            # Simple clustering: sort and assign cluster IDs
            sorted_pred = torch.argsort(event_pred.squeeze())
            cluster_ids = torch.zeros_like(sorted_pred)
            current_cluster = 1

            # Assign cluster IDs based on gaps in predictions
            for j in range(1, len(sorted_pred)):
                if event_pred[sorted_pred[j]] - event_pred[sorted_pred[j-1]] > 0.5:
                    current_cluster += 1
                cluster_ids[sorted_pred[j]] = current_cluster

            # Assign to original hit order
            predictions.append(cluster_ids)
            start_idx = end_idx

        # Convert to single tensor with -1 for noise (cluster_id == 0)
        # Here we'll mark very small clusters as noise
        final_preds = []
        for pred in predictions:
            unique, counts = torch.unique(pred, return_counts=True)
            noise_mask = torch.isin(pred, unique[counts < 4])
            pred[noise_mask] = -1
            final_preds.append(pred)

        return torch.cat(final_preds).long()

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()  # Using MSE for the regression head

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        epoch_val_loss = 0
        correct_train = 0
        total_train = 0
        correct_val = 0
        total_val = 0

        # Training
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            optimizer.zero_grad()

            # Convert labels to regression targets (normalized track IDs)
            # We'll use the mean of the track ID as target for each hit
            targets = []
            for y in view.batch_y:
                unique_tracks = torch.unique(y[y > 0])
                track_means = {}
                for track in unique_tracks:
                    track_means[track.item()] = track.float() / len(unique_tracks)

                target = torch.zeros_like(y, dtype=torch.float32)
                for track in unique_tracks:
                    mask = y == track
                    target[mask] = track_means[track.item()]
                targets.append(target.unsqueeze(1))

            targets = torch.cat(targets).to(device)

            outputs = model(view.batch_x)
            # For loss calculation, we need to convert cluster IDs back to regression targets
            # This is a simplified approach - in practice you'd want a better mapping
            loss_targets = []
            for i, pred in enumerate(outputs):
                unique_preds = torch.unique(pred[pred > 0])
                if len(unique_preds) == 0:
                    loss_targets.append(torch.zeros(1, device=device))
                    continue

                # Map cluster IDs to normalized track IDs
                pred_to_target = {}
                for j, p in enumerate(unique_preds):
                    pred_to_target[p.item()] = j / len(unique_preds)

                target = torch.zeros_like(pred, dtype=torch.float32)
                for p in unique_preds:
                    mask = pred == p
                    target[mask] = pred_to_target[p.item()]
                loss_targets.append(target.unsqueeze(0))

            loss_targets = torch.cat(loss_targets).to(device)

            loss = criterion(outputs.float(), loss_targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * len(view.batch_x)

            # Calculate accuracy (simplified)
            for i, pred in enumerate(outputs):
                y_true = view.batch_y[i]
                mask = y_true > 0
                if mask.sum() > 0:
                    # Count correct assignments (simplified)
                    correct_train += (pred[mask] == y_true[mask]).float().sum().item()
                    total_train += mask.sum().item()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)

                # Same target preparation as training
                targets = []
                for y in view.batch_y:
                    unique_tracks = torch.unique(y[y > 0])
                    track_means = {}
                    for track in unique_tracks:
                        track_means[track.item()] = track.float() / len(unique_tracks)

                    target = torch.zeros_like(y, dtype=torch.float32)
                    for track in unique_tracks:
                        mask = y == track
                        target[mask] = track_means[track.item()]
                    targets.append(target.unsqueeze(1))

                targets = torch.cat(targets).to(device)

                outputs = model(view.batch_x)

                # Calculate loss
                loss_targets = []
                for i, pred in enumerate(outputs):
                    unique_preds = torch.unique(pred[pred > 0])
                    if len(unique_preds) == 0:
                        loss_targets.append(torch.zeros(1, device=device))
                        continue

                    pred_to_target = {}
                    for j, p in enumerate(unique_preds):
                        pred_to_target[p.item()] = j / len(unique_preds)

                    target = torch.zeros_like(pred, dtype=torch.float32)
                    for p in unique_preds:
                        mask = pred == p
                        target[mask] = pred_to_target[p.item()]
                    loss_targets.append(target.unsqueeze(0))

                loss_targets = torch.cat(loss_targets).to(device)
                loss = criterion(outputs.float(), loss_targets)
                epoch_val_loss += loss.item() * len(view.batch_x)

                # Calculate accuracy
                for i, pred in enumerate(outputs):
                    y_true = view.batch_y[i]
                    mask = y_true > 0
                    if mask.sum() > 0:
                        correct_val += (pred[mask] == y_true[mask]).float().sum().item()
                        total_val += mask.sum().item()

        # Calculate metrics
        train_loss = epoch_train_loss / len(train_loader.dataset)
        val_loss = epoch_val_loss / len(val_loader.dataset)
        train_acc = correct_train / total_train if total_train > 0 else 0
        val_acc = correct_val / total_val if total_val > 0 else 0

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

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Load best model weights
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

