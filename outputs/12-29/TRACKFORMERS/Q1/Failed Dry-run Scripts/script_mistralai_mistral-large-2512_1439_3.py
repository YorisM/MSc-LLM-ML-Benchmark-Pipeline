
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
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_max, scatter_add
from sklearn.preprocessing import StandardScaler
import hdbscan
from scipy.spatial.distance import pdist, squareform

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_mean = None
        self.layer_std = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,

            "collate": "ragged_xy",

            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False, "num_workers": 0}
        }

    def fit(self, Xs):
        # Compute global statistics for normalization
        all_X = torch.cat(Xs, dim=0).numpy()
        self.scaler.fit(all_X[:, :3])  # Only scale r, theta, z

        # Compute layer-wise statistics
        layer_ids = all_X[:, 3]
        unique_layers = np.unique(layer_ids)
        self.layer_mean = {layer: np.mean(all_X[layer_ids == layer, :3], axis=0)
                          for layer in unique_layers}
        self.layer_std = {layer: np.std(all_X[layer_ids == layer, :3], axis=0)
                         for layer in unique_layers}

        return self

    def transform(self, X):
        # X: [N_hits, 4] tensor
        X = X.clone()

        # Normalize r, theta, z
        X[:, :3] = torch.from_numpy(self.scaler.transform(X[:, :3].numpy())).float()

        # Add layer-aware features
        layer_id = X[:, 3].long()
        for i in range(3):
            X = torch.cat([
                X,
                (X[:, i:i+1] - torch.tensor([self.layer_mean[l][i] for l in layer_id.numpy()],
                                           dtype=torch.float32).unsqueeze(1)) /
                torch.tensor([self.layer_std[l][i] if self.layer_std[l][i] > 0 else 1.0
                             for l in layer_id.numpy()], dtype=torch.float32).unsqueeze(1)
            ], dim=1)

        # Add cylindrical coordinate features
        X = torch.cat([
            X,
            torch.cos(X[:, 1:2]),  # cos(theta)
            torch.sin(X[:, 1:2]),  # sin(theta)
            X[:, 0:1] * torch.cos(X[:, 1:2]),  # x = r*cos(theta)
            X[:, 0:1] * torch.sin(X[:, 1:2]),  # y = r*sin(theta)
        ], dim=1)

        return X  # [N_hits, 4 + 3 + 5 = 12]

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()

        # Determine input features from example batch
        if isinstance(example_batch_x, list):
            in_features = example_batch_x[0].shape[1]
        else:
            in_features = example_batch_x.shape[1]

        # Graph neural network layers
        self.conv1 = EdgeConv(in_features, 64)
        self.conv2 = EdgeConv(64, 128)
        self.conv3 = EdgeConv(128, 256)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Output layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)  # Will predict cluster assignment scores

        # Track embedding
        self.track_embedding = nn.Embedding(1, 256)  # Dummy embedding for track initialization

        # Dropout
        self.dropout = nn.Dropout(0.2)

    def build_graph(self, x):
        # x: [N_hits, F]
        N = x.size(0)

        # Create edges based on spatial proximity
        pos = x[:, :3]  # r, theta, z
        dist = torch.cdist(pos, pos)
        threshold = 0.5  # Distance threshold for edge creation
        edge_index = (dist < threshold).nonzero(as_tuple=False).t()

        # Add self-loops
        edge_index = add_self_loops(edge_index, num_nodes=N)[0]

        return edge_index

    def forward(self, batch_x):
        if isinstance(batch_x, list):
            # Process each event in the batch separately
            outputs = []
            for x in batch_x:
                x = x.to(device)
                out = self.forward_single(x)
                outputs.append(out)
            return outputs
        else:
            return self.forward_single(batch_x)

    def forward_single(self, x):
        # x: [N_hits, F]
        N = x.size(0)

        # Build graph
        edge_index = self.build_graph(x)

        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))

        # Attention mechanism
        attn_weights = F.softmax(self.attention(x), dim=0)
        x = x * attn_weights

        # MLP layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        # Output cluster assignment scores
        scores = self.fc_out(x).squeeze()  # [N_hits]

        # Post-processing with HDBSCAN
        try:
            # Convert to numpy for HDBSCAN
            x_np = x.detach().cpu().numpy()
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=4,
                min_samples=1,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            cluster_labels = clusterer.fit_predict(x_np)

            # Convert to torch and ensure labels are > 0 (0 is noise)
            cluster_labels = torch.from_numpy(cluster_labels).to(device)
            cluster_labels[cluster_labels >= 0] += 1  # Make labels start from 1
            cluster_labels[cluster_labels < 0] = -1   # Noise remains -1

            return cluster_labels
        except:
            # Fallback to simple clustering if HDBSCAN fails
            scores = scores - scores.min()
            scores = scores / (scores.max() + 1e-8)
            threshold = 0.5
            cluster_labels = (scores > threshold).long()
            cluster_labels[cluster_labels == 0] = -1
            return cluster_labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    best_val_acc = 0
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()

            if isinstance(xb, list):
                # Process each event separately
                batch_loss = 0
                batch_correct = 0
                batch_samples = 0

                for x, y in zip(xb, yb):
                    x, y = x.to(device), y.to(device)
                    out = model(x.unsqueeze(0))[0]  # Remove batch dim

                    # Convert to classification problem
                    # We'll use the track_id as class labels (shifted by 1 to account for noise)
                    y_shifted = y.clone()
                    y_shifted[y_shifted > 0] += 1  # Make room for noise class
                    y_shifted[y_shifted == 0] = 0   # Noise remains 0

                    # Create a mapping from track_id to class index
                    unique_tracks = torch.unique(y_shifted)
                    track_to_class = {track.item(): i for i, track in enumerate(unique_tracks)}

                    # Convert labels to class indices
                    class_labels = torch.zeros_like(y_shifted)
                    for track, class_idx in track_to_class.items():
                        class_labels[y_shifted == track] = class_idx

                    # Create prediction tensor
                    pred_labels = out
                    if pred_labels.dim() == 1:
                        pred_labels = pred_labels.unsqueeze(1)

                    # Calculate loss
                    loss = criterion(pred_labels, class_labels.long())

                    # Calculate accuracy
                    pred_class = pred_labels.argmax(dim=1)
                    correct = (pred_class == class_labels).sum().item()
                    samples = y.numel()

                    batch_loss += loss.item() * samples
                    batch_correct += correct
                    batch_samples += samples

                    loss.backward()

                total_loss += batch_loss
                total_correct += batch_correct
                total_samples += batch_samples
            else:
                # Process as single batch (not expected in this problem)
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)

                y_shifted = yb.clone()
                y_shifted[yb > 0] += 1
                y_shifted[yb == 0] = 0

                unique_tracks = torch.unique(y_shifted)
                track_to_class = {track.item(): i for i, track in enumerate(unique_tracks)}

                class_labels = torch.zeros_like(y_shifted)
                for track, class_idx in track_to_class.items():
                    class_labels[y_shifted == track] = class_idx

                loss = criterion(out, class_labels.long())
                total_loss += loss.item() * yb.numel()

                pred_class = out.argmax(dim=1)
                total_correct += (pred_class == class_labels).sum().item()
                total_samples += yb.numel()

                loss.backward()

            optimizer.step()

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                if isinstance(xb, list):
                    batch_loss = 0
                    batch_correct = 0
                    batch_samples = 0

                    for x, y in zip(xb, yb):
                        x, y = x.to(device), y.to(device)
                        out = model(x.unsqueeze(0))[0]

                        y_shifted = y.clone()
                        y_shifted[y > 0] += 1
                        y_shifted[y == 0] = 0

                        unique_tracks = torch.unique(y_shifted)
                        track_to_class = {track.item(): i for i, track in enumerate(unique_tracks)}

                        class_labels = torch.zeros_like(y_shifted)
                        for track, class_idx in track_to_class.items():
                            class_labels[y_shifted == track] = class_idx

                        pred_labels = out
                        if pred_labels.dim() == 1:
                            pred_labels = pred_labels.unsqueeze(1)

                        loss = criterion(pred_labels, class_labels.long())

                        pred_class = pred_labels.argmax(dim=1)
                        correct = (pred_class == class_labels).sum().item()
                        samples = y.numel()

                        batch_loss += loss.item() * samples
                        batch_correct += correct
                        batch_samples += samples

                    val_loss += batch_loss
                    val_correct += batch_correct
                    val_samples += batch_samples
                else:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)

                    y_shifted = yb.clone()
                    y_shifted[yb > 0] += 1
                    y_shifted[yb == 0] = 0

                    unique_tracks = torch.unique(y_shifted)
                    track_to_class = {track.item(): i for i, track in enumerate(unique_tracks)}

                    class_labels = torch.zeros_like(y_shifted)
                    for track, class_idx in track_to_class.items():
                        class_labels[y_shifted == track] = class_idx

                    loss = criterion(out, class_labels.long())
                    val_loss += loss.item() * yb.numel()

                    pred_class = out.argmax(dim=1)
                    val_correct += (pred_class == class_labels).sum().item()
                    val_samples += yb.numel()

        val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
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

    # debugging
    print("NB meta:", view.meta, flush=True)
    print("type(batch_x):", type(view.batch_x), flush=True)
    if isinstance(view.batch_x, list) and len(view.batch_x) > 0:
        print("type(batch_x[0]):", type(view.batch_x[0]), flush=True)
        if isinstance(view.batch_x[0], (tuple, list)):
            print("len(batch_x[0]):", len(view.batch_x[0]), flush=True)
            print("inner types:", [type(z) for z in view.batch_x[0]], flush=True)

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

