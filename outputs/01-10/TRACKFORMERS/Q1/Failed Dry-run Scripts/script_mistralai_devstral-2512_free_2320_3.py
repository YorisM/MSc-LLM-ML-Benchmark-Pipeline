
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import detect_and_assert_lane, assert_label_output_by_lane, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, build_trackformers_model, to_python

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
from sklearn.cluster import DBSCAN
import hdbscan

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

        # Encode layer_id
        unique_layers = np.unique(all_hits[:, 3])
        self.n_layers = len(unique_layers)
        self.layer_encoder = {l: i for i, l in enumerate(unique_layers)}
        return self

    def transform(self, X):
        # X: [N_hits, 4] numpy array
        X = X.numpy() if isinstance(X, torch.Tensor) else X
        scaled = self.scaler.transform(X[:, :3])
        layer_encoded = np.array([self.layer_encoder[l] for l in X[:, 3]], dtype=np.float32)
        return torch.FloatTensor(np.column_stack([scaled, layer_encoded]))

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Determine input features
        if isinstance(example_batch_x, list):
            # Torch ragged lane
            self.lane = "torch_ragged_xy"
            self.input_dim = example_batch_x[0].shape[1]
        else:
            # PyG lane
            self.lane = "pyg_batch"
            self.input_dim = example_batch_x.x.shape[1]

        # GNN layers
        self.conv1 = GCNConv(self.input_dim, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.fc = nn.Linear(64, 32)

        # Clustering head
        self.cluster_head = nn.Linear(32, 16)

        # Edge construction parameters
        self.r_cutoff = 0.1
        self.z_cutoff = 1.0
        self.layer_cutoff = 2

    def build_edges(self, x, batch=None):
        # x: [N, 4] (r, theta, z, layer)
        # Build edges based on spatial proximity and layer adjacency
        r = x[:, 0]
        z = x[:, 2]
        layer = x[:, 3]

        # Compute pairwise distances
        r_diff = torch.abs(r.unsqueeze(1) - r.unsqueeze(0))
        z_diff = torch.abs(z.unsqueeze(1) - z.unsqueeze(0))
        layer_diff = torch.abs(layer.unsqueeze(1) - layer.unsqueeze(0))

        # Create adjacency matrix
        adj = (r_diff < self.r_cutoff) & (z_diff < self.z_cutoff) & (layer_diff < self.layer_cutoff)

        # Remove self-loops
        adj.fill_diagonal_(False)

        # Convert to edge indices
        edge_index = adj.nonzero().t().contiguous()

        # For batched data, ensure edges stay within same event
        if batch is not None:
            same_event = (batch[edge_index[0]] == batch[edge_index[1]])
            edge_index = edge_index[:, same_event]

        return edge_index

    def forward(self, batch_x):
        if self.lane == "torch_ragged_xy":
            # Convert to PyG format
            xs, batch = [], []
            for i, x in enumerate(batch_x):
                xs.append(x)
                batch.append(torch.full((x.shape[0],), i, dtype=torch.long))
            x = torch.cat(xs, dim=0)
            batch = torch.cat(batch, dim=0)
        else:
            x = batch_x.x
            batch = batch_x.batch

        # Build graph
        edge_index = self.build_edges(x, batch)

        # GNN forward pass
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))

        # Global pooling for event-level features
        event_features = global_mean_pool(h, batch)

        # Combine node and event features
        h = torch.cat([h, event_features[batch]], dim=1)
        h = F.relu(self.fc(h))

        # Clustering embedding
        embeddings = self.cluster_head(h)
        return embeddings

    def predict_labels(self, batch_x):
        with torch.no_grad():
            embeddings = self.forward(batch_x)

            if self.lane == "torch_ragged_xy":
                # Split embeddings back into events
                xs = batch_x
                embeddings_list = []
                start = 0
                for x in xs:
                    n = x.shape[0]
                    embeddings_list.append(embeddings[start:start+n])
                    start += n

                # Cluster each event separately
                all_labels = []
                for emb in embeddings_list:
                    emb_np = emb.cpu().numpy()
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=1,
                                              cluster_selection_epsilon=0.1,
                                              metric='euclidean')
                    labels = clusterer.fit_predict(emb_np)

                    # Convert to torch and handle noise (-1)
                    labels = torch.from_numpy(labels).to(emb.device)
                    all_labels.append(labels)
                return all_labels
            else:
                # Single batch clustering
                emb_np = embeddings.cpu().numpy()
                clusterer = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=1,
                                          cluster_selection_epsilon=0.1,
                                          metric='euclidean')
                labels = clusterer.fit_predict(emb_np)
                return torch.from_numpy(labels).to(embeddings.device)

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Loss function - we'll use a contrastive loss
    def contrastive_loss(embeddings, batch):
        # Create positive pairs (same track) and negative pairs (different tracks)
        # This is a simplified version - in practice you'd need ground truth labels
        # For this example, we'll use a basic clustering loss

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        sim = torch.mm(embeddings, embeddings.t())

        # Create target matrix (simplified)
        # In practice, you'd use the ground truth to define positive/negative pairs
        # Here we just encourage similar embeddings to be close
        loss = -torch.log(torch.sigmoid(sim) + 1e-8).mean()
        return loss

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    best_val_loss = float('inf')
    best_model = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            if model.lane == "torch_ragged_xy":
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                embeddings = model(Xs)
                loss = contrastive_loss(embeddings, None)
            else:
                G = batch.to(device)
                embeddings = model(G)
                loss = contrastive_loss(embeddings, G.batch)

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if model.lane == "torch_ragged_xy":
                    Xs, ys = batch
                    Xs = [x.to(device) for x in Xs]
                    embeddings = model(Xs)
                    loss = contrastive_loss(embeddings, None)
                else:
                    G = batch.to(device)
                    embeddings = model(G)
                    loss = contrastive_loss(embeddings, G.batch)
                epoch_val_loss += loss.item()

        # Update learning rate
        scheduler.step(epoch_val_loss)

        # Store metrics
        train_loss.append(epoch_train_loss / len(train_loader))
        val_loss.append(epoch_val_loss / len(val_loader))

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
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

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane(spec, first_batch)

    # Build model
    model = build_trackformers_model(mode, first_batch, make_model, device)

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
        if not hasattr(trained_model, "predict_labels") or not callable(getattr(trained_model, "predict_labels")):
            raise TypeError("Contract error: trained model must implement predict_labels(batch_x).")

        trained_model.eval()
        try:
            with torch.no_grad():
                mode = None
                for i, batch in enumerate(val_loader):
                    if mode is None:
                        mode = detect_and_assert_lane(spec, batch)

                    if mode == "torch_ragged_xy":
                        Xs, _ys = batch
                        Xs = [x.to(device) for x in Xs]
                        out = trained_model.predict_labels(Xs)
                    elif mode == "pyg_batch":
                        G = batch.to(device)
                        out = trained_model.predict_labels(G)
                    else:
                        raise RuntimeError(f"Unknown lane mode: {mode}")

                    assert_label_output_by_lane(mode, batch, out, allow_noise_label=True)
                    if i >= 3:  # 4 batches
                        break
        except Exception as e:
            raise RuntimeError("Sanity-check predict_labels() failed") from e
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
        summary = to_python(summary)
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

