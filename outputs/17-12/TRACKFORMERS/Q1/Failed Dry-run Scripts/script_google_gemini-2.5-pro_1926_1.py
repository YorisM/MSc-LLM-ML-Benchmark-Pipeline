
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy 
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader, split_X_y, EventDataset
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy, write_loaderspec
from utils.suffix_utils import base_from_argv0, write_json, plot_train_val, persist_artefacts

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

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import torch_geometric
from torch_geometric.data import Dataset as PyGDataset, Data as PyGData
from torch_geometric.nn import EdgeConv, knn_graph
import scipy.sparse
from utils.llm_io import split_X_y

# -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomPygDataset(PyGDataset):
    """
    Custom PyTorch Geometric dataset.
    For each event, it creates a graph where hits are nodes.
    Nodes are connected via k-NN.
    It prepares node features, edge indices, and edge labels for training.
    """
    def __init__(self, events, preprocessor, train: bool, k: int):
        super().__init__()
        self.events = events
        self.pre = preprocessor
        self.is_train = train
        self.k = k

    def len(self):
        return len(self.events)

    def get(self, idx):
        event = self.events[idx]
        X, y = split_X_y(event)

        X_tensor = torch.from_numpy(X).float()

        # Use unscaled cartesian coordinates for k-NN to capture geometric proximity.
        r, theta, z = X_tensor[:, 0], X_tensor[:, 1], X_tensor[:, 2]
        pos_x = r * torch.cos(theta)
        pos_y = r * torch.sin(theta)
        pos = torch.stack([pos_x, pos_y, z], dim=-1)

        edge_index = knn_graph(pos, self.k, loop=False)

        # Get preprocessed node features for the GNN.
        node_features = self.pre.transform_single(X_tensor)

        # Create edge-level ground truth labels for training.
        y_tensor = torch.from_numpy(y)
        row, col = edge_index
        edge_y = (y_tensor[row] == y_tensor[col]) & (y_tensor[row] > 0)

        return PyGData(x=node_features, edge_index=edge_index, y=edge_y.float(),
                       pos=pos, truth_labels=y_tensor.long(), num_nodes=X_tensor.shape[0])

def make_dataset(events, pre, train: bool, **kwargs):
    k = kwargs.get("k", 16)
    return CustomPygDataset(events, pre, train=train, k=k)

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit()
    #   - transform()

    def __init__(self):
        self.feature_means = None
        self.feature_stds = None
        self.num_layers = 20  # As per challenge_data_specs.json

    def _get_cartesian(self, X: torch.Tensor):
        r, theta = X[:, 0], X[:, 1]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return x, y

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "llm_script:make_dataset",
            "dataset_kwargs": {"k": 16},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 2 if torch.cuda.is_available() else 0, # Use workers if GPU is available
            "pin_memory": True,
            "collate": None,  # PyG loader supplies its own collate
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, data):
        # data is a list of X tensors [N, 4] from the training set.
        all_features = []
        for X_np in data:
            X = torch.from_numpy(X_np).float()
            x, y = self._get_cartesian(X)
            r, z = X[:, 0], X[:, 2]
            # Features to be scaled: x, y, z, r
            features = torch.stack([x, y, z, r], dim=-1) # [N_hits, 4]
            all_features.append(features)

        all_features = torch.cat(all_features, dim=0)
        self.feature_means = all_features.mean(dim=0)
        self.feature_stds = all_features.std(dim=0)
        return self

    def transform(self, data):
        # This is applied to the raw event list.
        # We perform the actual feature transformation in the Dataset's __getitem__
        # for on-the-fly graph construction. So this is an identity function.
        return data

    def transform_single(self, X: torch.Tensor):
        # X is a single event's feature tensor: [N, 4] float
        x, y = self._get_cartesian(X)
        z, r, layer_id = X[:, 2], X[:, 0], X[:, 3]

        # Scale continuous features
        cont_features = torch.stack([x, y, z, r], dim=-1) # [N, 4]
        cont_features = (cont_features - self.feature_means.to(X.device)) / self.feature_stds.to(X.device)

        # One-hot encode layer_id
        layer_id_one_hot = nn.functional.one_hot(layer_id.long(), num_classes=self.num_layers).float() # [N, 20]

        return torch.cat([cont_features, layer_id_one_hot], dim=-1) # [N, 24]

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # example_batch_x is a PyG Batch object
        input_dim = example_batch_x.num_node_features
        h_dim = 64

        # Stacked EdgeConv layers for deep feature learning
        self.gnn1 = EdgeConv(nn.Sequential(nn.Linear(2 * input_dim, h_dim), nn.ReLU()))
        self.gnn2 = EdgeConv(nn.Sequential(nn.Linear(2 * h_dim, h_dim), nn.ReLU()))
        self.gnn3 = EdgeConv(nn.Sequential(nn.Linear(2 * h_dim, h_dim), nn.ReLU()))

        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )
        self.register_buffer('dummy', torch.tensor(0))

    @property
    def device(self):
        return self.dummy.device

    def _get_edge_scores(self, batch: PyGData):
        h0 = batch.x # [N_total, F]
        h1 = self.gnn1(h0, batch.edge_index) # [N_total, h_dim]
        h2 = self.gnn2(h1, batch.edge_index) # [N_total, h_dim]
        h3 = self.gnn3(h2, batch.edge_index) # [N_total, h_dim]

        row, col = batch.edge_index # [2, E_total]
        edge_features = torch.cat([h3[row], h3[col]], dim=-1) # [E_total, 2*h_dim]
        return self.edge_predictor(edge_features)  # [E_total, 1]

    def forward(self, batch_x: PyGData):
        # batch_x is a PyG Batch object
        edge_logits = self._get_edge_scores(batch_x)
        if self.training:
            return edge_logits
        else:
            with torch.no_grad():
                return self.cluster_from_logits(batch_x, edge_logits)

    def cluster_from_logits(self, batch_x: PyGData, edge_logits: torch.Tensor):
        edge_scores = torch.sigmoid(edge_logits).squeeze()

        pred_labels = []
        for i in range(batch_x.num_graphs):
            node_mask = (batch_x.batch == i)
            num_nodes_in_graph = int(node_mask.sum())

            row, col = batch_x.edge_index
            edge_mask_for_graph = node_mask[row]

            graph_edges = batch_x.edge_index[:, edge_mask_for_graph]
            graph_edge_scores = edge_scores[edge_mask_for_graph]

            # Map global node indices to local-to-graph indices
            node_indices = torch.where(node_mask)[0]
            mapping = -torch.ones(batch_x.num_nodes, dtype=torch.long, device=self.device)
            mapping[node_indices] = torch.arange(num_nodes_in_graph, device=self.device)
            local_edges = mapping[graph_edges]

            # Filter edges with score > 0.5
            passing_edges = local_edges[:, graph_edge_scores > 0.5]

            if passing_edges.shape[1] > 0:
                adj = scipy.sparse.coo_matrix(
                    (torch.ones(passing_edges.shape[1]), 
                     (passing_edges[0].cpu().numpy(), passing_edges[1].cpu().numpy())),
                    shape=(num_nodes_in_graph, num_nodes_in_graph)
                )
                _, labels = scipy.sparse.csgraph.connected_components(
                    csgraph=adj, directed=False, return_labels=True
                )
                labels = torch.from_numpy(labels).long().to(self.device)
            else: # No edges passed the threshold
                labels = torch.arange(num_nodes_in_graph, device=self.device)

            # Filter out small clusters (< 4 hits)
            unique_labels, counts = torch.unique(labels, return_counts=True)
            small_clusters = unique_labels[counts < 4]
            final_labels = labels.clone()
            if len(small_clusters) > 0:
                is_small = torch.isin(labels, small_clusters)
                final_labels[is_small] = -1

            # Renumber valid clusters to be 1-indexed and contiguous
            valid_mask = (final_labels != -1)
            if valid_mask.any():
                uniques, inverse = torch.unique(final_labels[valid_mask], return_inverse=True)
                final_labels[valid_mask] = inverse + 1

            pred_labels.append(final_labels.long())
        return pred_labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 15
def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Give more weight to positive class (true edges) due to imbalance
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0).to(device))

    best_val_loss = float('inf')
    patience, patience_counter = 5, 0

    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss, total_acc, batches = 0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            edge_logits = model(batch)
            loss = criterion(edge_logits.squeeze(), batch.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            acc = ((torch.sigmoid(edge_logits).squeeze() > 0.5) == batch.y).float().mean()
            total_acc += acc.item()
            batches += 1

        train_loss.append(total_loss / batches)
        train_acc.append(total_acc / batches)

        model.eval()
        total_vloss, total_vacc, vbatches = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                # To get validation loss, we need the logits, not clustered labels.
                # We can call the internal method that just computes scores.
                edge_logits = model._get_edge_scores(batch)
                vloss = criterion(edge_logits.squeeze(), batch.y)
                total_vloss += vloss.item()
                vacc = ((torch.sigmoid(edge_logits).squeeze() > 0.5) == batch.y).float().mean()
                total_vacc += vacc.item()
                vbatches += 1

        current_val_loss = total_vloss / vbatches
        val_loss.append(current_val_loss)
        val_acc.append(total_vacc / vbatches)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}")

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break

    return model, train_loss, val_loss, train_acc, val_acc

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
        write_json(
            {"train_loss": tr_loss, "val_loss": va_loss, "train_acc": tr_acc, "val_acc": va_acc},
            out_path=os.path.join(SCRIPT_DIR, f"{base}_train_summary.json"),
        )

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

