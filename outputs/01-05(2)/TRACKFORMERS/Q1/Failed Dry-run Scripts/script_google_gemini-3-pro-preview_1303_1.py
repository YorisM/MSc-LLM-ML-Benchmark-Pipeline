
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import detect_and_assert_lane, assert_label_output_by_lane, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, build_trackformers_model

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

# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
#   REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
   def __init__(self, events, pre, train: bool = True, **kwargs):
       self.events = events
       self.pre = pre
       self.train = train

   def __len__(self):
       return len(self.events)

   def __getitem__(self, idx):
       # Retrieve raw event dict
       evt = self.events[idx]
       # Delegate processing to preprocessor to centralize logic
       return self.pre.process_event(evt)

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    def __init__(self):
        # Normalization constraints
        self.scale_r = 1000.0
        self.scale_z = 3000.0
        # Graph construction constraints
        self.k = 12

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:CustomDataset", 
            "dataset_kwargs": {},

            # PyG Lane Configuration
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 16,     # Batch size for Graph data
            "shuffle": True,
            "num_workers": 0,     # Avoid spawn/fork overhead issues in scripts
            "pin_memory": False,
            "collate": "None",    # PyG loader handles collation of Data objects

            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, F_raw]
        # X columns: hit_r, hit_theta, hit_z, layer_id

        # Estimate normalization scales from a subset of data
        max_r = 100.0
        max_z = 100.0

        # Sample every 10th event to be fast
        for x in Xs[::10]:
            # x is Tensor [N, 4]
            r = x[:, 0].abs().max().item()
            z = x[:, 2].abs().max().item()
            if r > max_r: max_r = r
            if z > max_z: max_z = z

        self.scale_r = max_r + 1.0
        self.scale_z = max_z + 1.0
        return self

    def transform(self, X):
        # This method is required by the harness interface but primarily used 
        # by the default EventDataset. We use CustomDataset, so we use process_event instead.
        # We return X as-is to satisfy return type hints if checked.
        return X

    def process_event(self, evt):
        # Convert raw dict to PyG Data object

        # 1. Extract and cast data
        # Inputs: r, theta, z, layer
        r = torch.from_numpy(evt["hit_r"].astype(np.float32))
        theta = torch.from_numpy(evt["hit_theta"].astype(np.float32))
        z = torch.from_numpy(evt["hit_z"].astype(np.float32))
        layer = torch.from_numpy(evt["layer_id"].astype(np.float32))
        y = torch.from_numpy(evt["track_id"].astype(np.int64))

        # 2. Coordinate Transformation (Cylindrical -> Cartesian)
        x_coord = r * torch.cos(theta)
        y_coord = r * torch.sin(theta)
        z_coord = z

        # 3. Features for Input Node (Normalized)
        # Using [x, y, z, r, layer]
        feat_x = x_coord / self.scale_r
        feat_y = y_coord / self.scale_r
        feat_z = z_coord / self.scale_z
        feat_r = r / self.scale_r
        feat_l = layer / 10.0 # Approximate layer scale

        # Node features: [N, 5]
        x = torch.stack([feat_x, feat_y, feat_z, feat_r, feat_l], dim=1)

        # 4. Graph Construction (k-NN)
        # Use normalized spatial coordinates for finding neighbors
        pos = torch.stack([feat_x, feat_y, feat_z], dim=1)

        # k=12 is a good tradeoff for efficiency vs recall
        edge_index = knn_graph(pos, k=self.k, loop=False)

        # 5. Label Generation (Edge Classification)
        src, dst = edge_index
        # An edge is True if: same track ID AND track ID is not noise (0)
        lid_src = y[src]
        lid_dst = y[dst]

        edge_label = (lid_src == lid_dst) & (lid_src != 0)
        edge_label = edge_label.float()

        # Return PyG Data
        return Data(x=x, edge_index=edge_index, y=y, edge_label=edge_label, num_nodes=x.size(0))

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # PyG Lane: example_batch_x is a Batch object (or Data)
        input_dim = 5 # x,y,z,r,layer
        hidden_dim = 96
        self.layers = 4

        # Feature Encoder
        self.node_enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # MPNN Layers
        self.edge_updates = nn.ModuleList()
        self.node_updates = nn.ModuleList()

        for _ in range(self.layers):
            # Edge Update Network: Concat(u, v) -> edge_hidden
            self.edge_updates.append(nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ))

            # Node Update Network: Concat(u, aggr_edge) -> node_hidden
            self.node_updates.append(nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ))

        # Final Edge Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch_x):
        # Unpack
        x = batch_x.x
        edge_index = batch_x.edge_index
        src, dst = edge_index

        # Encode
        x = self.node_enc(x)

        # Message Passing
        for i in range(self.layers):
            # 1. Edge Update
            edge_input = torch.cat([x[src], x[dst]], dim=1)
            edge_msg = self.edge_updates[i](edge_input)

            # 2. Aggregation (Sum)
            # Scatter sum edge messages to destination nodes
            aggr = torch.zeros_like(x)
            aggr.index_add_(0, dst, edge_msg)

            # 3. Node Update
            node_input = torch.cat([x, aggr], dim=1)
            dx = self.node_updates[i](node_input)
            x = x + dx # Residual

        # Final Classification
        # Using updated node embeddings
        edge_input = torch.cat([x[src], x[dst]], dim=1)
        logits = self.classifier(edge_input).squeeze(-1)

        return logits

    def predict_labels(self, batch_x):
        # 1. Run inference
        self.eval()
        with torch.no_grad():
            scores = self.forward(batch_x)
            probs = torch.sigmoid(scores)

        # 2. Thresholding for valid edges
        # Slightly higher threshold to prefer precision (purity) over recall
        threshold = 0.55
        mask = probs > threshold

        # 3. Connected Components
        # Use CPU filtering and Scipy
        n_nodes = batch_x.num_nodes
        edge_index = batch_x.edge_index

        # Filter predicted edges
        src = edge_index[0][mask].cpu().numpy()
        dst = edge_index[1][mask].cpu().numpy()

        # Build symmetric adjacency for undirected clustering (track is a line)
        # Using a boolean/ones matrix
        adj = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n_nodes, n_nodes))

        # Find components
        n_comps, labels = csgraph.connected_components(adj, directed=False)

        # Return LongTensor of labels
        return torch.from_numpy(labels).long().to(batch_x.x.device)

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 12
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Loss: weighted BCE because real edges are sparse in KNN graph
    # Weight ~ (k / 2) roughly, k=12 -> 6. Conservative: 5.0
    pos_weight = torch.tensor([5.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_hist_loss, val_hist_loss = [], []
    train_hist_acc, val_hist_acc = [], []

    for epoch in range(epochs):
        model.train()
        t_loss, t_acc, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)
            loss = criterion(out, batch.edge_label)

            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            with torch.no_grad():
                pred = (out > 0.0).float()
                acc = (pred == batch.edge_label).float().mean()
                t_acc += acc.item()
            n_batches += 1

        avg_t_loss = t_loss / max(n_batches, 1)
        avg_t_acc = t_acc / max(n_batches, 1)
        train_hist_loss.append(avg_t_loss)
        train_hist_acc.append(avg_t_acc)

        scheduler.step()

        # Validation
        model.eval()
        v_loss, v_acc, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.edge_label)

                v_loss += loss.item()
                pred = (out > 0.0).float()
                acc = (pred == batch.edge_label).float().mean()
                v_acc += acc.item()
                n_val += 1

        avg_v_loss = v_loss / max(n_val, 1)
        avg_v_acc = v_acc / max(n_val, 1)
        val_hist_loss.append(avg_v_loss)
        val_hist_acc.append(avg_v_acc)

        print(f"Epoch {epoch+1} | T_Loss: {avg_t_loss:.4f} Acc: {avg_t_acc:.4f} | V_Loss: {avg_v_loss:.4f} Acc: {avg_v_acc:.4f}")

    return model, train_hist_loss, val_hist_loss, train_hist_acc, val_hist_acc
# <end code template>

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
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

