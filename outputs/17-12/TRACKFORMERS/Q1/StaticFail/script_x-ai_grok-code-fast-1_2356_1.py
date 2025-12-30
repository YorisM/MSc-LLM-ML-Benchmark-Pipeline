
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
# <LLM: Import modules>
from torch_geometric.nn import GCNConv
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
import torch.nn.functional as F
import hdbscan
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# -------- (OPTIONAL) CUSTOM DATASET  --------
# def make_dataset(events, pre, train: bool, **kwargs):
#   REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:make_dataset"
#   k = kwargs.get("k", 16)
#   <LLM: Insert custom dataset logic here>
#   return CustomDataset(events, pre, train=train, k=k)

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit()
    #   - transform()

    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    # TIPS
    #   When modifying data features or feature engineering: annotate tensor size as comments after each tensor operation to reduce dimension mismatches.

    # <LLM: Write code to preprocess the data> 
    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        pass

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "llm_script:make_dataset",
            "dataset_kwargs": {"k": 5},

            "loader_class": "torch_geometric.loader:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 4,  # small batch size due to variable graphs
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one:
            "collate": None,  # None for PyG default

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, data):
        # <LLM: Extract statistics or fit transform>
        return self

    def transform(self, data):
        # <LLM: Apply preprocessing logic, return torch.Tensor>
        return data # must return an indexable, picklable object

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # IMPORTANT: Default harness input:
        #   batch_x is ragged list[Tensor], one per event, each shaped [N_hits, F].
        #   Infer F from example_batch_x (do NOT assume an int is passed).

        # <LLM: Define and initialize any stateful components here>
        # e.g. infer in_features from example_batch_x and build layers

        in_features = 4  # r, theta, z, layer_id
        self.conv1 = GCNConv(in_features, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 128)

    def get_embeddings(self, batch_x):
        x_emb = self.conv1(batch_x.x, batch_x.edge_index)
        x_emb = F.relu(x_emb)
        x_emb = self.conv2(x_emb, batch_x.edge_index)
        x_emb = F.relu(x_emb)
        x_emb = self.conv3(x_emb, batch_x.edge_index)
        return x_emb  # [N, 128]

    def forward(self, batch_x):
        # IMPORTANT Input contract:
        #   forward() MUST handle ragged list[Tensor] and may optionally support a single padded Tensor / PyG Batch.
        #   Harness calls:
        #       view = normalise_batch(batch, device=device)
        #       out  = model(view.batch_x)
        # 
        # IMPORTANT Output contract:
        #   forward(batch_x) must return predicted integer labels (dtype long/int64) with one label per hit (>0); predicted noise may be -1.

        # <LLM: Define your model's forward pass here>
        x_emb = self.get_embeddings(batch_x)
        device = batch_x.x.device
        labels = torch.zeros(len(batch_x.x), dtype=torch.long, device=device) - 1  # [N], default -1 for noise
        for graph_id in torch.unique(batch_x.batch, sorted=True):
            mask = batch_x.batch == graph_id
            embed_graph = x_emb[mask]  # [N_graph, 128]
            clusterer = hdbscan.HDBSCAN(min_cluster_size=4, prediction_data=True)
            cluster_labels_np = clusterer.fit_predict(embed_graph.detach().cpu().numpy())  # numpy array
            cluster_labels = torch.from_numpy(cluster_labels_np).to(device)  # [N_graph], int (-1 or 0,1,...)
            unique_clusters = torch.unique(cluster_labels)
            unique_clusters = unique_clusters[unique_clusters >= 0]
            # assign labels starting from 1 for each cluster
            for new_label, cl in enumerate(unique_clusters):
                cluster_mask = cluster_labels == cl
                labels[mask][cluster_mask] = new_label + 1  # 1, 2, 3, ...
        # split into list per graph [N_i]
        labels_list = []
        for graph_id in torch.unique(batch_x.batch, sorted=True):
            mask = batch_x.batch == graph_id
            labels_list.append(labels[mask])
        return labels_list

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50   # <LLM: adjust if you wish>   
def train_model(model, train_loader, val_loader, epochs):
    # If your method is non-parametric, train_model may be a no-op that returns the unmodified model and empty metric lists, otherwise:

    # REQUIREMENTS 
    #   - Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #   - Must return trained_model, train_loss, val_loss, train_acc, val_acc
    #   - Implement early-stopping.
    #   - Use CUDA - torch.cuda.is_available()
    #   - Forward signature must match.

    # <LLM: Write code to define training loop>
    # <LLM: Implement early stopping if possible>
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=max(epochs//3, 1), gamma=0.5)
    tr_loss, va_loss = [], []
    tr_acc, va_acc = [], []  # dummy placeholders
    patience = 10
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_tr_loss = 0.0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            x_emb = model.get_embeddings(view.batch_x)
            loss = 0.0
            graphs = torch.unique(view.batch_x.batch)
            for g_id in graphs:
                mask = view.batch_x.batch == g_id
                y = view.batch_x.y[mask]
                emb = x_emb[mask]
                mask2 = y > 0
                if mask2.sum() == 0:
                    continue
                y2 = y[mask2]
                emb2 = emb[mask2]
                num_tracks = len(torch.unique(y2))
                if num_tracks == 0:
                    continue
                # means [num_tracks, 128]
                means = pyg_utils.scatter('mean', emb2, y2 - 1, dim=0, dim_size=num_tracks)
                loss += F.mse_loss(emb2, means[y2 - 1].expand_as(emb2))
            if len(graphs) > 0:
                loss /= len(graphs)
            else:
                loss = 0.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_tr_loss += loss.item()
        tr_loss.append(epoch_tr_loss / len(train_loader))

        # validation loss
        model.eval()
        epoch_va_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                x_emb = model.get_embeddings(view.batch_x)
                loss = 0.0
                graphs = torch.unique(view.batch_x.batch)
                for g_id in graphs:
                    mask = view.batch_x.batch == g_id
                    y = view.batch_x.y[mask]
                    emb = x_emb[mask]
                    mask2 = y > 0
                    if mask2.sum() == 0:
                        continue
                    y2 = y[mask2]
                    emb2 = emb[mask2]
                    num_tracks = len(torch.unique(y2))
                    if num_tracks == 0:
                        continue
                    means = pyg_utils.scatter('mean', emb2, y2 - 1, dim=0, dim_size=num_tracks)
                    loss += F.mse_loss(emb2, means[y2 - 1].expand_as(emb2))
                if len(graphs) > 0:
                    loss /= len(graphs)
                else:
                    loss = 0.0
                epoch_va_loss += loss.item()
        va_loss.append(epoch_va_loss / len(val_loader))

        # early stopping on val_loss
        if va_loss[-1] < best_loss:
            best_loss = va_loss[-1]
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    # dummy acc
    tr_acc = [0.0] * len(tr_loss)
    va_acc = [0.0] * len(va_loss)

    return model, tr_loss, va_loss, tr_acc, va_acc

def make_dataset(events, pre, train: bool, **kwargs):
    k = kwargs.get("k", 5)
    from torch_geometric.data import Dataset as PyGDataset
    data_list = []
    for evt in events:
        X, y = split_X_y(evt)
        # X: [N,4] float32
        r, theta, z, layer_id = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        pos = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=1)  # [N, 3]
        edge_index = pyg_utils.knn_graph(pos.float(), k=k)
        d = Data(x=X.float(), y=y.long(), pos=pos.float(), edge_index=edge_index)
        data_list.append(d)

    class CustomDataset(PyGDataset):
        def __init__(self, data_list):
            super().__init__()
            self.data_list = data_list
        def len(self):
            return len(self.data_list)
        def get(self, idx):
            return self.data_list[idx]

    return CustomDataset(data_list)

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

