
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy 
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader
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
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
# <LLM: Import modules>
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

# -------- (OPTIONAL) CUSTOM DATASET  --------
def make_dataset(events, pre, train: bool, **kwargs):
    # REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:make_dataset"
    # k = kwargs.get("k", 16)
    # <LLM: Insert custom dataset logic here>
    dataset = []
    for evt in events:
        X, y_true = split_X_y(evt)
        if pre is not None:
            X = pre.transform(X)

        # Build graph for the event
        # Find unique tracks > 0
        unique_tracks = sorted(set(y_true.numpy()) - {0}) 
        num_tracks = len(unique_tracks) 
        # Safety: cap at 50, remap 0=0, 1 to num_tracks
        num_classes = min(num_tracks, 50) + 1  # 0 for noise

        # Create mapping: original track_id > 0 to new labels 1 to num_classes-1
        track_to_label = {tid: i+1 for i, tid in enumerate(unique_tracks[:50])}  # cap at 50

        y_mapped = []
        for tid in y_true.numpy():
            if tid == 0:
                y_mapped.append(0)
            else:
                y_mapped.append(track_to_label.get(tid, 0))  # if overflow, treat as noise

        y = torch.tensor(y_mapped, dtype=torch.long)

        # Build edge_index: connect between layers with close spatial proximity
        r, theta, z, layer = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        layer = layer.long()
        unique_layers = sorted(torch.unique(layer).tolist())
        edges = []
        for i in range(len(unique_layers)-1):
            l1 = unique_layers[i]
            l2 = unique_layers[i+1]
            idx_l1 = (layer == l1)
            idx_l2 = (layer == l2)
            hits_l1 = torch.nonzero(idx_l1).squeeze()
            hits_l2 = torch.nonzero(idx_l2).squeeze()
            for i_hit in hits_l1:
                for j_hit in hits_l2:
                    # Simple proximity: delta_r < 10, delta_theta < 0.1, delta_z < 50
                    delta_r = abs(r[i_hit] - r[j_hit])
                    delta_theta = min(abs(theta[i_hit] - theta[j_hit]), 2*np.pi - abs(theta[i_hit] - theta[j_hit]))
                    delta_z = abs(z[i_hit] - z[j_hit])
                    if delta_r < 10 and delta_theta < 0.1 and delta_z < 50:
                        edges.append([i_hit, j_hit])
                        edges.append([j_hit, i_hit])  # undirected

        if not edges:
            # If no edges, fully connect or something, but for now, singletons
            edge_index = torch.empty(2, 0, dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            # Remove self-loops if any, but unlikely

        data = Data(x=X, edge_index=edge_index, y=y, num_classes=num_classes)
        dataset.append(data)
    return dataset

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
    #   - IMPORTANT Default data flow: events[idx] -> split_X_y(evt) -> X, y
    #   - When modifying data features or feature engineering: annotate tensor size as comments after each tensor operation to reduce dimension mismatches.

    # <LLM: Write code to preprocess the data> 

    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        self.scaler = StandardScaler()

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "llm_script:make_dataset",
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 1,  # process one event at a time for varying num_classes
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one:
            "collate": None,  # identity for PyG

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, F_raw]
        # Concat all for global stats
        all_X = torch.cat(Xs, dim=0)  # [total_hits, 4]
        self.scaler.fit(all_X.numpy())  # fit scaler on full data
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, F_raw]
        # Apply standard scaling
        X_scaled = torch.tensor(self.scaler.transform(X.numpy()), dtype=torch.float32)
        return X_scaled  # [N_hits, 4]

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # example_batch_x is a torch_geometric.Batch or Data
        self.in_features = example_batch_x.x.size(1)  # 4
        self.hidden = 64
        self.num_classes = 51  # fixed, assume max 50 tracks + noise

        # Simple GCN + classifier
        self.gcn1 = GCNConv(self.in_features, self.hidden)
        self.gcn2 = GCNConv(self.hidden, self.hidden)
        self.classifier = nn.Linear(self.hidden, self.num_classes)

    def forward(self, batch_x):
        # batch_x is a Data or Batch object
        x, edge_index = batch_x.x, batch_x.edge_index
        # Pass through GCN
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        # Classify each hit
        out = self.classifier(x)  # [N, 51]
        # Since batch_size=1, no issue
        # Get predicted labels: argmax, 0 to 50
        labels = torch.argmax(out, dim=1)  # [N] long
        # But model must return predicted integer labels, dtype long/int64
        return labels  # [N] long

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50   # adjust for convergence
def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()  # we handle per event

    best_val_loss = float('inf')
    patience = 5
    counter = 0

    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0

        for batch in train_loader:
            batch = batch.to(device)  # batch is Data, since batch_size=1
            optimizer.zero_grad()

            x = batch.x
            edge_index = batch.edge_index
            # Forward: get logits
            h1 = F.relu(model.gcn1(x, edge_index))
            h2 = F.relu(model.gcn2(x, edge_index))
            logits = model.classifier(h2)  # [N, 51]

            y = batch.y  # [N] long, 0 to num_classes-1 per event
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * x.size(0)
            epoch_train_total += x.size(0)
            preds = torch.argmax(logits, dim=1)

            # But since y has arbitrary labels within 0-50, compute accuracy as match to y (ignoring permutation)
            # For simple acc, number of matches
            epoch_train_correct += (preds == y).sum().item()

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_train_acc = epoch_train_correct / epoch_train_total  # simple per hit accuracy
        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)

        # Val
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x = batch.x
                edge_index = batch.edge_index
                h1 = F.relu(model.gcn1(x, edge_index))
                h2 = F.relu(model.gcn2(x, edge_index))
                logits = model.classifier(h2)
                y = batch.y
                loss = criterion(logits, y)

                epoch_val_loss += loss.item() * x.size(0)
                epoch_val_total += x.size(0)
                preds = torch.argmax(logits, dim=1)
                epoch_val_correct += (preds == y).sum().item()

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        avg_val_acc = epoch_val_correct / epoch_val_total
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Train Acc {avg_train_acc:.4f}, Val Loss {avg_val_loss:.4f}, Val Acc {avg_val_acc:.4f}")

        # Early stopping on val_loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            best_model = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                model.load_state_dict(best_model)
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

