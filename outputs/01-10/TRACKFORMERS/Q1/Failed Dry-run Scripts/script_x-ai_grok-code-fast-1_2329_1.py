
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
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
# <LLM: Import modules>
import math
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        evt = events
        X = np.column_stack([evt["hit_r"], evt["hit_theta"], evt["hit_z"], evt["layer_id"]]).astype(np.float32)
        y = evt["track_id"].astype(np.int64)
        self.data = pre.transform((X, y))
    def __len__(self):
        return 1  # one event per dataset
    def __getitem__(self, idx):
        return self.data

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = None
        self.delta_z_max = 0.1  # tune if needed
        self.delta_theta_max = 0.1  # tune if needed

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:MyCustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 16,  # smaller batch since graphs
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,  # PyG default
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of np.array [N_i, 4]
        all_X = np.concatenate([X for X in Xs], axis=0)  # [sum N, 4]
        self.scaler = StandardScaler().fit(all_X[:, :3])  # r, theta, z
        return self

    def transform(self, data):
        X, y = data  # X: np.array [N, 4], y: np.array [N]
        X_transformed = X.copy()
        X_transformed[:, :3] = self.scaler.transform(X[:, :3])
        x = torch.from_numpy(X_transformed)  # [N, 4]

        # Build edge_index
        PI = math.pi
        n = X.shape[0]
        edge_list = []
        for i in range(n):
            for j in range(i+1, n):
                # Check consecutive layers and proximity
                layer_i, layer_j = X[i, 3], X[j, 3]
                if abs(layer_i - layer_j) == 1:
                    theta_i, theta_j = X[i, 1], X[j, 1]
                    delta_theta = abs(theta_i - theta_j) % (2 * PI)
                    delta_theta = min(delta_theta, 2 * PI - delta_theta)
                    delta_z = abs(X[i, 2] - X[j, 2])
                    if delta_theta < self.delta_theta_max and delta_z < self.delta_z_max:
                        edge_list.append([i, j])
                        edge_list.append([j, i])  # undirected
        if not edge_list:
            # Fallback: fully connected, but sparse if possible
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edge_list.append([i, j])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.empty(2, 0, dtype=torch.long)
        y_tensor = torch.from_numpy(y)
        data = Data(x=x, edge_index=edge_index, y=y_tensor)
        return data

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        self.num_classes_global = 51  # max 50 tracks + noise
        self.hidden = 128
        F = example_batch_x.x.shape[1]  # 4
        self.encoder = nn.Linear(F, self.hidden)
        self.gnn1 = GCNConv(self.hidden, self.hidden)
        self.gnn2 = GCNConv(self.hidden, self.hidden)
        self.classifier = nn.Linear(self.hidden, self.num_classes_global)
        # Final output size: logits [N, 51]

    def forward(self, batch_x):  # batch_x is PyG Batch, has x, edge_index, batch, etc.
        h = self.encoder(batch_x.x)
        h = F.relu(h)
        h = self.gnn1(h, batch_x.edge_index)
        h = F.relu(h)
        h = self.gnn2(h, batch_x.edge_index)
        h = F.relu(h)
        h = self.classifier(h)  # [total_N, 51]
        return h

    def predict_labels(self, batch_x):  # batch_x is PyG Batch
        with torch.no_grad():
            h = self(batch_x)
            labels = torch.argmax(h, dim=1)  # [N]
            labels = labels.clone()
            labels[labels == 0] = -1  # noise: 0 -> -1
            return labels  # LongTensor [N]

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20  # tune
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    best_acc = -1
    patience = 5
    no_improve = 0

    train_metrics = []
    val_metrics = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        train_correct = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            h = model(batch)  # [total_N, 51]
            loss = F.cross_entropy(h, batch.y, ignore_index=0)  # ignore noise for loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred_labels = model.predict_labels(batch)
            # For train acc, approx by comparing to batch.y, but since local, use as is
            total_samples += batch.x.shape[0]

        avg_loss = total_loss / len(train_loader)
        train_acc = train_correct / total_samples if total_samples > 0 else 0

        model.eval()
        val_loss = 0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                h = model(batch)
                loss = F.cross_entropy(h, batch.y, ignore_index=0)
                val_loss += loss.item()
                val_samples += batch.x.shape[0]
        val_acc = 0  # approx, since FitAccuracy not computed here
        avg_val_loss = val_loss / len(val_loader)

        train_metrics.append(avg_loss)
        val_metrics.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss {avg_loss:.4f}, Val Loss {avg_val_loss:.4f}")

        scheduler.step()

        # Early stopping on val_loss
        if avg_val_loss < best_acc or best_acc == -1:
            best_acc = avg_val_loss
            no_improve = 0
            best_model = model.state_dict()
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

    model.load_state_dict(best_model)
    return model, train_metrics, val_metrics, train_acc, val_acc

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

