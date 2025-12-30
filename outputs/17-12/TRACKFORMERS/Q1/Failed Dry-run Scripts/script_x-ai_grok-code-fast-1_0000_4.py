
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
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import torch_geometric.nn as TGnn

# -------- (OPTIONAL) CUSTOM DATASET  --------
def make_dataset(events, pre, train: bool, **kwargs):
    k = kwargs.get("k", 16)
    dataset = []
    for evt in events:
        X, y = pre.transform(evt)  # X: [N_hits, 4], y: [N_hits]
        X = X.float()
        # convert to cartesian for KNN
        r = X[:, 0]
        theta = X[:, 1]
        z = X[:, 2]
        x_cart = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=1).numpy()  # [N, 3]
        N = x_cart.shape[0]
        if N == 0:
            data = Data(x=X, y=y, edge_index=torch.empty(2, 0, dtype=torch.long))
        else:
            nn_model = NearestNeighbors(n_neighbors=min(k+1, N), metric='euclidean').fit(x_cart)
            _, indices = nn_model.kneighbors(x_cart)  # [N, k+1], first is self
            neighbors = indices[:, 1:]  # [N, k], exclude self
            row = torch.arange(N).repeat_interleave(k)
            col = torch.from_numpy(neighbors).flatten()
            edge_index = torch.stack([row, col], dim=0)  # [2, N*k]
            data = Data(x=X, y=y, edge_index=edge_index)
        dataset.append(data)
    return dataset

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "llm_script:make_dataset",
            "dataset_kwargs": {"k": 16},

            "loader_class": "torch_geometric.loader:DataLoader",    # Use PyG DataLoader
            "batch_size": 1,  # One event per batch
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": None,  # PyG handles collate

            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False}
        }

    def fit(self, data):
        # data is list of events, each event is dict
        # In harness, data is list of X
        # Extract all X to fit normalization
        all_X = []
        for evt in data:
            X, _ = split_X_y(evt)
            all_X.append(X)
        if all_X:
            all_X = torch.cat(all_X, dim=0)  # [total_N, 4]
            self.mean = all_X.mean(dim=0)  # [4]
            self.std = all_X.std(dim=0)  # [4]
        return self

    def transform(self, data):
        # data is event dict
        # Return (X, y) where X is torch.Tensor [N, 4], y [N]
        X, y = split_X_y(data)
        X = X.float()
        if self.mean is not None:
            X = (X - self.mean.to(X.device)) / (self.std.to(X.device) + 1e-6)
        y = torch.from_numpy(y).long()  # [N]
        return X, y

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # example_batch_x is a Data object, x.shape = [N, 4]
        in_features = example_batch_x.x.shape[1]  # should be 4
        self.encoder = nn.Sequential(
            TGnn.GCNConv(in_features, 64),
            nn.ReLU(),
            TGnn.GCNConv(64, 128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128, 51)  # 0-49 for tracks 1-50, 50 for noise

    def forward(self, batch_x):
        # Assume batch_x is a Data object
        x = batch_x.x  # [N, 4]
        edge_index = batch_x.edge_index  # [2, E]
        z = self.encoder(x, edge_index)  # [N, 128]
        logits = self.classifier(z)  # [N, 51]

        if self.training:
            # During training, return logits for loss computation
            return logits
        else:
            # During inference, return predicted labels
            pred = logits.argmax(dim=1)  # [N] 0-50
            pred = torch.where(pred == 50, -1, pred + 1)  # 0->1 (track1), ..., 49->50 (track50), 50->-1 (noise)
            return pred  # int64

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 80  # Adjust for convergence
def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        total_train_hits = 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            logits = model(batch)  # [N, 51]
            y_true = batch.y.long()  # [N]

            # Map track_ids to classes, noise=0 -> -1 (ignore), tracks 1-50 -> 0-49, noise=0->50
            unique_tracks = torch.unique(y_true[y_true > 0])
            y_mapped = torch.full_like(y_true, -1, dtype=torch.long)  # default ignore
            y_mapped[y_true == 0] = 50  # noise
            for j, t in enumerate(sorted(unique_tracks)):
                y_mapped[y_true == t] = j  # 0-49 for tracks

            loss = criterion(logits, y_mapped)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * batch.num_nodes

            # For accuracy, predict and compare
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                pred_mapped_back = torch.where(pred == 50, 0, -1)  # placeholder, accuracy is not critical here
                # Accuracy placeholder, as metric is FitAccuracy
                epoch_train_correct += (pred == y_mapped).sum().item()
            total_train_hits += batch.num_nodes

        train_loss = epoch_train_loss / total_train_hits
        train_acc = epoch_train_correct / total_train_hits  # not actual acc, but tracked

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_correct = 0
        total_val_hits = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)  # even though eval, but since training=False, wait no, model.eval() sets self.training=False, so returns int
                # Mistake: during validation, we need loss, but model returns labels
                # Need to get logits somehow. Perhaps compute logits separately or make forward always return labels but compute loss outside
                # To fix, since batch_size=1, re-run with model.training=True temp
                model.training = True  # temp
                logits = model(batch)
                model.training = False

                y_true = batch.y.long()
                unique_tracks = torch.unique(y_true[y_true > 0])
                y_mapped = torch.full_like(y_true, -1)
                y_mapped[y_true == 0] = 50
                for j, t in enumerate(sorted(unique_tracks)):
                    y_mapped[y_true == t] = j

                loss = criterion(logits, y_mapped)
                epoch_val_loss += loss.item() * batch.num_nodes

                # Placeholder acc
                pred = logits.argmax(dim=1)
                epoch_val_correct += (pred == y_mapped).sum().item()
                total_val_hits += batch.num_nodes

        val_loss = epoch_val_loss / total_val_hits if total_val_hits else 0
        val_acc = epoch_val_correct / total_val_hits

        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        # Early stopping
        if scheduler.num_bad_epochs >= 10:
            break

    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

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

