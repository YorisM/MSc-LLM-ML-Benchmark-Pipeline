
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
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.neighbors import kneighbors_graph
import hdbscan

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True):
        self.events = events
        self.pre = pre

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        X, y = split_X_y(self.events[idx])
        if self.pre is not None:
            X = self.pre.transform(X)
        # X: [N, 4] float32, x, y, z, layer
        pos = X.numpy()
        k = 10
        adj_coo = kneighbors_graph(pos, k, mode='connectivity')
        edge_index, _ = from_scipy_sparse_matrix(adj_coo)
        data = Data(x=X.float(), y=y.long(), edge_index=edge_index)
        return data


# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": None,
            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of [N_i, F_raw]
        all_X = torch.cat(Xs, dim=0)  # [total_N, 4]
        self.mean = all_X.mean(dim=0)
        self.std = all_X.std(dim=0)
        return self

    def transform(self, X):
        # X: [N, 4] raw: r, theta, z, layer
        r, theta, z, layer = X.unbind(dim=1)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        X_cart = torch.stack([x, y, z, layer], dim=1)  # [N, 4] x,y,z,layer
        X_cart = (X_cart - self.mean) / (self.std + 1e-8)
        return X_cart  # [N, 4]


def make_preprocessor():
    return MyPreprocessor()


# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        from torch_geometric.nn import GATConv
        input_dim = example_batch_x.shape[1]
        self.conv1 = GATConv(input_dim, 64, heads=8)
        self.conv2 = GATConv(64 * 8, 64, heads=4)
        self.conv3 = GATConv(64 * 4, 32)

    def forward(self, G):
        # G.x: [N, 4], G.edge_index: [2, E]
        x, edge_index = G.x, G.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return x  # [N, 32] embeddings

    def predict_labels(self, G):
        # G: Batch
        with torch.no_grad():
            emb = self.forward(G)  # [N, 32]
            batch = G.batch  # [N]
            unique_events = batch.unique()
            labels = torch.zeros_like(G.y, dtype=torch.long) - 1
            for e in unique_events:
                mask = batch == e
                e_emb = emb[mask].cpu().numpy()
                clusterer = hdbscan.HDBSCAN(min_cluster_size=4, metric='euclidean')
                e_labels = clusterer.fit_predict(e_emb)
                e_labels[e_labels == -1] = -1  # noise remains -1
                labels[mask] = torch.tensor(e_labels, dtype=torch.long, device=G.x.device)
            return labels  # [N] long tensor


def make_model(example_batch_x):
    return HitClassifier(example_batch_x)


# ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        num_events = 0
        for G in train_loader:
            G = G.to(device)
            emb = model(G)  # [N, 32]
            ys = G.y  # [N]
            batch = G.batch  # [N]
            unique_events = batch.unique()
            loss = 0
            acc_sum = 0
            events = 0
            for e in unique_events:
                mask = batch == e
                e_emb = emb[mask]
                e_ys = ys[mask]
                unique_track_ids = e_ys.unique()
                track_ids_no_noise = unique_track_ids[unique_track_ids > 0]
                n_tracks = len(track_ids_no_noise)
                if n_tracks == 0:
                    continue
                # compute centroids
                centroids = []
                for tid in track_ids_no_noise:
                    centroids.append(e_emb[e_ys == tid].mean(dim=0))
                centroids = torch.stack(centroids)  # [n_tracks, 32]
                # only non-noise hits
                valid_mask = e_ys > 0
                e_emb_valid = e_emb[valid_mask]  # [N_valid, 32]
                e_ys_valid = e_ys[valid_mask]  # [N_valid]
                if len(e_emb_valid) == 0:
                    continue
                dists = torch.cdist(e_emb_valid, centroids)  # [N_valid, n_tracks]
                # map tid to 0..n_tracks-1
                tid_to_idx = {tid.item(): i for i, tid in enumerate(track_ids_no_noise)}
                target = torch.tensor([tid_to_idx[tid.item()] for tid in e_ys_valid],
                                      dtype=torch.long, device=device)
                logits = -dists
                loss += F.cross_entropy(logits, target)
                pred = logits.argmax(dim=1)
                acc_sum += (pred == target).float().sum().item() / len(target)
                events += 1
            if events > 0:
                loss /= events
                total_loss += loss.item()
                total_acc += acc_sum / events
                num_events += 1
            else:
                loss = torch.tensor(0., device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if num_events > 0:
            train_loss.append(total_loss / num_events)
            train_acc.append(total_acc / num_events)
        else:
            train_loss.append(0.)
            train_acc.append(0.)

        # validation
        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        num_val_events = 0
        with torch.no_grad():
            for G in val_loader:
                G = G.to(device)
                emb = model(G)
                ys = G.y
                batch = G.batch
                unique_events = batch.unique()
                val_loss = 0
                val_acc_sum = 0
                events = 0
                for e in unique_events:
                    mask = batch == e
                    e_emb = emb[mask]
                    e_ys = ys[mask]
                    unique_track_ids = e_ys.unique()
                    track_ids_no_noise = unique_track_ids[unique_track_ids > 0]
                    n_tracks = len(track_ids_no_noise)
                    if n_tracks == 0:
                        continue
                    centroids = []
                    for tid in track_ids_no_noise:
                        centroids.append(e_emb[e_ys == tid].mean(dim=0))
                    centroids = torch.stack(centroids)
                    valid_mask = e_ys > 0
                    e_emb_valid = e_emb[valid_mask]
                    e_ys_valid = e_ys[valid_mask]
                    if len(e_emb_valid) == 0:
                        continue
                    dists = torch.cdist(e_emb_valid, centroids)
                    tid_to_idx = {tid.item(): i for i, tid in enumerate(track_ids_no_noise)}
                    target = torch.tensor([tid_to_idx[tid.item()] for tid in e_ys_valid],
                                          dtype=torch.long, device=device)
                    logits = -dists
                    val_loss += F.cross_entropy(logits, target).item()
                    pred = logits.argmax(dim=1)
                    val_acc_sum += (pred == target).float().sum().item() / len(target)
                    events += 1
                if events > 0:
                    val_loss /= events
                    total_val_loss += val_loss
                    total_val_acc += val_acc_sum / events
                    num_val_events += 1
        if num_val_events > 0:
            val_loss.append(total_val_loss / num_val_events)
            val_acc.append(total_val_acc / num_val_events)
        else:
            val_loss.append(0.)
            val_acc.append(0.)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")

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

