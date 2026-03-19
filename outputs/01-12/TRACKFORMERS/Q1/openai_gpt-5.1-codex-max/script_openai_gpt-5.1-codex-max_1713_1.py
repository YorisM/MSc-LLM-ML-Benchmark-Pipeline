
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
import math
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.cluster import DBSCAN

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    # TIPS
    #   - IMPORTANT Default data flow: events[idx] -> split_X_y(evt) -> X, y
    #   - When modifying data features or feature engineering: annotate tensor size as comments after each tensor operation to reduce dimension mismatches.

    # Define and initialize any stateful components here

    def __init__(self):
        self.mean = None
        self.std = None

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "utils.llm_io:EventDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one: 
            "collate": "ragged_xy",  # or "identity" or "None"
            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, F_raw]
        all_X = torch.cat(Xs, dim=0)  # [N_total, 4]
        r = all_X[:, 0]
        theta = all_X[:, 1]
        z = all_X[:, 2]
        layer = all_X[:, 3]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        feats = torch.stack([r, theta, z, layer, x, y], dim=1)  # [N_total, 6]
        self.mean = feats.mean(dim=0)
        self.std = feats.std(dim=0) + 1e-6
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, F_raw]
        r = X[:, 0]
        theta = X[:, 1]
        z = X[:, 2]
        layer = X[:, 3]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        feats = torch.stack([r, theta, z, layer, x, y], dim=1)  # [N_hits, 6]
        normed = (feats - self.mean) / self.std  # [N_hits, 6]
        sin_theta = torch.sin(theta).unsqueeze(1)  # [N_hits,1]
        cos_theta = torch.cos(theta).unsqueeze(1)  # [N_hits,1]
        out = torch.cat([normed, sin_theta, cos_theta], dim=1)  # [N_hits,8]
        return out.to(torch.float32)  # MUST return torch.FloatTensor [N_hits, F_out] for the default EventDataset path.

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        if isinstance(example_batch_x, list):
            in_dim = example_batch_x[0].shape[1]
        else:
            in_dim = example_batch_x.x.shape[1]
        hidden = 64
        emb_dim = 16
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim)
        )

    def forward(self, batch_x):
        # Returns list of embeddings per event for ragged input
        if isinstance(batch_x, list):
            embs = []
            for x in batch_x:
                embs.append(self.mlp(x))  # [N_i, emb_dim]
            return embs
        else:
            # PyG Batch
            return self.mlp(batch_x.x)

    def predict_labels(self, batch_x):
        self.eval()
        preds = []
        if isinstance(batch_x, list):
            with torch.no_grad():
                embs_list = self.forward(batch_x)
                for emb in embs_list:
                    emb_cpu = emb.detach().cpu().numpy()
                    # DBSCAN clustering on embeddings
                    clustering = DBSCAN(eps=0.5, min_samples=3, metric='euclidean', n_jobs=1)
                    labels = clustering.fit_predict(emb_cpu)  # [-1, 0, 1, ...]
                    preds.append(torch.from_numpy(labels.astype('int64')))
        else:
            # PyG Batch case
            with torch.no_grad():
                emb = self.forward(batch_x)
                emb_cpu = emb.detach().cpu().numpy()
                clustering = DBSCAN(eps=0.5, min_samples=3, metric='euclidean', n_jobs=1)
                labels = clustering.fit_predict(emb_cpu)
                preds = torch.from_numpy(labels.astype('int64')).to(batch_x.x.device)
        return preds

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 8   # adjust if you wish

def contrastive_loss(embs_list, labels_list, temperature=0.2):
    total_loss = 0.0
    total_count = 0
    for emb, y in zip(embs_list, labels_list):
        mask = y > 0  # ignore noise
        if mask.sum().item() <= 1:
            continue
        emb_pos = emb[mask]  # [N_pos, D]
        y_pos = y[mask]      # [N_pos]
        N = emb_pos.shape[0]
        norm_emb = F.normalize(emb_pos, dim=1)  # [N_pos, D]
        sim = torch.matmul(norm_emb, norm_emb.t()) / temperature  # [N_pos, N_pos]
        diag = torch.eye(N, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(diag, -1e9)
        y_mat = y_pos.unsqueeze(0) == y_pos.unsqueeze(1)  # [N_pos, N_pos]
        for i in range(N):
            pos_mask = y_mat[i] & (~diag[i])
            if pos_mask.sum().item() == 0:
                continue
            neg_mask = (~diag[i])
            denom = torch.logsumexp(sim[i][neg_mask], dim=0)
            pos_sim = sim[i][pos_mask]
            loss_i = -(pos_sim - denom).mean()
            total_loss += loss_i
            total_count += 1
    if total_count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / total_count

def compute_fit_accuracy(y_true, y_pred):
    # y_true: LongTensor [N], y_pred: LongTensor [N]
    truth_mask = y_true > 0
    total_true_hits = truth_mask.sum().item()
    if total_true_hits == 0:
        return 0.0
    correct_hits = 0
    unique_pred = torch.unique(y_pred)
    for plab in unique_pred:
        if plab.item() == -1:
            continue
        pred_mask = (y_pred == plab)
        pred_hits = pred_mask.sum().item()
        if pred_hits == 0:
            continue
        true_ids, counts = torch.unique(y_true[pred_mask], return_counts=True)
        if true_ids.numel() == 0:
            continue
        max_idx = torch.argmax(counts)
        true_id = true_ids[max_idx]
        if true_id.item() == 0:
            continue
        hits_from_true = counts[max_idx].item()
        total_hits_true_track = (y_true == true_id).sum().item()
        purity = hits_from_true / pred_hits
        efficiency = hits_from_true / total_hits_true_track
        if pred_hits >= 4 and purity >= 0.5 and efficiency >= 0.5:
            correct_hits += hits_from_true
    return correct_hits / total_true_hits

def evaluate_accuracy(model, loader):
    model.eval()
    total_correct = 0
    total_truth = 0
    with torch.no_grad():
        for batch in loader:
            Xs, ys = batch
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]
            preds = model.predict_labels(Xs)
            for y_true, y_pred in zip(ys, preds):
                acc_hits = compute_fit_accuracy(y_true.cpu(), y_pred.cpu())
                total_correct += acc_hits * (y_true > 0).sum().item()
                total_truth += (y_true > 0).sum().item()
    if total_truth == 0:
        return 0.0
    return total_correct / total_truth

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            Xs, ys = batch
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]
            optimizer.zero_grad()
            embs_list = model(Xs)  # list of embeddings
            loss = contrastive_loss(embs_list, ys)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        train_loss = epoch_loss / max(batches, 1)
        train_loss_history.append(train_loss)
        # Validation loss
        model.eval()
        val_epoch_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]
                embs_list = model(Xs)
                loss = contrastive_loss(embs_list, ys)
                val_epoch_loss += loss.item()
                val_batches += 1
        val_loss = val_epoch_loss / max(val_batches, 1)
        val_loss_history.append(val_loss)
        # Accuracy estimation
        train_acc = evaluate_accuracy(model, train_loader)
        val_acc = evaluate_accuracy(model, val_loader)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

