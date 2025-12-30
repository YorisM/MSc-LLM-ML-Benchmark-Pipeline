
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

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.mins = None
        self.maxs = None
        self.max_tracks = 51  # 1 for noise + 50 max tracks

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": "ragged_xy",
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, 4]
        all_X = torch.cat(Xs, dim=0)  # [total_hits, 4]
        self.mins = all_X.min(dim=0).values  # [4]
        self.maxs = all_X.max(dim=0).values  # [4]
        # Handle constant features to avoid division by zero
        self.ranges = self.maxs - self.mins
        self.ranges[self.ranges == 0] = 1.0
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, 4]
        # Apply min-max normalization: (X - min) / (max - min)
        X_norm = (X - self.mins) / self.ranges
        return X_norm  # [N_hits, 4]

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        self.d_model = 64
        self.num_classes = 51  # 1 (noise) + 50 max tracks
        self.num_layers_emb = 21  # Assume layer_id from 0 to 20
        self.layer_emb = nn.Embedding(self.num_layers_emb, self.d_model)
        self.pos_emb = nn.Linear(3, self.d_model)  # r, theta, z
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=8,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        self.head = nn.Linear(self.d_model, self.num_classes)

    def forward(self, batch_x):
        # batch_x: list of [N_hits_i, 4]
        outs = []
        for X in batch_x:
            if X.size(0) == 0:
                outs.append(torch.empty(0, dtype=torch.long, device=device))
                continue
            # Extract features: r, theta, z -> [N, 3]; layer_id -> [N]
            other = X[:, :3]  # [N, 3]
            layer = X[:, 3].long()  # [N], cast to long for embedding
            # Embed and combine
            emb_layer = self.layer_emb(layer)  # [N, d_model]
            emb_pos = self.pos_emb(other)  # [N, d_model]
            emb = emb_pos + emb_layer  # [N, d_model]
            # Apply Transformer Encoder: add batch dim, encode, remove batch dim
            encoded = self.encoder(emb.unsqueeze(0)).squeeze(0)  # [N, d_model]
            # Predict logits: [N, num_classes]
            logits = self.head(encoded)
            # Argmax for predicted labels: 0 (noise), 1 to num_classes-1 (tracks)
            pred = torch.argmax(logits, dim=-1).long()  # [N]
            outs.append(pred)
        return outs

# ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss, total_acc, total_hits = 0.0, 0.0, 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            out = model(xb)  # list of pred tensors
            loss_sum = 0.0
            acc_sum = 0.0
            event_hits = 0
            for i, (pred, true_y) in enumerate(zip(out, yb)):
                # Remap true_y: 0 for noise, 1 to M for unique tracks
                true_unique = torch.unique(true_y[true_y > 0])
                true_mapped = torch.zeros_like(true_y, dtype=torch.long)
                for idx, tid in enumerate(true_unique):
                    true_mapped[true_y == tid] = idx + 1  # 1 to len(true_unique)
                valid_mask = true_y > 0
                if valid_mask.sum() > 0:
                    # Compute cross-entropy, ignoring noise (mapped to 0)
                    loss = F.cross_entropy(out[i], true_mapped, ignore_index=0)
                    loss_sum += loss.item()
                    # Compute accuracy: fraction correct on valid hits
                    correct = (pred == true_mapped)[valid_mask].sum().item()
                    acc_sum += correct
                    event_hits += valid_mask.sum().item()
            if event_hits > 0:
                batch_loss = loss_sum / len(xb)
                batch_acc = acc_sum / event_hits
                total_loss += batch_loss
                total_acc += batch_acc
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss = total_loss / len(train_loader)
        train_acc = total_acc / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_hits = 0.0, 0.0, 0
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                out = model(xb)
                loss_sum, acc_sum, event_hits = 0.0, 0.0, 0
                for i, (pred, true_y) in enumerate(zip(out, yb)):
                    true_unique = torch.unique(true_y[true_y > 0])
                    true_mapped = torch.zeros_like(true_y, dtype=torch.long)
                    for idx, tid in enumerate(true_unique):
                        true_mapped[true_y == tid] = idx + 1
                    valid_mask = true_y > 0
                    if valid_mask.sum() > 0:
                        loss = F.cross_entropy(out[i], true_mapped, ignore_index=0)
                        loss_sum += loss.item()
                        correct = (pred == true_mapped)[valid_mask].sum().item()
                        acc_sum += correct
                        event_hits += valid_mask.sum().item()
                if event_hits > 0:
                    val_loss += loss_sum / len(xb)
                    val_acc += acc_sum / event_hits
                val_hits += event_hits
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

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

