
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
from sklearn.preprocessing import StandardScaler
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

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
        # Concatenate all Xs for fitting scaler
        all_X = torch.cat(Xs, dim=0).numpy()
        self.scaler.fit(all_X)
        return self

    def transform(self, X):
        # X: [N, 4]
        X_np = X.numpy()
        X_scaled = self.scaler.transform(X_np)
        return torch.from_numpy(X_scaled).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        self.embed = nn.Linear(4, 64)
        encoder_layers = TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layers, num_layers=4)
        self.classifier = nn.Linear(64, 51)  # 0 noise, 1-50 tracks

    def forward(self, batch_x):
        # batch_x: list of [N_i, 4]
        logits_list = []
        for x in batch_x:
            x_emb = self.embed(x)  # [N, 64]
            x_emb = x_emb.unsqueeze(0)  # [1, N, 64] for batch_first=False, but adjust to batch_first=True
            # TransformerEncoder expects [seq_len, batch_size, emb_size] or batch_first=True
            x_encoded = self.encoder(x_emb)  # [1, N, 64]
            logits = self.classifier(x_encoded.squeeze(0))  # [N, 51]
            logits_list.append(logits)
        return logits_list  # list of [N_i, 51]

    def predict_labels(self, batch_x):
        # batch_x: list of [N_i, 4]
        with torch.no_grad():
            logits_list = self.forward(batch_x)
        labels_list = []
        for logits in logits_list:
            preds = torch.argmax(logits, dim=-1)  # [N_i]
            preds = torch.where(preds == 0, -1, preds)  # 0 -> -1, others stay
            labels_list.append(preds)
        return labels_list

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore noise for loss? But noise is 0, tracks 1-50, wait no.
    # Actually, targets are 0 for noise, 1-50 for tracks, so we want to predict them all.
    # Change criterion to ignore_index=-1, but targets are 0-50, no -1.
    # Set ignore_index to something not used, say 99.
    criterion = nn.CrossEntropyLoss()

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        correct = 0
        total = 0
        for Xs, ys in train_loader:
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]
            optimizer.zero_grad()
            logits_list = model(Xs)
            loss = 0
            batch_correct = 0
            batch_total = 0
            for logits, y in zip(logits_list, ys):
                loss += criterion(logits, y)
                preds = torch.argmax(logits, dim=-1)
                mask_non_noise = y != 0  # only count non-noise
                batch_correct += (preds[mask_non_noise] == y[mask_non_noise]).sum().item()
                batch_total += mask_non_noise.sum().item()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            correct += batch_correct
            total += batch_total
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = correct / total if total > 0 else 0
        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)
        scheduler.step()

        model.eval()
        epoch_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]
                logits_list = model(Xs)
                loss = 0
                batch_correct = 0
                batch_total = 0
                for logits, y in zip(logits_list, ys):
                    loss += criterion(logits, y)
                    preds = torch.argmax(logits, dim=-1)
                    mask_non_noise = y != 0
                    batch_correct += (preds[mask_non_noise] == y[mask_non_noise]).sum().item()
                    batch_total += mask_non_noise.sum().item()
                epoch_val_loss += loss.item()
                correct += batch_correct
                total += batch_total
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_acc = correct / total if total > 0 else 0
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Train Acc {avg_train_acc:.4f}, Val Loss {avg_val_loss:.4f}, Val Acc {avg_val_acc:.4f}")

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

