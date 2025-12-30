
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
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence

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

    def __init__(self):
        self.MAX_CLASSES = 51  # Assuming max tracks ~50, with 0 for noise

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 32,  # Reduced for variable lengths
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one:
            "collate": "ragged_xy",  # or "identity" or None

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, data):
        # No fitting needed
        return self

    def transform(self, data):
        # Apply preprocessing logic, return torch.Tensor
        X, y = data
        y = y.numpy()
        unique_ids = np.unique(y[y > 0])
        track_map = {tid: i + 1 for i, tid in enumerate(sorted(unique_ids))}
        new_y = np.array([track_map.get(tid, 0) if tid > 0 else 0 for tid in y])
        return X, torch.from_numpy(new_y)  # must return an indexable, picklable object

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # IMPORTANT: Default harness input:
        #   batch_x is ragged list[Tensor], one per event, each shaped [N_hits, F].
        #   Infer F from example_batch_x (do NOT assume an int is passed).
        F = example_batch_x[0].shape[1]  # F=4
        self.num_classes = 51  # MAX_CLASSES +1
        self.embed = nn.Linear(F, 256)  # [N, F] -> [N, 256]
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.1), 
            num_layers=4
        )
        self.classifier = nn.Linear(256, self.num_classes)  # [N, 256] -> [N, 51]

    def forward_padded(self, x, mask=None):
        """Helper for padded input: x [B, S, F], mask [B, S] if provided"""
        embedded = self.embed(x)  # [B, S, 256]
        embedded = embedded.transpose(0, 1)  # [S, B, 256]
        transformed = self.transformer(embedded, src_key_padding_mask=mask)  # [S, B, 256]
        transformed = transformed.transpose(0, 1)  # [B, S, 256]
        logits = self.classifier(transformed)  # [B, S, 51]
        return logits

    def forward(self, batch_x):
        # IMPORTANT Input contract:
        #   forward() MUST handle ragged list[Tensor] and may optionally support a single padded Tensor / PyG Batch.
        #   Harness calls:
        #       view = normalise_batch(batch, device=device)
        #       out  = model(view.batch_x)
        # 
        # IMPORTANT Output contract:
        #   forward(batch_x) must return predicted integer labels (dtype long/int64) with one label per hit (>0); predicted noise may be -1.
        if isinstance(batch_x, list):
            Xs = batch_x
            max_N = max(len(x) for x in Xs)
            Xs_padded = []
            masks = []
            for x in Xs:
                pad_len = max_N - len(x)
                if pad_len > 0:
                    pad = torch.zeros(pad_len, x.shape[1], device=x.device)
                    x_pad = torch.cat([x, pad], dim=0)
                else:
                    x_pad = x
                Xs_padded.append(x_pad)
                mask = torch.cat([torch.zeros(len(x), dtype=bool, device=x.device), 
                                  torch.ones(pad_len, dtype=bool, device=x.device)]) if pad_len > 0 else torch.zeros(len(x), dtype=bool, device=x.device)
                masks.append(mask)
            Xs_tensor = torch.stack(Xs_padded, dim=0)  # [B, max_N, F]
            masks_tensor = torch.stack(masks, dim=0)  # [B, max_N]
            logits = self.forward_padded(Xs_tensor, masks_tensor)  # [B, max_N, 51]
            preds = torch.argmax(logits, dim=-1)  # [B, max_N]
            out = []
            for i, (pred, mask) in enumerate(zip(preds, masks_tensor)):
                valid_pred = pred[~mask[:len(pred)]]
                # Map 0 to -1 for noise, others as is
                valid_pred = torch.where(valid_pred == 0, -1, valid_pred)
                out.append(valid_pred)
            return out
        else:
            # Single padded tensor, for training
            logits = self.forward_padded(batch_x)  # [B, S, 51]
            return torch.argmax(logits, dim=-1)  # [B, S]

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50   
def train_model(model, train_loader, val_loader, epochs):
    # If your method is non-parametric, train_model may be a no-op that returns the unmodified model and empty metric lists, otherwise:

    # REQUIREMENTS 
    #   Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #   Must return trained_model, train_loss, val_loss, train_acc, val_acc
    #   Implement early-stopping.
    #   Use CUDA - torch.cuda.is_available()
    #   Forward signature must match.
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0
    patience = 10
    no_improve = 0

    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        total_train_hits = 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            batch_x = view.batch_x
            batch_y = view.batch_y  # list[Tensor [N]]

            # Pad batch_x and batch_y
            max_N = max(len(x) for x in batch_x)
            Xs_padded = pad_sequence(batch_x, batch_first=True, padding_value=0).to(device)
            ys_padded = []
            masks = []
            for y in batch_y:
                pad_len = max_N - len(y)
                y_pad = torch.cat([y.to(device), torch.full((pad_len,), -1, dtype=y.dtype, device=device)], dim=0)
                ys_padded.append(y_pad)
                mask = torch.cat([torch.zeros(len(y), dtype=bool, device=device), 
                                  torch.ones(pad_len, dtype=bool, device=device)]) if pad_len > 0 else torch.zeros(len(y), dtype=bool, device=device)
                masks.append(mask)
            ys_tensor = torch.stack(ys_padded, dim=0)  # [B, max_N]
            masks_tensor = torch.stack(masks, dim=0)  # [B, max_N]

            optimizer.zero_grad()
            logits = model.forward_padded(Xs_padded, masks_tensor)  # [B, S, 51]
            mask_flat = masks_tensor.view(-1)
            logits_flat = logits.view(-1, 51)[~mask_flat]
            targets_flat = ys_tensor.view(-1)[~mask_flat]
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            # Acc for monitoring
            preds_flat = torch.argmax(logits_flat, dim=-1)
            correct = (preds_flat == targets_flat).sum().item()
            epoch_train_acc += correct
            total_train_hits += len(targets_flat)

        scheduler.step()
        train_loss.append(epoch_train_loss / len(train_loader))
        train_acc.append(epoch_train_acc / total_train_hits if total_train_hits > 0 else 0.0)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        total_val_hits = 0
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                batch_x = view.batch_x
                batch_y = view.batch_y

                max_N = max(len(x) for x in batch_x)
                Xs_padded = pad_sequence(batch_x, batch_first=True, padding_value=0).to(device)
                ys_padded = []
                masks = []
                for y in batch_y:
                    pad_len = max_N - len(y)
                    y_pad = torch.cat([y.to(device), torch.full((pad_len,), -1, dtype=y.dtype, device=device)], dim=0)
                    ys_padded.append(y_pad)
                    mask = torch.cat([torch.zeros(len(y), dtype=bool, device=device), 
                                      torch.ones(pad_len, dtype=bool, device=device)]) if pad_len > 0 else torch.zeros(len(y), dtype=bool, device=device)
                    masks.append(mask)
                ys_tensor = torch.stack(ys_padded, dim=0)
                masks_tensor = torch.stack(masks, dim=0)

                logits = model.forward_padded(Xs_padded, masks_tensor)
                mask_flat = masks_tensor.view(-1)
                logits_flat = logits.view(-1, 51)[~mask_flat]
                targets_flat = ys_tensor.view(-1)[~mask_flat]
                loss = criterion(logits_flat, targets_flat)

                epoch_val_loss += loss.item()

                preds_flat = torch.argmax(logits_flat, dim=-1)
                correct = (preds_flat == targets_flat).sum().item()
                epoch_val_acc += correct
                total_val_hits += len(targets_flat)

        val_loss.append(epoch_val_loss / len(val_loader))
        val_acc.append(epoch_val_acc / total_val_hits if total_val_hits > 0 else 0.0)

        # Early stopping
        if val_acc[-1] > best_val_acc:
            best_val_acc = val_acc[-1]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Return all epochs, even if early stopped
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

