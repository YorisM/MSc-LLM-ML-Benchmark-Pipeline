
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

# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
# <LLM: Import modules>
import math
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

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
    #   - IMPORTANT Default data flow: events[idx] -> split_X_y(evt) -> X, y
    #   - When modifying data features or feature engineering: annotate tensor size as comments after each tensor operation to reduce dimension mismatches.

    # <LLM: Write code to preprocess the data> 

    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        self.scaler = StandardScaler()

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one:
            "collate": "ragged_xy",  # or "identity" or None

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, F_raw]
        # Stack and fit scaler
        X_all = np.vstack([x.numpy() for x in Xs])  # [total_hits, 4]
        self.scaler.fit(X_all)
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, F_raw]
        # Transform with scaler, then sort by z (index 2)
        X_np = X.numpy()
        X_scaled = self.scaler.transform(X_np)  # [N, 4]
        sort_indices = np.argsort(X_scaled[:, 2])  # sort by z
        X_sorted = X_scaled[sort_indices]  # [N, 4]
        return torch.from_numpy(X_sorted)  # return [N,4] tensor

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # IMPORTANT: Default harness input:
        #   batch_x is ragged list[Tensor], one per event, each shaped [N_hits, F].
        #   Infer F from example_batch_x (do NOT assume an int is passed).
        self.feat_dim = example_batch_x[0].shape[1]  # F=4
        self.d_model = 128
        self.num_classes = 101  # 0 for noise, 1-100 for tracks
        self.embed = nn.Linear(self.feat_dim, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward_logits(self, batch_x):
        # Return list of [N, num_classes]
        logits_list = []
        for x in batch_x:
            N = x.shape[0]
            emd = self.embed(x)  # [N, d_model]
            pos_emb = self._positional_encoding(N, self.d_model, x.device).unsqueeze(0)  # [1, N, d_model]
            emd = emd.unsqueeze(0) + pos_emb  # [1, N, d_model]
            out = self.transformer(emd)  # [1, N, d_model]
            emd_out = out.squeeze(0)  # [N, d_model]
            logits = self.classifier(emd_out)  # [N, num_classes]
            logits_list.append(logits)
        return logits_list

    def _positional_encoding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
        pos_emb = torch.zeros(seq_len, d_model, device=device)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        return pos_emb

    def forward(self, batch_x):
        # IMPORTANT Output contract:
        #   forward(batch_x) must return predicted integer labels (dtype long/int64) with one label per hit (>0); predicted noise may be -1.
        logits_list = self.forward_logits(batch_x)
        preds_list = []
        for logits in logits_list:
            preds = torch.argmax(logits, dim=-1)
            preds = torch.where(preds == 0, torch.tensor(-1, dtype=torch.long), preds)
            preds_list.append(preds)
        return preds_list

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20   # <LLM: adjust if you wish>   
def train_model(model, train_loader, val_loader, epochs):
    # If your method is non-parametric, train_model may be a no-op that returns the unmodified model and empty metric lists, otherwise:

    # REQUIREMENTS 
    #   Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #   Must return trained_model, train_loss, val_loss, train_acc, val_acc
    #   Implement early-stopping.
    #   Use CUDA - torch.cuda.is_available()
    #   Forward signature must match.

    # <LLM: Write code to define training loop>
    # <LLM: Implement early stopping if possible>
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    early_stopping_patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_hits = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            Xs, ys = view.batch_x, view.batch_y

            optimizer.zero_grad()
            logits_list = model.forward_logits(Xs)
            loss = 0
            for i, logits in enumerate(logits_list):
                y = ys[i].to(device)
                y_mapped = torch.clamp(y, 0, 100).long()  # clamp to 0-100
                loss += F.cross_entropy(logits, y_mapped, reduction='sum')
                preds = torch.argmax(logits, dim=-1)
                total_train_correct += (preds == y_mapped).sum().item()
                total_train_hits += y.numel()

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = total_train_correct / total_train_hits if total_train_hits > 0 else 0

        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_hits = 0
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                Xs, ys = view.batch_x, view.batch_y

                logits_list = model.forward_logits(Xs)
                for i, logits in enumerate(logits_list):
                    y = ys[i].to(device)
                    y_mapped = torch.clamp(y, 0, 100).long()
                    total_val_loss += F.cross_entropy(logits, y_mapped, reduction='sum').item()
                    preds = torch.argmax(logits, dim=-1)
                    total_val_correct += (preds == y_mapped).sum().item()
                    total_val_hits += y.numel()

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = total_val_correct / total_val_hits if total_val_hits > 0 else 0

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Acc={train_acc:.4f}; Val Loss={avg_val_loss:.4f}, Acc={val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

        scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
# <end code template>

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

