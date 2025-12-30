
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, torch, torch_geometric
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

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
# <LLM: Import modules>
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

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
        self.scaler = StandardScaler()

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 8,
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
        all_X = torch.cat(Xs, dim=0)  # [total_hits, 4]
        self.scaler.fit(all_X.numpy()[:, :3])
        self.layer_offset = torch.cat([X[:, 3] for X in Xs]).min().item()
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, F_raw]
        X_scaled = X.clone()
        X_scaled[:, :3] = torch.from_numpy(self.scaler.transform(X[:, :3].numpy())).float()
        X_scaled[:, 3] = X[:, 3] - self.layer_offset
        return X_scaled  # MUST return torch.FloatTensor [N_hits, 4] for the default EventDataset path.

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # example_batch_x: ragged list of [N_hits, 4] tensors (after preproc)
        self.num_classes = 64  # fixed number of classes, 0 for noise, 1 to 63 for tracks
        self.max_len = 1024
        F = example_batch_x[0].shape[1]  # 4
        layer_ids = example_batch_x[0][:, 3].long()
        self.num_layers = torch.unique(layer_ids).max().item() + 1
        self.layer_embed = nn.Embedding(self.num_layers, 16)
        embed_dim = 64 + 16
        self.hit_embed = nn.Linear(3, 64)  # for r, theta, z
        self.pos_embed = PositionalEncoding(embed_dim, dropout=0.1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=256, dropout=0.1),
            num_layers=6
        )
        self.classifier = nn.Linear(embed_dim, self.num_classes)

    def forward(self, batch_x):
        # batch_x: list of [N_hits, 4] tensors
        batch_output = []
        for X in batch_x:
            N = X.shape[0]
            if N == 0:
                batch_output.append(torch.tensor([], dtype=torch.int64))
                continue
            r_theta_z = X[:, :3]  # [N, 3]
            layer = X[:, 3].long()  # [N]
            pad_len = self.max_len - N
            r_theta_z_padded = torch.cat([r_theta_z, torch.zeros(pad_len, 3, device=X.device)], dim=0)  # [max_len, 3]
            layer_padded = torch.cat([layer, torch.zeros(pad_len, device=X.device, dtype=torch.long)], dim=0)  # [max_len]
            layer_emb = self.layer_embed(layer_padded)  # [max_len, 16]
            hit_emb = self.hit_embed(r_theta_z_padded)  # [max_len, 64]
            concatenated = torch.cat([hit_emb, layer_emb], dim=-1)  # [max_len, 80]
            seq = concatenated.unsqueeze(1)  # [max_len, 1, 80]
            seq = seq.transpose(0, 1)  # [1, max_len, 80]
            src_key_padding_mask = torch.zeros(1, self.max_len, dtype=torch.bool, device=X.device)
            src_key_padding_mask[0, N:] = True
            out = self.transformer(seq, src_key_padding_mask=src_key_padding_mask)  # [1, max_len, 80]
            out = out.squeeze(0)  # [max_len, 80]
            out_class = self.classifier(out)  # [max_len, 64]
            predicted = out_class.argmax(dim=-1)  # [max_len]
            predicted = predicted[:N].clone()  # [N]
            # map to >0 for tracks, -1 for noise (if any predicted 0; but 0 is noise, so for tracks >=1, set to >0 or -1? Wait, output must be >0 for tracks, -1 for noise.
            # Since 0 is noise, set predicted == 0 to -1, else keep (1 to 63 are tracks)
            predicted = torch.where(predicted == 0, -1, predicted)
            batch_output.append(predicted)
        return batch_output

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model, train_loader, val_loader, epochs):
    # If your method is non-parametric, train_model may be a no-op that returns the unmodified model and empty metric lists, otherwise:

    # REQUIREMENTS 
    #   Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #   Must return trained_model, train_loss, val_loss, train_acc, val_acc
    #   Implement early-stopping.
    #   Use CUDA - torch.cuda.is_available()
    #   Forward signature must match.

    # <LLM: Write code to define training loop>
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
    num_classes = model.num_classes

    best_model_weights = None
    best_acc = -1.0
    patience = 5
    trigger = 0
    epochs = epochs if epochs is not None else EPOCHS

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_hits = 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            batch_x = view.batch_x
            ys = view.batch_y
            model.zero_grad()
            pred_logits_per_event = model(batch_x)
            loss_sum = 0.0
            for i, pred_logits in enumerate(pred_logits_per_event):
                N_i = len(ys[i])
                y = ys[i]
                # Renumber y: unique track_ids >0, sorted, map to 1 to M, 0 stays 0
                unique_tids = torch.unique(y[y > 0])
                if len(unique_tids) == 0:
                    M = 0
                else:
                    unique_tids_sorted = torch.sort(unique_tids)[0]
                    M = len(unique_tids_sorted)
                map_tid = {tid.item(): j + 1 for j, tid in enumerate(unique_tids_sorted)}  # 1 to M
                renumbered = torch.tensor([map_tid.get(y[j].item(), 0) if y[j] != 0 else 0 for j in range(N_i)], dtype=torch.long, device=device)
                pred_logits_subset = pred_logits[:N_i]
                loss = F.cross_entropy(pred_logits_subset, renumbered, ignore_index=-1)
                loss_sum += loss * N_i
                predicted = pred_logits_subset.argmax(dim=-1)
                correct = (predicted == renumbered) & (renumbered != -1)
                total_correct += correct.sum().item()
                total_hits += N_i
            loss_sum.backward()
            optimizer.step()
            total_loss += loss_sum.item()
        scheduler.step()
        train_loss = total_loss / total_hits if total_hits > 0 else 0
        train_acc = total_correct / total_hits if total_hits > 0 else 0
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_hits = 0
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                batch_x = view.batch_x
                ys = view.batch_y
                pred_logits_per_event = model(batch_x)
                loss_sum = 0.0
                for i, pred_logits in enumerate(pred_logits_per_event):
                    N_i = len(ys[i])
                    y = ys[i]
                    # Same renumbering
                    unique_tids = torch.unique(y[y > 0])
                    if len(unique_tids) == 0:
                        M = 0
                    else:
                        unique_tids_sorted = torch.sort(unique_tids)[0]
                        M = len(unique_tids_sorted)
                    map_tid = {tid.item(): j + 1 for j, tid in enumerate(unique_tids_sorted)}
                    renumbered = torch.tensor([map_tid.get(y[j].item(), 0) if y[j] != 0 else 0 for j in range(N_i)], dtype=torch.long, device=device)
                    pred_logits_subset = pred_logits[:N_i]
                    loss = F.cross_entropy(pred_logits_subset, renumbered, ignore_index=-1)
                    loss_sum += loss * N_i
                    predicted = pred_logits_subset.argmax(dim=-1)
                    correct = (predicted == renumbered) & (renumbered != -1)
                    total_val_correct += correct.sum().item()
                    total_val_hits += N_i
                total_val_loss += loss_sum.item()
        val_loss = total_val_loss / total_val_hits if total_val_hits > 0 else 0
        val_acc = total_val_correct / total_val_hits if total_val_hits > 0 else 0
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_weights = model.state_dict().copy()
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                break

    # Load best model
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

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

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

