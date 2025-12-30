
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.max_N = None
        self.max_num_tracks = None
        self.num_classes = None  # max_num_tracks + 1 (for noise)
        self.padder_value = 0.0  # for features
        self.mask_value = -1  # for labels

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, 4]
        self.max_N = max(len(X) for X in Xs)
        # Assume ys are not needed for fit, but compute from globals if possible
        # In this setup, ys are not passed, so assume max_num_tracks = 50 (from problem: 10-50)
        self.max_num_tracks = 50  # conservative max
        self.num_classes = self.max_num_tracks + 1  # +1 for noise
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, 4], y not here
        # To sort by theta for position info
        # Assume X is numpy or tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        sorted_indices = np.argsort(X[:, 1])  # sort by theta
        X_sorted = X[sorted_indices]
        X_padded = np.pad(X_sorted, ((0, self.max_N - len(X_sorted)), (0, 0)), mode='constant', constant_values=self.padder_value)
        return torch.tensor(X_padded, dtype=torch.float32)

    def process_y_for_loader(self, y):
        # y: tensor [N]
        if y is None:
            return None
        unique_tracks = torch.unique(y[y > 0])
        label_map = {int(old.item()): int(new) + 1 for new, old in enumerate(unique_tracks)}
        encoded = torch.zeros_like(y, dtype=torch.long)
        for old, new in label_map.items():
            encoded[y == old] = new
        # noise is 0
        encoded_padded = torch.full((self.max_N,), self.mask_value, dtype=torch.long)
        encoded_padded[:len(encoded)] = encoded
        return encoded_padded

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 32,  # smaller for variable length
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,  # custom collate
        }

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x, max_N=500, num_classes=51, d_model=16, nhead=4, num_layers=4):
        super().__init__()
        self.max_N = max_N
        self.num_classes = num_classes
        num_layer_ids = 10  # assume
        self.layer_emb = nn.Embedding(num_layer_ids, 4)
        self.cont_emb = nn.Linear(3, 12)  # r, theta, z -> 12
        self.pos_enc = PositionalEncoding(d_model=16)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=64, dropout=0.1),
            num_layers=4
        )
        self.clf = nn.Linear(16, num_classes)

    def forward(self, batch_x):
        # batch_x: list of tensors [N_i, 4]
        lengths = [len(x) for x in batch_x]
        padded_x = pad_sequence(batch_x, batch_first=True, padding_value=0)  # [batch, max_N, 4]
        batch_size, seq_len, _ = padded_x.shape
        # embed
        cont = padded_x[:, :, :3]  # [batch, seq, 3]
        layer = padded_x[:, :, 3].long()  # assume layer_id is int-like
        # clamp to num_layer_ids
        layer = torch.clamp(layer, 0, 9)
        layer_emb = self.layer_emb(layer)  # [batch, seq, 4]
        cont_emb = self.cont_emb(cont)  # [batch, seq, 12]
        emb = torch.cat([cont_emb, layer_emb], dim=-1)  # [batch, 16]
        emb = emb + self.pos_enc(emb)  # pos
        src_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=emb.device)
        for i, l in enumerate(lengths):
            src_key_padding_mask[i, l:] = True
        # transformer
        out = self.encoder(emb.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask)  # [seq, batch, 16]
        out = out.permute(1, 0, 2)  # [batch, seq, 16]
        logits = self.clf(out)  # [batch, seq, num_classes]
        # for inference, get pred labels
        preds = []
        for i in range(batch_size):
            valid_logits = logits[i, :lengths[i]]  # [l_i, num_classes]
            pred_cls = torch.argmax(valid_logits, dim=1)  # [l_i]
            # map: noise (0) -> -1, tracks (1..) -> 1..
            pred_label = torch.where(pred_cls == 0, torch.tensor(-1).to(pred_cls.device), pred_cls)
            preds.append(pred_label.to(torch.int64))
        return preds  # list of [N_i] int64

def make_model(example_batch_x):
    # example_batch_x is list of tensors
    lengths = [len(x) for x in example_batch_x]
    max_N = max(lengths) if lengths else 512  # default
    return HitClassifier(example_batch_x, max_N=max_N, num_classes=51)

# ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, encoded_yb = view.batch_x, view.batch_y  # assume encoded_yb is padded
            optimizer.zero_grad()
            lengths = [len(x) for x in xb]
            padded_x = pad_sequence(xb, batch_first=True, padding_value=0)  # [batch, max_N, 4]
            logits = model.forward_padded(padded_x, lengths)  # need to modify model to have forward_padded
            loss = loss_fn(logits.view(-1, model.num_classes), encoded_yb.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred_cls = torch.argmax(logits, dim=-1)
            correct = ((pred_cls == encoded_yb) & (encoded_yb >= 0)).sum().item()
            total_acc += correct
            total_samples += (encoded_yb >= 0).sum().item()
        scheduler.step()
        tr_l = total_loss / len(train_loader)
        tr_a = total_acc / total_samples
        train_loss.append(tr_l)
        train_acc.append(tr_a)
        # val
        model.eval()
        with torch.no_grad():
            va_l, va_a = 0.0, 0.0
            va_samples = 0
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, encoded_yb = view.batch_x, view.batch_y
                lengths = [len(x) for x in xb]
                padded_x = pad_sequence(xb, batch_first=True, padding_value=0)
                logits = model.forward_padded(padded_x, lengths)
                loss = loss_fn(logits.view(-1, model.num_classes), encoded_yb.view(-1))
                va_l += loss.item()
                pred_cls = torch.argmax(logits, dim=-1)
                correct = ((pred_cls == encoded_yb) & (encoded_yb >= 0)).sum().item()
                va_a += correct
                va_samples += (encoded_yb >= 0).sum().item()
            va_l /= len(val_loader)
            va_a /= va_samples
        val_loss.append(va_l)
        val_acc.append(va_a)
        model.train()
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

