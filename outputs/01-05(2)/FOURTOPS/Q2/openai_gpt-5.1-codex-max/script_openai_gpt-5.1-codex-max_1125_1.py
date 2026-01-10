
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
                        
DATASET = {
    "X_train": "./challenges/FOURTOPS/data/train/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/train/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/train/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/train/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv(DATASET["X_train"], dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv(DATASET["Y_train"], dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv(DATASET["X_val"], dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv(DATASET['Y_val'], dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class FourTopsDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        X2 = pre.transform(X) if pre is not None else X
        if not torch.is_tensor(X2):
            X2 = torch.as_tensor(X2)
        self.X = X2.float()
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        self.y = y.long()
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MyPreprocessor:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        if torch.is_tensor(X):
            arr = X.numpy()
        else:
            arr = np.asarray(X)
        arr = arr.astype(np.float32)
        means = np.zeros(arr.shape[1], dtype=np.float32)
        stds = np.ones(arr.shape[1], dtype=np.float32)
        # global features indices 0,1
        for i in range(arr.shape[1]):
            if i < 2:
                col = arr[:, i]
                means[i] = col.mean()
                stds[i] = col.std()
            else:
                obj_idx = (i - 2) // 5
                id_col = 2 + obj_idx * 5
                mask = arr[:, id_col] != 0
                if mask.any():
                    col = arr[mask, i]
                    means[i] = col.mean()
                    stds[i] = col.std()
                else:
                    means[i] = 0.0
                    stds[i] = 1.0
            if stds[i] < 1e-6:
                stds[i] = 1.0
        self.mean_ = means
        self.std_ = stds
        return self

    def transform(self, X):
        if torch.is_tensor(X):
            arr = X.numpy()
        else:
            arr = np.asarray(X)
        arr = arr.astype(np.float32)
        arr = (arr - self.mean_) / self.std_
        return arr

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        d_model = 64
        self.d_model = d_model
        self.obj_mlp = nn.Sequential(
            nn.Linear(9, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.final_mlp = nn.Sequential(
            nn.Linear(d_model + 4, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x: [B,92]
        B = batch_x.size(0)
        global_feats = batch_x[:, 0:2]  # [B,2]
        met = global_feats[:, 0]  # [B]
        phi_m = global_feats[:, 1]  # [B]
        g_feats = torch.stack(
            [met, torch.log1p(torch.clamp(met, min=0.0)), torch.sin(phi_m), torch.cos(phi_m)],
            dim=1
        )  # [B,4]

        objs = batch_x[:, 2:]  # [B,90]
        objs = objs.view(B, 18, 5)  # [B,18,5]
        obj_id = objs[:, :, 0]  # [B,18]
        pad_mask = (obj_id == 0)  # True where padding
        valid_mask = ~pad_mask  # [B,18]
        # build object features
        E = objs[:, :, 1]
        pT = objs[:, :, 2]
        eta = objs[:, :, 3]
        phi = objs[:, :, 4]
        logE = torch.log1p(torch.clamp(E, min=0.0))
        logpT = torch.log1p(torch.clamp(pT, min=0.0))
        sinphi = torch.sin(phi)
        cosphi = torch.cos(phi)
        obj_feat = torch.stack(
            [obj_id, E, pT, eta, phi, logE, logpT, sinphi, cosphi],
            dim=-1
        )  # [B,18,9]
        obj_emb = self.obj_mlp(obj_feat)  # [B,18,d_model]
        obj_emb = self.transformer(obj_emb, src_key_padding_mask=pad_mask)  # [B,18,d_model]
        mask_float = valid_mask.unsqueeze(-1).float()  # [B,18,1]
        summed = (obj_emb * mask_float).sum(dim=1)  # [B,d_model]
        denom = mask_float.sum(dim=1)  # [B,1]
        denom = denom.clamp(min=1.0)
        pooled = summed / denom  # [B,d_model]
        combined = torch.cat([pooled, g_feats], dim=1)  # [B,d_model+4]
        out = self.final_mlp(combined).squeeze(-1)  # [B]
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 12
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            optimizer.zero_grad()
            outputs = model(xb)  # [B]
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds.cpu() == yb.cpu().long()).sum().item()
            total += xb.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total if total > 0 else 0.0
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).float()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_running_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (preds.cpu() == yb.cpu().long()).sum().item()
                val_total += xb.size(0)
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    return model, train_losses, val_losses, train_accs, val_accs

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:20]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre     = make_preprocessor().fit(X_train, Y_train)
    
    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec, require_torch_collate=False)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, (X_train, Y_train), pre, train=True)
    val_ds       = build_dataset(spec, (X_val,   Y_val),   pre, train=False)
    train_loader = build_dataloader(spec, train_ds, is_eval=False)
    val_loader   = build_dataloader(spec, val_ds,   is_eval=True)

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane_fourtops(spec, first_batch)
    view = make_view_by_lane_fourtops(mode, first_batch, device)

    # Build model
    model = make_model(view.batch_x).to(device)

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
                mode = detect_and_assert_lane_fourtops(spec, first_batch)
                view = make_view_by_lane_fourtops(mode, first_batch, device)
                out  = trained_model(view.batch_x)
                scores, kind = assert_binary_output(view, out)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e

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

