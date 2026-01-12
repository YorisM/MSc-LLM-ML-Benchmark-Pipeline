
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, to_python
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops, dryrun_finite_check_fourtops

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
from torch import nn
import torch.nn.functional as F

class MyPreprocessor:
    def __init__(self):
        pass

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
        return self

    def transform(self, X):
        if torch.is_tensor(X):
            X2 = X.clone().float()
            X2[:, 0] = X2[:, 0] / 1000.0
            for i in range(18):
                base = 2 + i * 5
                X2[:, base + 1:base + 3] = X2[:, base + 1:base + 3] / 1000.0
        else:
            X2 = X.astype(np.float32, copy=True)
            X2[:, 0] = X2[:, 0] / 1000.0
            for i in range(18):
                base = 2 + i * 5
                X2[:, base + 1:base + 3] = X2[:, base + 1:base + 3] / 1000.0
        return X2

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.d_model = 128
        self.embed_dim = 16
        self.obj_embed = nn.Embedding(20, self.embed_dim, padding_idx=0)
        self.cont_proj = nn.Linear(5, self.d_model - self.embed_dim)
        self.met_proj = nn.Linear(3, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, norm=nn.LayerNorm(self.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, batch_x):
        # batch_x: [B,92]
        x = batch_x
        B = x.shape[0]
        met_mag = x[:, 0]  # [B]
        met_phi = x[:, 1]  # [B]
        met_feat = torch.stack([torch.log1p(torch.clamp(met_mag, min=0)), torch.sin(met_phi), torch.cos(met_phi)], dim=-1)  # [B,3]
        met_emb = self.met_proj(met_feat)  # [B,d_model]
        met_emb = met_emb.unsqueeze(1)  # [B,1,d_model]
        objs = x[:, 2:].reshape(B, 18, 5)  # [B,18,5]
        obj_id = objs[:, :, 0].long()  # [B,18]
        E = torch.clamp(objs[:, :, 1], min=0)  # [B,18]
        pT = torch.clamp(objs[:, :, 2], min=0)  # [B,18]
        eta = objs[:, :, 3]  # [B,18]
        phi = objs[:, :, 4]  # [B,18]
        cont_features = torch.stack([
            torch.log1p(E),
            torch.log1p(pT),
            eta,
            torch.sin(phi),
            torch.cos(phi)
        ], dim=-1)  # [B,18,5]
        cont_emb = self.cont_proj(cont_features)  # [B,18,d_model-embed_dim]
        id_emb = self.obj_embed(torch.clamp(obj_id, min=0))  # [B,18,embed_dim]
        obj_emb = torch.cat([id_emb, cont_emb], dim=-1)  # [B,18,d_model]
        cls_tok = self.cls_token.expand(B, -1, -1)  # [B,1,d_model]
        seq = torch.cat([cls_tok, met_emb, obj_emb], dim=1)  # [B,20,d_model]
        mask_obj = obj_id <= 0  # [B,18]
        pad_mask = torch.cat([torch.zeros(B, 2, device=seq.device, dtype=torch.bool), mask_obj], dim=1)  # [B,20]
        enc_out = self.encoder(seq, src_key_padding_mask=pad_mask)  # [B,20,d_model]
        cls_out = enc_out[:, 0, :]  # [B,d_model]
        logits = self.head(cls_out).squeeze(-1)  # [B]
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).long()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb.float())
                v_loss += loss.item() * yb.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).long()
                v_correct += (preds == yb).sum().item()
                v_total += yb.size(0)
        v_epoch_loss = v_loss / v_total
        v_epoch_acc = v_correct / v_total
        val_loss.append(v_epoch_loss)
        val_acc.append(v_epoch_acc)
        scheduler.step()
    trained_model = model
    return trained_model, train_loss, val_loss, train_acc, val_acc

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    X_fit, Y_fit = X_train, Y_train
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:200]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre = make_preprocessor().fit(X_fit, Y_fit)
    
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
    n_epochs = 10 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        try:
            dryrun_finite_check_fourtops(trained_model, spec, val_loader, device, batches=10)
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
        summary = to_python(summary)
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

