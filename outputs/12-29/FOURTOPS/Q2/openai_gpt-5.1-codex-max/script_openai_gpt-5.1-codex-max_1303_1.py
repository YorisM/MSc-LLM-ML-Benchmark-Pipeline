
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

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
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        x = self.X[idx]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x, self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

# ---------- IMPORTS ----------
import torch.nn.functional as F
import numpy as np
import torch

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.field_means = {}
        self.field_stds = {}

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
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        self.global_mean = X_np[:, :2].mean(axis=0)
        self.global_std = X_np[:, :2].std(axis=0) + 1e-6
        ids = X_np[:, 2::5]
        mask = ids > 0
        field_names = ["E", "pT", "eta", "phi"]
        for i, name in enumerate(field_names):
            arr = X_np[:, (3 + i)::5]
            if mask.any():
                vals = arr[mask]
                mean = vals.mean() if vals.size > 0 else 0.0
                std = vals.std() + 1e-6 if vals.size > 0 else 1.0
            else:
                mean = 0.0
                std = 1.0
            self.field_means[name] = mean
            self.field_stds[name] = std
        return self

    def transform(self, X):
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        X_np = X_np.astype(np.float32, copy=True)
        # Normalize global features
        X_np[:, :2] = (X_np[:, :2] - self.global_mean) / self.global_std
        ids = X_np[:, 2::5]
        mask = ids > 0
        field_names = ["E", "pT", "eta", "phi"]
        for i, name in enumerate(field_names):
            arr = X_np[:, (3 + i)::5]
            arr = (arr - self.field_means[name]) / self.field_stds[name]
            arr *= mask
            X_np[:, (3 + i)::5] = arr
        return X_np

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.num_obj = 18
        self.embed_dim = 8
        self.d_model = 64
        self.obj_emb = nn.Embedding(num_embeddings=100, embedding_dim=self.embed_dim, padding_idx=0)
        self.obj_proj = nn.Sequential(
            nn.Linear(self.embed_dim + 4, self.d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.global_proj = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, batch_x):
        x = batch_x
        if x.dim() == 1:
            x = x.unsqueeze(0)
        global_feat = x[:, :2]  # [B,2]
        obj = x[:, 2:].view(x.size(0), self.num_obj, 5)  # [B,18,5]
        obj_ids = obj[:, :, 0].long()  # [B,18]
        cont = obj[:, :, 1:]  # [B,18,4]
        emb = self.obj_emb(obj_ids)  # [B,18,8]
        obj_inp = torch.cat([emb, cont], dim=-1)  # [B,18,12]
        obj_feat = self.obj_proj(obj_inp)  # [B,18,64]
        mask = obj_ids > 0  # [B,18]
        obj_enc = self.encoder(obj_feat, src_key_padding_mask=~mask)  # [B,18,64]
        mask_f = mask.unsqueeze(-1).float()  # [B,18,1]
        summed = (obj_enc * mask_f).sum(dim=1)  # [B,64]
        denom = mask_f.sum(dim=1).clamp(min=1.0)  # [B,1]
        pooled = summed / denom  # [B,64]
        g_proj = self.global_proj(global_feat)  # [B,16]
        feats = torch.cat([pooled, g_proj], dim=-1)  # [B,80]
        logits = self.classifier(feats).squeeze(-1)  # [B]
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 12
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y.float()
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            preds = (torch.sigmoid(out) > 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total += yb.size(0)
        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        train_loss.append(avg_loss)
        train_acc.append(acc)
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y.float()
                out = model(xb)
                loss = criterion(out, yb)
                val_running_loss += loss.item() * yb.size(0)
                preds = (torch.sigmoid(out) > 0.5).long()
                val_correct += (preds == yb.long()).sum().item()
                val_total += yb.size(0)
        val_avg_loss = val_running_loss / val_total if val_total > 0 else 0.0
        val_acc_epoch = val_correct / val_total if val_total > 0 else 0.0
        val_loss.append(val_avg_loss)
        val_acc.append(val_acc_epoch)
    return model, train_loss, val_loss, train_acc, val_acc

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
                view = normalise_batch(first_batch, device=device)
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

