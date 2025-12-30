
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, pickle, torch, torch_geometric, gc, json, importlib, scipy
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.llm_io import normalise_batch, assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy, write_loaderspec
from utils.suffix_utils import base_from_argv0, write_json, plot_train_val, persist_artefacts

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
        return self.X[idx], self.y[idx]


# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
import math, copy
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self, max_obj_types: int = 12):
        self.max_obj_types = int(max_obj_types)
        self.n_types = None  # includes padding-id 0
        self.feature_dim = None

    def make_loader_cfg(self):
        use_cuda = torch.cuda.is_available()
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 768 if use_cuda else 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": bool(use_cuda),

            "collate": None,

            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # X: torch.Tensor [N, 92]
        # object IDs are at indices 2,7,12,...,87
        with torch.no_grad():
            obj_ids = X[:, 2::5]  # [N, 18]
            max_id = int(torch.nan_to_num(obj_ids, nan=0.0).max().item())
            n_types = max_id + 1
            n_types = max(2, min(n_types, self.max_obj_types))
            self.n_types = int(n_types)

            # Feature dim formula:
            # global_dim G=6
            # counts dim = n_types
            # per-object dim = (n_types + 7)
            # 18 objects
            # total = 6 + n_types + 18*(n_types+7) = 132 + 19*n_types
            self.feature_dim = 132 + 19 * self.n_types
        return self

    def _featurize(self, X):
        # X: torch.Tensor [N, 92] float32/float64 on CPU
        # returns: torch.Tensor [N, D] float32
        X = X.to(torch.float32)
        N = X.shape[0]

        # Globals
        met = torch.clamp(X[:, 0], min=0.0) / 1000.0  # [N]
        met_phi = X[:, 1]  # [N]
        logmet = torch.log1p(met)  # [N]
        sinmet = torch.sin(met_phi)  # [N]
        cosmet = torch.cos(met_phi)  # [N]

        # Objects
        objs = X[:, 2:].reshape(N, 18, 5)  # [N, 18, 5]
        oid_f = torch.nan_to_num(objs[:, :, 0], nan=0.0)  # [N, 18]
        oid = torch.clamp(oid_f, min=0.0)
        mask = oid > 0.5  # [N, 18] bool
        maskf = mask.to(torch.float32)  # [N, 18]

        E = torch.clamp(torch.nan_to_num(objs[:, :, 1], nan=0.0), min=0.0) / 1000.0  # [N, 18] GeV
        pT = torch.clamp(torch.nan_to_num(objs[:, :, 2], nan=0.0), min=0.0) / 1000.0  # [N, 18] GeV
        eta_raw = torch.clamp(torch.nan_to_num(objs[:, :, 3], nan=0.0), min=-5.0, max=5.0)  # [N, 18]
        eta = eta_raw / 5.0  # [N, 18] scaled
        phi = torch.nan_to_num(objs[:, :, 4], nan=0.0)  # [N, 18]

        # Continuous per-object features (masked to 0 for padding)
        logE = torch.log1p(E) * maskf  # [N, 18]
        logpT = torch.log1p(pT) * maskf  # [N, 18]
        eta_s = eta * maskf  # [N, 18]
        sinphi = torch.sin(phi) * maskf  # [N, 18]
        cosphi = torch.cos(phi) * maskf  # [N, 18]

        # Approx invariant mass from (E, pT, eta)
        # p = pT * cosh(eta)
        p = pT * torch.cosh(eta_raw)  # [N, 18]
        m2 = (E * E - p * p).clamp(min=0.0) * maskf  # [N, 18]
        logm = torch.log1p(torch.sqrt(m2 + 1e-12)) * maskf  # [N, 18]

        # One-hot object types (masked so padding contributes 0)
        oid_int = oid.to(torch.int64)
        oid_int = torch.clamp(oid_int, min=0, max=self.n_types - 1)  # [N, 18]
        one_hot = F.one_hot(oid_int, num_classes=self.n_types).to(torch.float32)  # [N, 18, n_types]
        one_hot = one_hot * maskf.unsqueeze(-1)  # [N, 18, n_types]

        # Counts per type (normalized)
        counts = one_hot.sum(dim=1) / 18.0  # [N, n_types]

        # Global summaries
        nobj = maskf.sum(dim=1) / 18.0  # [N]
        HT = (pT * maskf).sum(dim=1)  # [N]
        sumE = (E * maskf).sum(dim=1)  # [N]
        logHT = torch.log1p(HT)  # [N]
        logsumE = torch.log1p(sumE)  # [N]
        glob = torch.stack([logmet, sinmet, cosmet, nobj, logHT, logsumE], dim=1)  # [N, 6]

        cont = torch.stack([logE, logpT, eta_s, sinphi, cosphi, logm], dim=-1)  # [N, 18, 6]
        obj_feat = torch.cat([one_hot, cont, maskf.unsqueeze(-1)], dim=-1)  # [N, 18, n_types+7]
        obj_flat = obj_feat.reshape(N, -1)  # [N, 18*(n_types+7)]

        feats = torch.cat([glob, counts, obj_flat], dim=1)  # [N, 6 + n_types + 18*(n_types+7)]
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)  # [N, D]
        return feats

    def transform(self, X):
        # Single pass (keep deterministic)
        return self._featurize(X)

def make_preprocessor():
    return MyPreprocessor(max_obj_types=12)

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        in_dim = int(sample_object.shape[-1])

        # Deduce n_types from D = 132 + 19*n_types (G=6, S=18, per-object extra=7)
        n_types = None
        for k in range(2, 65):
            if 132 + 19 * k == in_dim:
                n_types = k
                break
        if n_types is None:
            # Fallback rounding (should not happen if preprocessor is consistent)
            n_types = int(round((in_dim - 132) / 19))
            n_types = max(2, n_types)

        self.n_types = int(n_types)
        self.global_dim = 6
        self.d_obj = self.n_types + 7  # one_hot + 6 cont + mask
        self.seq_len = 18

        d_model = 96
        nhead = 4
        ff = 256
        n_layers = 2
        drop = 0.12

        self.obj_ln0 = nn.LayerNorm(self.d_obj)
        self.obj_proj = nn.Linear(self.d_obj, d_model)
        self.obj_act = nn.SiLU()
        self.obj_drop = nn.Dropout(drop)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        head_in = self.global_dim + self.n_types + 2 * d_model  # globals + counts + (mean,max)
        self.head = nn.Sequential(
            nn.Linear(head_in, 320),
            nn.BatchNorm1d(320),
            nn.SiLU(),
            nn.Dropout(0.20),

            nn.Linear(320, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Dropout(0.18),

            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.SiLU(),
            nn.Dropout(0.10),

            nn.Linear(96, 1),
        )

    def forward(self, batch_x):
        # batch_x: [B, D]
        x = batch_x
        B = x.shape[0]

        # Parse layout: [globals(6), counts(n_types), obj_flat(18*(n_types+7))]
        glob = x[:, :6]  # [B, 6]
        counts = x[:, 6:6 + self.n_types]  # [B, n_types]
        obj_flat = x[:, 6 + self.n_types:]  # [B, 18*(d_obj)]
        obj = obj_flat.view(B, self.seq_len, self.d_obj)  # [B, 18, d_obj]

        mask = obj[:, :, -1] > 0.5  # [B, 18] bool
        # Ensure at least one valid token to avoid all-masked edge cases
        n_valid = mask.sum(dim=1)  # [B]
        if (n_valid == 0).any():
            bad = (n_valid == 0).nonzero(as_tuple=False).squeeze(1)
            mask[bad, 0] = True

        tok = self.obj_ln0(obj)  # [B, 18, d_obj]
        tok = self.obj_proj(tok)  # [B, 18, d_model]
        tok = self.obj_act(tok)  # [B, 18, d_model]
        tok = self.obj_drop(tok)  # [B, 18, d_model]

        pad_mask = ~mask  # True for padding
        out = self.encoder(tok, src_key_padding_mask=pad_mask)  # [B, 18, d_model]

        maskf = mask.to(out.dtype)  # [B, 18]
        denom = maskf.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
        mean_pool = (out * maskf.unsqueeze(-1)).sum(dim=1) / denom  # [B, d_model]

        out_masked = out.masked_fill(pad_mask.unsqueeze(-1), -1e9)  # [B, 18, d_model]
        max_pool = out_masked.max(dim=1).values  # [B, d_model]

        h = torch.cat([glob, counts, mean_pool, max_pool], dim=1)  # [B, head_in]
        logits = self.head(h).squeeze(1)  # [B]
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    use_cuda = torch.cuda.is_available()
    use_amp = use_cuda

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.2e-3,
        weight_decay=7e-3,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    total_steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2.2e-3,
        total_steps=total_steps,
        pct_start=0.12,
        anneal_strategy="cos",
        div_factor=12.0,
        final_div_factor=60.0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = nn.BCEWithLogitsLoss()

    def _evaluate(loader):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_n = 0

        all_probs = []
        all_y = []

        with torch.no_grad():
            for batch in loader:
                view = normalise_batch(batch, device=device)
                x = view.batch_x  # [B, D]
                y = view.batch_y  # [B]
                y_f = y.to(torch.float32)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)  # [B]
                    loss = criterion(logits, y_f)

                bs = int(y.shape[0])
                total_loss += float(loss.item()) * bs
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).to(y.dtype)
                total_correct += int((preds == y).sum().item())
                total_n += bs

                all_probs.append(probs.detach().float().cpu())
                all_y.append(y.detach().cpu())

        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_y = torch.cat(all_y, dim=0).numpy()
        auc = float(roc_auc_score(all_y, all_probs))
        avg_loss = total_loss / max(1, total_n)
        acc = total_correct / max(1, total_n)
        return avg_loss, acc, auc

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    best_auc = -1.0
    best_state = None
    patience = 6
    bad_epochs = 0

    for epoch in range(int(epochs)):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_n = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            x = view.batch_x  # [B, D]
            y = view.batch_y  # [B]
            y_f = y.to(torch.float32)

            # small label smoothing for stability
            eps = 0.01
            y_smooth = y_f * (1.0 - eps) + 0.5 * eps  # [B]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)  # [B]
                loss = criterion(logits, y_smooth)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = int(y.shape[0])
            running_loss += float(loss.item()) * bs
            probs = torch.sigmoid(logits.detach())
            preds = (probs >= 0.5).to(y.dtype)
            running_correct += int((preds == y).sum().item())
            running_n += bs

        tr_loss = running_loss / max(1, running_n)
        tr_acc = running_correct / max(1, running_n)

        va_loss, va_acc, va_auc = _evaluate(val_loader)

        train_loss_hist.append(float(tr_loss))
        val_loss_hist.append(float(va_loss))
        train_acc_hist.append(float(tr_acc))
        val_acc_hist.append(float(va_acc))

        if va_auc > best_auc + 1e-4:
            best_auc = va_auc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    trained_model = model
    return trained_model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

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
            "epochs": n_epochs     if n_epochs else None,
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


