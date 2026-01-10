
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
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    """
    Produces a dense tensor with engineered global + per-object features.

    Output layout (per event):
      global: 7 floats
      objects: 18 * 11 floats, where per-object = [obj_id, 10 continuous engineered feats]
    Total F = 7 + 18*11 = 205
    """

    def __init__(self):
        self.n_obj = 18
        self.obj_stride_in = 5
        self.gdim = 7
        self.c_obj = 10
        self.per_obj_out = 1 + self.c_obj  # id + cont

        self.eps = 1e-6

        # Standardization params (numpy float32 for pickling)
        self.global_mean_ = None  # [7]
        self.global_std_ = None   # [7]
        self.obj_mean_ = None     # [10]
        self.obj_std_ = None      # [10]

    def make_loader_cfg(self) -> dict:
        bs = 1024 if torch.cuda.is_available() else 512
        pin = bool(torch.cuda.is_available())
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": bs,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": pin,

            "collate": None,
            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False, "batch_size": bs},
        }

    @staticmethod
    def _wrap_dphi(dphi: torch.Tensor) -> torch.Tensor:
        # dphi wrapped to [-pi, pi]
        return torch.atan2(torch.sin(dphi), torch.cos(dphi))

    def _engineer(self, X: torch.Tensor):
        """
        X: FloatTensor[N, 92]
        Returns:
          global_feats: FloatTensor[N, 7]
          obj_ids:      FloatTensor[N, 18]
          obj_cont:     FloatTensor[N, 18, 10]
          mask:         BoolTensor[N, 18]
        """
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)
        X = X.float()

        N = X.shape[0]
        # Global inputs
        met = X[:, 0] / 1000.0  # [N] in GeV
        met_phi = X[:, 1]       # [N]

        met_log = torch.log1p(torch.clamp(met, min=0.0))                 # [N]
        met_sin = torch.sin(met_phi)                                     # [N]
        met_cos = torch.cos(met_phi)                                     # [N]

        # Objects
        objs = X[:, 2:].reshape(N, self.n_obj, self.obj_stride_in)       # [N,18,5]
        obj_id = objs[:, :, 0]                                           # [N,18]
        E = objs[:, :, 1] / 1000.0                                       # [N,18] GeV
        pt = objs[:, :, 2] / 1000.0                                      # [N,18] GeV
        eta = objs[:, :, 3]                                              # [N,18]
        phi = objs[:, :, 4]                                              # [N,18]

        mask = (obj_id > 0.0) & (pt > 0.0)                               # [N,18] bool

        logE = torch.log1p(torch.clamp(E, min=0.0))                      # [N,18]
        logpt = torch.log1p(torch.clamp(pt, min=0.0))                    # [N,18]

        abs_eta = torch.abs(eta)                                         # [N,18]
        sinphi = torch.sin(phi)                                          # [N,18]
        cosphi = torch.cos(phi)                                          # [N,18]

        dphi_met = self._wrap_dphi(phi - met_phi[:, None])               # [N,18]
        sin_dphi = torch.sin(dphi_met)                                   # [N,18]
        cos_dphi = torch.cos(dphi_met)                                   # [N,18]

        # mass from E and p=pt*cosh(eta)
        p = pt * torch.cosh(eta)                                         # [N,18]
        mass2 = E * E - p * p                                            # [N,18]
        mass = torch.sqrt(torch.clamp(mass2, min=0.0))                   # [N,18]

        pt_over_E = pt / (E + self.eps)                                  # [N,18]

        # Event-level aggregates
        pt_masked = pt.masked_fill(~mask, 0.0)                           # [N,18]
        mass_masked = mass.masked_fill(~mask, 0.0)                       # [N,18]
        ht = pt_masked.sum(dim=1)                                        # [N]
        ht_log = torch.log1p(torch.clamp(ht, min=0.0))                   # [N]
        nobj = mask.sum(dim=1).float()                                   # [N]
        nobj_scaled = nobj / float(self.n_obj)                           # [N]
        max_pt = pt_masked.max(dim=1).values                             # [N]
        max_pt_log = torch.log1p(torch.clamp(max_pt, min=0.0))           # [N]
        sum_mass = mass_masked.sum(dim=1)                                # [N]
        sum_mass_log = torch.log1p(torch.clamp(sum_mass, min=0.0))       # [N]

        global_feats = torch.stack(
            [met_log, met_sin, met_cos, ht_log, nobj_scaled, max_pt_log, sum_mass_log],
            dim=1
        )  # [N,7]

        # Per-object continuous features [N,18,10]
        obj_cont = torch.stack(
            [logE, logpt, eta, abs_eta, sinphi, cosphi, mass, sin_dphi, cos_dphi, pt_over_E],
            dim=2
        )  # [N,18,10]

        return global_feats, obj_id, obj_cont, mask

    def fit(self, X, y=None):
        with torch.no_grad():
            global_feats, obj_id, obj_cont, mask = self._engineer(X)
            # Global mean/std over events
            g = global_feats.double()                                    # [N,7]
            g_mean = g.mean(dim=0)                                       # [7]
            g_var = (g - g_mean).pow(2).mean(dim=0)                      # [7]
            g_std = torch.sqrt(g_var + 1e-6)                             # [7]

            # Object mean/std over all real objects
            m = mask.unsqueeze(-1)                                       # [N,18,1] bool
            oc = obj_cont.double()                                       # [N,18,10]
            w = m.double()
            denom = w.sum(dim=(0, 1)).clamp(min=1.0)                     # [1]
            oc_mean = (oc * w).sum(dim=(0, 1)) / denom                   # [10]
            oc_var = ((oc - oc_mean) * w).pow(2).sum(dim=(0, 1)) / denom # [10]
            oc_std = torch.sqrt(oc_var + 1e-6)                           # [10]

            self.global_mean_ = g_mean.float().cpu().numpy()
            self.global_std_ = g_std.float().cpu().numpy()
            self.obj_mean_ = oc_mean.float().cpu().numpy()
            self.obj_std_ = oc_std.float().cpu().numpy()

        return self

    def transform(self, X):
        with torch.no_grad():
            global_feats, obj_id, obj_cont, mask = self._engineer(X)
            # Standardize
            gm = torch.from_numpy(self.global_mean_).to(global_feats.device)  # [7]
            gs = torch.from_numpy(self.global_std_).to(global_feats.device)   # [7]
            om = torch.from_numpy(self.obj_mean_).to(obj_cont.device)         # [10]
            os = torch.from_numpy(self.obj_std_).to(obj_cont.device)          # [10]

            global_z = (global_feats - gm) / gs                               # [N,7]
            obj_z = (obj_cont - om) / os                                      # [N,18,10]
            obj_z = obj_z * mask.unsqueeze(-1).float()                        # [N,18,10] (keep padded at 0)

            # Flatten per-object with id first
            obj_pack = torch.cat([obj_id.unsqueeze(-1), obj_z], dim=2)        # [N,18,11]
            obj_flat = obj_pack.reshape(obj_pack.shape[0], -1)               # [N,198]

            out = torch.cat([global_z, obj_flat], dim=1)                     # [N,205]
            return out.float()


def make_preprocessor():
    return MyPreprocessor()


# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        self.n_obj = 18
        self.gdim = 7
        self.per_obj = 11
        self.c_obj = 10

        in_dim = int(sample_object.shape[1])
        expected = self.gdim + self.n_obj * self.per_obj
        if in_dim != expected:
            raise ValueError(f"Unexpected input dim {in_dim}, expected {expected}")

        self.emb_size = 128
        self.type_emb_dim = 16
        self.d_model = 64

        self.type_emb = nn.Embedding(self.emb_size, self.type_emb_dim)

        tok_in = self.type_emb_dim + self.c_obj  # 16 + 10 = 26
        self.tok_proj = nn.Sequential(
            nn.Linear(tok_in, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(0.10),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )

        # Learned CLS token
        self.cls = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.global_mlp = nn.Sequential(
            nn.Linear(self.gdim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(0.10),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.LayerNorm(32),
        )

        # CLS (64) + mean pool (64) + max pool (64) + global (32) = 224
        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 64 + 32, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Dropout(0.15),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, batch_x):
        # batch_x: FloatTensor[B,205]
        x = batch_x
        B = x.shape[0]

        global_x = x[:, : self.gdim]  # [B,7]
        obj_flat = x[:, self.gdim :]  # [B,198]
        obj = obj_flat.reshape(B, self.n_obj, self.per_obj)  # [B,18,11]

        obj_id = obj[:, :, 0]  # [B,18] float
        obj_cont = obj[:, :, 1:]  # [B,18,10]

        # mask: padded objects have id==0
        mask = obj_id > 0.0  # [B,18] bool

        # embedding
        ids = torch.round(obj_id).long().clamp(0, self.emb_size - 1)  # [B,18]
        type_e = self.type_emb(ids)  # [B,18,16]

        tok = torch.cat([type_e, obj_cont], dim=2)  # [B,18,26]
        tok = self.tok_proj(tok)  # [B,18,64]
        tok = tok * mask.unsqueeze(-1).float()  # [B,18,64]

        # Transformer with CLS
        cls = self.cls.expand(B, 1, self.d_model)  # [B,1,64]
        seq = torch.cat([cls, tok], dim=1)  # [B,19,64]

        # src_key_padding_mask: True means "ignore"
        pad_mask = torch.cat([torch.zeros(B, 1, device=mask.device, dtype=torch.bool), ~mask], dim=1)  # [B,19]
        h = self.encoder(seq, src_key_padding_mask=pad_mask)  # [B,19,64]

        h_cls = h[:, 0, :]  # [B,64]
        h_obj = h[:, 1:, :]  # [B,18,64]

        # Masked mean/max pooling
        m = mask.unsqueeze(-1)  # [B,18,1]
        denom = m.float().sum(dim=1).clamp(min=1.0)  # [B,1]
        h_mean = (h_obj * m.float()).sum(dim=1) / denom  # [B,64]

        h_obj_for_max = h_obj.masked_fill(~m, -1e9)  # [B,18,64]
        h_max = h_obj_for_max.max(dim=1).values  # [B,64]
        # If an event had no objects (unlikely), max would be -1e9; clip to 0
        h_max = torch.where(torch.isfinite(h_max), h_max, torch.zeros_like(h_max))
        h_max = torch.clamp(h_max, min=-50.0, max=50.0)

        g = self.global_mlp(global_x)  # [B,32]

        feat = torch.cat([h_cls, h_mean, h_max, g], dim=1)  # [B,224]
        logit = self.head(feat).squeeze(1)  # [B]
        return logit


def make_model(example_object):
    return BinaryClassifier(example_object)


# ---------- MODEL TRAINING ----------
EPOCHS = 25


@torch.no_grad()
def _eval_epoch(model, loader, device):
    model.eval()
    losses = []
    logits_all = []
    y_all = []
    crit = nn.BCEWithLogitsLoss()

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float()
        out = model(xb)  # [B]
        loss = crit(out, yb)
        losses.append(loss.detach().float().item())
        logits_all.append(out.detach().float().cpu())
        y_all.append(yb.detach().float().cpu())

    logits = torch.cat(logits_all, dim=0).numpy()
    y = torch.cat(y_all, dim=0).numpy().astype(np.int64)

    try:
        auc = float(roc_auc_score(y, logits))
    except Exception:
        auc = float("nan")

    preds = (logits >= 0.0).astype(np.int64)
    acc = float((preds == y).mean())
    return float(np.mean(losses)), acc, auc


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    crit = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-2)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1.5e-3,
        epochs=max(1, epochs),
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,
        div_factor=15.0,
        final_div_factor=50.0,
        anneal_strategy="cos",
    )

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    best_auc = -1.0
    best_state = None
    patience = 6
    bad = 0

    for ep in range(int(epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(xb)                       # [B]
                loss = crit(out, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.detach().float().item() * xb.size(0)
            pred = (out.detach() >= 0.0).long()
            correct += (pred == yb.long()).sum().item()
            total += xb.size(0)

        tr_loss = running_loss / max(1, total)
        tr_acc = correct / max(1, total)

        va_loss, va_acc, va_auc = _eval_epoch(model, val_loader, device)

        train_loss_hist.append(float(tr_loss))
        val_loss_hist.append(float(va_loss))
        train_acc_hist.append(float(tr_acc))
        val_acc_hist.append(float(va_acc))

        if va_auc > best_auc + 1e-5:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    trained_model = model
    return trained_model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

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

