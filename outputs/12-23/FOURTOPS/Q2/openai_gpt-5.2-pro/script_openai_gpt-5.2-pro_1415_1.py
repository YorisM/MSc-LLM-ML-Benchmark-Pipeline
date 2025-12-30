
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
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
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

import math
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch import nn


def _wrap_phi(phi: torch.Tensor) -> torch.Tensor:
    # phi: [...], radians
    two_pi = 2.0 * math.pi
    return torch.remainder(phi + math.pi, two_pi) - math.pi


class MyPreprocessor:
    def __init__(self):
        # stateless (keep picklable)
        self._fitted = False

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        self._fitted = True
        return self

    def transform(self, X):
        # Input X: [N, 92] float32
        # Output:  [N, 92] float32 (same shape), but with physics-friendly transforms:
        #   - energies/pT: MeV -> GeV, log1p
        #   - met: MeV -> GeV, log1p
        #   - eta clipped
        #   - phi wrapped to [-pi, pi]
        #   - keep zero-padding exactly zero for kinematics when obj_id == 0
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)

        out = X.clone()  # [N, 92]
        N = out.shape[0]

        # Global
        out[:, 0] = torch.log1p(out[:, 0] / 1000.0)  # [N]
        out[:, 1] = _wrap_phi(out[:, 1])             # [N]

        # Objects
        objs = out[:, 2:].view(N, 18, 5)            # [N, 18, 5]
        obj_id = objs[:, :, 0]                      # [N, 18]
        mask = obj_id > 0                           # [N, 18] bool

        E = torch.log1p(objs[:, :, 1] / 1000.0)     # [N, 18]
        pT = torch.log1p(objs[:, :, 2] / 1000.0)    # [N, 18]
        eta = objs[:, :, 3].clamp(-5.0, 5.0)        # [N, 18]
        phi = _wrap_phi(objs[:, :, 4])              # [N, 18]

        # Preserve exact padding zeros
        m = mask.to(out.dtype)                      # [N, 18]
        E = E * m
        pT = pT * m
        eta = eta * m
        phi = phi * m

        objs[:, :, 1] = E
        objs[:, :, 2] = pT
        objs[:, :, 3] = eta
        objs[:, :, 4] = phi

        return out


def make_preprocessor():
    return MyPreprocessor()


class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Input batch_x: [B, 92]
        self.n_obj = 18
        self.obj_feat = 5

        self.topk = 6
        self.n_types = 32

        self.d_model = 128
        self.nhead = 8
        self.nlayers = 4
        self.ffn_dim = 4 * self.d_model
        self.dropout = 0.10

        # Object continuous features (per token):
        #   logE, logpT, eta, sinphi, cosphi, logm
        self.obj_cont_dim = 6
        self.obj_cont_proj = nn.Sequential(
            nn.Linear(self.obj_cont_dim, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.type_emb = nn.Embedding(self.n_types, self.d_model)

        # CLS extra engineered features:
        # met_log(1), met_sin(1), met_cos(1), nobj_norm(1), ht_log(1),
        # top3_pt_log(3), counts_id_1..10(10), pairwise(topk=6): 15 pairs * (logm, dR)=30
        self.cls_feat_dim = 1 + 2 + 1 + 1 + 3 + 10 + (self.topk * (self.topk - 1) // 2) * 2  # 48
        self.cls_proj = nn.Sequential(
            nn.Linear(self.cls_feat_dim, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.nlayers)
        self.post_ln = nn.LayerNorm(self.d_model)

        self.head = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Precompute triangle indices for pairwise features (topk fixed)
        tri = torch.triu_indices(self.topk, self.topk, offset=1)
        self.register_buffer("_tri_i", tri[0], persistent=False)
        self.register_buffer("_tri_j", tri[1], persistent=False)

    def forward(self, batch_x):
        # batch_x: [B, 92]
        x = batch_x
        B = x.shape[0]

        met_log = x[:, 0]                              # [B]
        met_phi = x[:, 1]                              # [B]

        objs = x[:, 2:].view(B, self.n_obj, self.obj_feat)  # [B, 18, 5]
        obj_id_f = objs[:, :, 0]                        # [B, 18]
        obj_id = obj_id_f.to(torch.long)                # [B, 18]
        mask = obj_id > 0                               # [B, 18] bool

        logE = objs[:, :, 1]                            # [B, 18]
        logpT = objs[:, :, 2]                           # [B, 18]
        eta = objs[:, :, 3]                             # [B, 18]
        phi = objs[:, :, 4]                             # [B, 18]

        # Recover E, pT in GeV for physics features
        E = torch.expm1(logE).clamp_min(0.0)            # [B, 18]
        pT = torch.expm1(logpT).clamp_min(0.0)          # [B, 18]

        sinphi = torch.sin(phi)                         # [B, 18]
        cosphi = torch.cos(phi)                         # [B, 18]

        # mass per object: m^2 = E^2 - p^2, with p = pT*cosh(eta)
        p = pT * torch.cosh(eta)                        # [B, 18]
        m2 = (E * E - p * p).clamp_min(0.0)             # [B, 18]
        m = torch.sqrt(m2)                               # [B, 18]
        logm = torch.log1p(m)                           # [B, 18]

        # Object token continuous features
        obj_cont = torch.stack([logE, logpT, eta, sinphi, cosphi, logm], dim=-1)  # [B, 18, 6]
        obj_emb = self.obj_cont_proj(obj_cont)          # [B, 18, d_model]

        # Type embedding
        obj_id_clamped = obj_id.clamp(min=0, max=self.n_types - 1)  # [B, 18]
        type_emb = self.type_emb(obj_id_clamped)        # [B, 18, d_model]
        tok_obj = obj_emb + type_emb                    # [B, 18, d_model]

        # Engineered CLS features
        met_sin = torch.sin(met_phi)                    # [B]
        met_cos = torch.cos(met_phi)                    # [B]
        nobj = mask.sum(dim=1).to(x.dtype)              # [B]
        nobj_norm = nobj / float(self.n_obj)            # [B]
        ht = (pT * mask.to(x.dtype)).sum(dim=1)         # [B]
        ht_log = torch.log1p(ht)                        # [B]

        # Top-3 pT (GeV) among real objects
        pt_masked = pT.masked_fill(~mask, -1.0)         # [B, 18]
        top3_vals, _ = torch.topk(pt_masked, k=3, dim=1)  # [B, 3]
        top3_vals = top3_vals.clamp_min(0.0)            # [B, 3]
        top3_log = torch.log1p(top3_vals)               # [B, 3]

        # Counts for ids 1..10
        counts = []
        for t in range(1, 11):
            counts.append((obj_id == t).sum(dim=1).to(x.dtype) / float(self.n_obj))  # [B]
        counts = torch.stack(counts, dim=1)             # [B, 10]

        # Pairwise features for topK objects: (log m_ij, dR_ij) flattened
        K = self.topk
        topk_vals, topk_idx = torch.topk(pt_masked, k=K, dim=1)  # topk_vals: [B, K], idx: [B, K]
        topk_valid = topk_vals > -0.5                    # [B, K] bool

        # Gather kinematics for topK
        gE = E.gather(1, topk_idx).clamp_min(0.0)        # [B, K]
        gpT = pT.gather(1, topk_idx).clamp_min(0.0)      # [B, K]
        geta = eta.gather(1, topk_idx)                   # [B, K]
        gphi = phi.gather(1, topk_idx)                   # [B, K]

        # 4-vectors
        gpx = gpT * torch.cos(gphi)                      # [B, K]
        gpy = gpT * torch.sin(gphi)                      # [B, K]
        gpz = gpT * torch.sinh(geta)                     # [B, K]

        # Pairwise masks
        vi = topk_valid[:, self._tri_i]                  # [B, n_pairs]
        vj = topk_valid[:, self._tri_j]                  # [B, n_pairs]
        vpair = (vi & vj).to(x.dtype)                    # [B, n_pairs]

        # Compute pairwise invariant masses
        Ei = gE[:, :, None]                              # [B, K, 1]
        Ej = gE[:, None, :]                              # [B, 1, K]
        px_i = gpx[:, :, None]                           # [B, K, 1]
        px_j = gpx[:, None, :]                           # [B, 1, K]
        py_i = gpy[:, :, None]                           # [B, K, 1]
        py_j = gpy[:, None, :]                           # [B, 1, K]
        pz_i = gpz[:, :, None]                           # [B, K, 1]
        pz_j = gpz[:, None, :]                           # [B, 1, K]

        Esum = Ei + Ej                                   # [B, K, K]
        pxsum = px_i + px_j                              # [B, K, K]
        pysum = py_i + py_j                              # [B, K, K]
        pzsum = pz_i + pz_j                              # [B, K, K]
        m2ij = (Esum * Esum - (pxsum * pxsum + pysum * pysum + pzsum * pzsum)).clamp_min(0.0)  # [B, K, K]
        mij = torch.sqrt(m2ij)                           # [B, K, K]

        # Pairwise dR
        etai = geta[:, :, None]                          # [B, K, 1]
        etaj = geta[:, None, :]                          # [B, 1, K]
        phii = gphi[:, :, None]                          # [B, K, 1]
        phij = gphi[:, None, :]                          # [B, 1, K]
        dphi = torch.remainder(phii - phij + math.pi, 2.0 * math.pi) - math.pi  # [B, K, K]
        deta = etai - etaj                                # [B, K, K]
        dR = torch.sqrt((deta * deta + dphi * dphi).clamp_min(0.0))             # [B, K, K]

        # Extract upper triangle pairs
        mij_u = mij[:, self._tri_i, self._tri_j]          # [B, n_pairs]
        dR_u = dR[:, self._tri_i, self._tri_j]            # [B, n_pairs]
        logm_u = torch.log1p(mij_u)                       # [B, n_pairs]

        # Mask invalid pairs to 0
        logm_u = logm_u * vpair                           # [B, n_pairs]
        dR_u = dR_u * vpair                               # [B, n_pairs]
        pair_flat = torch.cat([logm_u, dR_u], dim=1)       # [B, 2*n_pairs] = [B, 30]

        cls_feat = torch.cat(
            [
                met_log[:, None],                         # [B, 1]
                met_sin[:, None],                         # [B, 1]
                met_cos[:, None],                         # [B, 1]
                nobj_norm[:, None],                       # [B, 1]
                ht_log[:, None],                          # [B, 1]
                top3_log,                                 # [B, 3]
                counts,                                   # [B, 10]
                pair_flat,                                # [B, 30]
            ],
            dim=1,
        )                                                 # [B, 48]

        cls_emb = self.cls_proj(cls_feat).unsqueeze(1)    # [B, 1, d_model]
        cls_tok = self.cls_token.expand(B, 1, self.d_model) + cls_emb  # [B, 1, d_model]

        tokens = torch.cat([cls_tok, tok_obj], dim=1)     # [B, 19, d_model]
        tokens = self.post_ln(tokens)                     # [B, 19, d_model]

        # Padding mask: True means ignore (pad)
        pad_mask = torch.cat(
            [torch.zeros(B, 1, device=x.device, dtype=torch.bool), ~mask],
            dim=1,
        )                                                 # [B, 19]

        h = self.encoder(tokens, src_key_padding_mask=pad_mask)  # [B, 19, d_model]
        h_cls = h[:, 0, :]                                # [B, d_model]

        # Mean pool objects (valid only)
        h_obj = h[:, 1:, :]                               # [B, 18, d_model]
        m_float = mask.to(h_obj.dtype).unsqueeze(-1)      # [B, 18, 1]
        h_mean = (h_obj * m_float).sum(dim=1) / (m_float.sum(dim=1).clamp_min(1.0))  # [B, d_model]

        h_cat = torch.cat([h_cls, h_mean], dim=1)          # [B, 2*d_model]
        logits = self.head(h_cat).squeeze(-1)              # [B]
        return logits


def make_model(example_object):
    return BinaryClassifier(example_object)


EPOCHS = 20


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    is_cuda = (device.type == "cuda")

    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=1e-2,
    )

    total_steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=6e-4,
        total_steps=total_steps,
        pct_start=0.08,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=50.0,
    )

    scaler = torch.amp.GradScaler(enabled=is_cuda)

    train_loss_hist, val_loss_hist = [], []
    train_auc_hist, val_auc_hist = [], []

    best_auc = -1.0
    best_state = None
    patience = 6
    bad = 0

    for epoch in range(int(epochs)):
        model.train()
        tr_losses = []
        tr_logits = []
        tr_targets = []

        for batch in train_loader:
            bx, by = batch
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=is_cuda):
                logits = model(bx)                        # [B]
                loss = criterion(logits, by)              # scalar

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tr_losses.append(loss.detach().item())
            tr_logits.append(logits.detach().float().cpu())
            tr_targets.append(by.detach().float().cpu())

        tr_logits = torch.cat(tr_logits, dim=0).numpy()
        tr_targets = torch.cat(tr_targets, dim=0).numpy()
        tr_probs = 1.0 / (1.0 + np.exp(-tr_logits))
        try:
            tr_auc = float(roc_auc_score(tr_targets, tr_probs))
        except Exception:
            tr_auc = float("nan")

        train_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")

        model.eval()
        va_losses = []
        va_logits = []
        va_targets = []
        with torch.no_grad():
            for batch in val_loader:
                bx, by = batch
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True).float()

                with torch.amp.autocast(device_type=device.type, enabled=is_cuda):
                    logits = model(bx)                    # [B]
                    loss = criterion(logits, by)

                va_losses.append(loss.detach().item())
                va_logits.append(logits.detach().float().cpu())
                va_targets.append(by.detach().float().cpu())

        va_logits = torch.cat(va_logits, dim=0).numpy()
        va_targets = torch.cat(va_targets, dim=0).numpy()
        va_probs = 1.0 / (1.0 + np.exp(-va_logits))
        val_loss = float(np.mean(va_losses)) if va_losses else float("nan")
        try:
            val_auc = float(roc_auc_score(va_targets, va_probs))
        except Exception:
            val_auc = float("nan")

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_auc_hist.append(tr_auc)
        val_auc_hist.append(val_auc)

        improved = (val_auc > best_auc + 1e-4) if np.isfinite(val_auc) else False
        if improved:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    trained_model = model
    train_loss = train_loss_hist
    val_loss = val_loss_hist
    train_acc = train_auc_hist  # (use AUC for "acc" curves)
    val_acc = val_auc_hist      # (use AUC for "acc" curves)
    return trained_model, train_loss, val_loss, train_acc, val_acc

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


