
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
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

import math, copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# Module-level params set by preprocessor.fit() for model construction
PREPROC_PARAMS = {}


class MyPreprocessor:
    def __init__(self, chunk_size: int = 16384):
        self.chunk_size = int(chunk_size)

        # Learned stats (Python floats -> picklable)
        self.mean_metlog = 0.0
        self.std_metlog = 1.0
        self.mean_elog = 0.0
        self.std_elog = 1.0
        self.mean_ptlog = 0.0
        self.std_ptlog = 1.0
        self.mean_eta = 0.0
        self.std_eta = 1.0

        self.max_obj_id = 0

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
        # X: torch.Tensor [N,92]
        X = X if torch.is_tensor(X) else torch.as_tensor(X)
        N = int(X.shape[0])

        sum_met = 0.0
        sumsq_met = 0.0
        cnt_met = 0.0

        sum_e = 0.0
        sumsq_e = 0.0
        sum_pt = 0.0
        sumsq_pt = 0.0
        sum_eta = 0.0
        sumsq_eta = 0.0
        cnt_obj = 0.0

        max_id = 0

        for i in range(0, N, self.chunk_size):
            Xi = X[i : i + self.chunk_size].to(torch.float32)

            met = Xi[:, 0]  # [B]
            metlog = torch.log1p(torch.clamp(met, min=0.0) / 1000.0)  # [B]
            sum_met += float(metlog.sum().item())
            sumsq_met += float((metlog * metlog).sum().item())
            cnt_met += float(metlog.numel())

            ids = Xi[:, 2::5]  # [B,18]
            # max id over chunk
            max_id = max(max_id, int(torch.clamp(ids, min=0.0).amax().item()))

            mask = ids != 0.0  # [B,18] bool
            mf = mask.to(torch.float32)

            E = Xi[:, 3::5]  # [B,18]
            pt = Xi[:, 4::5]  # [B,18]
            eta = torch.clamp(Xi[:, 5::5], min=-7.0, max=7.0)  # [B,18]

            elog = torch.log1p(torch.clamp(E, min=0.0) / 1000.0)  # [B,18]
            ptlog = torch.log1p(torch.clamp(pt, min=0.0) / 1000.0)  # [B,18]

            sum_e += float((elog * mf).sum().item())
            sumsq_e += float((elog * elog * mf).sum().item())
            sum_pt += float((ptlog * mf).sum().item())
            sumsq_pt += float((ptlog * ptlog * mf).sum().item())
            sum_eta += float((eta * mf).sum().item())
            sumsq_eta += float((eta * eta * mf).sum().item())
            cnt_obj += float(mf.sum().item())

        def _mean_std(sumv, sumsqv, cnt, eps=1e-6):
            cnt = max(cnt, 1.0)
            mean = sumv / cnt
            var = max(sumsqv / cnt - mean * mean, eps)
            std = math.sqrt(var)
            return float(mean), float(std)

        self.mean_metlog, self.std_metlog = _mean_std(sum_met, sumsq_met, cnt_met)
        self.mean_elog, self.std_elog = _mean_std(sum_e, sumsq_e, cnt_obj)
        self.mean_ptlog, self.std_ptlog = _mean_std(sum_pt, sumsq_pt, cnt_obj)
        self.mean_eta, self.std_eta = _mean_std(sum_eta, sumsq_eta, cnt_obj)

        self.max_obj_id = int(max_id)

        global PREPROC_PARAMS
        PREPROC_PARAMS = {
            "mean_metlog": self.mean_metlog,
            "std_metlog": self.std_metlog,
            "mean_elog": self.mean_elog,
            "std_elog": self.std_elog,
            "mean_ptlog": self.mean_ptlog,
            "std_ptlog": self.std_ptlog,
            "mean_eta": self.mean_eta,
            "std_eta": self.std_eta,
            "max_obj_id": self.max_obj_id,
        }
        return self

    def transform(self, X):
        # Return torch.Tensor [N,92] float32
        X = X if torch.is_tensor(X) else torch.as_tensor(X)
        Xo = X.to(torch.float32).clone()  # [N,92]

        ids = Xo[:, 2::5]  # [N,18]
        mask = ids != 0.0  # [N,18] bool

        # Global: met -> standardized log1p(GeV)
        metlog = torch.log1p(torch.clamp(Xo[:, 0], min=0.0) / 1000.0)  # [N]
        Xo[:, 0] = (metlog - self.mean_metlog) / max(self.std_metlog, 1e-6)  # [N]

        # Objects: E, pt -> standardized log1p(GeV); eta -> standardized
        Elog = torch.log1p(torch.clamp(Xo[:, 3::5], min=0.0) / 1000.0)  # [N,18]
        PTlog = torch.log1p(torch.clamp(Xo[:, 4::5], min=0.0) / 1000.0)  # [N,18]
        ETA = torch.clamp(Xo[:, 5::5], min=-7.0, max=7.0)  # [N,18]

        Elog = (Elog - self.mean_elog) / max(self.std_elog, 1e-6)  # [N,18]
        PTlog = (PTlog - self.mean_ptlog) / max(self.std_ptlog, 1e-6)  # [N,18]
        ETA = (ETA - self.mean_eta) / max(self.std_eta, 1e-6)  # [N,18]

        # Zero out padded objects (prevents padding leakage)
        Elog = Elog.masked_fill(~mask, 0.0)  # [N,18]
        PTlog = PTlog.masked_fill(~mask, 0.0)  # [N,18]
        ETA = ETA.masked_fill(~mask, 0.0)  # [N,18]

        Xo[:, 3::5] = Elog
        Xo[:, 4::5] = PTlog
        Xo[:, 5::5] = ETA

        # Keep metphi (idx 1), obj_id (2::5), phi (6::5) unchanged
        return Xo


def make_preprocessor():
    return MyPreprocessor()


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims, out_dim: int, dropout: float = 0.1, act=nn.GELU):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            if b != out_dim:
                layers.append(act())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        params = PREPROC_PARAMS or {}
        max_obj_id = int(params.get("max_obj_id", 64))
        self.max_obj_id = max(1, max_obj_id)

        # Store un/standardization params as buffers (for physics-derived features)
        self.register_buffer("mean_metlog", torch.tensor(float(params.get("mean_metlog", 0.0)), dtype=torch.float32))
        self.register_buffer("std_metlog", torch.tensor(float(params.get("std_metlog", 1.0)), dtype=torch.float32))
        self.register_buffer("mean_elog", torch.tensor(float(params.get("mean_elog", 0.0)), dtype=torch.float32))
        self.register_buffer("std_elog", torch.tensor(float(params.get("std_elog", 1.0)), dtype=torch.float32))
        self.register_buffer("mean_ptlog", torch.tensor(float(params.get("mean_ptlog", 0.0)), dtype=torch.float32))
        self.register_buffer("std_ptlog", torch.tensor(float(params.get("std_ptlog", 1.0)), dtype=torch.float32))
        self.register_buffer("mean_eta", torch.tensor(float(params.get("mean_eta", 0.0)), dtype=torch.float32))
        self.register_buffer("std_eta", torch.tensor(float(params.get("std_eta", 1.0)), dtype=torch.float32))

        d_type = 16
        d_model = 64
        n_heads = 4
        n_layers = 2
        dropout = 0.12

        self.type_emb = nn.Embedding(self.max_obj_id + 1, d_type)

        # Per-object engineered features (see forward): C_obj = 12
        self.obj_ln = nn.LayerNorm(12 + d_type)
        self.obj_mlp = nn.Sequential(
            nn.Linear(12 + d_type, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.post_ln = nn.LayerNorm(d_model)

        self.attn = nn.Linear(d_model, 1)

        # Global features: 13 + K_types, with K_types=min(max_obj_id, 12)
        self.K_types = int(min(self.max_obj_id, 12))
        glob_dim = 13 + self.K_types  # [B,13+K]

        self.head_ln = nn.LayerNorm(3 * d_model + glob_dim)
        self.head = nn.Sequential(
            nn.Linear(3 * d_model + glob_dim, 192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, 1),
        )

    @staticmethod
    def _wrap_dphi(dphi):
        # returns in [-pi, pi]
        return torch.atan2(torch.sin(dphi), torch.cos(dphi))

    def forward(self, batch_x):
        # batch_x: torch.Tensor [B,92]
        x = batch_x
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.mean_metlog.device)
        x = x.to(torch.float32)

        B = x.shape[0]

        met_std = x[:, 0]  # [B]
        metphi = x[:, 1]  # [B]

        objs = x[:, 2:].view(B, 18, 5)  # [B,18,5]
        ids_f = objs[:, :, 0]  # [B,18]
        ids = torch.clamp(ids_f.round().to(torch.int64), min=0, max=self.max_obj_id)  # [B,18]
        mask = ids != 0  # [B,18] bool

        # Ensure at least one token valid for attention/transformer stability
        mask_any = mask.any(dim=1, keepdim=True)  # [B,1]
        mask2 = mask.clone()
        mask2 = torch.where(mask_any, mask2, torch.zeros_like(mask2))
        mask2[:, 0] = torch.where(mask_any[:, 0], mask2[:, 0], torch.ones_like(mask2[:, 0]))  # [B]

        # De-standardize log features to compute physical values
        E_std = objs[:, :, 1]  # [B,18]
        pt_std = objs[:, :, 2]  # [B,18]
        eta_std = objs[:, :, 3]  # [B,18]
        phi = objs[:, :, 4]  # [B,18]

        E_log = E_std * self.std_elog + self.mean_elog  # [B,18]
        pt_log = pt_std * self.std_ptlog + self.mean_ptlog  # [B,18]
        eta = eta_std * self.std_eta + self.mean_eta  # [B,18]

        # Physical (GeV) from log1p
        E = torch.expm1(E_log).clamp(min=0.0)  # [B,18]
        pt = torch.expm1(pt_log).clamp(min=0.0)  # [B,18]

        # mass from E^2 - p^2 with p = pt * cosh(eta)
        cosh_eta = torch.cosh(torch.clamp(eta, min=-7.0, max=7.0))  # [B,18]
        p = pt * cosh_eta  # [B,18]
        m2 = (E * E - p * p).clamp(min=0.0)  # [B,18]
        mass = torch.sqrt(m2 + 1e-9)  # [B,18]
        log_mass = torch.log1p(mass)  # [B,18]
        pt_over_E = pt / (E + 1e-6)  # [B,18]

        # Angles
        sin_phi = torch.sin(phi)  # [B,18]
        cos_phi = torch.cos(phi)  # [B,18]
        dphi = self._wrap_dphi(phi - metphi[:, None])  # [B,18]
        sin_dphi = torch.sin(dphi)  # [B,18]
        cos_dphi = torch.cos(dphi)  # [B,18]

        # Build per-object feature vector (C_obj=12)
        # obj_feats: [B,18,12]
        obj_feats = torch.stack(
            [
                E_std,  # standardized log1p(E/GeV)
                pt_std,  # standardized log1p(pt/GeV)
                eta_std,  # standardized eta
                sin_phi,
                cos_phi,
                sin_dphi,
                cos_dphi,
                log_mass,
                pt_over_E,
                pt_log,  # unstandardized log1p(pt/GeV)
                E_log,  # unstandardized log1p(E/GeV)
                torch.abs(eta),  # physical |eta|
            ],
            dim=-1,
        )  # [B,18,12]

        # Zero out engineered angle/mass features for padded objects to avoid leakage
        obj_feats = obj_feats * mask.unsqueeze(-1).to(obj_feats.dtype)  # [B,18,12]

        type_e = self.type_emb(ids)  # [B,18,d_type]
        h = torch.cat([obj_feats, type_e], dim=-1)  # [B,18,12+d_type]
        h = self.obj_ln(h)  # [B,18,12+d_type]
        h = self.obj_mlp(h)  # [B,18,d_model]

        # Transformer with key padding mask
        h = self.encoder(h, src_key_padding_mask=~mask2)  # [B,18,d_model]
        h = self.post_ln(h)  # [B,18,d_model]

        # Pooling
        attn_logits = self.attn(h).squeeze(-1)  # [B,18]
        attn_logits = attn_logits.masked_fill(~mask2, -1e9)
        w = torch.softmax(attn_logits, dim=1)  # [B,18]
        pool_attn = (w.unsqueeze(-1) * h).sum(dim=1)  # [B,d_model]

        cnt = mask2.sum(dim=1, keepdim=True).to(h.dtype).clamp(min=1.0)  # [B,1]
        pool_mean = (h * mask2.unsqueeze(-1).to(h.dtype)).sum(dim=1) / cnt  # [B,d_model]
        pool_max = h.masked_fill(~mask2.unsqueeze(-1), -1e9).max(dim=1).values  # [B,d_model]

        # Global engineered features
        met_log = met_std * self.std_metlog + self.mean_metlog  # [B]
        met_GeV = torch.expm1(met_log).clamp(min=0.0)  # [B]

        nobj = mask.sum(dim=1).to(h.dtype)  # [B]
        sum_pt = (pt * mask.to(pt.dtype)).sum(dim=1)  # [B]
        max_pt = pt.masked_fill(~mask2, 0.0).max(dim=1).values  # [B]
        sum_m = (mass * mask.to(mass.dtype)).sum(dim=1)  # [B]
        max_m = mass.masked_fill(~mask2, 0.0).max(dim=1).values  # [B]

        pt_masked_for_topk = pt.masked_fill(~mask, -1.0)  # [B,18]
        topk_pt = torch.topk(pt_masked_for_topk, k=4, dim=1).values.clamp(min=0.0)  # [B,4]
        topk_pt_log = torch.log1p(topk_pt)  # [B,4]

        # Type counts for ids 1..K
        if self.K_types > 0:
            counts = []
            for t in range(1, self.K_types + 1):
                counts.append(((ids == t) & mask).sum(dim=1).to(h.dtype))  # [B]
            type_counts = torch.stack(counts, dim=1)  # [B,K]
            # mild scaling
            type_counts = torch.log1p(type_counts)  # [B,K]
        else:
            type_counts = h.new_zeros((B, 0))  # [B,0]

        glob = torch.cat(
            [
                met_std[:, None],  # [B,1]
                torch.sin(metphi)[:, None],  # [B,1]
                torch.cos(metphi)[:, None],  # [B,1]
                met_log[:, None],  # [B,1]
                torch.log1p(nobj)[:, None],  # [B,1]
                torch.log1p(sum_pt)[:, None],  # [B,1]
                torch.log1p(max_pt + 1e-6)[:, None],  # [B,1]
                torch.log1p(sum_m)[:, None],  # [B,1]
                torch.log1p(max_m + 1e-6)[:, None],  # [B,1]
                topk_pt_log,  # [B,4]
                type_counts,  # [B,K]
            ],
            dim=1,
        )  # [B,13+K]

        z = torch.cat([pool_attn, pool_mean, pool_max, glob], dim=1)  # [B, 3*d_model + 13+K]
        z = self.head_ln(z)
        logits = self.head(z).squeeze(-1)  # [B]
        return logits


def make_model(example_object):
    return BinaryClassifier(example_object)


EPOCHS = 20 if (torch.cuda.is_available()) else 12


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1.0e-2)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, steps_per_epoch * int(epochs))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2.5e-3,
        total_steps=total_steps,
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
    patience = 5
    bad = 0

    for epoch in range(int(epochs)):
        model.train()
        tr_loss_sum = 0.0
        tr_n = 0
        tr_correct = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            yb = yb.to(torch.float32)  # [B]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)  # [B]
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = int(yb.shape[0])
            tr_loss_sum += float(loss.detach().item()) * bs
            tr_n += bs
            tr_correct += int(((logits.detach() > 0.0).to(torch.int64) == yb.to(torch.int64)).sum().item())

        train_loss = tr_loss_sum / max(tr_n, 1)
        train_acc = tr_correct / max(tr_n, 1)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        # Validation
        model.eval()
        va_loss_sum = 0.0
        va_n = 0
        va_correct = 0
        all_probs = []
        all_y = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                yb_f = yb.to(torch.float32)

                logits = model(xb)
                loss = criterion(logits, yb_f)

                bs = int(yb.shape[0])
                va_loss_sum += float(loss.item()) * bs
                va_n += bs
                va_correct += int(((logits > 0.0).to(torch.int64) == yb.to(torch.int64)).sum().item())

                probs = torch.sigmoid(logits).detach().to("cpu").to(torch.float32).numpy()
                all_probs.append(probs)
                all_y.append(yb.detach().to("cpu").numpy())

        val_loss = va_loss_sum / max(va_n, 1)
        val_acc = va_correct / max(va_n, 1)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        y_true = np.concatenate(all_y, axis=0)
        y_prob = np.concatenate(all_probs, axis=0)
        try:
            val_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            val_auc = float("nan")

        # Early stopping on AUC (primary metric)
        improved = (not math.isnan(val_auc)) and (val_auc > best_auc + 1e-4)
        if improved:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
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

