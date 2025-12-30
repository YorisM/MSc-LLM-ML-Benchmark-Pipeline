
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

import math
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class StructuredDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        data = pre.transform(X) if pre is not None else {"flat": X}
        self.data = data
        self.y = y

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        x = {k: v[idx] for k, v in self.data.items()}
        return x, self.y[idx]


class MyPreprocessor:
    def __init__(self, max_types: int = 256, top_k_types: int = 8, stats_sample: int = 60000):
        self.max_types = int(max_types)
        self.top_k_types = int(top_k_types)
        self.stats_sample = int(stats_sample)

        # Fitted state
        self.id_keys = None          # torch.int64 [n_unique_sorted]
        self.top_ids = []            # list[int] raw IDs
        self.obj_mean = None         # torch.float32 [6]
        self.obj_std = None          # torch.float32 [6]
        self.glob_mean = None        # torch.float32 [7]
        self.glob_std = None         # torch.float32 [7]
        self.K = 0

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:StructuredDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": bool(device.type == "cuda"),

            "collate": None,
            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False},
        }

    @staticmethod
    def _split_raw(X: torch.Tensor):
        # X: [N,92]
        met = X[:, 0]              # [N]
        met_phi = X[:, 1]          # [N]
        obj_id = X[:, 2::5]        # [N,18]
        E = X[:, 3::5]             # [N,18]
        pT = X[:, 4::5]            # [N,18]
        eta = X[:, 5::5]           # [N,18]
        phi = X[:, 6::5]           # [N,18]
        return met, met_phi, obj_id, E, pT, eta, phi

    def fit(self, X, y=None):
        with torch.no_grad():
            X = X.detach().cpu()
            N = int(X.shape[0])

            # --- ID vocab from full training set (robust to rare IDs) ---
            _, _, obj_id_f, _, _, _, _ = self._split_raw(X)
            raw_ids_full = torch.round(obj_id_f).to(torch.int64)  # [N,18]
            valid_full = raw_ids_full != 0                        # [N,18]
            flat = raw_ids_full[valid_full]                       # [n_valid]
            if flat.numel() > 0:
                unique, counts = torch.unique(flat, return_counts=True)
                # top-k by frequency
                order = torch.argsort(counts, descending=True)
                unique = unique[order]
                self.K = int(min(self.top_k_types, unique.numel()))
                self.top_ids = unique[: self.K].cpu().tolist()

                # keys for bucketize mapping (sorted asc)
                self.id_keys = torch.sort(unique).values.cpu().to(torch.int64)
            else:
                self.K = 0
                self.top_ids = []
                self.id_keys = torch.empty((0,), dtype=torch.int64)

            # --- Stats from subset (faster) ---
            n_stats = int(min(N, self.stats_sample))
            idx = torch.randperm(N)[:n_stats]
            Xs = X[idx]  # [n_stats,92]

            met, met_phi, obj_id, E, pT, eta, phi = self._split_raw(Xs)

            raw_ids = torch.round(obj_id).to(torch.int64)  # [n,18]
            valid = raw_ids != 0                            # [n,18]

            # Convert to GeV and clamp
            Egev = (E * 1e-3).clamp(min=0.0)    # [n,18]
            pTgev = (pT * 1e-3).clamp(min=0.0)  # [n,18]
            metgev = (met * 1e-3).clamp(min=0.0)  # [n]

            abs_eta = eta.abs()  # [n,18]
            p = pTgev * torch.cosh(eta)  # [n,18]
            m2 = (Egev * Egev - p * p).clamp(min=0.0)  # [n,18]
            m = torch.sqrt(m2)  # [n,18]

            logE = torch.log1p(Egev)      # [n,18]
            logpT = torch.log1p(pTgev)    # [n,18]
            logm = torch.log1p(m)         # [n,18]
            e_over_p = Egev / (p + 1e-6)  # [n,18]

            obj_cont = torch.stack([logE, logpT, eta, abs_eta, logm, e_over_p], dim=-1)  # [n,18,6]
            if valid.any():
                vals = obj_cont[valid]  # [n_valid,6]
                mean = vals.mean(dim=0)
                std = vals.std(dim=0, unbiased=False).clamp(min=1e-6)
            else:
                mean = torch.zeros((6,), dtype=torch.float32)
                std = torch.ones((6,), dtype=torch.float32)
            self.obj_mean = mean.to(torch.float32).cpu()
            self.obj_std = std.to(torch.float32).cpu()

            n_obj = valid.sum(dim=1).to(torch.float32)  # [n]
            HT = (pTgev * valid).sum(dim=1)             # [n]
            sumE = (Egev * valid).sum(dim=1)            # [n]
            max_pT = pTgev.masked_fill(~valid, 0.0).max(dim=1).values  # [n]
            mean_abs_eta_evt = (abs_eta * valid).sum(dim=1) / n_obj.clamp(min=1.0)      # [n]
            ST = HT + metgev  # [n]

            glob_cont = torch.stack(
                [
                    torch.log1p(metgev),
                    n_obj / 18.0,
                    torch.log1p(HT),
                    torch.log1p(sumE),
                    torch.log1p(max_pT),
                    mean_abs_eta_evt,
                    torch.log1p(ST),
                ],
                dim=-1,
            )  # [n,7]

            gmean = glob_cont.mean(dim=0)
            gstd = glob_cont.std(dim=0, unbiased=False).clamp(min=1e-6)
            self.glob_mean = gmean.to(torch.float32).cpu()
            self.glob_std = gstd.to(torch.float32).cpu()

        return self

    def transform(self, X):
        with torch.no_grad():
            X = X.detach().cpu()
            N = int(X.shape[0])

            met, met_phi, obj_id, E, pT, eta, phi = self._split_raw(X)

            # Raw IDs and valid mask
            raw_ids = torch.round(obj_id).to(torch.int64)  # [N,18]
            valid = raw_ids != 0                            # [N,18]

            # Map IDs to contiguous indices using bucketize over sorted unique keys
            if self.id_keys is None or self.id_keys.numel() == 0:
                mapped = torch.zeros_like(raw_ids, dtype=torch.int64)  # [N,18]
            else:
                keys = self.id_keys  # [n_keys]
                pos = torch.bucketize(raw_ids, keys)  # [N,18] in [0..n_keys]
                pos_clamped = pos.clamp(max=max(int(keys.numel()) - 1, 0))  # [N,18]
                matched = (raw_ids != 0) & (pos < keys.numel()) & (keys[pos_clamped] == raw_ids)  # [N,18]
                mapped = torch.where(matched, pos_clamped + 1, torch.zeros_like(pos_clamped))     # [N,18]
                mapped = mapped.clamp(max=self.max_types - 1).to(torch.int64)  # [N,18]

            # Convert to GeV, clamp
            Egev = (E * 1e-3).clamp(min=0.0)    # [N,18]
            pTgev = (pT * 1e-3).clamp(min=0.0)  # [N,18]
            metgev = (met * 1e-3).clamp(min=0.0)  # [N]

            # Sort objects by pT descending to canonicalize order
            sort_idx = torch.argsort(pTgev, dim=1, descending=True)  # [N,18]
            def g2(t2d):
                return torch.gather(t2d, 1, sort_idx)
            raw_ids = g2(raw_ids)    # [N,18]
            mapped = g2(mapped)      # [N,18]
            valid = g2(valid)        # [N,18]
            Egev = g2(Egev)          # [N,18]
            pTgev = g2(pTgev)        # [N,18]
            eta = g2(eta)            # [N,18]
            phi = g2(phi)            # [N,18]

            # Object-level derived features
            abs_eta = eta.abs()            # [N,18]
            p = pTgev * torch.cosh(eta)    # [N,18]
            m2 = (Egev * Egev - p * p).clamp(min=0.0)  # [N,18]
            m = torch.sqrt(m2)             # [N,18]

            logE = torch.log1p(Egev)       # [N,18]
            logpT = torch.log1p(pTgev)     # [N,18]
            logm = torch.log1p(m)          # [N,18]
            e_over_p = Egev / (p + 1e-6)   # [N,18]

            # Normalize continuous object channels
            obj_cont = torch.stack([logE, logpT, eta, abs_eta, logm, e_over_p], dim=-1)  # [N,18,6]
            mean = self.obj_mean.view(1, 1, 6)  # [1,1,6]
            std = self.obj_std.view(1, 1, 6)    # [1,1,6]
            obj_cont_n = (obj_cont - mean) / std  # [N,18,6]
            obj_cont_n = obj_cont_n.clamp(-6.0, 6.0)

            sin_phi = torch.sin(phi)  # [N,18]
            cos_phi = torch.cos(phi)  # [N,18]
            dphi = phi - met_phi.view(N, 1)  # [N,18]
            sin_dphi = torch.sin(dphi)  # [N,18]
            cos_dphi = torch.cos(dphi)  # [N,18]

            # Event sums for fractions
            HT = (pTgev * valid).sum(dim=1)  # [N]
            sumE = (Egev * valid).sum(dim=1)  # [N]
            pT_frac = pTgev / (HT.view(N, 1) + 1e-6)   # [N,18]
            E_frac = Egev / (sumE.view(N, 1) + 1e-6)   # [N,18]

            obj_feat = torch.cat(
                [
                    obj_cont_n,                             # [N,18,6]
                    sin_phi.unsqueeze(-1),                  # [N,18,1]
                    cos_phi.unsqueeze(-1),                  # [N,18,1]
                    sin_dphi.unsqueeze(-1),                 # [N,18,1]
                    cos_dphi.unsqueeze(-1),                 # [N,18,1]
                    pT_frac.unsqueeze(-1),                  # [N,18,1]
                    E_frac.unsqueeze(-1),                   # [N,18,1]
                ],
                dim=-1,
            )  # [N,18,12]
            obj_feat = obj_feat.masked_fill(~valid.unsqueeze(-1), 0.0).to(torch.float32)  # [N,18,12]

            pad_mask = (~valid).to(torch.bool)  # [N,18]

            # Global features
            n_obj = valid.sum(dim=1).to(torch.float32)  # [N]
            max_pT = pTgev.masked_fill(~valid, 0.0).max(dim=1).values  # [N]
            mean_abs_eta_evt = (abs_eta * valid).sum(dim=1) / n_obj.clamp(min=1.0)  # [N]
            ST = HT + metgev  # [N]

            glob_cont = torch.stack(
                [
                    torch.log1p(metgev),
                    n_obj / 18.0,
                    torch.log1p(HT),
                    torch.log1p(sumE),
                    torch.log1p(max_pT),
                    mean_abs_eta_evt,
                    torch.log1p(ST),
                ],
                dim=-1,
            ).to(torch.float32)  # [N,7]

            gmean = self.glob_mean.view(1, 7)  # [1,7]
            gstd = self.glob_std.view(1, 7)    # [1,7]
            glob_cont_n = ((glob_cont - gmean) / gstd).clamp(-6.0, 6.0)  # [N,7]

            sin_met_phi = torch.sin(met_phi).view(N, 1).to(torch.float32)  # [N,1]
            cos_met_phi = torch.cos(met_phi).view(N, 1).to(torch.float32)  # [N,1]

            # Type fractions for top IDs (+ other)
            if self.K > 0 and len(self.top_ids) > 0:
                counts = []
                for tid in self.top_ids:
                    counts.append((raw_ids == int(tid)).sum(dim=1).to(torch.float32) / 18.0)  # [N]
                top_fracs = torch.stack(counts, dim=1)  # [N,K]
                other_frac = (n_obj - (top_fracs * 18.0).sum(dim=1)).clamp(min=0.0) / 18.0  # [N]
                type_fracs = torch.cat([top_fracs, other_frac.view(N, 1)], dim=1).to(torch.float32)  # [N,K+1]
            else:
                type_fracs = (n_obj / 18.0).view(N, 1).to(torch.float32)  # [N,1] (acts as "other")

            global_feat = torch.cat([glob_cont_n, sin_met_phi, cos_met_phi, type_fracs], dim=1).to(torch.float32)
            # global_feat: [N, 7 + 2 + (K+1 or 1)] = [N, 10+K] or [N,10]

            return {
                "global": global_feat,         # float32 [N,G]
                "obj_id": mapped,              # int64   [N,18]
                "obj_feat": obj_feat,          # float32 [N,18,12]
                "pad_mask": pad_mask,          # bool    [N,18] True for padding
            }


def make_preprocessor():
    return MyPreprocessor(max_types=256, top_k_types=8, stats_sample=60000)


class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        if isinstance(sample_object, dict):
            gdim = int(sample_object["global"].shape[-1])
            ofeat_dim = int(sample_object["obj_feat"].shape[-1])
        else:
            raise TypeError("Expected dict-like batch_x from StructuredDataset")

        self.max_types = 256
        emb_dim = 16
        d_model = 96
        nhead = 6
        n_layers = 3
        ff_dim = 256
        dropout = 0.12

        self.type_emb = nn.Embedding(self.max_types, emb_dim)
        self.obj_in = nn.Sequential(
            nn.Linear(emb_dim + ofeat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        self.glob_in = nn.Sequential(
            nn.Linear(gdim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        # Positional embeddings for [CLS] + 18 pT-sorted objects
        self.pos_emb = nn.Embedding(19, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, 192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, batch_x):
        g = batch_x["global"]      # float [B,G]
        obj_id = batch_x["obj_id"]  # long  [B,18]
        obj_f = batch_x["obj_feat"]  # float [B,18,12]
        pad_mask = batch_x["pad_mask"]  # bool [B,18] True for padding

        B = int(g.shape[0])

        obj_id = obj_id.clamp(min=0, max=self.max_types - 1)
        emb = self.type_emb(obj_id)  # [B,18,emb_dim]
        obj_tok = torch.cat([emb, obj_f], dim=-1)  # [B,18,emb_dim+12]
        obj_tok = self.obj_in(obj_tok)  # [B,18,d_model]

        cls = self.glob_in(g).unsqueeze(1)  # [B,1,d_model]
        seq = torch.cat([cls, obj_tok], dim=1)  # [B,19,d_model]

        pos = torch.arange(19, device=seq.device).view(1, 19)  # [1,19]
        seq = seq + self.pos_emb(pos)  # [B,19,d_model]

        src_key_padding_mask = torch.cat(
            [torch.zeros((B, 1), dtype=torch.bool, device=seq.device), pad_mask],
            dim=1,
        )  # [B,19], True=pad

        out = self.encoder(seq, src_key_padding_mask=src_key_padding_mask)  # [B,19,d_model]

        cls_out = out[:, 0, :]  # [B,d_model]
        obj_out = out[:, 1:, :]  # [B,18,d_model]

        valid = (~pad_mask).to(obj_out.dtype)  # [B,18]
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B,1]

        mean_pool = (obj_out * valid.unsqueeze(-1)).sum(dim=1) / denom  # [B,d_model]
        max_pool = obj_out.masked_fill(pad_mask.unsqueeze(-1), -1e9).max(dim=1).values  # [B,d_model]
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        feat = torch.cat([cls_out, mean_pool, max_pool], dim=-1)  # [B,3*d_model]
        logit = self.head(feat).squeeze(-1)  # [B]
        return logit


def make_model(example_object):
    return BinaryClassifier(example_object)


EPOCHS = 12


@torch.no_grad()
def _eval_epoch(model: nn.Module, loader):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_n = 0
    correct = 0

    all_probs = []
    all_y = []

    for batch in loader:
        view = normalise_batch(batch, device=device)
        xb, yb = view.batch_x, view.batch_y
        ybf = yb.to(torch.float32)

        logits = model(xb).view(-1)  # [B]
        loss = crit(logits, ybf)

        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).to(yb.dtype)

        bs = int(yb.shape[0])
        total_loss += float(loss.item()) * bs
        total_n += bs
        correct += int((pred == yb).sum().item())

        all_probs.append(probs.detach().float().cpu())
        all_y.append(yb.detach().cpu())

    avg_loss = total_loss / max(total_n, 1)
    acc = correct / max(total_n, 1)

    probs = torch.cat(all_probs, dim=0).numpy()
    yy = torch.cat(all_y, dim=0).numpy()
    try:
        auc = float(roc_auc_score(yy, probs))
    except Exception:
        auc = float("nan")

    return avg_loss, acc, auc


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    crit = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    total_steps = int(max(1, epochs * len(train_loader)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2.0e-3,
        total_steps=total_steps,
        pct_start=0.10,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=100.0,
    )

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    best_auc = -1.0
    best_state = None
    patience = 4
    bad = 0

    step = 0
    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_n = 0
        correct = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            ybf = yb.to(torch.float32)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xb).view(-1)  # [B]
                    loss = crit(logits, ybf)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb).view(-1)  # [B]
                loss = crit(logits, ybf)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            step += 1

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred = (probs >= 0.5).to(yb.dtype)
                bs = int(yb.shape[0])
                total_loss += float(loss.item()) * bs
                total_n += bs
                correct += int((pred == yb).sum().item())

        tr_loss = total_loss / max(total_n, 1)
        tr_acc = correct / max(total_n, 1)

        va_loss, va_acc, va_auc = _eval_epoch(model, val_loader)

        train_loss_hist.append(tr_loss)
        train_acc_hist.append(tr_acc)
        val_loss_hist.append(va_loss)
        val_acc_hist.append(va_acc)

        if math.isfinite(va_auc) and va_auc > best_auc + 1e-5:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

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

