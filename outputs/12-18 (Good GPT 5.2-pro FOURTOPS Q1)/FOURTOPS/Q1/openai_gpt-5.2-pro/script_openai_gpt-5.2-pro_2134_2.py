
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

import math, copy
from typing import Optional, List

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.utils.data import DataLoader


if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # constants
        self.n_obj = 18
        self.obj_stride_in = 5  # [id, E, pt, eta, phi]
        self.energy_scale = 1000.0  # MeV -> GeV

        # object-type mapping (raw id -> encoded id)
        self.max_raw_id: int = 0
        self.top_raw_ids: List[int] = []
        self.other_idx: int = 1
        self.id_map_np: Optional[np.ndarray] = None  # length max_raw_id+1, maps raw->enc

        # stats for standardization (scalars)
        self.met_mean = 0.0
        self.met_std = 1.0

        self.logE_mean = 0.0
        self.logE_std = 1.0
        self.logpt_mean = 0.0
        self.logpt_std = 1.0
        self.eta_mean = 0.0
        self.eta_std = 1.0
        self.logm_mean = 0.0
        self.logm_std = 1.0

        self.sum_mean_np: Optional[np.ndarray] = None
        self.sum_std_np: Optional[np.ndarray] = None

        # feature layout
        self.global_dim = 3  # [log_met_z, sin(metphi), cos(metphi)]
        self.obj_feat_dim = 9  # [enc_id, logE_z, logpt_z, eta_z, sinphi, cosphi, logm_z, sin_dphi, cos_dphi]

        # config for top object types
        self.n_top_types = 12

    def make_loader_cfg(self):
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": bool(torch.cuda.is_available()),
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    @staticmethod
    def _safe_std(x: torch.Tensor, eps: float = 1e-6) -> float:
        s = x.float().std(unbiased=False)
        s = torch.clamp(s, min=eps)
        return float(s.item())

    def fit(self, X, y=None):
        # X: torch.float32 [N, 92]
        X = X.detach().cpu()

        N = X.shape[0]
        obj = X[:, 2:].reshape(N, self.n_obj, self.obj_stride_in)  # (N,18,5)
        raw_id = torch.round(obj[:, :, 0]).clamp(min=0).to(torch.long)  # (N,18)
        mask = raw_id > 0  # (N,18)

        self.max_raw_id = int(raw_id.max().item()) if raw_id.numel() else 0

        # Build raw-id frequency table
        if self.max_raw_id >= 1:
            flat_ids = raw_id.reshape(-1)
            counts = torch.bincount(flat_ids, minlength=self.max_raw_id + 1).float()  # (max_raw_id+1,)
            # exclude 0 (padding)
            counts_no0 = counts[1:]
            k = min(self.n_top_types, int((counts_no0 > 0).sum().item()))
            if k > 0:
                top = torch.topk(counts_no0, k=k, largest=True).indices + 1  # raw ids
                self.top_raw_ids = [int(v.item()) for v in top]
            else:
                self.top_raw_ids = []
        else:
            self.top_raw_ids = []

        self.other_idx = len(self.top_raw_ids) + 1  # 0 padding, 1..K top, K+1 other

        # Create mapping array for raw -> enc
        id_map = torch.full((self.max_raw_id + 1,), fill_value=self.other_idx, dtype=torch.long)
        id_map[0] = 0
        for i, rid in enumerate(self.top_raw_ids, start=1):
            if rid <= self.max_raw_id:
                id_map[rid] = i
        self.id_map_np = id_map.numpy()

        # Compute kinematics in GeV
        E = obj[:, :, 1] / self.energy_scale  # (N,18)
        pt = obj[:, :, 2] / self.energy_scale  # (N,18)
        eta = obj[:, :, 3]  # (N,18)

        # Per-object engineered scalars
        logE = torch.log1p(torch.clamp(E, min=0.0))  # (N,18)
        logpt = torch.log1p(torch.clamp(pt, min=0.0))  # (N,18)
        p = pt * torch.cosh(eta)  # (N,18)
        m2 = torch.clamp(E * E - p * p, min=0.0)
        logm = torch.log1p(torch.sqrt(m2))  # (N,18)

        # Stats: ignore padding objects
        m = mask
        if m.any():
            self.logE_mean = float(logE[m].mean().item())
            self.logE_std = self._safe_std(logE[m])
            self.logpt_mean = float(logpt[m].mean().item())
            self.logpt_std = self._safe_std(logpt[m])
            self.eta_mean = float(eta[m].mean().item())
            self.eta_std = self._safe_std(eta[m])
            self.logm_mean = float(logm[m].mean().item())
            self.logm_std = self._safe_std(logm[m])
        else:
            # fallback
            self.logE_mean, self.logE_std = 0.0, 1.0
            self.logpt_mean, self.logpt_std = 0.0, 1.0
            self.eta_mean, self.eta_std = 0.0, 1.0
            self.logm_mean, self.logm_std = 0.0, 1.0

        # Global MET stats
        met = X[:, 0] / self.energy_scale  # (N,)
        log_met = torch.log1p(torch.clamp(met, min=0.0))  # (N,)
        self.met_mean = float(log_met.mean().item())
        self.met_std = self._safe_std(log_met)

        # Summary features
        n_obj = mask.sum(dim=1).float()  # (N,)
        ht = (pt * mask.float()).sum(dim=1)  # (N,)
        sumE = (E * mask.float()).sum(dim=1)  # (N,)
        maxpt = pt.masked_fill(~mask, 0.0).max(dim=1).values  # (N,)
        mean_eta = (eta * mask.float()).sum(dim=1) / (n_obj + 1e-6)  # (N,)
        max_abs_eta = eta.abs().masked_fill(~mask, 0.0).max(dim=1).values  # (N,)

        counts_top = []
        for rid in self.top_raw_ids:
            counts_top.append((raw_id == rid).sum(dim=1).float())
        if len(counts_top) > 0:
            counts_top = torch.stack(counts_top, dim=1)  # (N,K)
            other_count = (n_obj - counts_top.sum(dim=1)).unsqueeze(1)  # (N,1)
            counts_all = torch.cat([counts_top, other_count], dim=1)  # (N,K+1)
        else:
            counts_all = n_obj.unsqueeze(1)  # (N,1) only "other" (all objects)

        summary = torch.cat(
            [
                n_obj.unsqueeze(1),         # (N,1)
                ht.unsqueeze(1),            # (N,1)
                sumE.unsqueeze(1),          # (N,1)
                maxpt.unsqueeze(1),         # (N,1)
                mean_eta.unsqueeze(1),      # (N,1)
                max_abs_eta.unsqueeze(1),   # (N,1)
                counts_all,                 # (N,K+1)
            ],
            dim=1,
        )  # (N, 6+K+1)

        sum_mean = summary.mean(dim=0)
        sum_std = summary.std(dim=0, unbiased=False).clamp(min=1e-6)

        self.sum_mean_np = sum_mean.numpy()
        self.sum_std_np = sum_std.numpy()

        return self

    def transform(self, X):
        # X: torch.float32 [N, 92]
        X = X.detach().cpu()
        N = X.shape[0]

        met = X[:, 0] / self.energy_scale  # (N,)
        met_phi = X[:, 1]  # (N,)

        log_met = torch.log1p(torch.clamp(met, min=0.0))  # (N,)
        log_met_z = (log_met - self.met_mean) / (self.met_std + 1e-6)  # (N,)

        sin_metphi = torch.sin(met_phi)  # (N,)
        cos_metphi = torch.cos(met_phi)  # (N,)

        global_feat = torch.stack([log_met_z, sin_metphi, cos_metphi], dim=1)  # (N,3)

        obj = X[:, 2:].reshape(N, self.n_obj, self.obj_stride_in)  # (N,18,5)
        raw_id = torch.round(obj[:, :, 0]).clamp(min=0).to(torch.long)  # (N,18)
        mask = raw_id > 0  # (N,18)
        mask_f = mask.float()  # (N,18)

        # Encode ids using learned mapping
        id_map = torch.from_numpy(self.id_map_np).to(torch.long)  # (max_raw_id+1,)
        enc_id = torch.empty_like(raw_id)
        too_big = raw_id > self.max_raw_id
        safe = ~too_big
        enc_id[too_big] = self.other_idx
        enc_id[safe] = id_map[raw_id[safe]]
        enc_id = enc_id * mask.to(torch.long)  # keep padding = 0
        enc_id_f = enc_id.float()  # (N,18)

        E = obj[:, :, 1] / self.energy_scale  # (N,18)
        pt = obj[:, :, 2] / self.energy_scale  # (N,18)
        eta = obj[:, :, 3]  # (N,18)
        phi = obj[:, :, 4]  # (N,18)

        logE = torch.log1p(torch.clamp(E, min=0.0))  # (N,18)
        logpt = torch.log1p(torch.clamp(pt, min=0.0))  # (N,18)

        p = pt * torch.cosh(eta)  # (N,18)
        m2 = torch.clamp(E * E - p * p, min=0.0)  # (N,18)
        logm = torch.log1p(torch.sqrt(m2))  # (N,18)

        logE_z = ((logE - self.logE_mean) / (self.logE_std + 1e-6)) * mask_f  # (N,18)
        logpt_z = ((logpt - self.logpt_mean) / (self.logpt_std + 1e-6)) * mask_f  # (N,18)
        eta_z = ((eta - self.eta_mean) / (self.eta_std + 1e-6)) * mask_f  # (N,18)
        logm_z = ((logm - self.logm_mean) / (self.logm_std + 1e-6)) * mask_f  # (N,18)

        sinphi = torch.sin(phi) * mask_f  # (N,18)
        cosphi = torch.cos(phi) * mask_f  # (N,18)

        dphi = phi - met_phi.unsqueeze(1)  # (N,18)
        dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))  # wrap to [-pi, pi]
        sin_dphi = torch.sin(dphi) * mask_f  # (N,18)
        cos_dphi = torch.cos(dphi) * mask_f  # (N,18)

        obj_feat = torch.stack(
            [
                enc_id_f,   # (N,18)
                logE_z,     # (N,18)
                logpt_z,    # (N,18)
                eta_z,      # (N,18)
                sinphi,     # (N,18)
                cosphi,     # (N,18)
                logm_z,     # (N,18)
                sin_dphi,   # (N,18)
                cos_dphi,   # (N,18)
            ],
            dim=2,
        )  # (N,18,9)

        obj_flat = obj_feat.reshape(N, self.n_obj * self.obj_feat_dim)  # (N,162)

        # Summary features
        n_obj = mask.sum(dim=1).float()  # (N,)
        ht = (pt * mask_f).sum(dim=1)  # (N,)
        sumE = (E * mask_f).sum(dim=1)  # (N,)
        maxpt = pt.masked_fill(~mask, 0.0).max(dim=1).values  # (N,)
        mean_eta = (eta * mask_f).sum(dim=1) / (n_obj + 1e-6)  # (N,)
        max_abs_eta = eta.abs().masked_fill(~mask, 0.0).max(dim=1).values  # (N,)

        counts_top = []
        for rid in self.top_raw_ids:
            counts_top.append((raw_id == rid).sum(dim=1).float())
        if len(counts_top) > 0:
            counts_top = torch.stack(counts_top, dim=1)  # (N,K)
            other_count = (n_obj - counts_top.sum(dim=1)).unsqueeze(1)  # (N,1)
            counts_all = torch.cat([counts_top, other_count], dim=1)  # (N,K+1)
        else:
            counts_all = n_obj.unsqueeze(1)  # (N,1)

        summary = torch.cat(
            [
                n_obj.unsqueeze(1),         # (N,1)
                ht.unsqueeze(1),            # (N,1)
                sumE.unsqueeze(1),          # (N,1)
                maxpt.unsqueeze(1),         # (N,1)
                mean_eta.unsqueeze(1),      # (N,1)
                max_abs_eta.unsqueeze(1),   # (N,1)
                counts_all,                 # (N,K+1)
            ],
            dim=1,
        )  # (N, 6+K+1)

        sum_mean = torch.from_numpy(self.sum_mean_np).float()  # (S,)
        sum_std = torch.from_numpy(self.sum_std_np).float()  # (S,)
        summary_z = (summary - sum_mean) / (sum_std + 1e-6)  # (N,S)

        out = torch.cat([global_feat, obj_flat, summary_z], dim=1)  # (N, 3+162+S)
        return out.to(torch.float32)


def make_preprocessor():
    return MyPreprocessor()


# ---------- MODEL DEFINITION ----------
class SetBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        # x: (B,N,D)
        h = self.ln1(x)  # (B,N,D)
        attn_out, _ = self.mha(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)  # (B,N,D)
        x = x + self.drop1(attn_out)  # (B,N,D)

        h2 = self.ln2(x)  # (B,N,D)
        x = x + self.drop2(self.ff(h2))  # (B,N,D)
        return x


class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        in_dim = int(sample_object.shape[-1])

        # fixed from preprocessor
        self.n_obj = 18
        self.global_dim = 3
        self.obj_stride = 9
        self.obj_flat_dim = self.n_obj * self.obj_stride

        assert in_dim >= self.global_dim + self.obj_flat_dim, f"Unexpected input dim: {in_dim}"

        self.summary_dim = in_dim - (self.global_dim + self.obj_flat_dim)

        # embedding for encoded object IDs (0 padding, 1..K types, plus "other")
        self.max_embed = 64
        emb_dim = 16
        self.obj_emb = nn.Embedding(self.max_embed, emb_dim, padding_idx=0)

        num_dim = self.obj_stride - 1  # (logE_z, logpt_z, eta_z, sinphi, cosphi, logm_z, sin_dphi, cos_dphi) => 8
        d_model = 128

        self.obj_in = nn.Sequential(
            nn.Linear(emb_dim + num_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        self.blocks = nn.ModuleList([SetBlock(d_model, n_heads=8, dropout=0.10) for _ in range(3)])
        self.final_ln = nn.LayerNorm(d_model)

        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        head_in = 3 * d_model + (self.global_dim + self.summary_dim)

        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, batch_x):
        # batch_x: (B,F)
        B = batch_x.shape[0]

        g = batch_x[:, : self.global_dim]  # (B,3)
        obj_flat = batch_x[:, self.global_dim : self.global_dim + self.obj_flat_dim]  # (B,162)
        s = batch_x[:, self.global_dim + self.obj_flat_dim :]  # (B,summary_dim)

        obj = obj_flat.view(B, self.n_obj, self.obj_stride)  # (B,18,9)

        obj_id = obj[:, :, 0].to(torch.long)  # (B,18)
        obj_id = obj_id.clamp(min=0, max=self.max_embed - 1)  # (B,18)
        mask = obj_id != 0  # (B,18)
        key_padding_mask = ~mask  # (B,18) True means "ignore"

        obj_num = obj[:, :, 1:]  # (B,18,8)
        emb = self.obj_emb(obj_id)  # (B,18,16)
        x = torch.cat([emb, obj_num], dim=-1)  # (B,18,24)
        x = self.obj_in(x)  # (B,18,128)

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)  # (B,18,128)

        x = self.final_ln(x)  # (B,18,128)

        mask_f = mask.unsqueeze(-1).float()  # (B,18,1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)  # (B,1)

        mean_pool = (x * mask_f).sum(dim=1) / denom  # (B,128)

        x_masked = x.masked_fill(~mask.unsqueeze(-1), -1e9)  # (B,18,128)
        max_pool = x_masked.max(dim=1).values  # (B,128)
        has_any = (mask.sum(dim=1) > 0).unsqueeze(-1)  # (B,1)
        max_pool = torch.where(has_any, max_pool, torch.zeros_like(max_pool))  # (B,128)

        scores = self.attn_pool(x).squeeze(-1)  # (B,18)
        scores = scores.masked_fill(~mask, -1e9)  # (B,18)
        w = torch.softmax(scores, dim=1)  # (B,18)
        attn_pool = torch.bmm(w.unsqueeze(1), x).squeeze(1)  # (B,128)

        pooled = torch.cat([mean_pool, max_pool, attn_pool], dim=1)  # (B,384)
        global_all = torch.cat([g, s], dim=1)  # (B, 3+summary_dim)
        h = torch.cat([pooled, global_all], dim=1)  # (B, head_in)

        logits = self.head(h).squeeze(-1)  # (B,)
        return logits


def make_model(example_object):
    return BinaryClassifier(example_object)


# ---------- MODEL TRAINING ----------
EPOCHS = 15


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    use_cuda = torch.cuda.is_available()
    dev = next(model.parameters()).device

    loss_fn = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    best_state = copy.deepcopy(model.state_dict())
    best_auc = -1.0
    patience = 4
    bad = 0

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    for ep in range(int(epochs)):
        model.train()
        tr_loss_sum = 0.0
        tr_n = 0
        tr_corr = 0

        for xb, yb in train_loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_cuda):
                logits = model(xb)  # (B,)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                tr_loss_sum += float(loss.item()) * xb.size(0)
                tr_n += xb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).to(torch.long)
                tr_corr += int((preds == yb.to(torch.long)).sum().item())

        sched.step()

        tr_loss = tr_loss_sum / max(1, tr_n)
        tr_acc = tr_corr / max(1, tr_n)
        train_loss_hist.append(tr_loss)
        train_acc_hist.append(tr_acc)

        # Validation
        model.eval()
        va_loss_sum = 0.0
        va_n = 0
        va_corr = 0
        all_scores = []
        all_y = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(dev, non_blocking=True)
                yb_f = yb.to(dev, non_blocking=True).float()

                with torch.cuda.amp.autocast(enabled=use_cuda):
                    logits = model(xb)
                    loss = loss_fn(logits, yb_f)

                va_loss_sum += float(loss.item()) * xb.size(0)
                va_n += xb.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).to(torch.long)
                va_corr += int((preds.cpu() == yb.cpu()).sum().item())

                all_scores.append(probs.detach().float().cpu().numpy())
                all_y.append(yb.detach().cpu().numpy())

        va_loss = va_loss_sum / max(1, va_n)
        va_acc = va_corr / max(1, va_n)
        val_loss_hist.append(va_loss)
        val_acc_hist.append(va_acc)

        y_true = np.concatenate(all_y, axis=0)
        y_score = np.concatenate(all_scores, axis=0)
        try:
            va_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            va_auc = 0.5

        # Early stopping on AUC
        if va_auc > best_auc + 1e-4:
            best_auc = va_auc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

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


