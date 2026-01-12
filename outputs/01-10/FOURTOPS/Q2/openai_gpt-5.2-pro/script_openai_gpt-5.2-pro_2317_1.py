
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
import copy
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score


class MyPreprocessor:
    # Output layout:
    #   global features (G=14):
    #     0  met_log_norm
    #     1  sin(met_phi)
    #     2  cos(met_phi)
    #     3  ht_log_norm
    #     4  sumE_log_norm
    #     5  nobj_norm
    #     6  min_dR_norm
    #     7  mean_dR_norm
    #     8  frac_dR_lt_04_norm
    #     9  max_m_log_norm
    #     10 mean_m_log_norm
    #     11 top1_m_log_norm
    #     12 top2_m_log_norm
    #     13 top3_m_log_norm
    #
    #   object features (Nobj=18, Dobj=8) flattened -> 18*8=144:
    #     per object: [id, logE_norm, logpt_norm, eta_norm, sinphi, cosphi, mass_log_norm, mask]

    def __init__(self):
        self.n_obj = 18
        self.obj_in_dim = 5
        self.gdim = 14
        self.obj_dim = 8
        self.out_dim = self.gdim + self.n_obj * self.obj_dim  # 14 + 144 = 158

        self._eps = 1e-6
        self._two_pi = float(2.0 * math.pi)
        self._pi = float(math.pi)

        # Stats (torch CPU tensors, picklable)
        self.mean_obj = None  # [4] for [logE, logpt, eta, mass_log]
        self.std_obj = None   # [4]
        self.mean_glob = None  # [4] for [met_log, ht_log, sumE_log, nobj]
        self.std_glob = None   # [4]
        self.mean_pair = None  # [8]
        self.std_pair = None   # [8]

        # Precomputed upper-triangular mask (no diagonal)
        triu = torch.triu(torch.ones(self.n_obj, self.n_obj, dtype=torch.bool), diagonal=1)  # [18,18]
        self._triu = triu

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
            "eval_overrides": {"shuffle": False, "batch_size": 2048},
        }

    @staticmethod
    def _as_float_tensor_cpu(X):
        if torch.is_tensor(X):
            return X.detach().to(dtype=torch.float32, device="cpu")
        return torch.as_tensor(X, dtype=torch.float32, device="cpu")

    def _split(self, Xb):
        # Xb: FloatTensor[B,92]
        # Returns:
        #   met: [B]
        #   metphi: [B]
        #   obj_id,E,pt,eta,phi: each [B,18]
        met = Xb[:, 0]  # [B]
        metphi = Xb[:, 1]  # [B]
        objs = Xb[:, 2:].view(-1, self.n_obj, self.obj_in_dim)  # [B,18,5]
        obj_id = objs[:, :, 0]  # [B,18]
        E = objs[:, :, 1]       # [B,18]
        pt = objs[:, :, 2]      # [B,18]
        eta = objs[:, :, 3]     # [B,18]
        phi = objs[:, :, 4]     # [B,18]
        return met, metphi, obj_id, E, pt, eta, phi

    def _wrap_dphi(self, dphi):
        # Wrap to [-pi, pi]
        return torch.remainder(dphi + self._pi, self._two_pi) - self._pi

    def _pairwise_aggs(self, obj_id, E, pt, eta, phi):
        # Inputs shapes: [B,18]
        # Output: [B,8] -> [min_dR, mean_dR, frac_dR_lt_04, max_m_log, mean_m_log, top1_m_log, top2_m_log, top3_m_log]
        B = obj_id.shape[0]
        mask = (obj_id > 0)  # [B,18] bool

        # Pair mask: valid objects and i<j
        mask_pairs = (mask[:, :, None] & mask[:, None, :] & self._triu[None, :, :])  # [B,18,18] bool
        num_pairs = mask_pairs.sum(dim=(1, 2)).to(dtype=torch.float32)  # [B]

        # dR
        deta = eta[:, :, None] - eta[:, None, :]  # [B,18,18]
        dphi = self._wrap_dphi(phi[:, :, None] - phi[:, None, :])  # [B,18,18]
        dR = torch.sqrt(deta * deta + dphi * dphi + 1e-12)  # [B,18,18]

        dR_for_min = dR.masked_fill(~mask_pairs, 1e9)  # [B,18,18]
        min_dR = dR_for_min.amin(dim=(1, 2))  # [B]
        sum_dR = (dR * mask_pairs.to(dtype=torch.float32)).sum(dim=(1, 2))  # [B]
        mean_dR = torch.where(num_pairs > 0, sum_dR / num_pairs, torch.zeros_like(sum_dR))  # [B]
        frac_lt = torch.where(
            num_pairs > 0,
            ((dR < 0.4) & mask_pairs).sum(dim=(1, 2)).to(dtype=torch.float32) / num_pairs,
            torch.zeros_like(num_pairs),
        )  # [B]
        min_dR = torch.where(num_pairs > 0, min_dR, torch.zeros_like(min_dR))  # [B]

        # Invariant mass
        px = pt * torch.cos(phi)  # [B,18]
        py = pt * torch.sin(phi)  # [B,18]
        pz = pt * torch.sinh(eta)  # [B,18]

        Es = E[:, :, None] + E[:, None, :]      # [B,18,18]
        pxs = px[:, :, None] + px[:, None, :]   # [B,18,18]
        pys = py[:, :, None] + py[:, None, :]   # [B,18,18]
        pzs = pz[:, :, None] + pz[:, None, :]   # [B,18,18]

        p2 = pxs * pxs + pys * pys + pzs * pzs  # [B,18,18]
        m2 = Es * Es - p2  # [B,18,18]
        m = torch.sqrt(torch.clamp(m2, min=0.0))  # [B,18,18]
        m_ut = m.masked_fill(~mask_pairs, 0.0)  # [B,18,18]

        max_m = m_ut.amax(dim=(1, 2))  # [B]
        sum_m = m_ut.sum(dim=(1, 2))   # [B]
        mean_m = torch.where(num_pairs > 0, sum_m / num_pairs, torch.zeros_like(sum_m))  # [B]
        max_m_log = torch.log1p(max_m)     # [B]
        mean_m_log = torch.log1p(mean_m)   # [B]

        m_flat = m_ut.view(B, -1)  # [B,324]
        top3 = torch.topk(m_flat, k=3, dim=1).values  # [B,3]
        top3_log = torch.log1p(top3)  # [B,3]

        feats = torch.stack([min_dR, mean_dR, frac_lt, max_m_log, mean_m_log], dim=1)  # [B,5]
        feats = torch.cat([feats, top3_log], dim=1)  # [B,8]
        return feats

    def fit(self, X, y=None):
        Xc = self._as_float_tensor_cpu(X)  # [N,92]
        met, metphi, obj_id, E, pt, eta, phi = self._split(Xc)

        # Object-level features for stats
        mask = (obj_id > 0).to(dtype=torch.float32)  # [N,18]
        logE = torch.log1p(torch.clamp(E, min=0.0))  # [N,18]
        logpt = torch.log1p(torch.clamp(pt, min=0.0))  # [N,18]
        p = torch.abs(pt) * torch.cosh(eta)  # [N,18]
        mass2 = E * E - p * p  # [N,18]
        mass = torch.sqrt(torch.clamp(mass2, min=0.0))  # [N,18]
        mass_log = torch.log1p(mass)  # [N,18]

        def masked_mean_std(x, m):
            # x,m: [N,18] float
            msum = m.sum().clamp_min(1.0)
            mean = (x * m).sum() / msum
            var = (x * x * m).sum() / msum - mean * mean
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            return mean, std

        m_logE, s_logE = masked_mean_std(logE, mask)
        m_logpt, s_logpt = masked_mean_std(logpt, mask)
        m_eta, s_eta = masked_mean_std(eta, mask)
        m_masslog, s_masslog = masked_mean_std(mass_log, mask)

        self.mean_obj = torch.tensor([m_logE, m_logpt, m_eta, m_masslog], dtype=torch.float32)  # [4]
        self.std_obj = torch.tensor([s_logE, s_logpt, s_eta, s_masslog], dtype=torch.float32)   # [4]

        # Global stats
        met_log = torch.log1p(torch.clamp(met, min=0.0))  # [N]
        ht = (pt * mask).sum(dim=1)  # [N]
        ht_log = torch.log1p(torch.clamp(ht, min=0.0))  # [N]
        sumE = (E * mask).sum(dim=1)  # [N]
        sumE_log = torch.log1p(torch.clamp(sumE, min=0.0))  # [N]
        nobj = mask.sum(dim=1)  # [N]

        def mean_std_1d(x):
            mean = x.mean()
            var = (x * x).mean() - mean * mean
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            return mean, std

        m_metlog, s_metlog = mean_std_1d(met_log)
        m_htlog, s_htlog = mean_std_1d(ht_log)
        m_sumElog, s_sumElog = mean_std_1d(sumE_log)
        m_nobj, s_nobj = mean_std_1d(nobj)

        self.mean_glob = torch.tensor([m_metlog, m_htlog, m_sumElog, m_nobj], dtype=torch.float32)  # [4]
        self.std_glob = torch.tensor([s_metlog, s_htlog, s_sumElog, s_nobj], dtype=torch.float32)   # [4]

        # Pairwise stats on a deterministic subset (speed)
        N = Xc.shape[0]
        subset = min(N, 50000)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(42)
        idx = torch.randperm(N, generator=gen)[:subset]
        obj_id_s = obj_id[idx]
        E_s = E[idx]
        pt_s = pt[idx]
        eta_s = eta[idx]
        phi_s = phi[idx]

        # Streaming sums for pairwise features
        sum_p = torch.zeros(8, dtype=torch.float64)
        sumsq_p = torch.zeros(8, dtype=torch.float64)
        count_p = 0

        chunk = 4096
        for start in range(0, subset, chunk):
            end = min(subset, start + chunk)
            feats = self._pairwise_aggs(obj_id_s[start:end], E_s[start:end], pt_s[start:end], eta_s[start:end], phi_s[start:end])  # [b,8]
            feats64 = feats.to(dtype=torch.float64)
            sum_p += feats64.sum(dim=0)
            sumsq_p += (feats64 * feats64).sum(dim=0)
            count_p += feats.shape[0]

        mean_p = sum_p / max(count_p, 1)
        var_p = sumsq_p / max(count_p, 1) - mean_p * mean_p
        std_p = torch.sqrt(torch.clamp(var_p, min=1e-6))

        self.mean_pair = mean_p.to(dtype=torch.float32)  # [8]
        self.std_pair = std_p.to(dtype=torch.float32)    # [8]

        return self

    def transform(self, X):
        Xc = self._as_float_tensor_cpu(X)  # [N,92]
        N = Xc.shape[0]
        out = torch.empty((N, self.out_dim), dtype=torch.float32, device="cpu")  # [N,158]

        # Unpack stats
        mean_obj = self.mean_obj  # [4]
        std_obj = torch.clamp(self.std_obj, min=1e-4)  # [4]
        mean_glob = self.mean_glob  # [4]
        std_glob = torch.clamp(self.std_glob, min=1e-4)  # [4]
        mean_pair = self.mean_pair  # [8]
        std_pair = torch.clamp(self.std_pair, min=1e-4)  # [8]

        chunk = 4096
        for start in range(0, N, chunk):
            end = min(N, start + chunk)
            Xb = Xc[start:end]  # [B,92]
            B = Xb.shape[0]

            met, metphi, obj_id, E, pt, eta, phi = self._split(Xb)
            mask = (obj_id > 0).to(dtype=torch.float32)  # [B,18]

            # Global
            met_log = torch.log1p(torch.clamp(met, min=0.0))  # [B]
            met_log_n = (met_log - mean_glob[0]) / std_glob[0]  # [B]
            met_sin = torch.sin(metphi)  # [B]
            met_cos = torch.cos(metphi)  # [B]

            ht = (pt * mask).sum(dim=1)  # [B]
            ht_log = torch.log1p(torch.clamp(ht, min=0.0))  # [B]
            ht_log_n = (ht_log - mean_glob[1]) / std_glob[1]  # [B]

            sumE = (E * mask).sum(dim=1)  # [B]
            sumE_log = torch.log1p(torch.clamp(sumE, min=0.0))  # [B]
            sumE_log_n = (sumE_log - mean_glob[2]) / std_glob[2]  # [B]

            nobj = mask.sum(dim=1)  # [B]
            nobj_n = (nobj - mean_glob[3]) / std_glob[3]  # [B]

            pair = self._pairwise_aggs(obj_id, E, pt, eta, phi)  # [B,8]
            pair_n = (pair - mean_pair[None, :]) / std_pair[None, :]  # [B,8]

            global_feat = torch.cat(
                [
                    met_log_n[:, None],     # [B,1]
                    met_sin[:, None],       # [B,1]
                    met_cos[:, None],       # [B,1]
                    ht_log_n[:, None],      # [B,1]
                    sumE_log_n[:, None],    # [B,1]
                    nobj_n[:, None],        # [B,1]
                    pair_n,                 # [B,8]
                ],
                dim=1,
            )  # [B,14]

            # Object features
            logE = torch.log1p(torch.clamp(E, min=0.0))  # [B,18]
            logpt = torch.log1p(torch.clamp(pt, min=0.0))  # [B,18]
            p = torch.abs(pt) * torch.cosh(eta)  # [B,18]
            mass2 = E * E - p * p  # [B,18]
            mass = torch.sqrt(torch.clamp(mass2, min=0.0))  # [B,18]
            mass_log = torch.log1p(mass)  # [B,18]

            logE_n = ((logE - mean_obj[0]) / std_obj[0]) * mask  # [B,18]
            logpt_n = ((logpt - mean_obj[1]) / std_obj[1]) * mask  # [B,18]
            eta_n = ((eta - mean_obj[2]) / std_obj[2]) * mask  # [B,18]
            sinphi = torch.sin(phi) * mask  # [B,18]
            cosphi = torch.cos(phi) * mask  # [B,18]
            masslog_n = ((mass_log - mean_obj[3]) / std_obj[3]) * mask  # [B,18]

            obj_feat = torch.stack(
                [
                    logE_n,        # [B,18]
                    logpt_n,       # [B,18]
                    eta_n,         # [B,18]
                    sinphi,        # [B,18]
                    cosphi,        # [B,18]
                    masslog_n,     # [B,18]
                    mask,          # [B,18]
                ],
                dim=-1,
            )  # [B,18,7]

            obj_all = torch.cat([obj_id[:, :, None], obj_feat], dim=-1)  # [B,18,8]
            obj_flat = obj_all.reshape(B, -1)  # [B,144]

            out[start:end] = torch.cat([global_feat, obj_flat], dim=1)  # [B,158]

        return out


def make_preprocessor():
    return MyPreprocessor()


class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object: FloatTensor[B,F], with F=158
        F = int(sample_object.shape[-1])
        self.gdim = 14
        self.n_obj = 18
        self.obj_dim = 8
        assert F == self.gdim + self.n_obj * self.obj_dim, f"Unexpected input dim {F}"

        # Model dims
        self.id_vocab = 32
        self.id_emb_dim = 16
        self.d_model = 64

        self.id_emb = nn.Embedding(self.id_vocab, self.id_emb_dim)
        self.obj_proj = nn.Sequential(
            nn.Linear(self.id_emb_dim + (self.obj_dim - 1), self.d_model),  # (16 + 7) -> 64
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.obj_dropout = nn.Dropout(0.10)

        self.global_mlp = nn.Sequential(
            nn.Linear(self.gdim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.10),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.cls_from_global = nn.Sequential(
            nn.Linear(self.gdim, self.d_model),
            nn.Tanh(),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=256,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)

        self.head = nn.Sequential(
            nn.Linear(self.d_model * 3 + 64, 192),  # cls + mean + max + global(64)
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
        # batch_x: FloatTensor[B,158]
        x = batch_x
        B = x.shape[0]

        global_x = x[:, : self.gdim]  # [B,14]
        obj_flat = x[:, self.gdim :]  # [B,144]
        obj = obj_flat.view(B, self.n_obj, self.obj_dim)  # [B,18,8]

        obj_id = obj[:, :, 0]  # [B,18] float
        obj_cont = obj[:, :, 1:]  # [B,18,7]
        mask = obj_cont[:, :, -1]  # [B,18] float in {0,1}
        key_padding_mask = mask < 0.5  # [B,18] bool (True means PAD)

        ids = torch.round(obj_id).clamp(0, self.id_vocab - 1).to(dtype=torch.long)  # [B,18]
        id_emb = self.id_emb(ids)  # [B,18,16]

        obj_in = torch.cat([id_emb, obj_cont], dim=-1)  # [B,18,23]
        obj_in = self.obj_proj(obj_in)  # [B,18,64]
        obj_in = self.obj_dropout(obj_in)  # [B,18,64]
        obj_in = obj_in.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)  # [B,18,64]

        # CLS token conditioned on global features
        cls_bias = self.cls_from_global(global_x).unsqueeze(1)  # [B,1,64]
        cls = self.cls_token.expand(B, 1, self.d_model) + cls_bias  # [B,1,64]
        seq = torch.cat([cls, obj_in], dim=1)  # [B,19,64]

        pad_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=x.device), key_padding_mask],
            dim=1,
        )  # [B,19]

        enc = self.transformer(seq, src_key_padding_mask=pad_mask)  # [B,19,64]
        cls_out = enc[:, 0, :]  # [B,64]
        obj_out = enc[:, 1:, :]  # [B,18,64]

        valid = (~key_padding_mask).to(dtype=obj_out.dtype).unsqueeze(-1)  # [B,18,1]
        den = valid.sum(dim=1).clamp_min(1.0)  # [B,1]
        mean_pool = (obj_out * valid).sum(dim=1) / den  # [B,64]

        obj_out_masked = obj_out.masked_fill(key_padding_mask.unsqueeze(-1), -1e9)  # [B,18,64]
        max_pool = obj_out_masked.max(dim=1).values  # [B,64]
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))  # [B,64]

        g = self.global_mlp(global_x)  # [B,64]
        feat = torch.cat([cls_out, mean_pool, max_pool, g], dim=1)  # [B,256]
        logit = self.head(feat).squeeze(-1)  # [B]
        return logit


def make_model(example_object):
    return BinaryClassifier(example_object)


EPOCHS = 15


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    use_amp = (device.type == "cuda")

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=max(1, epochs),
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,
        div_factor=10.0,
        final_div_factor=100.0,
        anneal_strategy="cos",
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = nn.BCEWithLogitsLoss()

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    best_auc = -1.0
    best_state = None
    patience = 5
    bad = 0

    for epoch in range(int(epochs)):
        # ---- train ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)  # [B,F]
            yb = yb.to(device, non_blocking=True).float()  # [B]
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

            running_loss += float(loss.detach().cpu()) * xb.size(0)
            preds = (logits.detach() > 0).to(dtype=torch.int64)
            correct += int((preds == yb.to(dtype=torch.int64)).sum().detach().cpu())
            total += int(xb.size(0))

        tr_loss = running_loss / max(total, 1)
        tr_acc = correct / max(total, 1)
        train_loss.append(tr_loss)
        train_acc.append(tr_acc)

        # ---- validate ----
        model.eval()
        v_running_loss = 0.0
        v_correct = 0
        v_total = 0
        all_scores = []
        all_true = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)

                v_running_loss += float(loss.detach().cpu()) * xb.size(0)
                preds = (logits > 0).to(dtype=torch.int64)
                v_correct += int((preds == yb.to(dtype=torch.int64)).sum().detach().cpu())
                v_total += int(xb.size(0))

                scores = torch.sigmoid(logits).detach().float().cpu().numpy()
                all_scores.append(scores)
                all_true.append(yb.detach().cpu().numpy())

        va_loss = v_running_loss / max(v_total, 1)
        va_acc = v_correct / max(v_total, 1)
        val_loss.append(va_loss)
        val_acc.append(va_acc)

        y_true = np.concatenate(all_true, axis=0)
        y_score = np.concatenate(all_scores, axis=0)
        try:
            va_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            va_auc = -1.0

        # Early stopping on AUC
        if va_auc > best_auc + 1e-4:
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

