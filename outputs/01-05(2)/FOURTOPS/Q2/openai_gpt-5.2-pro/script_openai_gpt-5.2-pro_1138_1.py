
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

import math, copy
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn


class MyPreprocessor:
    def __init__(self, max_objects: int = 18, obj_stride: int = 5, topk_pairwise: int = 6, chunk_size: int = 32768):
        self.max_objects = int(max_objects)
        self.obj_stride = int(obj_stride)
        self.topk = int(topk_pairwise)
        self.chunk_size = int(chunk_size)

        # Stats (set in fit)
        self.met_mu = None
        self.met_sigma = None

        # global continuous: [ht_log, sumE_log, maxpT_log, avg_pT_log, meanAbsEta] => 5
        self.glob_mu = None
        self.glob_sigma = None

        # pairwise => 12
        self.pair_mu = None
        self.pair_sigma = None

        # per-object channels to normalize: [logE, logpT, eta] => 3
        self.obj_mu = None
        self.obj_sigma = None

        self._fitted = False

    def make_loader_cfg(self) -> dict:
        pin = bool(torch.cuda.is_available())
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": pin,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 1024},
        }

    @staticmethod
    def _masked_mean_std(x, mask, eps=1e-6):
        # x: FloatTensor[B,P], mask: BoolTensor[B,P]
        mask_f = mask.float()
        cnt = mask_f.sum(dim=1, keepdim=True)  # [B,1]
        cnt_safe = cnt.clamp_min(1.0)
        mean = (x * mask_f).sum(dim=1, keepdim=True) / cnt_safe  # [B,1]
        var = ((x - mean) ** 2 * mask_f).sum(dim=1, keepdim=True) / cnt_safe  # [B,1]
        std = torch.sqrt(var.clamp_min(eps))
        mean = torch.where(cnt > 0, mean, torch.zeros_like(mean))
        std = torch.where(cnt > 0, std, torch.ones_like(std))
        return mean.squeeze(1), std.squeeze(1)  # [B], [B]

    @staticmethod
    def _masked_max_min(x, mask):
        # x: FloatTensor[B,P], mask: BoolTensor[B,P]
        neg_inf = torch.tensor(-float("inf"), dtype=x.dtype, device=x.device)
        pos_inf = torch.tensor(float("inf"), dtype=x.dtype, device=x.device)
        xm = x.masked_fill(~mask, neg_inf)
        xM = x.masked_fill(~mask, pos_inf)
        x_max = xm.max(dim=1).values  # [B]
        x_min = xM.min(dim=1).values  # [B]
        anyv = mask.any(dim=1)
        x_max = torch.where(anyv, x_max, torch.zeros_like(x_max))
        x_min = torch.where(anyv, x_min, torch.zeros_like(x_min))
        return x_max, x_min  # [B], [B]

    @staticmethod
    def _masked_topk2(x, mask):
        # returns top1, top2 with invalid -> 0
        neg_inf = torch.tensor(-float("inf"), dtype=x.dtype, device=x.device)
        xm = x.masked_fill(~mask, neg_inf)
        vals, _ = torch.topk(xm, k=2, dim=1)  # [B,2]
        vals = torch.where(torch.isfinite(vals), vals, torch.zeros_like(vals))
        return vals[:, 0], vals[:, 1]  # [B], [B]

    def _compute_pairwise_features(self, E, pT, eta, phi, obj_mask):
        # E,pT: [B,18] in MeV ; eta,phi: [B,18] ; obj_mask: [B,18] bool
        B = E.shape[0]
        K = self.topk
        # Select topK by pT (mask invalid)
        pT_masked = pT.masked_fill(~obj_mask, -1.0e9)  # [B,18]
        idx = torch.topk(pT_masked, k=K, dim=1, largest=True, sorted=False).indices  # [B,K]

        # Gather
        Esel = torch.gather(E, 1, idx)  # [B,K]
        pTsel = torch.gather(pT, 1, idx)  # [B,K]
        etasel = torch.gather(eta, 1, idx)  # [B,K]
        phisel = torch.gather(phi, 1, idx)  # [B,K]
        vsel = torch.gather(obj_mask, 1, idx)  # [B,K] bool

        # Build 4-vectors
        px = pTsel * torch.cos(phisel)  # [B,K]
        py = pTsel * torch.sin(phisel)  # [B,K]
        pz = pTsel * torch.sinh(etasel)  # [B,K]

        tri = torch.triu_indices(K, K, 1, device=E.device)
        i, j = tri[0], tri[1]
        P = int(i.numel())

        # Pair mask
        pair_mask = (vsel[:, i] & vsel[:, j])  # [B,P]

        Ei = Esel[:, i]  # [B,P]
        Ej = Esel[:, j]
        pxi = px[:, i]
        pxj = px[:, j]
        pyi = py[:, i]
        pyj = py[:, j]
        pzi = pz[:, i]
        pzj = pz[:, j]

        Es = Ei + Ej  # [B,P]
        pxs = pxi + pxj
        pys = pyi + pyj
        pzs = pzi + pzj

        m2 = Es * Es - (pxs * pxs + pys * pys + pzs * pzs)  # [B,P]
        m = torch.sqrt(m2.clamp_min(0.0)) / 1000.0  # [B,P] in GeV
        mlog = torch.log1p(m)  # [B,P]

        dphi = phisel[:, i] - phisel[:, j]  # [B,P]
        dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))  # wrap to [-pi,pi]
        deta = etasel[:, i] - etasel[:, j]  # [B,P]
        dR = torch.sqrt(deta * deta + dphi * dphi + 1e-12)  # [B,P]

        # Stats for dR (4) and mlog (4), plus top2 each (4) => 12
        dr_mean, dr_std = self._masked_mean_std(dR, pair_mask)  # [B], [B]
        dr_max, dr_min = self._masked_max_min(dR, pair_mask)    # [B], [B]
        dr_top1, dr_top2 = self._masked_topk2(dR, pair_mask)    # [B], [B]

        m_mean, m_std = self._masked_mean_std(mlog, pair_mask)  # [B], [B]
        m_max, m_min = self._masked_max_min(mlog, pair_mask)    # [B], [B]
        m_top1, m_top2 = self._masked_topk2(mlog, pair_mask)    # [B], [B]

        pair = torch.stack(
            [dr_mean, dr_std, dr_max, dr_min, m_mean, m_std, m_max, m_min, m_top1, m_top2, dr_top1, dr_top2],
            dim=1
        )  # [B,12]
        return pair

    def fit(self, X, y=None):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float32)
        X = X.float().cpu()
        N = int(X.shape[0])

        # Accumulators
        met_sum = 0.0
        met_sumsq = 0.0

        glob_sum = torch.zeros(5, dtype=torch.float64)     # [5]
        glob_sumsq = torch.zeros(5, dtype=torch.float64)   # [5]

        pair_sum = torch.zeros(12, dtype=torch.float64)    # [12]
        pair_sumsq = torch.zeros(12, dtype=torch.float64)  # [12]

        obj_sum = torch.zeros(3, dtype=torch.float64)      # [3]
        obj_sumsq = torch.zeros(3, dtype=torch.float64)    # [3]
        obj_count = 0.0

        for start in range(0, N, self.chunk_size):
            xb = X[start:start + self.chunk_size]  # [B,92]
            B = int(xb.shape[0])

            met = xb[:, 0]  # [B]
            met_log = torch.log1p((met / 1000.0).clamp_min(0.0))  # [B]
            met_sum += float(met_log.double().sum().item())
            met_sumsq += float((met_log.double() ** 2).sum().item())

            objs = xb[:, 2:].reshape(B, self.max_objects, self.obj_stride)  # [B,18,5]
            obj_id = objs[:, :, 0]  # [B,18]
            E = objs[:, :, 1]      # [B,18]
            pT = objs[:, :, 2]     # [B,18]
            eta = objs[:, :, 3]    # [B,18]
            phi = objs[:, :, 4]    # [B,18]
            obj_mask = obj_id != 0  # [B,18] bool

            # Per-object stats (valid only)
            E_log = torch.log1p((E / 1000.0).clamp_min(0.0))      # [B,18]
            pT_log = torch.log1p((pT / 1000.0).clamp_min(0.0))    # [B,18]

            mask_f = obj_mask.float()
            cnt = float(mask_f.sum().item())
            if cnt > 0:
                v0 = (E_log * mask_f).double().sum().item()
                v1 = (pT_log * mask_f).double().sum().item()
                v2 = (eta * mask_f).double().sum().item()
                obj_sum += torch.tensor([v0, v1, v2], dtype=torch.float64)

                s0 = ((E_log.double() ** 2) * mask_f.double()).sum().item()
                s1 = ((pT_log.double() ** 2) * mask_f.double()).sum().item()
                s2 = ((eta.double() ** 2) * mask_f.double()).sum().item()
                obj_sumsq += torch.tensor([s0, s1, s2], dtype=torch.float64)
                obj_count += cnt

            # Global features
            nobj = obj_mask.sum(dim=1).float()  # [B]
            nobj_frac = nobj / float(self.max_objects)  # [B]
            pT_GeV = pT / 1000.0  # [B,18]
            E_GeV = E / 1000.0    # [B,18]

            ht = (pT_GeV * obj_mask.float()).sum(dim=1)  # [B]
            sumE = (E_GeV * obj_mask.float()).sum(dim=1)  # [B]
            maxpT = pT_GeV.masked_fill(~obj_mask, -1.0).max(dim=1).values.clamp_min(0.0)  # [B]
            meanAbsEta = (eta.abs() * obj_mask.float()).sum(dim=1) / (nobj.clamp_min(1.0))  # [B]
            avg_pT = ht / (nobj.clamp_min(1.0))  # [B]

            ht_log = torch.log1p(ht.clamp_min(0.0))          # [B]
            sumE_log = torch.log1p(sumE.clamp_min(0.0))      # [B]
            maxpT_log = torch.log1p(maxpT.clamp_min(0.0))    # [B]
            avg_pT_log = torch.log1p(avg_pT.clamp_min(0.0))  # [B]

            glob = torch.stack([ht_log, sumE_log, maxpT_log, avg_pT_log, meanAbsEta], dim=1).double()  # [B,5]
            glob_sum += glob.sum(dim=0)
            glob_sumsq += (glob ** 2).sum(dim=0)

            # Pairwise features
            pair = self._compute_pairwise_features(E, pT, eta, phi, obj_mask).double()  # [B,12]
            pair_sum += pair.sum(dim=0)
            pair_sumsq += (pair ** 2).sum(dim=0)

        # finalize stats
        Nf = float(N)
        met_mu = met_sum / Nf
        met_var = met_sumsq / Nf - met_mu * met_mu
        met_sigma = float(max(met_var, 1e-8) ** 0.5)

        glob_mu = (glob_sum / Nf).numpy().astype(np.float32)  # [5]
        glob_var = (glob_sumsq / Nf - (glob_sum / Nf) ** 2).clamp_min(1e-8)
        glob_sigma = torch.sqrt(glob_var).numpy().astype(np.float32)  # [5]

        pair_mu = (pair_sum / Nf).numpy().astype(np.float32)  # [12]
        pair_var = (pair_sumsq / Nf - (pair_sum / Nf) ** 2).clamp_min(1e-8)
        pair_sigma = torch.sqrt(pair_var).numpy().astype(np.float32)  # [12]

        if obj_count <= 0:
            obj_mu = np.zeros(3, dtype=np.float32)
            obj_sigma = np.ones(3, dtype=np.float32)
        else:
            obj_mu_t = (obj_sum / obj_count)
            obj_var_t = (obj_sumsq / obj_count - obj_mu_t ** 2).clamp_min(1e-8)
            obj_mu = obj_mu_t.numpy().astype(np.float32)      # [3]
            obj_sigma = torch.sqrt(obj_var_t).numpy().astype(np.float32)  # [3]

        self.met_mu = float(met_mu)
        self.met_sigma = float(max(met_sigma, 1e-6))
        self.glob_mu = glob_mu
        self.glob_sigma = np.maximum(glob_sigma, 1e-6).astype(np.float32)
        self.pair_mu = pair_mu
        self.pair_sigma = np.maximum(pair_sigma, 1e-6).astype(np.float32)
        self.obj_mu = obj_mu
        self.obj_sigma = np.maximum(obj_sigma, 1e-6).astype(np.float32)

        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            return X
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float32)
        X = X.float().cpu()
        N = int(X.shape[0])

        F_global = 21  # [met(3) + global(6) + pair(12)]
        F_obj = 6      # [id + 5 cont]
        F = F_global + self.max_objects * F_obj  # 21 + 18*6 = 129

        out = torch.empty((N, F), dtype=torch.float32)  # [N,129]

        met_mu = self.met_mu
        met_sigma = self.met_sigma
        glob_mu = torch.as_tensor(self.glob_mu, dtype=torch.float32)      # [5]
        glob_sigma = torch.as_tensor(self.glob_sigma, dtype=torch.float32)  # [5]
        pair_mu = torch.as_tensor(self.pair_mu, dtype=torch.float32)      # [12]
        pair_sigma = torch.as_tensor(self.pair_sigma, dtype=torch.float32)  # [12]
        obj_mu = torch.as_tensor(self.obj_mu, dtype=torch.float32)        # [3]
        obj_sigma = torch.as_tensor(self.obj_sigma, dtype=torch.float32)  # [3]

        for start in range(0, N, self.chunk_size):
            xb = X[start:start + self.chunk_size]  # [B,92]
            B = int(xb.shape[0])

            met = xb[:, 0]  # [B]
            metphi = xb[:, 1]  # [B]
            met_log = torch.log1p((met / 1000.0).clamp_min(0.0))  # [B]
            met_log = (met_log - met_mu) / met_sigma  # [B]
            met_sin = torch.sin(metphi)  # [B]
            met_cos = torch.cos(metphi)  # [B]

            objs = xb[:, 2:].reshape(B, self.max_objects, self.obj_stride)  # [B,18,5]
            obj_id = objs[:, :, 0]  # [B,18]
            E = objs[:, :, 1]      # [B,18]
            pT = objs[:, :, 2]     # [B,18]
            eta = objs[:, :, 3]    # [B,18]
            phi = objs[:, :, 4]    # [B,18]
            obj_mask = obj_id != 0  # [B,18] bool
            mask_f = obj_mask.float()

            # Per-object engineered
            E_log = torch.log1p((E / 1000.0).clamp_min(0.0))   # [B,18]
            pT_log = torch.log1p((pT / 1000.0).clamp_min(0.0)) # [B,18]
            eta_v = eta  # [B,18]
            sinphi = torch.sin(phi)  # [B,18]
            cosphi = torch.cos(phi)  # [B,18]

            # Normalize per-object channels, then mask padding back to 0
            E_log = (E_log - obj_mu[0]) / obj_sigma[0]   # [B,18]
            pT_log = (pT_log - obj_mu[1]) / obj_sigma[1] # [B,18]
            eta_v = (eta_v - obj_mu[2]) / obj_sigma[2]   # [B,18]

            E_log = E_log * mask_f
            pT_log = pT_log * mask_f
            eta_v = eta_v * mask_f
            sinphi = sinphi * mask_f
            cosphi = cosphi * mask_f

            # Global features
            nobj = obj_mask.sum(dim=1).float()  # [B]
            nobj_frac = nobj / float(self.max_objects)  # [B]

            pT_GeV = pT / 1000.0  # [B,18]
            E_GeV = E / 1000.0    # [B,18]
            ht = (pT_GeV * mask_f).sum(dim=1)  # [B]
            sumE = (E_GeV * mask_f).sum(dim=1) # [B]
            maxpT = pT_GeV.masked_fill(~obj_mask, -1.0).max(dim=1).values.clamp_min(0.0)  # [B]
            meanAbsEta = (eta.abs() * mask_f).sum(dim=1) / (nobj.clamp_min(1.0))  # [B]
            avg_pT = ht / (nobj.clamp_min(1.0))  # [B]

            ht_log = torch.log1p(ht.clamp_min(0.0))
            sumE_log = torch.log1p(sumE.clamp_min(0.0))
            maxpT_log = torch.log1p(maxpT.clamp_min(0.0))
            avg_pT_log = torch.log1p(avg_pT.clamp_min(0.0))

            glob = torch.stack([ht_log, sumE_log, maxpT_log, avg_pT_log, meanAbsEta], dim=1)  # [B,5]
            glob = (glob - glob_mu[None, :]) / glob_sigma[None, :]  # [B,5]

            # Pairwise
            pair = self._compute_pairwise_features(E, pT, eta, phi, obj_mask)  # [B,12]
            pair = (pair - pair_mu[None, :]) / pair_sigma[None, :]  # [B,12]

            # Assemble global token features
            # global_vec: [met_log, met_sin, met_cos, nobj_frac, glob(5), pair(12)] => 3+1+5+12 = 21
            global_vec = torch.cat(
                [
                    met_log[:, None], met_sin[:, None], met_cos[:, None],  # [B,3]
                    nobj_frac[:, None],                                    # [B,1]
                    glob,                                                  # [B,5]
                    pair,                                                  # [B,12]
                ],
                dim=1
            )  # [B,21]

            # Assemble object flat: [id, Elog, pTlog, eta, sinphi, cosphi] => [B,18,6]
            obj_feat = torch.stack(
                [
                    obj_id.float(),
                    E_log,
                    pT_log,
                    eta_v,
                    sinphi,
                    cosphi,
                ],
                dim=2
            )  # [B,18,6]

            # Flatten and write
            out_chunk = torch.cat([global_vec, obj_feat.reshape(B, -1)], dim=1)  # [B,129]
            out[start:start + B] = out_chunk

        return out


def make_preprocessor():
    return MyPreprocessor()


class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.max_objects = 18
        self.obj_feat_dim = 6
        self.global_dim = 21
        self.d_model = 96
        self.nhead = 8
        self.nlayers = 4
        self.phi_aug = True

        self.id_vocab = 256
        self.id_emb_dim = 24
        self.id_emb = nn.Embedding(self.id_vocab, self.id_emb_dim, padding_idx=0)

        # global token projection
        self.global_proj = nn.Sequential(
            nn.Linear(self.global_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )

        # object token projection: [id_emb(24) + cont(5)] = 29 -> d_model
        self.obj_proj = nn.Sequential(
            nn.Linear(self.id_emb_dim + 5, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=4 * self.d_model,
            dropout=0.12,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.nlayers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.out_norm = nn.LayerNorm(self.d_model * 2 + self.global_dim)

        self.head = nn.Sequential(
            nn.Linear(self.d_model * 2 + self.global_dim, 256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def _apply_global_phi_rotation(self, global_vec, obj_cont, obj_mask):
        # global_vec: [B,21], obj_cont: [B,18,5] where cont=[Elog,pTlog,eta,sin,cos]
        # Rotate met sin/cos (global_vec[1], global_vec[2]) and obj sin/cos (cont[...,3:5])
        B = global_vec.shape[0]
        theta = (torch.rand(B, device=global_vec.device) * (2.0 * math.pi) - math.pi).to(global_vec.dtype)  # [B]
        c = torch.cos(theta)[:, None]  # [B,1]
        s = torch.sin(theta)[:, None]  # [B,1]

        g = global_vec.clone()
        met_sin = g[:, 1:2]
        met_cos = g[:, 2:3]
        g[:, 1:2] = met_sin * c + met_cos * s
        g[:, 2:3] = met_cos * c - met_sin * s

        oc = obj_cont.clone()
        sinp = oc[:, :, 3]  # [B,18]
        cosp = oc[:, :, 4]  # [B,18]
        # broadcast [B,1] to [B,18]
        sinp2 = sinp * c + cosp * s
        cosp2 = cosp * c - sinp * s
        sinp2 = sinp2 * obj_mask.float()
        cosp2 = cosp2 * obj_mask.float()
        oc[:, :, 3] = sinp2
        oc[:, :, 4] = cosp2

        return g, oc

    def forward(self, batch_x):
        # batch_x: FloatTensor[B,129]
        x = batch_x
        if x.dim() != 2:
            x = x.view(x.shape[0], -1)

        B = x.shape[0]
        global_vec = x[:, :self.global_dim]  # [B,21]
        obj_flat = x[:, self.global_dim:]    # [B,108]
        obj = obj_flat.view(B, self.max_objects, self.obj_feat_dim)  # [B,18,6]

        obj_id = obj[:, :, 0].round().clamp(0, self.id_vocab - 1).long()  # [B,18]
        obj_mask = obj_id != 0  # [B,18]
        obj_cont = obj[:, :, 1:]  # [B,18,5]

        if self.training and self.phi_aug:
            global_vec, obj_cont = self._apply_global_phi_rotation(global_vec, obj_cont, obj_mask)

        id_emb = self.id_emb(obj_id)  # [B,18,24]
        obj_in = torch.cat([id_emb, obj_cont], dim=-1)  # [B,18,29]
        obj_tok = self.obj_proj(obj_in)  # [B,18,96]

        global_tok = self.global_proj(global_vec).unsqueeze(1)  # [B,1,96]
        cls = self.cls_token.expand(B, 1, self.d_model)  # [B,1,96]

        tokens = torch.cat([cls, global_tok, obj_tok], dim=1)  # [B,20,96]
        pad = torch.zeros((B, 2), dtype=torch.bool, device=tokens.device)  # [B,2]
        src_key_padding_mask = torch.cat([pad, ~obj_mask], dim=1)  # [B,20], True=padded

        z = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)  # [B,20,96]
        cls_out = z[:, 0, :]  # [B,96]
        obj_out = z[:, 2:, :]  # [B,18,96]

        # masked mean pool over objects
        mf = obj_mask.float().unsqueeze(-1)  # [B,18,1]
        denom = mf.sum(dim=1).clamp_min(1.0)  # [B,1]
        pooled = (obj_out * mf).sum(dim=1) / denom  # [B,96]

        h = torch.cat([cls_out, pooled, global_vec], dim=1)  # [B,96+96+21=213]
        h = self.out_norm(h)  # [B,213]
        logit = self.head(h).squeeze(1)  # [B]
        return logit


def make_model(example_object):
    return BinaryClassifier(example_object)


EPOCHS = 15


@torch.no_grad()
def _eval_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    all_probs = []
    all_y = []
    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float()
        logits = model(xb)  # [B]
        loss = loss_fn(logits, yb)
        bs = int(yb.shape[0])
        total_loss += float(loss.item()) * bs
        total_n += bs
        probs = torch.sigmoid(logits).detach().float().cpu()
        all_probs.append(probs)
        all_y.append(yb.detach().cpu().long())

    probs = torch.cat(all_probs, dim=0).numpy()
    yy = torch.cat(all_y, dim=0).numpy()
    try:
        auc = float(roc_auc_score(yy, probs))
    except Exception:
        auc = float("nan")

    avg_loss = total_loss / max(total_n, 1)
    pred = (probs >= 0.5).astype(np.int64)
    acc = float((pred == yy).mean())
    return avg_loss, auc, acc


@torch.no_grad()
def _approx_train_auc(model, loader, device, max_batches=40):
    model.eval()
    all_probs = []
    all_y = []
    nb = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.sigmoid(logits).detach().float().cpu()
        all_probs.append(probs)
        all_y.append(yb.detach().cpu().long())
        nb += 1
        if nb >= max_batches:
            break
    probs = torch.cat(all_probs, dim=0).numpy()
    yy = torch.cat(all_y, dim=0).numpy()
    try:
        auc = float(roc_auc_score(yy, probs))
    except Exception:
        auc = float("nan")
    return auc


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-2, betas=(0.9, 0.98), eps=1e-8)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=6.0e-4,
        epochs=max(1, epochs),
        steps_per_epoch=steps_per_epoch,
        pct_start=0.12,
        div_factor=10.0,
        final_div_factor=200.0,
        anneal_strategy="cos",
    )

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_auc = -1.0
    best_state = None
    patience = 4
    bad = 0

    train_loss_hist = []
    val_loss_hist = []
    train_auc_hist = []
    val_auc_hist = []

    for ep in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(xb)  # [B]
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = int(yb.shape[0])
            total_loss += float(loss.item()) * bs
            total_n += bs

        tr_loss = total_loss / max(total_n, 1)
        va_loss, va_auc, va_acc = _eval_epoch(model, val_loader, device)
        tr_auc = _approx_train_auc(model, train_loader, device, max_batches=35)

        train_loss_hist.append(tr_loss)
        val_loss_hist.append(va_loss)
        train_auc_hist.append(tr_auc)
        val_auc_hist.append(va_auc)

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Return AUC histories in acc slots (challenge metric)
    trained_model = model
    train_loss = train_loss_hist
    val_loss = val_loss_hist
    train_acc = train_auc_hist
    val_acc = val_auc_hist
    return trained_model, train_loss, val_loss, train_acc, val_acc

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

