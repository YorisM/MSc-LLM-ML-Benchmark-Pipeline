
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
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import Dataset

# Enable TF32 on CUDA for speed (typically no AUC impact, but helps runtime).
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class MyPreprocessor:
    MAX_OBJECTS = 18
    OBJ_SLICE = 5
    MAX_ID = 16  # bucket all ids > MAX_ID into MAX_ID

    # Output layout (fixed):
    #   global: 3  -> [logMET_norm, sin(phi_MET), cos(phi_MET)]
    #   event : 12 -> normalized event-level engineered features
    #   counts: 16 -> normalized counts fraction per id (1..MAX_ID) / 18
    #   sumpt : 16 -> normalized log1p(sum pT per id)   (pT in GeV)
    #   objects: 18*7 -> [id, 6 cont feats] per object, sorted by pT desc
    # Total F = 3 + 12 + 16 + 16 + 18*7 = 173
    OUT_DIM = 3 + 12 + 16 + 16 + 18 * 7

    def __init__(self, chunk_size: int = 50000):
        self.chunk_size = int(chunk_size)

        # Global
        self.glob_mean = 0.0
        self.glob_std = 1.0

        # Event engineered (12,)
        self.event_mean = np.zeros((12,), dtype=np.float32)
        self.event_std = np.ones((12,), dtype=np.float32)

        # Counts (16,)
        self.count_mean = np.zeros((self.MAX_ID,), dtype=np.float32)
        self.count_std = np.ones((self.MAX_ID,), dtype=np.float32)

        # Sum pT per id (16,)
        self.sumpt_mean = np.zeros((self.MAX_ID,), dtype=np.float32)
        self.sumpt_std = np.ones((self.MAX_ID,), dtype=np.float32)

        # Object cont feats (6,)
        self.obj_mean = np.zeros((6,), dtype=np.float32)
        self.obj_std = np.ones((6,), dtype=np.float32)

    def make_loader_cfg(self) -> dict:
        bs = 2048 if (device.type == "cuda") else 512
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": int(bs),
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": bool(device.type == "cuda"),
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": int(bs)},
        }

    @staticmethod
    def _to_numpy(X):
        if torch.is_tensor(X):
            return X.detach().cpu().numpy()
        return np.asarray(X)

    @staticmethod
    def _wrap_dphi(dphi):
        # Wrap to [-pi, pi]
        return (dphi + np.pi) % (2.0 * np.pi) - np.pi

    def _compute_raw_features(self, Xc: np.ndarray):
        """
        Compute raw engineered features for a chunk.
        Returns dict of numpy arrays:
          logmet: (B,)
          sinmet: (B,)
          cosmet: (B,)
          event:  (B,12)
          counts: (B,16) counts fraction per id / 18
          sumpt:  (B,16) log1p(sum pT per id) (GeV)
          obj_ids_sorted: (B,18) int64 in [0, MAX_ID]
          obj_cont: (B,18,6) float32 raw cont feats: [logE, logpT, eta, sinphi, cosphi, logm]
          mask: (B,18) bool
        """
        B = Xc.shape[0]
        # Globals
        met = np.clip(Xc[:, 0], 0.0, None) / 1e3  # (B,) GeV
        logmet = np.log1p(met).astype(np.float32)  # (B,)
        phi_met = Xc[:, 1].astype(np.float32)  # (B,)
        sinmet = np.sin(phi_met).astype(np.float32)  # (B,)
        cosmet = np.cos(phi_met).astype(np.float32)  # (B,)

        objs = Xc[:, 2:].reshape(B, self.MAX_OBJECTS, self.OBJ_SLICE)  # (B,18,5)

        ids = np.rint(objs[:, :, 0]).astype(np.int64)  # (B,18)
        ids = np.clip(ids, 0, self.MAX_ID)  # (B,18)
        mask = ids > 0  # (B,18) bool

        E = np.clip(objs[:, :, 1], 0.0, None).astype(np.float32) / 1e3  # (B,18) GeV
        pT = np.clip(objs[:, :, 2], 0.0, None).astype(np.float32) / 1e3  # (B,18) GeV
        eta = objs[:, :, 3].astype(np.float32)  # (B,18)
        phi = objs[:, :, 4].astype(np.float32)  # (B,18)

        # Sort objects by descending pT, paddings at end
        pT_key = np.where(mask, pT, -1.0).astype(np.float32)  # (B,18)
        idx = np.argsort(-pT_key, axis=1, kind="mergesort")  # (B,18)
        ids = np.take_along_axis(ids, idx, axis=1)  # (B,18)
        mask = ids > 0  # (B,18)
        E = np.take_along_axis(E, idx, axis=1)  # (B,18)
        pT = np.take_along_axis(pT, idx, axis=1)  # (B,18)
        eta = np.take_along_axis(eta, idx, axis=1)  # (B,18)
        phi = np.take_along_axis(phi, idx, axis=1)  # (B,18)

        # Object continuous features
        logE = np.log1p(E).astype(np.float32)  # (B,18)
        logpT = np.log1p(pT).astype(np.float32)  # (B,18)
        sinphi = np.sin(phi).astype(np.float32)  # (B,18)
        cosphi = np.cos(phi).astype(np.float32)  # (B,18)

        # mass proxy: m^2 = E^2 - |p|^2, with |p| = pT*cosh(eta)
        # Use float32 but guard overflows with float64 intermediate for cosh/sinh if needed.
        cosh_eta = np.cosh(eta.astype(np.float64)).astype(np.float32)  # (B,18)
        p = (pT * cosh_eta).astype(np.float32)  # (B,18)
        m2 = np.maximum(E * E - p * p, 0.0).astype(np.float32)  # (B,18)
        logm = np.log1p(np.sqrt(m2).astype(np.float32)).astype(np.float32)  # (B,18)

        obj_cont = np.stack([logE, logpT, eta, sinphi, cosphi, logm], axis=2).astype(np.float32)  # (B,18,6)

        # Event engineered features (12,)
        n_obj = mask.sum(axis=1).astype(np.float32)  # (B,)
        ht = (pT * mask.astype(np.float32)).sum(axis=1).astype(np.float32)  # (B,)
        sumE = (E * mask.astype(np.float32)).sum(axis=1).astype(np.float32)  # (B,)
        maxpt = np.max(np.where(mask, pT, 0.0), axis=1).astype(np.float32)  # (B,)

        mean_abs_eta = (np.abs(eta).astype(np.float32) * mask.astype(np.float32)).sum(axis=1) / np.maximum(n_obj, 1.0)  # (B,)
        met_over_sqrt_ht = (met / np.sqrt(ht + 1e-3)).astype(np.float32)  # (B,)
        centrality = (ht / (sumE + 1e-6)).astype(np.float32)  # (B,)

        dphi = self._wrap_dphi(phi - phi_met[:, None].astype(np.float32)).astype(np.float32)  # (B,18)
        abs_dphi = np.abs(dphi).astype(np.float32)  # (B,18)
        abs_dphi_masked = np.where(mask, abs_dphi, np.pi).astype(np.float32)  # (B,18)
        min_abs_dphi = np.min(abs_dphi_masked, axis=1).astype(np.float32)  # (B,)
        mean_abs_dphi = (abs_dphi * mask.astype(np.float32)).sum(axis=1) / np.maximum(n_obj, 1.0)  # (B,)

        abs_dphi_lead = abs_dphi_masked[:, 0].astype(np.float32)  # (B,)

        have2 = (n_obj >= 2.0)  # (B,) bool

        # Pair features for top-2 by pT
        # deltaR12
        dphi12 = self._wrap_dphi(phi[:, 0] - phi[:, 1]).astype(np.float32)  # (B,)
        deta12 = (eta[:, 0] - eta[:, 1]).astype(np.float32)  # (B,)
        deltaR12 = np.sqrt(deta12 * deta12 + dphi12 * dphi12).astype(np.float32)  # (B,)
        deltaR12 = np.where(have2, deltaR12, 0.0).astype(np.float32)  # (B,)

        # m12 (using E and p derived from pT,phi,eta)
        # pz = pT*sinh(eta)
        sinh_eta0 = np.sinh(eta[:, 0].astype(np.float64)).astype(np.float32)
        sinh_eta1 = np.sinh(eta[:, 1].astype(np.float64)).astype(np.float32)
        px0 = (pT[:, 0] * np.cos(phi[:, 0])).astype(np.float32)
        py0 = (pT[:, 0] * np.sin(phi[:, 0])).astype(np.float32)
        pz0 = (pT[:, 0] * sinh_eta0).astype(np.float32)
        px1 = (pT[:, 1] * np.cos(phi[:, 1])).astype(np.float32)
        py1 = (pT[:, 1] * np.sin(phi[:, 1])).astype(np.float32)
        pz1 = (pT[:, 1] * sinh_eta1).astype(np.float32)

        E12 = (E[:, 0] + E[:, 1]).astype(np.float32)
        px12 = (px0 + px1).astype(np.float32)
        py12 = (py0 + py1).astype(np.float32)
        pz12 = (pz0 + pz1).astype(np.float32)
        m12_2 = np.maximum(E12 * E12 - (px12 * px12 + py12 * py12 + pz12 * pz12), 0.0).astype(np.float32)
        log_m12 = np.log1p(np.sqrt(m12_2).astype(np.float32)).astype(np.float32)
        log_m12 = np.where(have2, log_m12, 0.0).astype(np.float32)

        event = np.stack(
            [
                np.log1p(ht).astype(np.float32),          # 0 logHT
                np.log1p(sumE).astype(np.float32),        # 1 logSumE
                np.log1p(maxpt).astype(np.float32),       # 2 logMaxpT
                n_obj.astype(np.float32),                 # 3 nObj
                mean_abs_eta.astype(np.float32),          # 4 mean|eta|
                met_over_sqrt_ht.astype(np.float32),      # 5 MET/sqrt(HT)
                centrality.astype(np.float32),            # 6 HT/sumE
                min_abs_dphi.astype(np.float32),          # 7 min|dphi(obj,MET)|
                mean_abs_dphi.astype(np.float32),         # 8 mean|dphi(obj,MET)|
                abs_dphi_lead.astype(np.float32),         # 9 |dphi(lead,MET)|
                log_m12.astype(np.float32),               # 10 log m12(lead,sublead)
                deltaR12.astype(np.float32),              # 11 deltaR12(lead,sublead)
            ],
            axis=1,
        ).astype(np.float32)  # (B,12)

        # Per-id counts and sum pT
        counts = np.zeros((B, self.MAX_ID), dtype=np.float32)  # (B,16)
        sumpt = np.zeros((B, self.MAX_ID), dtype=np.float32)  # (B,16)
        # Note: ids in [0, MAX_ID], where MAX_ID includes overflow bucket.
        for k in range(1, self.MAX_ID + 1):
            m = (ids == k)  # (B,18)
            counts[:, k - 1] = m.sum(axis=1).astype(np.float32)
            sumpt[:, k - 1] = (pT * m.astype(np.float32)).sum(axis=1).astype(np.float32)

        counts_frac = (counts / float(self.MAX_OBJECTS)).astype(np.float32)  # (B,16)
        sumpt_log = np.log1p(sumpt).astype(np.float32)  # (B,16)

        return {
            "logmet": logmet,                 # (B,)
            "sinmet": sinmet,                 # (B,)
            "cosmet": cosmet,                 # (B,)
            "event": event,                   # (B,12)
            "counts": counts_frac,            # (B,16)
            "sumpt": sumpt_log,               # (B,16)
            "obj_ids_sorted": ids,            # (B,18) int64
            "obj_cont": obj_cont,             # (B,18,6)
            "mask": mask,                     # (B,18) bool
        }

    def fit(self, X, y=None):
        Xn = self._to_numpy(X).astype(np.float32, copy=False)
        N = Xn.shape[0]

        # Accumulators in float64 for stability
        glob_sum = 0.0
        glob_sumsq = 0.0
        glob_n = 0

        event_sum = np.zeros((12,), dtype=np.float64)
        event_sumsq = np.zeros((12,), dtype=np.float64)
        event_n = 0

        count_sum = np.zeros((self.MAX_ID,), dtype=np.float64)
        count_sumsq = np.zeros((self.MAX_ID,), dtype=np.float64)
        count_n = 0

        sumpt_sum = np.zeros((self.MAX_ID,), dtype=np.float64)
        sumpt_sumsq = np.zeros((self.MAX_ID,), dtype=np.float64)
        sumpt_n = 0

        obj_sum = np.zeros((6,), dtype=np.float64)
        obj_sumsq = np.zeros((6,), dtype=np.float64)
        obj_n = 0

        for s in range(0, N, self.chunk_size):
            e = min(N, s + self.chunk_size)
            feats = self._compute_raw_features(Xn[s:e])

            logmet = feats["logmet"].astype(np.float64)  # (B,)
            glob_sum += float(logmet.sum())
            glob_sumsq += float((logmet * logmet).sum())
            glob_n += int(logmet.shape[0])

            event = feats["event"].astype(np.float64)  # (B,12)
            event_sum += event.sum(axis=0)
            event_sumsq += (event * event).sum(axis=0)
            event_n += int(event.shape[0])

            counts = feats["counts"].astype(np.float64)  # (B,16)
            count_sum += counts.sum(axis=0)
            count_sumsq += (counts * counts).sum(axis=0)
            count_n += int(counts.shape[0])

            sumpt = feats["sumpt"].astype(np.float64)  # (B,16)
            sumpt_sum += sumpt.sum(axis=0)
            sumpt_sumsq += (sumpt * sumpt).sum(axis=0)
            sumpt_n += int(sumpt.shape[0])

            obj_cont = feats["obj_cont"]  # (B,18,6)
            mask = feats["mask"]  # (B,18)
            obj_flat = obj_cont.reshape(-1, 6).astype(np.float64)  # (B*18,6)
            mask_flat = mask.reshape(-1)  # (B*18,)
            if mask_flat.any():
                valid = obj_flat[mask_flat]  # (M,6)
                obj_sum += valid.sum(axis=0)
                obj_sumsq += (valid * valid).sum(axis=0)
                obj_n += int(valid.shape[0])

        # Compute means/stds
        glob_mean = glob_sum / max(glob_n, 1)
        glob_var = glob_sumsq / max(glob_n, 1) - glob_mean * glob_mean
        glob_std = math.sqrt(max(glob_var, 1e-8))

        self.glob_mean = float(glob_mean)
        self.glob_std = float(glob_std if glob_std > 1e-6 else 1.0)

        event_mean = event_sum / max(event_n, 1)
        event_var = event_sumsq / max(event_n, 1) - event_mean * event_mean
        event_std = np.sqrt(np.maximum(event_var, 1e-8))

        self.event_mean = event_mean.astype(np.float32)
        self.event_std = np.where(event_std.astype(np.float32) > 1e-6, event_std.astype(np.float32), 1.0).astype(np.float32)

        count_mean = count_sum / max(count_n, 1)
        count_var = count_sumsq / max(count_n, 1) - count_mean * count_mean
        count_std = np.sqrt(np.maximum(count_var, 1e-8))

        self.count_mean = count_mean.astype(np.float32)
        self.count_std = np.where(count_std.astype(np.float32) > 1e-6, count_std.astype(np.float32), 1.0).astype(np.float32)

        sumpt_mean = sumpt_sum / max(sumpt_n, 1)
        sumpt_var = sumpt_sumsq / max(sumpt_n, 1) - sumpt_mean * sumpt_mean
        sumpt_std = np.sqrt(np.maximum(sumpt_var, 1e-8))

        self.sumpt_mean = sumpt_mean.astype(np.float32)
        self.sumpt_std = np.where(sumpt_std.astype(np.float32) > 1e-6, sumpt_std.astype(np.float32), 1.0).astype(np.float32)

        obj_mean = obj_sum / max(obj_n, 1)
        obj_var = obj_sumsq / max(obj_n, 1) - obj_mean * obj_mean
        obj_std = np.sqrt(np.maximum(obj_var, 1e-8))

        self.obj_mean = obj_mean.astype(np.float32)
        self.obj_std = np.where(obj_std.astype(np.float32) > 1e-6, obj_std.astype(np.float32), 1.0).astype(np.float32)

        return self

    def transform(self, X):
        Xn = self._to_numpy(X).astype(np.float32, copy=False)
        N = Xn.shape[0]
        out = np.zeros((N, self.OUT_DIM), dtype=np.float32)  # (N,173)

        for s in range(0, N, self.chunk_size):
            e = min(N, s + self.chunk_size)
            feats = self._compute_raw_features(Xn[s:e])
            B = e - s

            logmet = feats["logmet"]  # (B,)
            sinmet = feats["sinmet"]  # (B,)
            cosmet = feats["cosmet"]  # (B,)
            event = feats["event"]    # (B,12)
            counts = feats["counts"]  # (B,16)
            sumpt = feats["sumpt"]    # (B,16)

            obj_ids = feats["obj_ids_sorted"].astype(np.float32)  # (B,18)
            obj_cont = feats["obj_cont"]  # (B,18,6)
            mask = feats["mask"]  # (B,18)

            # Normalize
            logmet_n = ((logmet - self.glob_mean) / self.glob_std).astype(np.float32)  # (B,)
            event_n = ((event - self.event_mean[None, :]) / self.event_std[None, :]).astype(np.float32)  # (B,12)
            counts_n = ((counts - self.count_mean[None, :]) / self.count_std[None, :]).astype(np.float32)  # (B,16)
            sumpt_n = ((sumpt - self.sumpt_mean[None, :]) / self.sumpt_std[None, :]).astype(np.float32)  # (B,16)
            obj_cont_n = ((obj_cont - self.obj_mean[None, None, :]) / self.obj_std[None, None, :]).astype(np.float32)  # (B,18,6)

            # Zero-out padded objects
            obj_cont_n[~mask] = 0.0
            obj_ids = np.where(mask, obj_ids, 0.0).astype(np.float32)  # (B,18)

            # Pack object block
            obj_block = np.concatenate([obj_ids[:, :, None], obj_cont_n], axis=2).astype(np.float32)  # (B,18,7)
            obj_flat = obj_block.reshape(B, self.MAX_OBJECTS * 7).astype(np.float32)  # (B,126)

            # Assemble final vector: 3 + 12 + 16 + 16 + 126 = 173
            # global3
            out[s:e, 0] = logmet_n
            out[s:e, 1] = sinmet
            out[s:e, 2] = cosmet
            # event12
            out[s:e, 3:15] = event_n
            # counts16
            out[s:e, 15:31] = counts_n
            # sumpt16
            out[s:e, 31:47] = sumpt_n
            # objects126
            out[s:e, 47:173] = obj_flat

        return out


def make_preprocessor():
    return MyPreprocessor(chunk_size=50000)


class BinaryClassifier(nn.Module):
    MAX_OBJECTS = 18
    MAX_ID = MyPreprocessor.MAX_ID

    def __init__(self, sample_object):
        super().__init__()
        in_dim = int(sample_object.shape[1])

        # Expected: 173 features from our preprocessor
        # Layout indices:
        #   [0:3]      global3
        #   [3:15]     event12
        #   [15:31]    counts16
        #   [31:47]    sumpt16
        #   [47:173]   objects (18*7)
        self.global_dim = 3
        self.event_dim = 12
        self.count_dim = 16
        self.sumpt_dim = 16
        self.obj_token_dim = 7
        self.obj_cont_dim = 6
        self.obj_flat_dim = self.MAX_OBJECTS * self.obj_token_dim  # 126
        self.expected_in_dim = 3 + 12 + 16 + 16 + self.obj_flat_dim  # 173

        # If input dim differs (shouldn't), adapt by assuming objects are at end and fixed size.
        # Keep robust without crashing.
        self.input_dim = in_dim
        self.obj_start = max(0, in_dim - self.obj_flat_dim)

        emb_dim = 12
        cont_hidden = 48
        d_model = 64

        self.id_emb = nn.Embedding(32, emb_dim)  # ids in [0..16], extra room
        self.cont_mlp = nn.Sequential(
            nn.Linear(self.obj_cont_dim, cont_hidden),
            nn.SiLU(),
            nn.Linear(cont_hidden, cont_hidden),
            nn.SiLU(),
            nn.LayerNorm(cont_hidden),
        )
        self.tok_proj = nn.Sequential(
            nn.Linear(emb_dim + cont_hidden, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.pos_emb = nn.Embedding(self.MAX_OBJECTS, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=192,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.attn_q = nn.Parameter(torch.randn(d_model) * 0.02)

        # Head: pooled (3*d_model) + (global/event/count/sumpt dims if present)
        # We will concatenate whatever non-object prefix exists.
        self.prefix_dim = self.obj_start
        head_in = 3 * d_model + self.prefix_dim  # (B, 3d + prefix)

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 192),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(96, 1),
        )

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch_x):
        # batch_x: FloatTensor[B, F]
        x = batch_x
        B = x.shape[0]

        prefix = x[:, : self.obj_start]  # FloatTensor[B, prefix_dim]

        obj_flat = x[:, self.obj_start : self.obj_start + self.obj_flat_dim]  # FloatTensor[B,126]
        obj = obj_flat.view(B, self.MAX_OBJECTS, self.obj_token_dim)  # FloatTensor[B,18,7]

        ids = obj[:, :, 0].round().long().clamp(min=0, max=self.MAX_ID)  # LongTensor[B,18]
        cont = obj[:, :, 1:]  # FloatTensor[B,18,6]

        mask = ids > 0  # BoolTensor[B,18]
        n = mask.sum(dim=1)  # LongTensor[B]
        if (n == 0).any():
            # Ensure at least one token to avoid NaNs in masked softmax/pooling
            mask = mask.clone()
            mask[n == 0, 0] = True

        id_emb = self.id_emb(ids)  # FloatTensor[B,18,emb_dim]
        cont_h = self.cont_mlp(cont)  # FloatTensor[B,18,cont_hidden]
        tok = torch.cat([id_emb, cont_h], dim=-1)  # FloatTensor[B,18,emb_dim+cont_hidden]
        tok = self.tok_proj(tok)  # FloatTensor[B,18,d_model]

        pos = self.pos_emb(torch.arange(self.MAX_OBJECTS, device=x.device))  # FloatTensor[18,d_model]
        tok = tok + pos.unsqueeze(0)  # FloatTensor[B,18,d_model]

        tok = self.transformer(tok, src_key_padding_mask=~mask)  # FloatTensor[B,18,d_model]

        mask_f = mask.float().unsqueeze(-1)  # FloatTensor[B,18,1]
        denom = mask_f.sum(dim=1).clamp(min=1.0)  # FloatTensor[B,1]
        mean_pool = (tok * mask_f).sum(dim=1) / denom  # FloatTensor[B,d_model]

        tok_masked = tok.masked_fill(~mask.unsqueeze(-1), -1e9)  # FloatTensor[B,18,d_model]
        max_pool = tok_masked.max(dim=1).values  # FloatTensor[B,d_model]
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        scores = (tok * self.attn_q.view(1, 1, -1)).sum(dim=-1) / math.sqrt(tok.shape[-1])  # FloatTensor[B,18]
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)  # FloatTensor[B,18,1]
        attn_pool = (tok * w).sum(dim=1)  # FloatTensor[B,d_model]

        pooled = torch.cat([mean_pool, max_pool, attn_pool], dim=1)  # FloatTensor[B,3*d_model]
        feats = torch.cat([pooled, prefix], dim=1)  # FloatTensor[B, 3*d_model + prefix_dim]

        out = self.head(feats).squeeze(-1)  # FloatTensor[B]
        return out


def make_model(example_object):
    return BinaryClassifier(example_object)


EPOCHS = 12


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    model = model.to(device)

    # Optimizer / schedule
    base_lr = 3e-4 if device.type == "cuda" else 2.5e-4
    max_lr = 1.2e-3 if device.type == "cuda" else 6.0e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=8e-4)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=max(1, epochs),
        steps_per_epoch=steps_per_epoch,
        pct_start=0.12,
        div_factor=max_lr / max(base_lr, 1e-8),
        final_div_factor=40.0,
        anneal_strategy="cos",
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    criterion = nn.BCEWithLogitsLoss()

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    best_auc = -1.0
    best_state = None
    patience = 4
    bad_epochs = 0

    for epoch in range(int(epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()  # FloatTensor[B]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(Xb)  # FloatTensor[B]
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.detach().item()) * int(yb.shape[0])
            probs = torch.sigmoid(logits.detach())
            preds = (probs >= 0.5).long()
            correct += int((preds == yb.long()).sum().item())
            total += int(yb.shape[0])

        tr_loss = running_loss / max(total, 1)
        tr_acc = correct / max(total, 1)
        train_loss_hist.append(float(tr_loss))
        train_acc_hist.append(float(tr_acc))

        # Validation
        model.eval()
        v_running_loss = 0.0
        v_correct = 0
        v_total = 0
        all_logits = []
        all_y = []

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float()

                logits = model(Xb)  # FloatTensor[B]
                loss = criterion(logits, yb)

                v_running_loss += float(loss.detach().item()) * int(yb.shape[0])
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                v_correct += int((preds == yb.long()).sum().item())
                v_total += int(yb.shape[0])

                all_logits.append(logits.detach().float().cpu())
                all_y.append(yb.detach().float().cpu())

        va_loss = v_running_loss / max(v_total, 1)
        va_acc = v_correct / max(v_total, 1)
        val_loss_hist.append(float(va_loss))
        val_acc_hist.append(float(va_acc))

        # AUC for early stopping (ranking metric)
        try:
            y_true = torch.cat(all_y).numpy()
            y_score = torch.sigmoid(torch.cat(all_logits)).numpy()
            va_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            va_auc = float("nan")

        if np.isfinite(va_auc) and (va_auc > best_auc + 5e-5):
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
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

