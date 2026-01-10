
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
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.utils.data import Dataset


# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    """
    Produces a fixed-length dense vector per event, designed for a permutation-invariant set model.

    Output layout (Float32), total dim = 246:
      - Global features: G = 48
      - Object features: MAX_OBJS=18, per-object DOBJ=11 => 198
      => 48 + 18*11 = 246

    Per-object (11):
      [0] present (0/1)
      [1] type_index (0..K), kept as float but used as int in the model
      [2:11] 9 continuous features (standardized; multiplied by present to keep padding at 0)
    """

    MAX_OBJS = 18
    SLICE = 5
    K = 16  # mapped categories 1..K (K includes "other"), 0 is padding
    BASE_GLOBAL = 16  # fixed
    G = BASE_GLOBAL + 2 * K  # 16 + 32 = 48
    OBJ_CONT = 9
    DOBJ = 2 + OBJ_CONT  # present + type_idx + 9 cont = 11
    OUT_DIM = G + MAX_OBJS * DOBJ  # 48 + 18*11 = 246

    def __init__(self):
        self.top_ids_ = None                 # list[int] length <= (K-1)
        self.global_mean_ = None             # (G,)
        self.global_std_ = None              # (G,)
        self.obj_cont_mean_ = None           # (9,)
        self.obj_cont_std_ = None            # (9,)
        self._pin_memory = bool(torch.cuda.is_available())

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": self._pin_memory,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 2048},
        }

    @staticmethod
    def _to_numpy(X):
        if torch.is_tensor(X):
            return X.detach().cpu().numpy()
        return np.asarray(X)

    @staticmethod
    def _wrap_phi(dphi):
        # returns in [-pi, pi]
        return np.arctan2(np.sin(dphi), np.cos(dphi))

    def fit(self, X, y=None):
        Xn = self._to_numpy(X).astype(np.float32, copy=False)  # (N, 92)
        N = Xn.shape[0]

        objs = Xn[:, 2:].reshape(N, self.MAX_OBJS, self.SLICE)  # (N,18,5)
        raw_id = objs[:, :, 0].astype(np.int64, copy=False)     # (N,18)
        mask = raw_id != 0

        # Choose top (K-1) raw IDs by frequency (exclude 0)
        nonzero = raw_id[mask]
        if nonzero.size == 0:
            self.top_ids_ = []
        else:
            u, c = np.unique(nonzero, return_counts=True)
            order = np.argsort(-c)
            u = u[order]
            self.top_ids_ = [int(v) for v in u[: max(0, self.K - 1)]]

        # Build engineered features (unscaled) to compute stats
        global_raw, obj_type_idx, obj_cont_raw, obj_present = self._engineer_parts(Xn)  # shapes: (N,G), (N,18), (N,18,9), (N,18)

        # Global stats over events
        gmean = np.mean(global_raw, axis=0, dtype=np.float64).astype(np.float32)  # (G,)
        gstd = np.std(global_raw, axis=0, dtype=np.float64).astype(np.float32)   # (G,)
        gstd = np.maximum(gstd, 1e-6).astype(np.float32)

        # Object cont stats over present objects only
        cont = obj_cont_raw.copy()  # (N,18,9)
        cont[obj_present < 0.5] = np.nan
        cm = np.nanmean(cont.reshape(-1, self.OBJ_CONT), axis=0).astype(np.float32)  # (9,)
        cs = np.nanstd(cont.reshape(-1, self.OBJ_CONT), axis=0).astype(np.float32)  # (9,)
        cs = np.maximum(cs, 1e-6).astype(np.float32)

        # Replace NaN means/stds if pathological
        cm = np.where(np.isfinite(cm), cm, 0.0).astype(np.float32)
        cs = np.where(np.isfinite(cs), cs, 1.0).astype(np.float32)

        self.global_mean_ = gmean
        self.global_std_ = gstd
        self.obj_cont_mean_ = cm
        self.obj_cont_std_ = cs
        return self

    def _map_ids(self, raw_id, present_mask):
        # raw_id: (N,18) int64, present_mask: bool (N,18)
        mapped = np.zeros_like(raw_id, dtype=np.int64)  # 0 padding
        if self.top_ids_:
            for i, rid in enumerate(self.top_ids_, start=1):
                mapped[raw_id == rid] = i
        # Other: any nonzero not mapped -> K
        other = present_mask & (mapped == 0)
        mapped[other] = self.K
        return mapped  # (N,18) in [0..K]

    def _engineer_parts(self, Xn):
        """
        Returns:
          global_raw: (N,G) float32
          obj_type_idx: (N,18) int64 in [0..K]
          obj_cont_raw: (N,18,9) float32 (unscaled), padded entries 0
          obj_present: (N,18) float32 in {0,1}
        """
        N = Xn.shape[0]
        glob_et = Xn[:, 0].astype(np.float32, copy=False)     # (N,)
        glob_phi = Xn[:, 1].astype(np.float32, copy=False)    # (N,)

        objs = Xn[:, 2:].reshape(N, self.MAX_OBJS, self.SLICE)  # (N,18,5)
        raw_id = objs[:, :, 0].astype(np.int64, copy=False)     # (N,18)
        E = objs[:, :, 1].astype(np.float32, copy=False)        # (N,18)
        pT = objs[:, :, 2].astype(np.float32, copy=False)       # (N,18)
        eta = objs[:, :, 3].astype(np.float32, copy=False)      # (N,18)
        phi = objs[:, :, 4].astype(np.float32, copy=False)      # (N,18)

        present = (raw_id != 0)
        obj_present = present.astype(np.float32)                # (N,18)
        obj_type_idx = self._map_ids(raw_id, present)           # (N,18)

        # Kinematics
        pT_pos = np.clip(pT, 0.0, None)
        E_pos = np.clip(E, 0.0, None)

        logE = np.log1p(E_pos)                                  # (N,18)
        logpT = np.log1p(pT_pos)                                # (N,18)
        sinphi = np.sin(phi)                                    # (N,18)
        cosphi = np.cos(phi)                                    # (N,18)

        phi_m = glob_phi[:, None]                               # (N,1)
        dphi = self._wrap_phi(phi - phi_m)                      # (N,18)
        sindphi = np.sin(dphi)                                  # (N,18)
        cosdphi = np.cos(dphi)                                  # (N,18)

        # Derived mass: m^2 = E^2 - p^2, with p = pT * cosh(eta)
        # Use float64 intermediate for stability, then float32
        p = (pT_pos.astype(np.float64) * np.cosh(eta.astype(np.float64)))          # (N,18)
        m2 = E_pos.astype(np.float64) ** 2 - p ** 2                                # (N,18)
        mass = np.sqrt(np.clip(m2, 0.0, None)).astype(np.float32)                  # (N,18)
        logmass = np.log1p(mass)                                                   # (N,18)

        # Signed log pz
        pz = (pT_pos.astype(np.float64) * np.sinh(eta.astype(np.float64))).astype(np.float32)  # (N,18)
        logpz = np.sign(pz) * np.log1p(np.abs(pz))                                 # (N,18)

        # Pack object continuous features (9): (N,18,9)
        obj_cont_raw = np.stack(
            [logE, logpT, eta, sinphi, cosphi, logmass, logpz, sindphi, cosdphi],
            axis=-1,
        ).astype(np.float32, copy=False)  # (N,18,9)

        # Zero-out padding cont features
        obj_cont_raw *= obj_present[:, :, None]  # (N,18,9)

        # -------- Global engineered features --------
        # Base global (16)
        log_etmiss = np.log1p(np.clip(glob_et, 0.0, None))       # (N,)
        sin_phi_m = np.sin(glob_phi)                             # (N,)
        cos_phi_m = np.cos(glob_phi)                             # (N,)

        nobj = obj_present.sum(axis=1).astype(np.float32)        # (N,)

        pT_masked = pT_pos * obj_present                         # (N,18)
        E_masked = E_pos * obj_present                           # (N,18)

        HT = pT_masked.sum(axis=1).astype(np.float32)            # (N,)
        sumE = E_masked.sum(axis=1).astype(np.float32)           # (N,)

        # Top-2 pT
        # sort along axis=1 (18 elements) is fine
        sorted_pT = np.sort(pT_masked, axis=1)                   # (N,18)
        maxpT = sorted_pT[:, -1].astype(np.float32)              # (N,)
        secondpT = sorted_pT[:, -2].astype(np.float32)           # (N,)

        logHT = np.log1p(HT)
        logSumE = np.log1p(sumE)
        logMaxpT = np.log1p(maxpT)
        logSecondpT = np.log1p(secondpT)

        # eta moments
        denom = np.maximum(nobj, 1.0).astype(np.float32)         # (N,)
        mean_eta = (eta * obj_present).sum(axis=1) / denom       # (N,)
        mean_eta2 = ((eta * eta) * obj_present).sum(axis=1) / denom
        var_eta = np.maximum(mean_eta2 - mean_eta * mean_eta, 0.0).astype(np.float32)
        std_eta = np.sqrt(var_eta).astype(np.float32)

        # pT vector sum
        pX = (pT_masked * cosphi).sum(axis=1).astype(np.float32)  # (N,)
        pY = (pT_masked * sinphi).sum(axis=1).astype(np.float32)  # (N,)
        ptvec_mag = np.sqrt(pX * pX + pY * pY).astype(np.float32)  # (N,)
        log_ptvec_mag = np.log1p(ptvec_mag)

        ptvec_phi = np.arctan2(pY.astype(np.float64), pX.astype(np.float64)).astype(np.float32)  # (N,)
        dphi_ptvec_met = self._wrap_phi(ptvec_phi - glob_phi)  # (N,)
        cos_dphi_ptvec_met = np.cos(dphi_ptvec_met).astype(np.float32)
        sin_dphi_ptvec_met = np.sin(dphi_ptvec_met).astype(np.float32)

        log_ratio = (log_etmiss - logHT).astype(np.float32)

        # pT-weighted average direction relative to MET
        HT_safe = np.maximum(HT, 1.0).astype(np.float32)
        sumCosDphi = ((pT_masked * cosdphi).sum(axis=1) / HT_safe).astype(np.float32)  # (N,)
        sumSinDphi = ((pT_masked * sindphi).sum(axis=1) / HT_safe).astype(np.float32)  # (N,)

        base_global = np.stack(
            [
                log_etmiss,          # 0
                sin_phi_m,           # 1
                cos_phi_m,           # 2
                nobj,                # 3
                logHT,               # 4
                logSumE,             # 5
                logMaxpT,            # 6
                logSecondpT,         # 7
                mean_eta.astype(np.float32),  # 8
                std_eta,             # 9
                log_ptvec_mag,       # 10
                cos_dphi_ptvec_met,  # 11
                sin_dphi_ptvec_met,  # 12
                log_ratio,           # 13
                sumCosDphi,          # 14
                sumSinDphi,          # 15
            ],
            axis=1,
        ).astype(np.float32)  # (N,16)

        # Per-type counts and sum pT (K each): counts (N,K), log1p(sum pT) (N,K)
        counts = np.zeros((N, self.K), dtype=np.float32)
        sumpt = np.zeros((N, self.K), dtype=np.float32)
        for t in range(1, self.K + 1):
            m = (obj_type_idx == t) & present  # (N,18)
            c = m.sum(axis=1).astype(np.float32)              # (N,)
            s = (pT_pos * m.astype(np.float32)).sum(axis=1)   # (N,)
            counts[:, t - 1] = c
            sumpt[:, t - 1] = np.log1p(s.astype(np.float32))

        global_raw = np.concatenate([base_global, counts, sumpt], axis=1)  # (N,48)
        return global_raw, obj_type_idx, obj_cont_raw, obj_present

    def transform(self, X):
        if self.global_mean_ is None:
            raise RuntimeError("MyPreprocessor must be fitted before calling transform().")

        Xn = self._to_numpy(X).astype(np.float32, copy=False)
        N = Xn.shape[0]

        global_raw, obj_type_idx, obj_cont_raw, obj_present = self._engineer_parts(Xn)
        # Standardize global: (N,G)
        global_scaled = (global_raw - self.global_mean_[None, :]) / self.global_std_[None, :]  # (N,48)

        # Standardize object continuous features (present-only stats), keep padding at 0:
        # obj_cont_raw: (N,18,9)
        cont_scaled = (obj_cont_raw - self.obj_cont_mean_[None, None, :]) / self.obj_cont_std_[None, None, :]  # (N,18,9)
        cont_scaled *= obj_present[:, :, None]  # (N,18,9), padding stays 0

        # Build object feature tensor: (N,18,11)
        # [present, type_idx, cont(9)]
        obj_feat = np.concatenate(
            [
                obj_present[:, :, None].astype(np.float32),                     # (N,18,1)
                obj_type_idx[:, :, None].astype(np.float32),                    # (N,18,1)
                cont_scaled.astype(np.float32, copy=False),                     # (N,18,9)
            ],
            axis=-1,
        ).astype(np.float32, copy=False)  # (N,18,11)

        out = np.concatenate([global_scaled, obj_feat.reshape(N, -1)], axis=1)  # (N,246)
        if out.shape[1] != self.OUT_DIM:
            raise RuntimeError(f"Unexpected transformed dim {out.shape[1]} != {self.OUT_DIM}")
        return torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))


def make_preprocessor():
    return MyPreprocessor()


# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    MAX_OBJS = 18
    K = 16
    G = 48
    DOBJ = 11
    OBJ_CONT = 9

    def __init__(self, sample_object):
        super().__init__()
        in_dim = int(sample_object.shape[-1])
        if in_dim != (self.G + self.MAX_OBJS * self.DOBJ):
            raise ValueError(f"Expected input dim {self.G + self.MAX_OBJS * self.DOBJ}, got {in_dim}")

        self.type_emb_dim = 8
        self.model_dim = 96

        self.type_emb = nn.Embedding(self.K + 1, self.type_emb_dim, padding_idx=0)

        obj_in_dim = 1 + self.type_emb_dim + self.OBJ_CONT  # present + type_emb + cont => 1+8+9=18
        self.obj_proj = nn.Sequential(
            nn.Linear(obj_in_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=4,
            dim_feedforward=4 * self.model_dim,  # 384
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.obj_encoder = nn.TransformerEncoder(enc_layer, num_layers=3)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))

        self.global_mlp = nn.Sequential(
            nn.Linear(self.G, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.GELU(),
        )

        head_in = self.model_dim * 4  # global + cls + mean + max
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.type_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.type_emb.weight[0].fill_(0.0)

    def forward(self, batch_x):
        # batch_x: FloatTensor[B, 246]
        x = batch_x
        B = x.shape[0]

        g = x[:, : self.G]  # (B,48)
        obj_flat = x[:, self.G :]  # (B, 18*11=198)
        obj = obj_flat.view(B, self.MAX_OBJS, self.DOBJ)  # (B,18,11)

        present = obj[:, :, 0].clamp(0.0, 1.0)  # (B,18)
        type_idx = obj[:, :, 1].round().clamp(0, self.K).long()  # (B,18)
        cont = obj[:, :, 2:]  # (B,18,9)

        type_e = self.type_emb(type_idx)  # (B,18,8)
        obj_in = torch.cat([present.unsqueeze(-1), type_e, cont], dim=-1)  # (B,18,18)

        tok = self.obj_proj(obj_in)  # (B,18,96)

        # Padding mask: True for padded tokens (to be ignored)
        pad_mask = present < 0.5  # (B,18) bool

        cls = self.cls_token.expand(B, 1, self.model_dim)  # (B,1,96)
        seq = torch.cat([cls, tok], dim=1)  # (B,19,96)
        seq_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad_mask], dim=1)  # (B,19)

        enc = self.obj_encoder(seq, src_key_padding_mask=seq_mask)  # (B,19,96)
        cls_out = enc[:, 0, :]  # (B,96)
        obj_out = enc[:, 1:, :]  # (B,18,96)

        # Pooling (masked)
        denom = present.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1)
        masked = obj_out * present.unsqueeze(-1)  # (B,18,96)
        mean_pool = masked.sum(dim=1) / denom  # (B,96)

        max_pool = obj_out.masked_fill(pad_mask.unsqueeze(-1), -1e9).max(dim=1).values  # (B,96)
        has_any = (~pad_mask).any(dim=1, keepdim=True)  # (B,1)
        max_pool = torch.where(has_any, max_pool, torch.zeros_like(max_pool))  # (B,96)

        g_emb = self.global_mlp(g)  # (B,96)

        feat = torch.cat([g_emb, cls_out, mean_pool, max_pool], dim=1)  # (B,384)
        logit = self.head(feat).squeeze(-1)  # (B,)
        return logit


def make_model(example_object):
    return BinaryClassifier(example_object)


# ---------- MODEL TRAINING ----------
EPOCHS = 12


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1.0e-3)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2.5e-3,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.12,
        div_factor=15.0,
        final_div_factor=200.0,
    )

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    best_auc = -1.0
    best_state = None
    patience = 3
    bad_epochs = 0

    for epoch in range(int(epochs)):
        # ---- Train ----
        model.train()
        tr_loss_sum = 0.0
        tr_correct = 0
        tr_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()  # (B,)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)  # (B,)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = int(yb.shape[0])
            tr_loss_sum += float(loss.item()) * bs
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).long()
                tr_correct += int((preds == yb.long()).sum().item())
                tr_total += bs

        tr_loss = tr_loss_sum / max(1, tr_total)
        tr_acc = tr_correct / max(1, tr_total)
        train_loss_hist.append(tr_loss)
        train_acc_hist.append(tr_acc)

        # ---- Validate ----
        model.eval()
        va_loss_sum = 0.0
        va_correct = 0
        va_total = 0

        all_probs = []
        all_true = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float()

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)

                bs = int(yb.shape[0])
                va_loss_sum += float(loss.item()) * bs
                probs = torch.sigmoid(logits).detach().float().cpu().numpy()
                ytrue = yb.detach().long().cpu().numpy()

                all_probs.append(probs)
                all_true.append(ytrue)

                preds = (probs > 0.5).astype(np.int64)
                va_correct += int((preds == ytrue).sum())
                va_total += bs

        va_loss = va_loss_sum / max(1, va_total)
        va_acc = va_correct / max(1, va_total)
        val_loss_hist.append(va_loss)
        val_acc_hist.append(va_acc)

        # AUC for early stopping
        try:
            y_true = np.concatenate(all_true, axis=0)
            y_prob = np.concatenate(all_probs, axis=0)
            va_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            va_auc = -1.0

        if va_auc > best_auc + 1e-4:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        model.to(device)

    trained_model = model
    train_loss, val_loss = train_loss_hist, val_loss_hist
    train_acc, val_acc = train_acc_hist, val_acc_hist
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
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

