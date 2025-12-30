
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
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score


class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X
        self.y = y

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        x = {k: v[idx] for k, v in self.X.items()}
        return x, self.y[idx]


class MyPreprocessor:
    def __init__(self):
        self.K_TOP = 10
        self.K_ID_MAP = 254  # map most common raw IDs -> 1..254, unknown -> 255, pad -> 0
        self.global_mean = None
        self.global_std = None
        self.obj_mean = None
        self.obj_std = None
        self.top_ids = None
        self.max_raw_id = None
        self._use_tensor_map = True
        self._id_map_tensor = None
        self._id_map_dict = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    @staticmethod
    def _split(X: torch.Tensor):
        # X: [N, 92]
        met = X[:, 0]  # [N]
        met_phi = X[:, 1]  # [N]
        obj = X[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]
        return met, met_phi, obj

    @staticmethod
    def _wrap_obj(obj: torch.Tensor):
        # obj: [N, 18, 5] = [id, E, pT, eta, phi]
        raw_id = obj[:, :, 0].round().to(torch.int64)  # [N, 18]
        E = obj[:, :, 1]  # [N, 18]
        pT = obj[:, :, 2]  # [N, 18]
        eta = obj[:, :, 3]  # [N, 18]
        phi = obj[:, :, 4]  # [N, 18]
        mask = (raw_id > 0) & (pT > 0)  # [N, 18] (bool)
        return raw_id, E, pT, eta, phi, mask

    @staticmethod
    def _sort_by_pt(raw_id, E, pT, eta, phi, mask):
        # All: [N, 18] except mask bool
        pT_masked = pT.masked_fill(~mask, -1e9)  # [N, 18]
        idx = torch.argsort(pT_masked, dim=1, descending=True)  # [N, 18]
        gather = lambda t: torch.gather(t, 1, idx)
        raw_id_s = gather(raw_id)  # [N, 18]
        E_s = gather(E)  # [N, 18]
        pT_s = gather(pT)  # [N, 18]
        eta_s = gather(eta)  # [N, 18]
        phi_s = gather(phi)  # [N, 18]
        mask_s = gather(mask.to(torch.int64)).to(torch.bool)  # [N, 18]
        return raw_id_s, E_s, pT_s, eta_s, phi_s, mask_s

    @staticmethod
    def _build_object_cont(E, pT, eta, phi, met_phi):
        # E,pT,eta,phi: [N, 18], met_phi: [N]
        E_GeV = torch.clamp(E, min=0.0) / 1000.0  # [N, 18]
        pT_GeV = torch.clamp(pT, min=0.0) / 1000.0  # [N, 18]
        abs_eta = torch.abs(eta)  # [N, 18]

        logE = torch.log1p(E_GeV)  # [N, 18]
        logpT = torch.log1p(pT_GeV)  # [N, 18]

        sin_phi = torch.sin(phi)  # [N, 18]
        cos_phi = torch.cos(phi)  # [N, 18]

        dphi = phi - met_phi[:, None]  # [N, 18]
        sin_dphi = torch.sin(dphi)  # [N, 18]
        cos_dphi = torch.cos(dphi)  # [N, 18]

        # obj_cont: [N, 18, 8]
        obj_cont = torch.stack([logE, logpT, eta, abs_eta, sin_phi, cos_phi, sin_dphi, cos_dphi], dim=-1)
        return obj_cont, E_GeV, pT_GeV, abs_eta, cos_dphi

    def _build_global(self, met, met_phi, raw_id_s, E_GeV_s, pT_GeV_s, abs_eta_s, cos_dphi_s, mask_s):
        # met, met_phi: [N]; others: [N, 18]
        met_GeV = torch.clamp(met, min=0.0) / 1000.0  # [N]
        logMET = torch.log1p(met_GeV)  # [N]
        sin_met_phi = torch.sin(met_phi)  # [N]
        cos_met_phi = torch.cos(met_phi)  # [N]

        m = mask_s.to(torch.float32)  # [N, 18]
        n_valid = m.sum(dim=1)  # [N]
        sum_pT = (pT_GeV_s * m).sum(dim=1)  # [N]
        sum_E = (E_GeV_s * m).sum(dim=1)  # [N]

        max_pT = pT_GeV_s.masked_fill(~mask_s, -1e9).max(dim=1).values  # [N]
        max_pT = torch.where(n_valid > 0, max_pT, torch.zeros_like(max_pT))  # [N]
        max_abs_eta = abs_eta_s.masked_fill(~mask_s, -1e9).max(dim=1).values  # [N]
        max_abs_eta = torch.where(n_valid > 0, max_abs_eta, torch.zeros_like(max_abs_eta))  # [N]

        # Top-4 (sorted already): [N, 4]
        k = 4
        top_pT = pT_GeV_s[:, :k]  # [N, 4]
        top_abs_eta = abs_eta_s[:, :k]  # [N, 4]
        top_cos_dphi = cos_dphi_s[:, :k]  # [N, 4]
        top_logpT = torch.log1p(torch.clamp(top_pT, min=0.0))  # [N, 4]

        # Type counts & sum pT for frequent raw IDs
        type_counts = []
        type_sumpt = []
        for tid in self.top_ids:
            sel = (raw_id_s == int(tid)) & mask_s  # [N, 18]
            sel_f = sel.to(torch.float32)  # [N, 18]
            type_counts.append(sel_f.sum(dim=1))  # [N]
            type_sumpt.append((pT_GeV_s * sel_f).sum(dim=1))  # [N]
        if len(type_counts) > 0:
            type_counts = torch.stack(type_counts, dim=1)  # [N, K_TOP]
            type_sumpt = torch.stack(type_sumpt, dim=1)  # [N, K_TOP]
        else:
            type_counts = met[:, None].new_zeros((met.shape[0], 0))  # [N, 0]
            type_sumpt = met[:, None].new_zeros((met.shape[0], 0))  # [N, 0]

        # global: [N, 3 + 5 + 12 + 2*K_TOP] = [N, 20 + 2*K_TOP]
        global_feat = torch.cat(
            [
                torch.stack([logMET, sin_met_phi, cos_met_phi], dim=1),  # [N, 3]
                torch.stack([n_valid, sum_pT, sum_E, max_pT, max_abs_eta], dim=1),  # [N, 5]
                top_logpT,  # [N, 4]
                top_abs_eta,  # [N, 4]
                top_cos_dphi,  # [N, 4]
                type_counts,  # [N, K_TOP]
                type_sumpt,  # [N, K_TOP]
            ],
            dim=1,
        )
        return global_feat

    def fit(self, X, y=None):
        X = X.detach().cpu()
        met, met_phi, obj = self._split(X)  # met: [N], met_phi: [N], obj: [N, 18, 5]
        raw_id, E, pT, eta, phi, mask = self._wrap_obj(obj)  # each [N, 18]
        raw_id_s, E_s, pT_s, eta_s, phi_s, mask_s = self._sort_by_pt(raw_id, E, pT, eta, phi, mask)

        # Choose top IDs for global hist features
        flat_ids = raw_id_s[mask_s].to(torch.int64)
        if flat_ids.numel() > 0:
            unique, counts = torch.unique(flat_ids, return_counts=True)
            order = torch.argsort(counts, descending=True)
            unique = unique[order]
            top = unique[: self.K_TOP].tolist()
        else:
            top = []
        self.top_ids = [int(x) for x in top]

        # Build ID mapping for embedding indices:
        # pad=0 -> 0, most common -> 1..K_ID_MAP, unknown nonzero -> 255
        self.max_raw_id = int(raw_id_s.max().item()) if raw_id_s.numel() > 0 else 0
        # Determine most common IDs for mapping
        if flat_ids.numel() > 0:
            unique2, counts2 = torch.unique(flat_ids, return_counts=True)
            order2 = torch.argsort(counts2, descending=True)
            common = unique2[order2][: self.K_ID_MAP].tolist()
        else:
            common = []
        common = [int(x) for x in common if int(x) != 0]

        if self.max_raw_id <= 10000:
            self._use_tensor_map = True
            map_t = torch.full((self.max_raw_id + 1,), 255, dtype=torch.int64)  # unknown default
            map_t[0] = 0
            for i, rid in enumerate(common, start=1):
                if 0 <= rid <= self.max_raw_id:
                    map_t[rid] = i
            self._id_map_tensor = map_t
            self._id_map_dict = None
        else:
            self._use_tensor_map = False
            d = {0: 0}
            for i, rid in enumerate(common, start=1):
                d[int(rid)] = int(i)
            self._id_map_dict = d
            self._id_map_tensor = None

        # Object cont features (for mean/std)
        obj_cont, E_GeV_s, pT_GeV_s, abs_eta_s, cos_dphi_s = self._build_object_cont(E_s, pT_s, eta_s, phi_s, met_phi)
        # obj_cont: [N, 18, 8]
        valid = mask_s  # [N, 18]
        if valid.any():
            cont_flat = obj_cont[valid]  # [N_valid, 8]
            mean = cont_flat.to(torch.float64).mean(dim=0)  # [8]
            std = cont_flat.to(torch.float64).std(dim=0, unbiased=False)  # [8]
        else:
            mean = torch.zeros((obj_cont.shape[-1],), dtype=torch.float64)
            std = torch.ones((obj_cont.shape[-1],), dtype=torch.float64)
        std = torch.clamp(std, min=1e-6)
        self.obj_mean = mean.to(torch.float32).numpy()
        self.obj_std = std.to(torch.float32).numpy()

        # Global features (for mean/std)
        global_feat = self._build_global(met, met_phi, raw_id_s, E_GeV_s, pT_GeV_s, abs_eta_s, cos_dphi_s, mask_s)
        g_mean = global_feat.to(torch.float64).mean(dim=0)
        g_std = global_feat.to(torch.float64).std(dim=0, unbiased=False)
        g_std = torch.clamp(g_std, min=1e-6)
        self.global_mean = g_mean.to(torch.float32).numpy()
        self.global_std = g_std.to(torch.float32).numpy()

        return self

    def _map_ids(self, raw_id_s: torch.Tensor):
        # raw_id_s: [N, 18] int64
        # output: [N, 18] int16, range 0..255
        if self._use_tensor_map:
            rid = raw_id_s
            rid = torch.where((rid >= 0) & (rid <= self.max_raw_id), rid, torch.zeros_like(rid))  # [N, 18]
            mapped = self._id_map_tensor[rid]  # [N, 18] int64
        else:
            # dict-based mapping (rare)
            mapped = torch.full_like(raw_id_s, 255, dtype=torch.int64)
            mapped[raw_id_s == 0] = 0
            for k, v in self._id_map_dict.items():
                if k == 0:
                    continue
                mapped[raw_id_s == int(k)] = int(v)
        mapped = torch.clamp(mapped, 0, 255).to(torch.int16)
        return mapped

    def transform(self, X):
        X = X.detach().cpu()
        met, met_phi, obj = self._split(X)  # met: [N], met_phi: [N], obj: [N, 18, 5]
        raw_id, E, pT, eta, phi, mask = self._wrap_obj(obj)  # each [N, 18]
        raw_id_s, E_s, pT_s, eta_s, phi_s, mask_s = self._sort_by_pt(raw_id, E, pT, eta, phi, mask)

        obj_cont, E_GeV_s, pT_GeV_s, abs_eta_s, cos_dphi_s = self._build_object_cont(E_s, pT_s, eta_s, phi_s, met_phi)
        # obj_cont: [N, 18, 8]
        global_feat = self._build_global(met, met_phi, raw_id_s, E_GeV_s, pT_GeV_s, abs_eta_s, cos_dphi_s, mask_s)
        # global_feat: [N, G]

        # Normalize
        obj_mean = torch.from_numpy(self.obj_mean)  # [8]
        obj_std = torch.from_numpy(self.obj_std)  # [8]
        obj_cont_n = (obj_cont - obj_mean[None, None, :]) / obj_std[None, None, :]  # [N, 18, 8]
        obj_cont_n = obj_cont_n.masked_fill(~mask_s.unsqueeze(-1), 0.0)  # [N, 18, 8]

        g_mean = torch.from_numpy(self.global_mean)  # [G]
        g_std = torch.from_numpy(self.global_std)  # [G]
        global_n = (global_feat - g_mean[None, :]) / g_std[None, :]  # [N, G]

        id_mapped = self._map_ids(raw_id_s)  # [N, 18] int16

        return {
            "g": global_n.to(torch.float32),            # [N, G]
            "x": obj_cont_n.to(torch.float32),          # [N, 18, 8]
            "id": id_mapped,                            # [N, 18] int16
            "mask": mask_s.to(torch.bool),              # [N, 18] bool
        }


def make_preprocessor():
    return MyPreprocessor()


class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        g_dim = int(sample_object["g"].shape[-1])
        x_dim = int(sample_object["x"].shape[-1])  # 8
        self.seq_len = int(sample_object["x"].shape[1])  # 18

        d_model = 128
        id_dim = 24

        self.id_emb = nn.Embedding(256, id_dim)  # indices 0..255
        self.id_proj = nn.Linear(id_dim, d_model, bias=False)
        self.cont_proj = nn.Linear(x_dim, d_model, bias=True)

        self.pos_emb = nn.Embedding(self.seq_len, d_model)

        self.in_norm = nn.LayerNorm(d_model)
        self.in_drop = nn.Dropout(0.10)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=256,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)

        self.gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.global_mlp = nn.Sequential(
            nn.LayerNorm(g_dim),
            nn.Linear(g_dim, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 3 + 64),
            nn.Linear(d_model * 3 + 64, 192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, batch_x):
        # batch_x:
        #   g:    [B, G]
        #   x:    [B, 18, 8]
        #   id:   [B, 18]
        #   mask: [B, 18] bool
        g = batch_x["g"]
        x = batch_x["x"]
        ids = batch_x["id"].long()
        mask = batch_x["mask"]

        B, L, _ = x.shape
        if L != self.seq_len:
            raise RuntimeError(f"Unexpected sequence length: got {L}, expected {self.seq_len}")

        # Ensure at least one valid token per event to avoid all-masked attention softmax NaNs
        n_valid = mask.sum(dim=1)  # [B]
        if (n_valid == 0).any():
            mask = mask.clone()
            mask[n_valid == 0, 0] = True
            x = x.clone()
            x[n_valid == 0, 0, :] = 0.0
            ids = ids.clone()
            ids[n_valid == 0, 0] = 0

        cont = self.cont_proj(x)  # [B, 18, 128]
        idv = self.id_proj(self.id_emb(ids))  # [B, 18, 128]

        pos_idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)  # [B, 18]
        pos = self.pos_emb(pos_idx)  # [B, 18, 128]

        h = cont + idv + pos  # [B, 18, 128]
        h = self.in_drop(self.in_norm(h))  # [B, 18, 128]

        key_padding_mask = ~mask  # [B, 18]
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # [B, 18, 128]

        # Pooling: masked mean, masked max, gated attention
        m = mask.to(h.dtype)  # [B, 18]
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]

        mean_pool = (h * m.unsqueeze(-1)).sum(dim=1) / denom  # [B, 128]

        h_max = h.masked_fill(~mask.unsqueeze(-1), -1e9)  # [B, 18, 128]
        max_pool = h_max.max(dim=1).values  # [B, 128]
        max_pool = torch.where((m.sum(dim=1) > 0).unsqueeze(-1), max_pool, torch.zeros_like(max_pool))  # [B, 128]

        attn_logits = self.gate(h).squeeze(-1)  # [B, 18]
        attn_logits = attn_logits.masked_fill(~mask, -1e9)  # [B, 18]
        attn_w = torch.softmax(attn_logits, dim=1)  # [B, 18]
        attn_pool = (attn_w.unsqueeze(-1) * h).sum(dim=1)  # [B, 128]

        g_emb = self.global_mlp(g)  # [B, 64]
        feat = torch.cat([g_emb, mean_pool, max_pool, attn_pool], dim=1)  # [B, 64+384] = [B, 448]
        logits = self.head(feat).squeeze(-1)  # [B]
        return logits


def make_model(example_object):
    return BinaryClassifier(example_object)


EPOCHS = 15


@torch.no_grad()
def _eval_epoch(model, loader):
    model.eval()
    losses = []
    all_p = []
    all_y = []
    total = 0
    correct = 0

    bce = nn.BCEWithLogitsLoss(reduction="mean")

    for batch in loader:
        view = normalise_batch(batch, device=device)
        xb, yb = view.batch_x, view.batch_y  # yb: [B]
        yb_f = yb.float()

        logits = model(xb)  # [B]
        loss = bce(logits, yb_f)
        losses.append(float(loss.item()))

        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).to(yb.dtype)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())

        all_p.append(prob.detach().cpu())
        all_y.append(yb.detach().cpu())

    p = torch.cat(all_p).numpy()
    y = torch.cat(all_y).numpy()
    auc = float(roc_auc_score(y, p)) if (np.unique(y).size > 1) else 0.5
    acc = correct / max(1, total)
    return float(np.mean(losses)) if losses else 0.0, acc, auc


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    lr = 3e-3
    wd = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=max(1, epochs),
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,
        div_factor=10.0,
        final_div_factor=50.0,
        anneal_strategy="cos",
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    bce = nn.BCEWithLogitsLoss()

    train_loss_hist, val_loss_hist = [], []
    train_auc_hist, val_auc_hist = [], []

    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    patience = 4
    bad = 0

    for epoch in range(int(epochs)):
        model.train()
        running = 0.0
        n_batches = 0

        # Approx train AUC on limited number of batches for speed
        approx_p = []
        approx_y = []
        approx_max_batches = 60

        for bi, batch in enumerate(train_loader):
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            yb_f = yb.float()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(xb)  # [B]
                loss = bce(logits, yb_f)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.item())
            n_batches += 1

            if bi < approx_max_batches:
                prob = torch.sigmoid(logits.detach())
                approx_p.append(prob.detach().cpu())
                approx_y.append(yb.detach().cpu())

        tr_loss = running / max(1, n_batches)
        train_loss_hist.append(tr_loss)

        if approx_p:
            p = torch.cat(approx_p).numpy()
            y = torch.cat(approx_y).numpy()
            tr_auc = float(roc_auc_score(y, p)) if (np.unique(y).size > 1) else 0.5
        else:
            tr_auc = 0.5
        train_auc_hist.append(tr_auc)

        va_loss, va_acc, va_auc = _eval_epoch(model, val_loader)
        val_loss_hist.append(va_loss)
        val_auc_hist.append(va_auc)

        if va_auc > best_auc + 1e-4:
            best_auc = va_auc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model, train_loss_hist, val_loss_hist, train_auc_hist, val_auc_hist

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

