
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
import contextlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


class MyPreprocessor:
    def __init__(self):
        self._fitted = False

    def make_loader_cfg(self) -> dict:
        use_cuda = torch.cuda.is_available()
        bs = 1024 if use_cuda else 512
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": bs,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": bool(use_cuda),
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": bs},
        }

    def fit(self, X, y=None):
        self._fitted = True
        return self

    def transform(self, X):
        # Keep raw physics values intact (for invariant mass construction in the model).
        # Just sanitize NaNs/Infs deterministically.
        if torch.is_tensor(X):
            X2 = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return X2
        X = np.asarray(X)
        X2 = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X2.astype(np.float32, copy=False)


def make_preprocessor():
    return MyPreprocessor()


def _delta_phi(dphi: torch.Tensor) -> torch.Tensor:
    # Wrap to [-pi, pi]
    return torch.atan2(torch.sin(dphi), torch.cos(dphi))


class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        in_dim = int(sample_object.shape[-1])
        assert in_dim >= 2 and (in_dim - 2) % 5 == 0, f"Unexpected input feature length: {in_dim}"
        self.n_obj = (in_dim - 2) // 5  # 18
        self.obj_stride = 5
        self.k_top = 6  # pairwise computed on top-K pT objects (keeps compute small)

        self.n_id = 32  # clip object id into [0, n_id-1]
        self.d_model = 64
        self.cont_dim = 8  # per-object continuous features below

        self.id_emb = nn.Embedding(self.n_id, self.d_model)
        self.obj_linear = nn.Sequential(
            nn.Linear(self.cont_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=128,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Global feature head
        # global feats:
        #   MET: 3
        #   event sums: 7
        #   id counts: n_id
        #   pairwise: 12
        # total = 3 + 7 + 32 + 12 = 54
        self.global_dim = 54

        mlp_in = self.d_model * 3 + self.global_dim  # cls + mean + max + global
        self.head = nn.Sequential(
            nn.LayerNorm(mlp_in),
            nn.Linear(mlp_in, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, batch_x):
        # batch_x: FloatTensor[B, 92]
        x = batch_x
        B = x.shape[0]

        met = x[:, 0]  # [B]
        met_phi = x[:, 1]  # [B]

        objs = x[:, 2:].view(B, self.n_obj, self.obj_stride)  # [B, 18, 5]
        obj_id_f = objs[:, :, 0]  # [B, 18]
        E = objs[:, :, 1]  # [B, 18]
        pT = objs[:, :, 2]  # [B, 18]
        eta = objs[:, :, 3]  # [B, 18]
        phi = objs[:, :, 4]  # [B, 18]

        # Valid mask (padding is all zeros; object id is 0 for padding)
        obj_id = obj_id_f.round().clamp(min=0).long()  # [B, 18]
        mask = obj_id.ne(0)  # [B, 18] bool
        mask_f = mask.float()  # [B, 18]

        # Basic 4-vector components from (pT, eta, phi)
        # Use these both for per-object features and pairwise masses.
        cosphi = torch.cos(phi)  # [B, 18]
        sinphi = torch.sin(phi)  # [B, 18]
        px = pT * cosphi  # [B, 18]
        py = pT * sinphi  # [B, 18]
        pz = pT * torch.sinh(eta)  # [B, 18]
        p2 = pT * pT + pz * pz  # [B, 18]
        m2 = (E * E - p2).clamp_min(0.0)  # [B, 18]
        m = torch.sqrt(m2 + 1e-12)  # [B, 18]

        # Per-object continuous features
        logE = torch.log1p(E.clamp_min(0.0))  # [B, 18]
        logpT = torch.log1p(pT.clamp_min(0.0))  # [B, 18]
        logm = torch.log1p(m)  # [B, 18]
        abs_eta = eta.abs()  # [B, 18]
        pt_over_E = pT / (E.abs() + 1.0)  # [B, 18]
        # cont: [B, 18, 8]
        cont = torch.stack(
            [logE, logpT, eta, abs_eta, sinphi, cosphi, logm, pt_over_E],
            dim=-1,
        )
        cont = cont * mask_f.unsqueeze(-1)  # [B, 18, 8]

        # Object tokens: [B, 18, d_model]
        ids_clip = obj_id.clamp(0, self.n_id - 1)  # [B, 18]
        tok = self.obj_linear(cont) + self.id_emb(ids_clip)  # [B, 18, 64]

        # Transformer sequence with CLS
        cls = self.cls_token.expand(B, 1, self.d_model)  # [B, 1, 64]
        seq = torch.cat([cls, tok], dim=1)  # [B, 19, 64]
        pad_mask = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.bool), ~mask], dim=1)  # [B, 19]

        out = self.transformer(seq, src_key_padding_mask=pad_mask)  # [B, 19, 64]
        cls_out = out[:, 0, :]  # [B, 64]
        obj_out = out[:, 1:, :]  # [B, 18, 64]

        # Masked pooling
        denom = mask_f.sum(dim=1).clamp_min(1.0).unsqueeze(-1)  # [B, 1]
        mean_pool = (obj_out * mask_f.unsqueeze(-1)).sum(dim=1) / denom  # [B, 64]
        obj_out_masked = obj_out.masked_fill(~mask.unsqueeze(-1), -1e9)  # [B, 18, 64]
        max_pool = obj_out_masked.max(dim=1).values  # [B, 64]
        no_obj = mask_f.sum(dim=1).eq(0.0)  # [B]
        max_pool = torch.where(no_obj.unsqueeze(-1), torch.zeros_like(max_pool), max_pool)  # [B, 64]

        # Global event features
        logmet = torch.log1p(met.clamp_min(0.0))  # [B]
        met_sin = torch.sin(met_phi)  # [B]
        met_cos = torch.cos(met_phi)  # [B]

        ht = (pT.clamp_min(0.0) * mask_f).sum(dim=1)  # [B]
        sumE = (E.clamp_min(0.0) * mask_f).sum(dim=1)  # [B]
        lead_pT = (pT.masked_fill(~mask, 0.0)).max(dim=1).values  # [B]
        nobj = mask_f.sum(dim=1)  # [B]
        mean_abs_eta = (abs_eta * mask_f).sum(dim=1) / nobj.clamp_min(1.0)  # [B]
        met_over_ht = met.clamp_min(0.0) / (ht + 1.0)  # [B]
        loght = torch.log1p(ht.clamp_min(0.0))  # [B]
        logsumE = torch.log1p(sumE.clamp_min(0.0))  # [B]
        logleadpt = torch.log1p(lead_pT.clamp_min(0.0))  # [B]
        frac_nobj = nobj / float(self.n_obj)  # [B]

        # ID counts (normalized): [B, n_id]
        onehot = F.one_hot(ids_clip, num_classes=self.n_id).float()  # [B, 18, 32]
        onehot = onehot * mask_f.unsqueeze(-1)  # [B, 18, 32]
        id_counts = onehot.sum(dim=1) / float(self.n_obj)  # [B, 32]

        # Pairwise features on top-K pT objects
        pt_masked = pT.masked_fill(~mask, -1e9)  # [B, 18]
        top_pt, top_idx = pt_masked.topk(self.k_top, dim=1)  # [B, K], [B, K]
        valid_k = top_pt > -1e8  # [B, K]

        def _gather_feat(feat_2d):
            # feat_2d: [B, 18] -> [B, K]
            return torch.gather(feat_2d, dim=1, index=top_idx)

        Ek = _gather_feat(E)  # [B, K]
        pTk = _gather_feat(pT)  # [B, K]
        etak = _gather_feat(eta)  # [B, K]
        phik = _gather_feat(phi)  # [B, K]

        cosk = torch.cos(phik)  # [B, K]
        sink = torch.sin(phik)  # [B, K]
        pxk = pTk * cosk  # [B, K]
        pyk = pTk * sink  # [B, K]
        pzk = pTk * torch.sinh(etak)  # [B, K]

        # Pair matrices: [B, K, K]
        dEta = etak.unsqueeze(2) - etak.unsqueeze(1)  # [B, K, K]
        dPhi = _delta_phi(phik.unsqueeze(2) - phik.unsqueeze(1))  # [B, K, K]
        dR = torch.sqrt(dEta * dEta + dPhi * dPhi + 1e-12)  # [B, K, K]

        EiEj = Ek.unsqueeze(2) + Ek.unsqueeze(1)  # [B, K, K]
        pxij = pxk.unsqueeze(2) + pxk.unsqueeze(1)  # [B, K, K]
        pyij = pyk.unsqueeze(2) + pyk.unsqueeze(1)  # [B, K, K]
        pzij = pzk.unsqueeze(2) + pzk.unsqueeze(1)  # [B, K, K]
        m2ij = (EiEj * EiEj - pxij * pxij - pyij * pyij - pzij * pzij).clamp_min(0.0)  # [B, K, K]
        mij = torch.sqrt(m2ij + 1e-12)  # [B, K, K]

        iu = torch.triu_indices(self.k_top, self.k_top, offset=1, device=x.device)
        dr_pairs = dR[:, iu[0], iu[1]]  # [B, P]
        m_pairs = mij[:, iu[0], iu[1]]  # [B, P]

        valid_pairs = valid_k[:, iu[0]] & valid_k[:, iu[1]]  # [B, P]
        vp_f = valid_pairs.float()  # [B, P]
        npair = vp_f.sum(dim=1).clamp_min(1.0)  # [B]

        m_sum = (m_pairs * vp_f).sum(dim=1)  # [B]
        m2_sum = ((m_pairs * m_pairs) * vp_f).sum(dim=1)  # [B]
        m_mean = m_sum / npair  # [B]
        m_var = (m2_sum / npair - m_mean * m_mean).clamp_min(0.0)  # [B]
        m_std = torch.sqrt(m_var + 1e-12)  # [B]
        m_max = m_pairs.masked_fill(~valid_pairs, -1e9).max(dim=1).values  # [B]
        has_pair = valid_pairs.any(dim=1)  # [B]
        m_max = torch.where(has_pair, m_max, torch.zeros_like(m_max))  # [B]
        m_top3 = m_pairs.masked_fill(~valid_pairs, -1e9).topk(3, dim=1).values  # [B, 3]
        m_top3 = m_top3.clamp_min(0.0)  # [B, 3]

        dr_sum = (dr_pairs * vp_f).sum(dim=1)  # [B]
        dr2_sum = ((dr_pairs * dr_pairs) * vp_f).sum(dim=1)  # [B]
        dr_mean = dr_sum / npair  # [B]
        dr_var = (dr2_sum / npair - dr_mean * dr_mean).clamp_min(0.0)  # [B]
        dr_std = torch.sqrt(dr_var + 1e-12)  # [B]
        dr_max = dr_pairs.masked_fill(~valid_pairs, -1e9).max(dim=1).values  # [B]
        dr_max = torch.where(has_pair, dr_max, torch.zeros_like(dr_max))  # [B]
        dr_top3 = dr_pairs.masked_fill(~valid_pairs, -1e9).topk(3, dim=1).values  # [B, 3]
        dr_top3 = dr_top3.clamp_min(0.0)  # [B, 3]

        # Assemble global features: [B, 54]
        global_feats = torch.cat(
            [
                torch.stack([logmet, met_sin, met_cos], dim=1),  # [B, 3]
                torch.stack(
                    [
                        loght,
                        logsumE,
                        logleadpt,
                        mean_abs_eta,
                        met_over_ht,
                        frac_nobj,
                        torch.log1p((met.clamp_min(0.0) + ht).clamp_min(0.0)),
                    ],
                    dim=1,
                ),  # [B, 7]
                id_counts,  # [B, 32]
                torch.stack([m_mean, m_max, m_std, dr_mean, dr_max, dr_std], dim=1),  # [B, 6]
                m_top3,  # [B, 3]
                dr_top3,  # [B, 3]
            ],
            dim=1,
        )  # [B, 54]

        feats = torch.cat([cls_out, mean_pool, max_pool, global_feats], dim=1)  # [B, 64*3 + 54] = [B, 246]
        logits = self.head(feats).squeeze(-1)  # [B]
        return logits


def make_model(example_object):
    return BinaryClassifier(example_object)


EPOCHS = 12


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    bce = nn.BCEWithLogitsLoss()

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    val_auc_hist = []

    best_auc = -float("inf")
    best_state = None
    patience = 3
    bad = 0

    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)

            ctx = torch.cuda.amp.autocast(enabled=(device.type == "cuda"))
            with ctx:
                logits = model(xb)  # [B]
                loss = bce(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            bs = int(yb.numel())
            total_loss += float(loss.detach()) * bs
            preds = (logits.detach() > 0.0).long()
            total_correct += int((preds == yb.long()).sum().item())
            total_seen += bs

        sched.step()

        tr_loss = total_loss / max(1, total_seen)
        tr_acc = total_correct / max(1, total_seen)
        train_loss_hist.append(tr_loss)
        train_acc_hist.append(tr_acc)

        model.eval()
        v_total_loss = 0.0
        v_total_correct = 0
        v_total_seen = 0
        all_logits = []
        all_y = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True).float()
                yb_t = yb.to(device, non_blocking=True).float()

                ctx = torch.cuda.amp.autocast(enabled=(device.type == "cuda"))
                with ctx:
                    logits = model(xb)  # [B]
                    loss = bce(logits, yb_t)

                bs = int(yb_t.numel())
                v_total_loss += float(loss.detach()) * bs
                preds = (logits.detach() > 0.0).long()
                v_total_correct += int((preds == yb.to(device)).sum().item())
                v_total_seen += bs

                all_logits.append(logits.detach().float().cpu())
                all_y.append(yb.detach().cpu())

        va_loss = v_total_loss / max(1, v_total_seen)
        va_acc = v_total_correct / max(1, v_total_seen)
        val_loss_hist.append(va_loss)
        val_acc_hist.append(va_acc)

        logits_cat = torch.cat(all_logits, dim=0).numpy()
        y_cat = torch.cat(all_y, dim=0).numpy()
        try:
            va_auc = float(roc_auc_score(y_cat, logits_cat))
        except Exception:
            va_auc = float("nan")
        val_auc_hist.append(va_auc)

        # Early stopping on AUC
        improved = (va_auc == va_auc) and (va_auc > best_auc + 1e-4)
        if improved:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

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

