
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
        x = self.X[idx]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x, self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

import math
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score


class MyPreprocessor:
    def __init__(self):
        pass

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
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def make_preprocessor():
    return MyPreprocessor()


class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Data layout
        self.n_obj = 18
        self.obj_stride = 5

        # Hyperparams
        self.num_types = 32
        self.type_emb_dim = 16
        self.obj_in_dim = 7 + self.type_emb_dim  # [logE, logpT, eta, sinphi, cosphi, pTfrac, log(E/(pTcosh))] + emb
        self.d_model = 128
        self.nhead = 8
        self.nlayers = 3
        self.dropout = 0.12

        # Transformer
        self.type_emb = nn.Embedding(self.num_types, self.type_emb_dim)

        self.obj_in = nn.Sequential(
            nn.Linear(self.obj_in_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=256,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.nlayers)

        self.pool_score = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1),
        )

        # Pairwise feature config
        self.topk = 5
        triu = torch.triu(torch.ones(self.n_obj, self.n_obj, dtype=torch.bool), diagonal=1)
        self.register_buffer("triu_mask", triu, persistent=False)

        # Extra top-pt
        self.toppt = 4

        # Final MLP input dim calculation
        # global_dim: 11
        # counts_dim: 10
        # pool_dim: 3*d_model
        # pair_dim: 21 (count_frac + 5 stats + 15 topk)
        # toppt_dim: 4
        global_dim = 11
        counts_dim = 10
        pool_dim = 3 * self.d_model
        pair_dim = 21
        toppt_dim = self.toppt
        final_in = global_dim + counts_dim + pool_dim + pair_dim + toppt_dim  # = 11+10+384+21+4 = 430

        self.head = nn.Sequential(
            nn.LayerNorm(final_in),
            nn.Linear(final_in, 256),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _wrap_dphi(dphi):
        # dphi in radians -> wrap to [-pi, pi]
        two_pi = 2.0 * math.pi
        return torch.remainder(dphi + math.pi, two_pi) - math.pi

    def _masked_topk(self, vals_flat, mask_flat, k, largest=True):
        # vals_flat: [B, M], mask_flat: [B, M] bool
        if largest:
            v = vals_flat.masked_fill(~mask_flat, float("-inf"))
            out = torch.topk(v, k=k, dim=1, largest=True).values
            out = torch.nan_to_num(out, neginf=0.0, posinf=0.0)
            return out
        else:
            # k smallest: topk on -vals (largest), masking invalid to -inf so they don't appear
            v = (-vals_flat).masked_fill(~mask_flat, float("-inf"))
            out = torch.topk(v, k=k, dim=1, largest=True).values
            out = (-out)
            out = torch.nan_to_num(out, neginf=0.0, posinf=0.0)
            return out

    def _pairwise_features(self, E, pT, eta, phi, valid):
        # Inputs: [B, N] float, valid: [B, N] bool
        # Returns: [B, 21]
        B, N = E.shape

        # Force float32 for stability (also if autocast is enabled)
        E = E.float()
        pT = pT.float()
        eta = eta.float()
        phi = phi.float()

        mask_f = valid.float()  # [B, N]
        E = E * mask_f
        pT = pT * mask_f

        px = pT * torch.cos(phi) * mask_f  # [B, N]
        py = pT * torch.sin(phi) * mask_f  # [B, N]
        pz = pT * torch.sinh(eta) * mask_f  # [B, N]

        # Broadcast to pairs: [B, N, N]
        Ei = E[:, :, None]
        Ej = E[:, None, :]
        Es = Ei + Ej

        pxs = px[:, :, None] + px[:, None, :]
        pys = py[:, :, None] + py[:, None, :]
        pzs = pz[:, :, None] + pz[:, None, :]

        m2 = Es * Es - (pxs * pxs + pys * pys + pzs * pzs)
        m = torch.sqrt(torch.clamp(m2, min=0.0) + 1e-6)  # [B, N, N]
        logm = torch.log1p(m)  # [B, N, N]

        dphi = self._wrap_dphi(phi[:, :, None] - phi[:, None, :])  # [B, N, N]
        deta = eta[:, :, None] - eta[:, None, :]  # [B, N, N]
        dr = torch.sqrt(deta * deta + dphi * dphi + 1e-12)  # [B, N, N]

        mask_pair = valid[:, :, None] & valid[:, None, :] & self.triu_mask[None, :, :]  # [B, N, N]
        mask_pair_f = mask_pair.float()

        count = mask_pair_f.sum(dim=(1, 2))  # [B]
        denom = count.clamp(min=1.0)

        # Stats (5)
        logm_mean = (logm * mask_pair_f).sum(dim=(1, 2)) / denom  # [B]
        logm_max = logm.masked_fill(~mask_pair, float("-inf")).amax(dim=(1, 2))
        logm_max = torch.nan_to_num(logm_max, neginf=0.0, posinf=0.0)

        dr_mean = (dr * mask_pair_f).sum(dim=(1, 2)) / denom
        dr_min = dr.masked_fill(~mask_pair, float("inf")).amin(dim=(1, 2))
        dr_min = torch.nan_to_num(dr_min, posinf=0.0, neginf=0.0)
        dr_max = dr.masked_fill(~mask_pair, float("-inf")).amax(dim=(1, 2))
        dr_max = torch.nan_to_num(dr_max, neginf=0.0, posinf=0.0)

        # TopK (15): logm_topk (5), dr_small (5), dr_large (5)
        mask_flat = mask_pair.view(B, -1)  # [B, N*N]
        logm_flat = logm.view(B, -1)
        dr_flat = dr.view(B, -1)

        logm_top = self._masked_topk(logm_flat, mask_flat, k=self.topk, largest=True)  # [B, 5]
        dr_small = self._masked_topk(dr_flat, mask_flat, k=self.topk, largest=False)  # [B, 5]
        dr_large = self._masked_topk(dr_flat, mask_flat, k=self.topk, largest=True)  # [B, 5]

        # count fraction (1)
        total_pairs = float(N * (N - 1) // 2)
        count_frac = (count / total_pairs).unsqueeze(1)  # [B, 1]

        feats = torch.cat(
            [
                count_frac,  # [B, 1]
                logm_mean.unsqueeze(1),  # [B, 1]
                logm_max.unsqueeze(1),  # [B, 1]
                dr_mean.unsqueeze(1),  # [B, 1]
                dr_min.unsqueeze(1),  # [B, 1]
                dr_max.unsqueeze(1),  # [B, 1]
                logm_top,  # [B, 5]
                dr_small,  # [B, 5]
                dr_large,  # [B, 5]
            ],
            dim=1,
        )  # [B, 21]
        return feats

    def forward(self, batch_x):
        x = batch_x
        if isinstance(x, (list, tuple)):
            x = x[0]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        B = x.shape[0]

        # Global
        met = x[:, 0]  # [B]
        phi_met = x[:, 1]  # [B]

        # Objects
        objs = x[:, 2:].view(B, self.n_obj, self.obj_stride)  # [B, 18, 5]
        obj_id_f = objs[:, :, 0]  # [B, 18]
        obj_id = obj_id_f.round().long().clamp(0, self.num_types - 1)  # [B, 18]
        E = objs[:, :, 1]  # [B, 18]
        pT = objs[:, :, 2]  # [B, 18]
        eta = objs[:, :, 3]  # [B, 18]
        phi = objs[:, :, 4]  # [B, 18]

        valid = (obj_id != 0) & (pT > 0)  # [B, 18]
        valid_f = valid.float()  # [B, 18]
        nobj = valid_f.sum(dim=1)  # [B]
        nobj_clamp = nobj.clamp(min=1.0)

        # Event-level scalars
        HT = (pT * valid_f).sum(dim=1)  # [B]
        logHT = torch.log1p(HT.clamp(min=0.0))  # [B]
        logmet = torch.log1p(met.clamp(min=0.0))  # [B]

        # MET-object relations
        dphi_met_obj = self._wrap_dphi(phi - phi_met[:, None])  # [B, 18]
        adphi_met_obj = dphi_met_obj.abs()  # [B, 18]
        adphi_min = adphi_met_obj.masked_fill(~valid, float("inf")).amin(dim=1)
        adphi_min = torch.nan_to_num(adphi_min, posinf=0.0, neginf=0.0)
        adphi_max = adphi_met_obj.masked_fill(~valid, float("-inf")).amax(dim=1)
        adphi_max = torch.nan_to_num(adphi_max, posinf=0.0, neginf=0.0)
        adphi_mean = (adphi_met_obj * valid_f).sum(dim=1) / nobj_clamp
        adphi_mean = torch.nan_to_num(adphi_mean, posinf=0.0, neginf=0.0)

        mt = torch.sqrt(torch.clamp(2.0 * pT * met[:, None] * (1.0 - torch.cos(dphi_met_obj)), min=0.0) + 1e-6)  # [B, 18]
        mt_max = mt.masked_fill(~valid, float("-inf")).amax(dim=1)
        mt_max = torch.nan_to_num(mt_max, neginf=0.0, posinf=0.0)

        # Leading pT (top2)
        pT_masked = pT.masked_fill(~valid, float("-inf"))  # [B, 18]
        top2 = torch.topk(pT_masked, k=2, dim=1).values  # [B, 2]
        top2 = torch.nan_to_num(top2, neginf=0.0, posinf=0.0)
        lead_pt = top2[:, 0]
        sublead_pt = top2[:, 1]

        # Top pT list for head
        toppt_vals = torch.topk(pT_masked, k=self.toppt, dim=1).values  # [B, 4]
        toppt_vals = torch.nan_to_num(toppt_vals, neginf=0.0, posinf=0.0)
        toppt_feats = torch.log1p(toppt_vals.clamp(min=0.0))  # [B, 4]

        # Type counts (ids 1..10), normalized
        counts = []
        for k in range(1, 11):
            counts.append(((obj_id == k) & valid).sum(dim=1, dtype=torch.float32) / float(self.n_obj))
        type_counts = torch.stack(counts, dim=1)  # [B, 10]

        # Object features for transformer
        logE = torch.log1p(E.clamp(min=0.0))  # [B, 18]
        logpT = torch.log1p(pT.clamp(min=0.0))  # [B, 18]
        sinphi = torch.sin(phi)  # [B, 18]
        cosphi = torch.cos(phi)  # [B, 18]
        pTfrac = pT / (HT[:, None] + 1e-6)  # [B, 18]
        denom = (pT * torch.cosh(eta).clamp(min=1e-6) + 1e-6)  # [B, 18]
        e_ratio = (E / denom).clamp(min=1e-6)  # [B, 18]
        log_er = torch.log(e_ratio)  # [B, 18]

        obj_base = torch.stack(
            [logE, logpT, eta, sinphi, cosphi, pTfrac, log_er],
            dim=2,
        )  # [B, 18, 7]
        obj_emb = self.type_emb(obj_id)  # [B, 18, 16]
        obj_feat = torch.cat([obj_base, obj_emb], dim=2)  # [B, 18, 23]

        h0 = self.obj_in(obj_feat)  # [B, 18, 128]

        # Transformer with key padding mask: True means "ignore"
        h = self.encoder(h0, src_key_padding_mask=~valid)  # [B, 18, 128]

        # Masked pooling
        vmask = valid[:, :, None]  # [B, 18, 1]
        h_sum = (h * vmask).sum(dim=1)  # [B, 128]
        h_mean = h_sum / nobj_clamp[:, None]  # [B, 128]
        h_max = h.masked_fill(~vmask, float("-inf")).amax(dim=1)  # [B, 128]
        h_max = torch.nan_to_num(h_max, neginf=0.0, posinf=0.0)

        scores = self.pool_score(h).squeeze(-1)  # [B, 18]
        scores = scores.masked_fill(~valid, float("-inf"))
        w = torch.softmax(scores, dim=1)  # [B, 18]
        h_attn = (h * w[:, :, None]).sum(dim=1)  # [B, 128]

        # Pairwise engineered features (in float32)
        pair_feats = self._pairwise_features(E, pT, eta, phi, valid)  # [B, 21]

        # Global features (11)
        global_feats = torch.stack(
            [
                logmet,
                torch.sin(phi_met),
                torch.cos(phi_met),
                logHT,
                (nobj / float(self.n_obj)).clamp(0.0, 1.0),
                torch.log1p(lead_pt.clamp(min=0.0)),
                torch.log1p(sublead_pt.clamp(min=0.0)),
                adphi_min,
                adphi_mean,
                adphi_max,
                torch.log1p(mt_max.clamp(min=0.0)),
            ],
            dim=1,
        )  # [B, 11]

        z = torch.cat(
            [
                global_feats,       # [B, 11]
                type_counts,        # [B, 10]
                h_attn,             # [B, 128]
                h_mean,             # [B, 128]
                h_max,              # [B, 128]
                pair_feats,         # [B, 21]
                toppt_feats,        # [B, 4]
            ],
            dim=1,
        )  # [B, 430]

        logits = self.head(z).squeeze(1)  # [B]
        return logits


def make_model(example_object):
    return BinaryClassifier(example_object)


EPOCHS = 15


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    lr = 3.0e-4
    wd = 1.5e-2

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    steps_per_epoch = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        epochs=max(1, epochs),
        steps_per_epoch=steps_per_epoch,
        pct_start=0.12,
        div_factor=12.0,
        final_div_factor=200.0,
        anneal_strategy="cos",
    )

    criterion = nn.BCEWithLogitsLoss()

    use_cuda = next(model.parameters()).is_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    best_auc = -1.0
    best_state = None
    patience = 4
    bad = 0
    smooth = 0.02

    def _eval(loader):
        model.eval()
        losses = []
        correct = 0
        total = 0
        all_logits = []
        all_y = []
        with torch.no_grad():
            for batch in loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                y = yb.float()
                y_s = y * (1.0 - smooth) + 0.5 * smooth

                logits = model(xb)  # [B]
                loss = criterion(logits, y_s)
                losses.append(loss.detach().float().cpu().item())

                pred = (logits > 0).long()
                correct += (pred == yb).sum().item()
                total += int(yb.numel())

                all_logits.append(logits.detach().float().cpu())
                all_y.append(yb.detach().cpu())

        mean_loss = float(np.mean(losses)) if losses else 0.0
        acc = correct / max(1, total)

        logits_cat = torch.cat(all_logits, dim=0).numpy()
        y_cat = torch.cat(all_y, dim=0).numpy()
        try:
            auc = float(roc_auc_score(y_cat, logits_cat))
        except Exception:
            auc = 0.5
        return mean_loss, acc, auc

    for epoch in range(int(epochs)):
        model.train()
        losses = []
        correct = 0
        total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            y = yb.float()
            y_s = y * (1.0 - smooth) + 0.5 * smooth

            opt.zero_grad(set_to_none=True)

            if use_cuda:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(xb)  # [B]
                    loss = criterion(logits, y_s)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, y_s)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            sched.step()

            losses.append(loss.detach().float().cpu().item())
            pred = (logits.detach() > 0).long()
            correct += (pred == yb).sum().item()
            total += int(yb.numel())

        tr_loss = float(np.mean(losses)) if losses else 0.0
        tr_acc = correct / max(1, total)

        va_loss, va_acc, va_auc = _eval(val_loader)

        train_loss_hist.append(tr_loss)
        val_loss_hist.append(va_loss)
        train_acc_hist.append(tr_acc)
        val_acc_hist.append(va_acc)

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

