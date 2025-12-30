
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
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # Constants describing the raw event layout
        self.n_objects = 18
        self.obj_stride = 5

        # Engineered feature sizes
        self.K_types = 8  # counts for IDs 1..8 + other
        self.gs_dim = 21  # 3 global + 18 summary
        self.obj_num_dim = 7
        self.obj_block_dim = 1 + 1 + self.obj_num_dim  # id + mask + numeric = 9
        self.out_dim = self.gs_dim + self.n_objects * self.obj_block_dim  # 21 + 18*9 = 183

        # Embedding safety
        self.max_embed = 32  # clamp ids to [0, 31] for embedding

        # Fit state (picklable)
        self.gs_mean = None
        self.gs_std = None
        self.obj_mean = None
        self.obj_std = None
        self.max_obj_id_seen = None

    def make_loader_cfg(self):
        pin = bool(torch.cuda.is_available())
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": pin,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    @staticmethod
    def _safe_std(var, eps=1e-6):
        return torch.sqrt(torch.clamp(var, min=eps))

    @staticmethod
    def _dphi(phi1, phi2):
        # returns value in [-pi, pi]
        return torch.atan2(torch.sin(phi1 - phi2), torch.cos(phi1 - phi2))

    def _split_raw(self, X):
        # X: [N, 92]
        met = X[:, 0]  # [N]
        met_phi = X[:, 1]  # [N]
        objs = X[:, 2:].view(-1, self.n_objects, self.obj_stride)  # [N,18,5]
        obj_id = objs[:, :, 0]  # [N,18]
        E = objs[:, :, 1]       # [N,18]
        pT = objs[:, :, 2]      # [N,18]
        eta = objs[:, :, 3]     # [N,18]
        phi = objs[:, :, 4]     # [N,18]
        return met, met_phi, obj_id, E, pT, eta, phi

    def _sort_by_pt(self, obj_id, E, pT, eta, phi):
        # obj_id,E,pT,eta,phi: [N,18]
        mask = (obj_id > 0).to(pT.dtype)  # [N,18]
        pT_sort = pT.clone()
        pT_sort = pT_sort.masked_fill(mask < 0.5, -1.0)  # padding goes to end
        idx = torch.argsort(pT_sort, dim=1, descending=True)  # [N,18]
        gather_idx = idx  # [N,18]

        def g(a):
            return torch.gather(a, 1, gather_idx)

        return g(obj_id), g(E), g(pT), g(eta), g(phi)

    def _featurize(self, X):
        # Returns:
        #   gs: [N,21]
        #   obj_block: [N,18,9] = [id, mask, 7 numeric]
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)
        X = X.to(dtype=torch.float32, device="cpu")

        met, met_phi, obj_id, E, pT, eta, phi = self._split_raw(X)

        # Sort objects by pT descending to give stable ranks
        obj_id, E, pT, eta, phi = self._sort_by_pt(obj_id, E, pT, eta, phi)

        # Mask
        mask = (obj_id > 0).to(torch.float32)  # [N,18]

        # Unit conversion to GeV where useful
        met_g = met / 1000.0  # [N]
        E_g = E / 1000.0      # [N,18]
        pT_g = pT / 1000.0    # [N,18]

        # Basic global features
        log_met = torch.log1p(torch.clamp(met_g, min=0.0))  # [N]
        sin_met = torch.sin(met_phi)  # [N]
        cos_met = torch.cos(met_phi)  # [N]

        # Per-object derived kinematics
        eta_c = torch.clamp(eta, -5.0, 5.0)  # [N,18]
        sin_phi = torch.sin(phi)  # [N,18]
        cos_phi = torch.cos(phi)  # [N,18]

        # Momentum components in GeV
        px = pT_g * cos_phi  # [N,18]
        py = pT_g * sin_phi  # [N,18]
        pz = pT_g * torch.sinh(eta_c)  # [N,18]

        # Mass (GeV): m^2 = E^2 - |p|^2
        p2 = pT_g * pT_g + pz * pz  # [N,18]
        m2 = torch.clamp(E_g * E_g - p2, min=0.0)  # [N,18]
        m = torch.sqrt(m2)  # [N,18]
        log_m = torch.log1p(m)  # [N,18]

        log_pt = torch.log1p(torch.clamp(pT_g, min=0.0))  # [N,18]
        log_E = torch.log1p(torch.clamp(E_g, min=0.0))    # [N,18]
        pt_over_E = pT_g / (E_g + 1e-6)  # [N,18]

        # Summary features
        nobj = (mask.sum(dim=1) / float(self.n_objects))  # [N]
        HT = (pT_g * mask).sum(dim=1)  # [N]
        sumE = (E_g * mask).sum(dim=1)  # [N]
        lead_pt = pT_g[:, 0]  # [N]
        sublead_pt = pT_g[:, 1]  # [N]
        ST = HT + met_g  # [N]

        log_HT = torch.log1p(torch.clamp(HT, min=0.0))       # [N]
        log_sumE = torch.log1p(torch.clamp(sumE, min=0.0))   # [N]
        log_lead = torch.log1p(torch.clamp(lead_pt, min=0.0))      # [N]
        log_sub = torch.log1p(torch.clamp(sublead_pt, min=0.0))    # [N]
        log_ST = torch.log1p(torch.clamp(ST, min=0.0))       # [N]

        # deltaR and dijet mass for leading pair
        m12_mask = (mask[:, 0] * mask[:, 1])  # [N]
        deta12 = eta_c[:, 0] - eta_c[:, 1]  # [N]
        dphi12 = self._dphi(phi[:, 0], phi[:, 1])  # [N]
        dR12 = torch.sqrt(deta12 * deta12 + dphi12 * dphi12) * m12_mask  # [N]

        # m12 in GeV
        E12 = E_g[:, 0] + E_g[:, 1]  # [N]
        px12 = px[:, 0] + px[:, 1]   # [N]
        py12 = py[:, 0] + py[:, 1]   # [N]
        pz12 = pz[:, 0] + pz[:, 1]   # [N]
        m2_12 = torch.clamp(E12 * E12 - (px12 * px12 + py12 * py12 + pz12 * pz12), min=0.0)  # [N]
        m12 = torch.sqrt(m2_12) * m12_mask  # [N]
        log_m12 = torch.log1p(m12)  # [N]

        # m1234 from leading 4 objects (masked)
        m4 = mask[:, :4]  # [N,4]
        E4 = (E_g[:, :4] * m4).sum(dim=1)  # [N]
        px4 = (px[:, :4] * m4).sum(dim=1)  # [N]
        py4 = (py[:, :4] * m4).sum(dim=1)  # [N]
        pz4 = (pz[:, :4] * m4).sum(dim=1)  # [N]
        m2_4 = torch.clamp(E4 * E4 - (px4 * px4 + py4 * py4 + pz4 * pz4), min=0.0)  # [N]
        m1234 = torch.sqrt(m2_4)  # [N]
        log_m1234 = torch.log1p(m1234)  # [N]

        # Type counts (IDs 1..K_types, plus "other"), normalized by n_objects
        counts = []
        for t in range(1, self.K_types + 1):
            counts.append(((obj_id == float(t)).to(torch.float32).sum(dim=1) / float(self.n_objects)))  # [N]
        counts = torch.stack(counts, dim=1)  # [N,K_types]
        known = counts.sum(dim=1)  # [N]
        other = torch.clamp(nobj - known, min=0.0)  # [N]
        counts_all = torch.cat([counts, other.unsqueeze(1)], dim=1)  # [N,K_types+1] = [N,9]

        # gs: [N, 21] = 3 global + 9 summary scalars + 9 counts
        gs = torch.cat(
            [
                log_met.unsqueeze(1), sin_met.unsqueeze(1), cos_met.unsqueeze(1),  # [N,3]
                nobj.unsqueeze(1), log_HT.unsqueeze(1), log_sumE.unsqueeze(1), log_lead.unsqueeze(1), log_sub.unsqueeze(1),
                log_ST.unsqueeze(1), dR12.unsqueeze(1), log_m12.unsqueeze(1), log_m1234.unsqueeze(1),  # [N,9]
                counts_all,  # [N,9]
            ],
            dim=1,
        )  # [N,21]

        # Per-object numeric: [N,18,7]
        obj_num = torch.stack([log_pt, log_E, eta_c, sin_phi, cos_phi, log_m, pt_over_E], dim=2)  # [N,18,7]

        # Pack object block [id, mask, scaled_numeric] later; here keep id float + mask
        obj_block = torch.cat([obj_id.unsqueeze(2), mask.unsqueeze(2), obj_num], dim=2)  # [N,18,9]

        return gs, obj_block

    def fit(self, X, y=None):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)
        X = X.to(dtype=torch.float32, device="cpu")

        # Track max object ID (for diagnostics)
        raw_ids = X[:, 2::5]
        try:
            self.max_obj_id_seen = int(raw_ids.max().item())
        except Exception:
            self.max_obj_id_seen = None

        # Streaming accumulation for gs stats and obj numeric stats
        gs_sum = torch.zeros(self.gs_dim, dtype=torch.float64)
        gs_sumsq = torch.zeros(self.gs_dim, dtype=torch.float64)
        gs_count = 0

        obj_sum = torch.zeros(self.obj_num_dim, dtype=torch.float64)
        obj_sumsq = torch.zeros(self.obj_num_dim, dtype=torch.float64)
        obj_count = 0.0

        N = X.shape[0]
        bs = 8192
        for i in range(0, N, bs):
            xb = X[i : i + bs]  # [B,92]
            gs, obj_block = self._featurize(xb)  # gs: [B,21], obj_block: [B,18,9]
            B = gs.shape[0]

            gs64 = gs.to(torch.float64)
            gs_sum += gs64.sum(dim=0)
            gs_sumsq += (gs64 * gs64).sum(dim=0)
            gs_count += B

            mask = obj_block[:, :, 1:2]  # [B,18,1]
            obj_num = obj_block[:, :, 2:]  # [B,18,7]
            # Weighted by mask to ignore padded objects
            w = mask.to(torch.float64)  # [B,18,1]
            obj64 = obj_num.to(torch.float64)  # [B,18,7]
            obj_sum += (obj64 * w).sum(dim=(0, 1))
            obj_sumsq += ((obj64 * obj64) * w).sum(dim=(0, 1))
            obj_count += float(w.sum().item())

        gs_mean = gs_sum / max(gs_count, 1)
        gs_var = gs_sumsq / max(gs_count, 1) - gs_mean * gs_mean
        gs_std = self._safe_std(gs_var.to(torch.float32)).to(torch.float32)

        if obj_count < 1.0:
            obj_mean = torch.zeros(self.obj_num_dim, dtype=torch.float32)
            obj_std = torch.ones(self.obj_num_dim, dtype=torch.float32)
        else:
            obj_mean = (obj_sum / obj_count).to(torch.float32)
            obj_var = (obj_sumsq / obj_count).to(torch.float32) - obj_mean * obj_mean
            obj_std = self._safe_std(obj_var).to(torch.float32)

        # Avoid tiny stds
        gs_std = torch.clamp(gs_std, min=1e-3)
        obj_std = torch.clamp(obj_std, min=1e-3)

        # Store as numpy for pickle stability
        self.gs_mean = gs_mean.to(torch.float32).cpu().numpy()
        self.gs_std = gs_std.cpu().numpy()
        self.obj_mean = obj_mean.cpu().numpy()
        self.obj_std = obj_std.cpu().numpy()

        return self

    def transform(self, X):
        gs, obj_block = self._featurize(X)  # gs: [N,21], obj_block: [N,18,9]

        # Clamp object IDs to embedding range, keep as float for storage
        obj_id = obj_block[:, :, 0].clamp(0, float(self.max_embed - 1))  # [N,18]
        mask = obj_block[:, :, 1]  # [N,18]
        obj_num = obj_block[:, :, 2:]  # [N,18,7]

        # Scale
        gs_mean = torch.as_tensor(self.gs_mean, dtype=torch.float32, device=gs.device)  # [21]
        gs_std = torch.as_tensor(self.gs_std, dtype=torch.float32, device=gs.device)    # [21]
        obj_mean = torch.as_tensor(self.obj_mean, dtype=torch.float32, device=gs.device)  # [7]
        obj_std = torch.as_tensor(self.obj_std, dtype=torch.float32, device=gs.device)    # [7]

        gs_scaled = (gs - gs_mean) / gs_std  # [N,21]

        obj_scaled = (obj_num - obj_mean.view(1, 1, -1)) / obj_std.view(1, 1, -1)  # [N,18,7]
        obj_scaled = obj_scaled * mask.unsqueeze(2)  # [N,18,7] keep padding at 0

        obj_pack = torch.cat([obj_id.unsqueeze(2), mask.unsqueeze(2), obj_scaled], dim=2)  # [N,18,9]
        obj_flat = obj_pack.reshape(obj_pack.shape[0], -1)  # [N,162]

        out = torch.cat([gs_scaled, obj_flat], dim=1)  # [N,183]
        return out


def make_preprocessor():
    return MyPreprocessor()


# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        in_dim = int(sample_object.shape[-1])

        self.gs_dim = 21
        self.n_objects = 18
        self.obj_block_dim = 9
        self.obj_num_dim = 7

        assert in_dim == self.gs_dim + self.n_objects * self.obj_block_dim, f"Unexpected input dim: {in_dim}"

        emb_dim = 8
        self.type_emb = nn.Embedding(32, emb_dim)

        obj_in_dim = emb_dim + self.obj_num_dim  # 8 + 7 = 15
        h = 96

        self.obj_ln = nn.LayerNorm(obj_in_dim)
        self.obj_mlp = nn.Sequential(
            nn.Linear(obj_in_dim, h),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.attn = nn.Linear(h, 1)

        head_in = self.gs_dim + 3 * h  # gs + (attn, mean, max)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, batch_x):
        # batch_x: [B,183]
        B = batch_x.shape[0]
        gs = batch_x[:, : self.gs_dim]  # [B,21]
        obj_flat = batch_x[:, self.gs_dim :]  # [B,162]
        objs = obj_flat.view(B, self.n_objects, self.obj_block_dim)  # [B,18,9]

        obj_id = objs[:, :, 0].clamp(0, 31).long()  # [B,18]
        mask = objs[:, :, 1]  # [B,18]
        obj_num = objs[:, :, 2:]  # [B,18,7]

        emb = self.type_emb(obj_id)  # [B,18,8]
        obj_in = torch.cat([emb, obj_num], dim=-1)  # [B,18,15]
        obj_in = self.obj_ln(obj_in)  # [B,18,15]

        obj_h = self.obj_mlp(obj_in)  # [B,18,96]

        # Attention pooling
        attn_logits = self.attn(obj_h).squeeze(-1)  # [B,18]
        attn_logits = attn_logits.masked_fill(mask < 0.5, -1e9)
        w = torch.softmax(attn_logits, dim=1)  # [B,18]
        attn_pool = (obj_h * w.unsqueeze(-1)).sum(dim=1)  # [B,96]

        # Masked mean pooling
        msum = mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)  # [B,1]
        mean_pool = (obj_h * mask.unsqueeze(-1)).sum(dim=1) / msum  # [B,96]

        # Masked max pooling
        obj_h_masked = obj_h.masked_fill(mask.unsqueeze(-1) < 0.5, -1e9)  # [B,18,96]
        max_pool = obj_h_masked.max(dim=1).values  # [B,96]
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))  # [B,96]

        pooled = torch.cat([attn_pool, mean_pool, max_pool], dim=1)  # [B,288]
        feat = torch.cat([gs, pooled], dim=1)  # [B,309]
        logit = self.head(feat).squeeze(-1)  # [B]
        return logit


def make_model(example_object):
    return BinaryClassifier(example_object)


# ---------- MODEL TRAINING ----------
EPOCHS = 16


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    from utils.llm_io import normalise_batch

    model = model.to(device)
    use_amp = (device.type == "cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=8.0e-5)
    total_steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3.0e-3,
        total_steps=total_steps,
        pct_start=0.10,
        anneal_strategy="cos",
        div_factor=15.0,
        final_div_factor=250.0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    best_state = None
    best_auc = -1.0
    patience = 5
    bad_epochs = 0

    for epoch in range(int(epochs)):
        # ---- train ----
        model.train()
        running_loss = 0.0
        correct = 0
        n = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            x = view.batch_x  # [B,183]
            y = view.batch_y  # [B]
            y_f = y.float()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)  # [B]
                loss = F.binary_cross_entropy_with_logits(logits, y_f)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = int(y.shape[0])
            running_loss += float(loss.item()) * bs
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).to(torch.int64)
                correct += int((preds == y).sum().item())
                n += bs

        tr_loss = running_loss / max(n, 1)
        tr_acc = correct / max(n, 1)

        # ---- validate ----
        model.eval()
        v_running_loss = 0.0
        v_correct = 0
        v_n = 0

        all_scores = []
        all_y = []
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                x = view.batch_x
                y = view.batch_y
                y_f = y.float()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    loss = F.binary_cross_entropy_with_logits(logits, y_f)

                bs = int(y.shape[0])
                v_running_loss += float(loss.item()) * bs
                probs = torch.sigmoid(logits).detach()

                preds = (probs > 0.5).to(torch.int64)
                v_correct += int((preds == y).sum().item())
                v_n += bs

                all_scores.append(probs.float().cpu().numpy())
                all_y.append(y.cpu().numpy())

        va_loss = v_running_loss / max(v_n, 1)
        va_acc = v_correct / max(v_n, 1)

        scores = np.concatenate(all_scores, axis=0)
        ys = np.concatenate(all_y, axis=0)
        try:
            va_auc = float(roc_auc_score(ys, scores))
        except Exception:
            va_auc = -1.0

        train_loss_hist.append(tr_loss)
        val_loss_hist.append(va_loss)
        train_acc_hist.append(tr_acc)
        val_acc_hist.append(va_acc)

        # Early stopping on AUC
        if va_auc > best_auc + 1e-4:
            best_auc = va_auc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
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


