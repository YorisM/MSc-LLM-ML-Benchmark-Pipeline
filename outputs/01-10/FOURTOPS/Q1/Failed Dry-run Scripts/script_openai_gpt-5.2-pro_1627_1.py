
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
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Fixed event structure
    N_OBJ = 18
    OBJ_STRIDE = 5

    # Engineered feature sizes
    C_OBJ = 10          # per-object continuous engineered features
    T_COUNT = 12        # counts for object ids 1..T_COUNT as extra global features
    G_BASE = 7          # base global engineered features (before counts)
    G_DIM = G_BASE + T_COUNT
    OUT_DIM = G_DIM + N_OBJ * (1 + C_OBJ)  # 19 + 18*11 = 217

    # Numeric stability / scaling
    SCALE = 1e3         # MeV -> GeV
    CLIP_Z = 6.0
    ETA_CLIP = 5.0
    EPS = 1e-6

    def __init__(self):
        self.global_mean = None  # FloatTensor[G_DIM]
        self.global_std = None   # FloatTensor[G_DIM]
        self.obj_mean = None     # FloatTensor[C_OBJ]
        self.obj_std = None      # FloatTensor[C_OBJ]
        self.max_obj_id_seen = 0

    def make_loader_cfg(self) -> dict:
        bs = 1024 if torch.cuda.is_available() else 512
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": bs,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": bool(torch.cuda.is_available()),

            "collate": None,
            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False, "batch_size": bs}
        }

    @staticmethod
    def _wrap_dphi(dphi: torch.Tensor) -> torch.Tensor:
        # dphi wrapped into [-pi, pi]
        return torch.atan2(torch.sin(dphi), torch.cos(dphi))

    def _featurize_components(self, X: torch.Tensor):
        """
        X: FloatTensor[B,92]
        Returns:
          global_feat: FloatTensor[B,G_DIM]
          obj_id:      LongTensor[B,N_OBJ]
          cont:        FloatTensor[B,N_OBJ,C_OBJ]
          pad_mask:    BoolTensor[B,N_OBJ]  True where padded / invalid object
        """
        B = int(X.shape[0])
        X = X.float()

        met = (X[:, 0] / self.SCALE).clamp_min(0.0)                           # [B]
        met_phi = X[:, 1]                                                     # [B]
        log_met = torch.log1p(met)                                            # [B]
        sin_met = torch.sin(met_phi)                                          # [B]
        cos_met = torch.cos(met_phi)                                          # [B]

        objs = X[:, 2:].view(B, self.N_OBJ, self.OBJ_STRIDE)                  # [B,18,5]
        obj_id_f = objs[:, :, 0]                                              # [B,18]
        obj_id = torch.round(obj_id_f).clamp_(0, 63).long()                   # [B,18]
        pad_mask = (obj_id == 0)                                              # [B,18]
        live = (~pad_mask).float()                                            # [B,18]

        E = (objs[:, :, 1] / self.SCALE).clamp_min(0.0)                       # [B,18] GeV
        pT = (objs[:, :, 2] / self.SCALE).clamp_min(0.0)                      # [B,18] GeV
        eta = objs[:, :, 3].clamp(-self.ETA_CLIP, self.ETA_CLIP)              # [B,18]
        phi = objs[:, :, 4]                                                   # [B,18]

        logE = torch.log1p(E)                                                 # [B,18]
        logpT = torch.log1p(pT)                                               # [B,18]
        sinphi = torch.sin(phi)                                               # [B,18]
        cosphi = torch.cos(phi)                                               # [B,18]

        # p = pT * cosh(eta), m^2 = E^2 - p^2
        p = pT * torch.cosh(eta)                                              # [B,18]
        m2 = (E * E - p * p).clamp_min(0.0)                                   # [B,18]
        m = torch.sqrt(m2 + self.EPS)                                         # [B,18]
        logm = torch.log1p(m)                                                 # [B,18]

        dphi = self._wrap_dphi(phi - met_phi[:, None])                        # [B,18]
        cosdphi = torch.cos(dphi)                                             # [B,18]
        sindphi = torch.sin(dphi)                                             # [B,18]

        pt_over_met = pT / (met[:, None] + 0.1)                               # [B,18]
        log_pt_over_met = torch.log1p(pt_over_met.clamp_min(0.0))             # [B,18]

        e_over_p = (E / (p + 0.1)).clamp(0.0, 5.0)                            # [B,18]

        cont = torch.stack(
            [logE, logpT, eta, sinphi, cosphi, logm, cosdphi, sindphi, log_pt_over_met, e_over_p],
            dim=-1
        )                                                                     # [B,18,10]

        # Event-level engineered features
        nobj = live.sum(dim=1) / float(self.N_OBJ)                            # [B]
        ht = (pT * live).sum(dim=1)                                           # [B]
        sum_e = (E * live).sum(dim=1)                                         # [B]
        log_ht = torch.log1p(ht)                                              # [B]
        log_sum_e = torch.log1p(sum_e)                                        # [B]
        met_over_ht = met / (ht + 0.1)                                        # [B]
        log_met_over_ht = torch.log1p(met_over_ht.clamp_min(0.0))             # [B]

        global_base = torch.stack(
            [log_met, sin_met, cos_met, nobj, log_ht, log_sum_e, log_met_over_ht],
            dim=-1
        )                                                                     # [B,7]

        # Object id counts (1..T_COUNT)
        if self.T_COUNT > 0:
            ids_for_counts = obj_id.clone()                                   # [B,18]
            ids_for_counts[(ids_for_counts > self.T_COUNT)] = 0
            oh = F.one_hot(ids_for_counts, num_classes=self.T_COUNT + 1).float()  # [B,18,T+1]
            counts = oh.sum(dim=1)[:, 1:] / float(self.N_OBJ)                 # [B,T]
            global_feat = torch.cat([global_base, counts], dim=-1)            # [B,7+T]
        else:
            global_feat = global_base                                         # [B,7]

        return global_feat, obj_id, cont, pad_mask

    def fit(self, X, y=None):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)
        X = X.float()

        # Determine maximum object id seen (full scan only on id columns)
        with torch.no_grad():
            ids_all = X[:, 2::self.OBJ_STRIDE].float()
            self.max_obj_id_seen = int(torch.round(ids_all).clamp_(0, 1e9).max().item())

        # Subsample for statistics (deterministic)
        n = int(X.shape[0])
        max_fit = 80000
        if n > max_fit:
            g = torch.Generator(device="cpu")
            g.manual_seed(42)
            idx = torch.randperm(n, generator=g)[:max_fit]
            Xs = X.index_select(0, idx)
        else:
            Xs = X

        # Accumulate stats in chunks
        chunk = 16384
        sum_g = torch.zeros(self.G_DIM, dtype=torch.float64)
        sumsq_g = torch.zeros(self.G_DIM, dtype=torch.float64)
        count_g = 0

        sum_o = torch.zeros(self.C_OBJ, dtype=torch.float64)
        sumsq_o = torch.zeros(self.C_OBJ, dtype=torch.float64)
        count_o = 0.0

        with torch.no_grad():
            for i in range(0, int(Xs.shape[0]), chunk):
                Xc = Xs[i:i + chunk]
                gf, obj_id, cont, pad_mask = self._featurize_components(Xc)     # gf [b,G], cont [b,18,C]
                b = int(gf.shape[0])

                # Global stats over events
                sum_g += gf.double().sum(dim=0)
                sumsq_g += (gf.double() * gf.double()).sum(dim=0)
                count_g += b

                # Object stats over real objects only
                live = (~pad_mask).double().unsqueeze(-1)                      # [b,18,1]
                cont_d = cont.double()
                sum_o += (cont_d * live).sum(dim=(0, 1))                       # [C]
                sumsq_o += (cont_d * cont_d * live).sum(dim=(0, 1))            # [C]
                count_o += float(live.sum().item())

        # Finalize mean/std
        count_g = max(int(count_g), 1)
        mean_g = sum_g / float(count_g)
        var_g = (sumsq_g / float(count_g) - mean_g * mean_g).clamp_min(0.0)
        std_g = torch.sqrt(var_g + 1e-6).clamp_min(1e-3)

        if count_o < 1.0:
            mean_o = torch.zeros(self.C_OBJ, dtype=torch.float64)
            std_o = torch.ones(self.C_OBJ, dtype=torch.float64)
        else:
            mean_o = sum_o / float(count_o)
            var_o = (sumsq_o / float(count_o) - mean_o * mean_o).clamp_min(0.0)
            std_o = torch.sqrt(var_o + 1e-6).clamp_min(1e-3)

        self.global_mean = mean_g.float().contiguous()                          # [G_DIM]
        self.global_std = std_g.float().contiguous()                            # [G_DIM]
        self.obj_mean = mean_o.float().contiguous()                             # [C_OBJ]
        self.obj_std = std_o.float().contiguous()                               # [C_OBJ]
        return self

    def transform(self, X):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)
        X = X.float()

        out_list = []
        chunk = 32768
        with torch.no_grad():
            for i in range(0, int(X.shape[0]), chunk):
                Xc = X[i:i + chunk]
                gf, obj_id, cont, pad_mask = self._featurize_components(Xc)

                # Standardize
                gf = (gf - self.global_mean[None, :]) / self.global_std[None, :]   # [b,G_DIM]
                gf = gf.clamp(-self.CLIP_Z, self.CLIP_Z)

                cont = (cont - self.obj_mean[None, None, :]) / self.obj_std[None, None, :]  # [b,18,C]
                cont = cont.clamp(-self.CLIP_Z, self.CLIP_Z)

                # Zero-out padded objects after normalization
                live = (~pad_mask).float().unsqueeze(-1)                             # [b,18,1]
                cont = cont * live                                                   # [b,18,C]

                # Flatten: [global, (id + cont)*18]
                obj_block = torch.cat([obj_id.float().unsqueeze(-1), cont], dim=-1)  # [b,18,11]
                flat = obj_block.reshape(int(Xc.shape[0]), self.N_OBJ * (1 + self.C_OBJ))  # [b,198]
                feats = torch.cat([gf, flat], dim=-1)                                # [b,217]
                out_list.append(feats.contiguous())

        return torch.cat(out_list, dim=0)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        in_dim = int(sample_object.shape[-1])
        self.n_obj = MyPreprocessor.N_OBJ
        self.c_obj = MyPreprocessor.C_OBJ
        self.obj_block = 1 + self.c_obj
        self.global_dim = in_dim - self.n_obj * self.obj_block
        if self.global_dim <= 0:
            raise ValueError(f"Invalid inferred global_dim={self.global_dim} from in_dim={in_dim}")

        d_model = 64
        emb_dim = 8
        self.id_vocab = 64

        self.type_emb = nn.Embedding(self.id_vocab, emb_dim, padding_idx=0)

        self.obj_mlp = nn.Sequential(
            nn.Linear(emb_dim + self.c_obj, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.10),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.10),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)

        head_in = d_model * 4  # [cls, mean, max, global_emb]
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

        # Init
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, batch_x):
        # batch_x: FloatTensor[B,F]
        B = int(batch_x.shape[0])

        g = batch_x[:, :self.global_dim]                                         # [B,G]
        obj_flat = batch_x[:, self.global_dim:]                                  # [B,18*11]
        obj = obj_flat.view(B, self.n_obj, self.obj_block)                       # [B,18,11]

        obj_id = obj[:, :, 0].round().clamp(0, self.id_vocab - 1).long()         # [B,18]
        pad_mask = (obj_id == 0)                                                 # [B,18]
        cont = obj[:, :, 1:]                                                     # [B,18,10]

        global_emb = self.global_mlp(g)                                          # [B,64]

        t = self.type_emb(obj_id)                                                # [B,18,8]
        obj_in = torch.cat([t, cont], dim=-1)                                    # [B,18,18]
        obj_lat = self.obj_mlp(obj_in)                                           # [B,18,64]

        cls = self.cls_token.expand(B, 1, -1) + global_emb.unsqueeze(1)          # [B,1,64]
        seq = torch.cat([cls, obj_lat], dim=1)                                   # [B,19,64]

        key_padding_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=seq.device), pad_mask],
            dim=1
        )                                                                        # [B,19]

        enc = self.encoder(seq, src_key_padding_mask=key_padding_mask)           # [B,19,64]
        cls_out = enc[:, 0, :]                                                   # [B,64]
        obj_out = enc[:, 1:, :]                                                  # [B,18,64]

        live = (~pad_mask).float().unsqueeze(-1)                                 # [B,18,1]
        denom = live.sum(dim=1).clamp_min(1.0)                                   # [B,1]
        mean_pool = (obj_out * live).sum(dim=1) / denom                          # [B,64]

        obj_out_masked = obj_out.masked_fill(pad_mask.unsqueeze(-1), -1e9)       # [B,18,64]
        max_pool = obj_out_masked.max(dim=1).values                              # [B,64]

        h = torch.cat([cls_out, mean_pool, max_pool, global_emb], dim=-1)        # [B,256]
        logit = self.head(h).squeeze(-1)                                         # [B]
        return logit

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    lr = 2e-3
    wd = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    total_steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        total_steps=total_steps,
        pct_start=0.10,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    best_auc = -1.0
    best_state = None
    patience = 6
    bad = 0
    min_epochs = 5

    for epoch in range(int(epochs)):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)                                               # [B]
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                running_loss += float(loss.item()) * int(yb.shape[0])
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                running_correct += int((preds == yb.long()).sum().item())
                running_total += int(yb.shape[0])

        tr_loss = running_loss / max(1, running_total)
        tr_acc = running_correct / max(1, running_total)
        train_loss_hist.append(tr_loss)
        train_acc_hist.append(tr_acc)

        # Validation
        model.eval()
        vloss_sum = 0.0
        vtotal = 0
        vcorrect = 0
        all_logits = []
        all_y = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb_f = yb.to(device, non_blocking=True).float()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(xb)                                           # [B]
                    loss = loss_fn(logits, yb_f)

                vloss_sum += float(loss.item()) * int(yb.shape[0])
                vtotal += int(yb.shape[0])

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                vcorrect += int((preds.cpu() == yb.cpu()).sum().item())

                all_logits.append(logits.detach().float().cpu())
                all_y.append(yb.detach().cpu())

        va_loss = vloss_sum / max(1, vtotal)
        va_acc = vcorrect / max(1, vtotal)
        val_loss_hist.append(va_loss)
        val_acc_hist.append(va_acc)

        y_true = torch.cat(all_y).numpy()
        y_score = torch.cat(all_logits).numpy()
        try:
            va_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            va_auc = -1.0

        # Early stopping on AUC
        improved = (va_auc > best_auc + 1e-4)
        if improved:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch + 1 >= min_epochs and bad >= patience:
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

