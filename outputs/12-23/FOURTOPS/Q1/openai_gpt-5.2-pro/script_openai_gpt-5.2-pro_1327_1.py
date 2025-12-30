
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
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

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

import math
import copy
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score


class MyPreprocessor:
    # Total flat length per event (X_train & X_val): 92
    # Global:
    #   X[:, 0] : MET magnitude
    #   X[:, 1] : MET phi
    # Objects: 18 objects, each 5 features, starting at X[:,2:]
    #   [obj_id, E, pT, eta, phi]
    #
    # Output tensor shape per event: [19, 8]
    #   token 0 (global token): [type_id, 7 global engineered features]
    #   tokens 1..18 (object tokens): [type_id, 7 object engineered features]
    #
    # Token feature layout (last dim=8):
    #   [:, :, 0]   type_id  (float, cast to long in model; 0 is padding)
    #   [:, :, 1:]  7 continuous engineered features (float, normalized)
    #
    # Shapes:
    #   X_out: [N, 19, 8]

    def __init__(self):
        self.global_id = 63  # reserved id for global token
        self.obj_clip_id = 62
        self.eps = 1e-6

        # normalization params (numpy float32, picklable)
        self.obj_mean = None  # (7,)
        self.obj_std = None   # (7,)
        self.glob_mean = None # (7,)
        self.glob_std = None  # (7,)

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
            "eval_overrides": {"shuffle": False},
        }

    @staticmethod
    def _parse(X: torch.Tensor):
        # X: [N, 92]
        N = X.shape[0]
        objs = X[:, 2:].view(N, 18, 5)  # [N, 18, 5]
        obj_id = torch.round(objs[:, :, 0]).clamp(min=0)  # [N, 18]
        E = objs[:, :, 1]  # [N, 18]
        pT = objs[:, :, 2]  # [N, 18]
        eta = objs[:, :, 3]  # [N, 18]
        phi = objs[:, :, 4]  # [N, 18]
        met = X[:, 0]  # [N]
        met_phi = X[:, 1]  # [N]
        return obj_id, E, pT, eta, phi, met, met_phi

    def _engineer(self, X: torch.Tensor):
        # returns:
        #   type_ids: [N, 18] float (0 padding)
        #   obj_cont: [N, 18, 7] float
        #   glob_cont: [N, 7] float
        obj_id, E, pT, eta, phi, met, met_phi = self._parse(X)

        # mask for present objects
        mask = (obj_id > 0)  # [N, 18] bool

        # Basic stabilized transforms
        Epos = E.clamp(min=0)          # [N, 18]
        pTpos = pT.clamp(min=0)        # [N, 18]
        logE = torch.log1p(Epos)       # [N, 18]
        logpT = torch.log1p(pTpos)     # [N, 18]
        eta_c = eta.clamp(-5.0, 5.0)   # [N, 18]
        sinphi = torch.sin(phi)        # [N, 18]
        cosphi = torch.cos(phi)        # [N, 18]

        # Invariant-mass proxy using (E, pT, eta); stable pz via clipped eta
        eta_m = eta_c.clamp(-4.0, 4.0)               # [N, 18]
        pz = pTpos * torch.sinh(eta_m)               # [N, 18]
        p2 = pTpos * pTpos + pz * pz                 # [N, 18]
        m2 = (Epos * Epos - p2).clamp(min=0)         # [N, 18]
        logm = torch.log1p(torch.sqrt(m2 + 1e-12))   # [N, 18]

        # MET-object angular correlation
        met_phi_u = met_phi.unsqueeze(1)             # [N, 1]
        cos_dphi_met = torch.cos(phi - met_phi_u)    # [N, 18]

        # Object continuous features: 7
        obj_cont = torch.stack(
            [logE, logpT, eta_c, sinphi, cosphi, logm, cos_dphi_met], dim=-1
        )  # [N, 18, 7]

        # Global engineered features: 7
        metpos = met.clamp(min=0)                    # [N]
        logmet = torch.log1p(metpos)                 # [N]
        sin_met = torch.sin(met_phi)                 # [N]
        cos_met = torch.cos(met_phi)                 # [N]

        HT = (pTpos * mask.to(pTpos.dtype)).sum(dim=1)       # [N]
        logHT = torch.log1p(HT)                               # [N]
        nobj = mask.sum(dim=1).to(torch.float32) / 18.0       # [N]

        # max log pT among present objects
        neg_inf = torch.full_like(logpT, -1e9)                # [N, 18]
        max_logpT = torch.where(mask, logpT, neg_inf).max(dim=1).values  # [N]
        max_logpT = torch.where(mask.any(dim=1), max_logpT, torch.zeros_like(max_logpT))

        # mean |eta| among present objects
        denom = mask.sum(dim=1).clamp(min=1).to(torch.float32)  # [N]
        mean_abs_eta = (eta_c.abs() * mask.to(eta_c.dtype)).sum(dim=1) / denom  # [N]

        glob_cont = torch.stack(
            [logmet, sin_met, cos_met, logHT, nobj, max_logpT, mean_abs_eta], dim=-1
        )  # [N, 7]

        # type ids (keep padding zeros)
        type_ids = obj_id.clamp(0, self.obj_clip_id).to(torch.float32)  # [N, 18]
        return type_ids, mask, obj_cont, glob_cont

    def fit(self, X, y=None):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)

        with torch.no_grad():
            type_ids, mask, obj_cont, glob_cont = self._engineer(X)

            # Object stats over present objects only
            if mask.any():
                flat = obj_cont[mask]  # [N_present, 7]
                obj_mean = flat.mean(dim=0)  # [7]
                obj_std = flat.std(dim=0, unbiased=False).clamp(min=self.eps)  # [7]
            else:
                obj_mean = torch.zeros(7, dtype=torch.float32)
                obj_std = torch.ones(7, dtype=torch.float32)

            # Global stats over events
            glob_mean = glob_cont.mean(dim=0)  # [7]
            glob_std = glob_cont.std(dim=0, unbiased=False).clamp(min=self.eps)  # [7]

            self.obj_mean = obj_mean.cpu().numpy().astype(np.float32, copy=True)
            self.obj_std = obj_std.cpu().numpy().astype(np.float32, copy=True)
            self.glob_mean = glob_mean.cpu().numpy().astype(np.float32, copy=True)
            self.glob_std = glob_std.cpu().numpy().astype(np.float32, copy=True)

        return self

    def transform(self, X):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)

        obj_mean = torch.from_numpy(self.obj_mean).to(dtype=X.dtype, device=X.device)  # [7]
        obj_std = torch.from_numpy(self.obj_std).to(dtype=X.dtype, device=X.device)    # [7]
        glob_mean = torch.from_numpy(self.glob_mean).to(dtype=X.dtype, device=X.device)  # [7]
        glob_std = torch.from_numpy(self.glob_std).to(dtype=X.dtype, device=X.device)    # [7]

        with torch.no_grad():
            type_ids, mask, obj_cont, glob_cont = self._engineer(X)

            # Normalize
            obj_cont = (obj_cont - obj_mean.view(1, 1, 7)) / obj_std.view(1, 1, 7)  # [N, 18, 7]
            glob_cont = (glob_cont - glob_mean.view(1, 7)) / glob_std.view(1, 7)    # [N, 7]

            # Clip to reduce rare outliers impact
            obj_cont = obj_cont.clamp(-6.0, 6.0)   # [N, 18, 7]
            glob_cont = glob_cont.clamp(-6.0, 6.0) # [N, 7]

            # Zero-out padded objects' continuous features
            obj_cont = obj_cont * mask.unsqueeze(-1).to(obj_cont.dtype)  # [N, 18, 7]

            # Build tokens
            N = X.shape[0]
            global_type = torch.full((N, 1, 1), float(self.global_id), dtype=X.dtype, device=X.device)  # [N, 1, 1]
            global_tok = torch.cat([global_type, glob_cont.unsqueeze(1)], dim=2)  # [N, 1, 8]

            obj_type = type_ids.unsqueeze(-1)  # [N, 18, 1]
            obj_tok = torch.cat([obj_type, obj_cont], dim=2)  # [N, 18, 8]

            out = torch.cat([global_tok, obj_tok], dim=1).to(torch.float32)  # [N, 19, 8]
            return out


def make_preprocessor():
    return MyPreprocessor()


class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # sample_object expected shape: [B, 19, 8]
        if sample_object.dim() != 3 or sample_object.shape[-1] != 8:
            raise ValueError(f"Expected batch_x with shape [B, 19, 8], got {tuple(sample_object.shape)}")

        self.n_types = 64
        self.global_id = 63
        self.d_model = 96

        self.type_emb = nn.Embedding(self.n_types, self.d_model, padding_idx=0)
        self.cont_proj = nn.Sequential(
            nn.Linear(7, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=self.d_model * 4,
            dropout=0.12,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.post_ln = nn.LayerNorm(self.d_model)

        # Combine: CLS/global token + mean pool + max pool => 3*d_model
        self.head = nn.Sequential(
            nn.Linear(self.d_model * 3, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, batch_x):
        # batch_x: [B, 19, 8]
        x = batch_x
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, L, F], got {tuple(x.shape)}")

        type_id = x[:, :, 0].clamp(0, self.n_types - 1).to(torch.long)  # [B, 19]
        cont = x[:, :, 1:]  # [B, 19, 7]

        tok = self.type_emb(type_id) + self.cont_proj(cont)  # [B, 19, d_model]

        # padding mask: True means "ignore"
        pad_mask = type_id.eq(0)  # [B, 19]
        if pad_mask.shape[1] >= 1:
            pad_mask[:, 0] = False  # never pad the global token

        h = self.encoder(tok, src_key_padding_mask=pad_mask)  # [B, 19, d_model]
        h = self.post_ln(h)  # [B, 19, d_model]

        cls = h[:, 0, :]  # [B, d_model]

        obj_h = h[:, 1:, :]          # [B, 18, d_model]
        obj_valid = ~pad_mask[:, 1:] # [B, 18] bool

        # mean pool
        denom = obj_valid.sum(dim=1, keepdim=True).clamp(min=1).to(obj_h.dtype)  # [B, 1]
        mean_pool = (obj_h * obj_valid.unsqueeze(-1).to(obj_h.dtype)).sum(dim=1) / denom  # [B, d_model]

        # max pool
        obj_h_masked = obj_h.masked_fill(~obj_valid.unsqueeze(-1), -1e9)  # [B, 18, d_model]
        max_pool = obj_h_masked.max(dim=1).values  # [B, d_model]
        any_obj = obj_valid.any(dim=1, keepdim=True)  # [B, 1]
        max_pool = torch.where(any_obj, max_pool, torch.zeros_like(max_pool))  # [B, d_model]

        feat = torch.cat([cls, mean_pool, max_pool], dim=1)  # [B, 3*d_model]
        logits = self.head(feat).squeeze(1)  # [B]
        return logits


def make_model(example_object):
    return BinaryClassifier(example_object)


EPOCHS = 20


def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1.5e-4)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3 if device.type == "cuda" else 1.5e-3,
        epochs=max(1, epochs),
        steps_per_epoch=steps_per_epoch,
        pct_start=0.12,
        div_factor=20.0,
        final_div_factor=200.0,
    )

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    best_auc = -1.0
    best_state = None
    patience = 6
    bad_epochs = 0

    for epoch in range(int(epochs)):
        # ---------------- TRAIN ----------------
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            x = view.batch_x
            y = view.batch_y.to(torch.float32)  # [B]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)  # [B]
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = int(y.shape[0])
            running_loss += float(loss.detach()) * bs
            preds = (logits.detach() > 0).to(torch.int64)
            running_correct += int((preds == y.to(torch.int64)).sum().item())
            running_total += bs

        epoch_train_loss = running_loss / max(1, running_total)
        epoch_train_acc = running_correct / max(1, running_total)
        train_loss_hist.append(epoch_train_loss)
        train_acc_hist.append(epoch_train_acc)

        # ---------------- VALIDATION ----------------
        model.eval()
        v_running_loss = 0.0
        v_correct = 0
        v_total = 0

        all_logits = []
        all_y = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                x = view.batch_x
                y = view.batch_y.to(torch.float32)

                logits = model(x)
                loss = criterion(logits, y)

                bs = int(y.shape[0])
                v_running_loss += float(loss.detach()) * bs
                preds = (logits > 0).to(torch.int64)
                v_correct += int((preds == y.to(torch.int64)).sum().item())
                v_total += bs

                all_logits.append(logits.detach().float().cpu())
                all_y.append(y.detach().float().cpu())

        epoch_val_loss = v_running_loss / max(1, v_total)
        epoch_val_acc = v_correct / max(1, v_total)
        val_loss_hist.append(epoch_val_loss)
        val_acc_hist.append(epoch_val_acc)

        # AUC for early stopping (optimize target metric)
        y_true = torch.cat(all_y, dim=0).numpy()
        y_score = torch.cat(all_logits, dim=0).numpy()
        try:
            val_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            val_auc = -1.0

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        model.to(device)

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


