
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

# ---------- IMPORTS ----------
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Total flat length per event (X_train & X_val): 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    # Indices  7-11 : object 2  ->  obj_2, E_2 , p_T_2 , eta_2 , phi_2
    # ...
    # Indices 87-91 : object 18 ->  obj_18, E_18 , p_T_18 , eta_18 , phi_18

    def __init__(self):
        self.mean = None
        self.std = None
        # precompute indices
        self.obj_indices = [2 + 5 * k for k in range(18)]
        self.E_indices = [3 + 5 * k for k in range(18)]
        self.pT_indices = [4 + 5 * k for k in range(18)]
        self.eta_indices = [5 + 5 * k for k in range(18)]
        self.phi_indices = [1] + [6 + 5 * k for k in range(18)]

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": torch.cuda.is_available(),
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        with torch.no_grad():
            feats = self._compute_features(X)  # [N, F]
            self.mean = feats.mean(dim=0)
            self.std = feats.std(dim=0)
            self.std[self.std < 1e-6] = 1.0
        return self

    def _compute_features(self, X):
        # X shape [N,92]
        x = X if torch.is_tensor(X) else torch.from_numpy(X)
        x = x.float()
        # original features
        feats = [x]  # [N,92]
        # energies and pT
        E_vals = x[:, self.E_indices]  # [N,18]
        pT_vals = x[:, self.pT_indices]  # [N,18]
        feats.append(torch.log1p(torch.clamp(E_vals, min=0.0)))  # [N,18]
        feats.append(torch.log1p(torch.clamp(pT_vals, min=0.0)))  # [N,18]
        # phi features
        phi_vals = x[:, self.phi_indices]  # [N,19]
        feats.append(torch.sin(phi_vals))  # [N,19]
        feats.append(torch.cos(phi_vals))  # [N,19]
        # mask for valid objects
        obj_ids = x[:, self.obj_indices]  # [N,18]
        mask = (obj_ids > 0).float()  # [N,18]
        count = mask.sum(dim=1, keepdim=True)  # [N,1]
        feats.append(count / 18.0)  # [N,1]
        sum_pT = (pT_vals * mask).sum(dim=1, keepdim=True)  # [N,1]
        sum_E = (E_vals * mask).sum(dim=1, keepdim=True)  # [N,1]
        feats.append(torch.log1p(sum_pT))  # [N,1]
        feats.append(torch.log1p(sum_E))  # [N,1]
        max_pT = (pT_vals * mask + (1 - mask) * (-1e9)).max(dim=1, keepdim=True).values  # [N,1]
        max_pT = torch.where(max_pT < -1e8, torch.zeros_like(max_pT), max_pT)
        feats.append(torch.log1p(torch.clamp(max_pT, min=0.0)))  # [N,1]
        eta_vals = x[:, self.eta_indices]  # [N,18]
        mean_abs_eta = (eta_vals.abs() * mask).sum(dim=1, keepdim=True) / (count + 1e-3)  # [N,1]
        feats.append(mean_abs_eta)
        return torch.cat(feats, dim=1)  # [N, F]

    def transform(self, X):
        feats = self._compute_features(X)
        if self.mean is not None and self.std is not None:
            feats = (feats - self.mean) / self.std
        return feats  # [N, F]

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # x: [B, D]
        out = self.fc1(x)  # [B, D]
        out = self.bn1(out)  # [B, D]
        out = self.act(out)  # [B, D]
        out = self.drop(out)  # [B, D]
        out = self.fc2(out)  # [B, D]
        out = self.bn2(out)  # [B, D]
        out = self.drop(out)  # [B, D]
        return self.act(out + x)  # [B, D]

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        in_dim = sample_object.shape[-1]
        hidden1 = 256
        hidden2 = 128
        hidden3 = 64
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.res1 = ResidualBlock(hidden1, drop=0.2)
        self.res2 = ResidualBlock(hidden1, drop=0.2)
        self.res3 = ResidualBlock(hidden1, drop=0.2)
        self.down1 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.res4 = ResidualBlock(hidden2, drop=0.2)
        self.down2 = nn.Sequential(
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.out_layer = nn.Linear(hidden3, 1)

    def forward(self, batch_x):
        x = batch_x  # [B, F]
        x = self.input_layer(x)  # [B,256]
        x = self.res1(x)  # [B,256]
        x = self.res2(x)  # [B,256]
        x = self.res3(x)  # [B,256]
        x = self.down1(x)  # [B,128]
        x = self.res4(x)  # [B,128]
        x = self.down2(x)  # [B,64]
        out = self.out_layer(x)  # [B,1]
        return out  # [B,1]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 15
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    train_loss_hist = []
    val_loss_hist = []
    train_auc_hist = []
    val_auc_hist = []

    best_val_auc = -1.0
    best_state = None
    patience = 5
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        all_train_preds = []
        all_train_targets = []
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            x = view.batch_x  # [B, F]
            y = view.batch_y.float()  # [B]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x).squeeze(-1)  # [B]
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * x.size(0)
            with torch.no_grad():
                preds = torch.sigmoid(logits).detach().cpu()  # [B]
                all_train_preds.append(preds)
                all_train_targets.append(y.detach().cpu())
        epoch_loss /= len(train_loader.dataset)
        train_loss_hist.append(epoch_loss)
        # compute train AUC
        train_preds = torch.cat(all_train_preds)  # [N_train]
        train_targets = torch.cat(all_train_targets)  # [N_train]
        try:
            train_auc = roc_auc_score(train_targets.numpy(), train_preds.numpy())
        except Exception:
            train_auc = 0.0
        train_auc_hist.append(train_auc)

        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                x = view.batch_x  # [B, F]
                y = view.batch_y.float()  # [B]
                logits = model(x).squeeze(-1)  # [B]
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = torch.sigmoid(logits).detach().cpu()  # [B]
                all_val_preds.append(preds)
                all_val_targets.append(y.detach().cpu())
        val_loss /= len(val_loader.dataset)
        val_loss_hist.append(val_loss)
        val_preds = torch.cat(all_val_preds)  # [N_val]
        val_targets = torch.cat(all_val_targets)  # [N_val]
        try:
            val_auc = roc_auc_score(val_targets.numpy(), val_preds.numpy())
        except Exception:
            val_auc = 0.0
        val_auc_hist.append(val_auc)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_loss_hist, val_loss_hist, train_auc_hist, val_auc_hist

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


