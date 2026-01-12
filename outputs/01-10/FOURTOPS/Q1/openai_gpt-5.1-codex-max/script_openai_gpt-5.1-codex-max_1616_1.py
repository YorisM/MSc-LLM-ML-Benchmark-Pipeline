
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
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 1024}
        }

    def _augment(self, X):
        xt = torch.as_tensor(X)
        if xt.dtype != torch.float32:
            xt = xt.float()
        X_proc = xt.clone()
        # Apply log1p to energy and pT features to reduce skewness
        energy_slice = slice(3, None, 5)
        pt_slice = slice(4, None, 5)
        X_proc[:, energy_slice] = torch.log1p(torch.clamp(X_proc[:, energy_slice], min=0.0))
        X_proc[:, pt_slice] = torch.log1p(torch.clamp(X_proc[:, pt_slice], min=0.0))
        # Extra features from raw (un-logged) energies and pT
        energies_raw = xt[:, 3::5]  # [N,18]
        pts_raw = xt[:, 4::5]       # [N,18]
        present = ((xt[:, 2::5] != 0) | (energies_raw != 0) | (pts_raw != 0)).float()  # [N,18]
        nobj = present.sum(dim=1, keepdim=True)                             # [N,1]
        sumE = energies_raw.sum(dim=1, keepdim=True)                        # [N,1]
        sumPt = pts_raw.sum(dim=1, keepdim=True)                            # [N,1]
        meanPt = sumPt / torch.clamp(nobj, min=1.0)                         # [N,1]
        maxPt = pts_raw.max(dim=1, keepdim=True).values                     # [N,1]
        varPt = ((pts_raw - meanPt) ** 2 * present).sum(dim=1, keepdim=True) / torch.clamp(nobj, min=1.0)  # [N,1]
        stdPt = torch.sqrt(varPt + 1e-6)                                    # [N,1]
        extras = torch.cat([nobj, sumE, sumPt, meanPt, maxPt, stdPt], dim=1)  # [N,6]
        return torch.cat([X_proc, extras], dim=1)                           # [N,98]

    def fit(self, X, y=None):
        with torch.no_grad():
            feats = self._augment(X)
            self.mean = feats.mean(dim=0)
            self.std = feats.std(dim=0) + 1e-6
        return self

    def transform(self, X):
        feats = self._augment(X)
        if self.mean is not None and self.std is not None:
            feats = (feats - self.mean) / self.std
        return feats

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        in_features = sample_object.shape[1]
        layers = [
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        ]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, batch_x):
        x = batch_x  # [B, F]
        out = self.net(x)  # [B,1]
        return out.squeeze(-1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 15
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loss_history = []
    val_loss_history = []
    train_auc_history = []
    val_auc_history = []
    best_val_auc = -float("inf")
    best_state = None
    patience = 4
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        all_train_logits = []
        all_train_labels = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)  # [B]
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * yb.size(0)
            all_train_logits.append(logits.detach().cpu())
            all_train_labels.append(yb.detach().cpu())
        avg_train_loss = total_loss / len(train_loader.dataset)
        try:
            train_probs = torch.sigmoid(torch.cat(all_train_logits)).numpy()
            train_labels = torch.cat(all_train_labels).numpy()
            train_auc = roc_auc_score(train_labels, train_probs)
        except Exception:
            train_auc = float('nan')
        model.eval()
        val_total_loss = 0.0
        val_logits = []
        val_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).float()
                logits = model(xb)  # [B]
                loss = criterion(logits, yb)
                val_total_loss += loss.item() * yb.size(0)
                val_logits.append(logits.cpu())
                val_labels.append(yb.cpu())
        avg_val_loss = val_total_loss / len(val_loader.dataset)
        try:
            val_probs = torch.sigmoid(torch.cat(val_logits)).numpy()
            val_lbls = torch.cat(val_labels).numpy()
            val_auc = roc_auc_score(val_lbls, val_probs)
        except Exception:
            val_auc = float('nan')
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_auc_history.append(train_auc)
        val_auc_history.append(val_auc)
        scheduler.step()
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_loss_history, val_loss_history, train_auc_history, val_auc_history

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

