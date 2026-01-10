
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

class MyPreprocessor:
    # Custom preprocessor with feature augmentation and normalization
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.num_obj_types = None
        self.obj_type_vals = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": torch.cuda.is_available(),
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        X_t = torch.as_tensor(X)
        if X_t.ndim != 2:
            X_t = X_t.view(X_t.shape[0], -1)
        with torch.no_grad():
            obj_ids_flat = X_t[:, 2:].reshape(-1, 5)[:, 0]
            max_id = int(obj_ids_flat.max().item()) if obj_ids_flat.numel() > 0 else 1
            if max_id < 1:
                max_id = 1
            self.num_obj_types = max_id
            self.obj_type_vals = torch.arange(1, self.num_obj_types + 1)
            aug = self._augment(X_t)
            self.mean_ = aug.mean(dim=0)
            self.std_ = aug.std(dim=0)
            self.std_ = torch.where(self.std_ < 1e-6, torch.ones_like(self.std_), self.std_)
        return self

    def _augment(self, X):
        X_t = torch.as_tensor(X).float()
        if X_t.ndim != 2:
            X_t = X_t.view(X_t.shape[0], -1)
        base0 = X_t[:, :2]                                      # [N,2]
        obj_feat = X_t[:, 2:].reshape(-1, 18, 5)                # [N,18,5]
        pts = obj_feat[:, :, 2]                                 # [N,18]
        idx = torch.argsort(pts, dim=1, descending=True)        # [N,18]
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, 5)           # [N,18,5]
        obj_sorted = torch.gather(obj_feat, 1, idx_exp)         # [N,18,5]
        X_sorted = torch.cat([base0, obj_sorted.reshape(X_t.size(0), -1)], dim=1)  # [N,92]
        mask = (obj_sorted[:, :, 0] > 0).float()                # [N,18]
        pt = obj_sorted[:, :, 2]                                # [N,18]
        energy = obj_sorted[:, :, 1]                            # [N,18]
        eta = obj_sorted[:, :, 3]                               # [N,18]
        num_obj = mask.sum(dim=1)                               # [N]
        sum_pt = (pt * mask).sum(dim=1)                         # [N]
        sum_energy = (energy * mask).sum(dim=1)                 # [N]
        max_pt = (pt * mask).max(dim=1).values                  # [N]
        mean_pt = sum_pt / num_obj.clamp(min=1)                 # [N]
        std_pt = torch.sqrt(((pt - mean_pt.unsqueeze(1)) ** 2 * mask).sum(dim=1) / num_obj.clamp(min=1) + 1e-6)  # [N]
        mean_eta = (eta * mask).sum(dim=1) / num_obj.clamp(min=1)  # [N]
        mean_energy = sum_energy / num_obj.clamp(min=1)            # [N]
        counts_list = []
        n_types = self.num_obj_types if self.num_obj_types is not None else int(obj_sorted[:, :, 0].max().item()) if obj_sorted.numel() > 0 else 1
        n_types = max(1, n_types)
        for k in range(1, n_types + 1):
            counts_list.append((obj_sorted[:, :, 0] == float(k)).sum(dim=1))  # each [N]
        counts = torch.stack(counts_list, dim=1)                               # [N, n_types]
        additional = torch.stack([num_obj, sum_pt, sum_energy, max_pt, mean_pt, std_pt, mean_eta, mean_energy], dim=1)  # [N,8]
        out = torch.cat([X_sorted, additional, counts], dim=1)                 # [N, 92+8+n_types]
        return out

    def transform(self, X):
        aug = self._augment(X)
        if self.mean_ is not None and self.std_ is not None:
            aug = (aug - self.mean_) / self.std_
        return aug

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        in_dim = sample_object.shape[-1]
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        x = batch_x
        out = self.net(x)
        return out.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 12
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    best_val_loss = float('inf')
    best_state = None
    patience = 4
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)                                   # [B]
            loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        epoch_train_loss = running_loss / max(1, total)
        epoch_train_acc = correct / max(1, total)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        model.eval()
        val_running_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb.float())
                val_running_loss += loss.item() * yb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).long()
                v_correct += (preds == yb).sum().item()
                v_total += yb.size(0)
        epoch_val_loss = val_running_loss / max(1, v_total)
        epoch_val_acc = v_correct / max(1, v_total)
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_val_loss - 1e-4:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_loss, val_loss, train_acc, val_acc

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

