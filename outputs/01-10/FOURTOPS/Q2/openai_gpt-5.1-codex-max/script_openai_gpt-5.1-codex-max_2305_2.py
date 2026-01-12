
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

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
        self.eps = 1e-6

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        if not torch.is_tensor(X):
            X_t = torch.as_tensor(X, dtype=torch.float32)
        else:
            X_t = X.clone().float()
        X_proc = self._preprocess_core(X_t)  # [N, F_aug]
        self.mean = X_proc.mean(dim=0)
        self.std = X_proc.std(dim=0, unbiased=False) + self.eps
        self.fitted = True
        return self

    def _preprocess_core(self, X):
        Xc = X.clone()  # [N,92]
        # log1p transform for positive-valued magnitudes
        energy_idx = torch.arange(3, 92, 5)
        pt_idx = torch.arange(4, 92, 5)
        Xc[:, 0] = torch.log1p(torch.clamp(Xc[:, 0], min=0))  # E_T_miss
        Xc[:, energy_idx] = torch.log1p(torch.clamp(Xc[:, energy_idx], min=0))
        Xc[:, pt_idx] = torch.log1p(torch.clamp(Xc[:, pt_idx], min=0))
        # Extra features
        obj_ids = Xc[:, 2:92:5]  # [N,18]
        mask = (obj_ids != 0).float()  # [N,18]
        count_objects = mask.sum(dim=1)  # [N]
        energies = Xc[:, 3:92:5]  # [N,18]
        pts = Xc[:, 4:92:5]  # [N,18]
        etas = Xc[:, 5:92:5]  # [N,18]
        sum_E = (energies * mask).sum(dim=1)  # [N]
        sum_pT = (pts * mask).sum(dim=1)  # [N]
        mean_eta = (etas * mask).sum(dim=1) / (count_objects + 1e-6)  # [N]
        var_eta = (((etas - mean_eta.unsqueeze(1)) ** 2) * mask).sum(dim=1) / (count_objects + 1e-6)  # [N]
        std_eta = torch.sqrt(var_eta)  # [N]
        extra_feats = torch.stack([count_objects, sum_E, sum_pT, mean_eta, std_eta], dim=1)  # [N,5]
        X_aug = torch.cat([Xc, extra_feats], dim=1)  # [N,97]
        return X_aug

    def transform(self, X):
        if not torch.is_tensor(X):
            X_t = torch.as_tensor(X, dtype=torch.float32)
        else:
            X_t = X.clone().float()
        X_proc = self._preprocess_core(X_t)
        if self.fitted and self.mean is not None and self.std is not None:
            X_proc = (X_proc - self.mean) / self.std
        return X_proc  # [N,97]

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[-1]
        hidden_dims = [256, 256, 128, 64]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.2))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, batch_x):
        logits = self.net(batch_x)  # [B,1]
        return logits.squeeze(1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 12
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []
        best_val_loss = float('inf')
        best_state = None
        patience = 5
        patience_counter = 0
        model.to(device)
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device).float()
                optimizer.zero_grad()
                outputs = model(xb)  # [B]
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.size(0)
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            train_loss_list.append(epoch_loss)
            train_acc_list.append(epoch_acc)
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device).float()
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    val_running_loss += loss.item() * xb.size(0)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    val_correct += (preds == yb).sum().item()
                    val_total += yb.size(0)
            val_epoch_loss = val_running_loss / val_total
            val_epoch_acc = val_correct / val_total
            val_loss_list.append(val_epoch_loss)
            val_acc_list.append(val_epoch_acc)
            scheduler.step(val_epoch_loss)
            if val_epoch_loss < best_val_loss - 1e-4:
                best_val_loss = val_epoch_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

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

