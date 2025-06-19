
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: Python 3.12, PyTorch 2.6.0, Torch_Geometric 2.6.1, NumPy 2.2.3, SciKit-Learn 1.6.1
import os, sys, pickle, torch, torch_geometric, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train),
            torch.from_numpy(Y_train),
            torch.from_numpy(X_val),
            torch.from_numpy(Y_val))

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        if isinstance(self.x, (tuple, list)):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])      

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = PairDataset(X_train, Y_train)
    val_ds   = PairDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

# 0. ---------- IMPORTS ----------
import math
import pickle
from typing import Tuple, List

from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # feature-wise mean and std (Tensor of shape (92,))
        self.mean = None
        self.std = None

    def fit(self, X: torch.Tensor, y=None):
        # X : (N, 92)
        with torch.no_grad():
            self.mean = X.mean(dim=0)                       # (92,)
            self.std  = X.std (dim=0)                       # (92,)
            # guard against zero std to avoid division errors
            self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, X: torch.Tensor):
        # works for both CPU & CUDA tensors
        return (X - self.mean) / self.std                   # (N, 92)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], *, use_mask: bool):
        super().__init__()
        self.use_mask = use_mask
        in_features = int(math.prod(input_shape))           # 92 for flat input
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.15),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.15),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.10),

            nn.Linear(64, 32),
            nn.LeakyReLU(),

            nn.Linear(32, 1)                               # logits
        )

    def forward(self, data: torch.Tensor, mask: torch.Tensor | None = None):
        # data: (B, F)
        if self.use_mask:
            # For flat model we ignore mask (not used)
            data = data
        logits = self.net(data).squeeze(-1)                 # (B,)
        return logits

def make_model(input_shape, *, use_mask=False):
    return BinaryClassifier(input_shape, use_mask=use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader  : torch.utils.data.DataLoader,
                epochs      : int
                ) -> Tuple[nn.Module, List[float], List[float], List[float], List[float]]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=2)

    train_loss, val_loss = [], []
    train_acc , val_acc  = [], []

    best_auc = -float('inf')
    patience, counter = 5, 0
    best_state_dict = None

    for epoch in range(epochs):
        # --- training ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            if isinstance(batch[0], (tuple, list)):
                data, lbl = batch[0][0], batch[1]
            else:
                data, lbl = batch[0], batch[1]

            data = data.to(device, non_blocking=True)
            lbl  = lbl.to(device, non_blocking=True).float()      # (B,)

            optimizer.zero_grad()
            logits = model(data)                                  # (B,)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(lbl)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == lbl.long()).sum().item()
            total   += len(lbl)

        epoch_train_loss = running_loss / total
        epoch_train_acc  = correct / total

        # --- validation ---
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        val_logits_all, val_lbl_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch[0], (tuple, list)):
                    data, lbl = batch[0][0], batch[1]
                else:
                    data, lbl = batch[0], batch[1]

                data = data.to(device, non_blocking=True)
                lbl  = lbl.to(device, non_blocking=True).float()

                logits = model(data)
                loss = criterion(logits, lbl)
                running_loss += loss.item() * len(lbl)

                val_logits_all.append(logits.cpu())
                val_lbl_all.append(lbl.cpu())

                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds == lbl.long()).sum().item()
                total   += len(lbl)

        epoch_val_loss = running_loss / total
        epoch_val_acc  = correct / total

        # Compute AUC for early stopping
        val_logits_cat = torch.cat(val_logits_all)            # (N_val,)
        val_lbl_cat    = torch.cat(val_lbl_all)               # (N_val,)
        val_prob       = torch.sigmoid(val_logits_cat).numpy()
        val_auc        = roc_auc_score(val_lbl_cat.numpy(), val_prob)

        scheduler.step(epoch_val_loss)

        train_loss.append(epoch_train_loss)
        val_loss  .append(epoch_val_loss)
        train_acc .append(epoch_train_acc)
        val_acc   .append(epoch_val_acc)

        # Early Stopping on AUC
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= patience:
                break

    # restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, train_loss, val_loss, train_acc, val_acc

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        X_train, Y_train, X_val, Y_val = X_train[:200], Y_train[:200], X_val[:20], Y_val[:20]
    pre = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train)                    # may be Tensor or Tuple
    X_val   = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    if isinstance(X_train, torch.Tensor):               # single-tensor case
        temp_ref    = X_train
        input_shape = temp_ref.shape[1:]                # e.g. (F,)
        use_mask    = False
    else:                                               # tuple => (data, mask)
        temp_ref    = X_train
        input_shape = temp_ref[0].shape[1:]             # e.g. (L, F)
        use_mask    = True                              
    model = make_model(input_shape, use_mask=use_mask)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single toy forward pass
    if dryrun:
        toy_data = torch.zeros(8, *input_shape, dtype=torch.float32)
        if use_mask:
            toy_mask = torch.zeros(8, input_shape[0], dtype=torch.bool)
            toy_batch = (toy_data, toy_mask)
        else:
            toy_batch = toy_data

        toy_transformed = pre.transform(toy_batch)
        try:
            _ = trained_model(*toy_transformed) if isinstance(toy_transformed, (tuple, list)) \
                else trained_model(toy_transformed)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 6. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
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

