
import os, sys, pickle, torch, gc, json
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
                        
# ----------------  START OF LLM BLOCK  ----------------

# 0. ---------- IMPORTS ----------
import copy
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
# Only import extra std-lib modules, torch or sklearn (sub-)modules you actually use.

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    """
    Simple z-score standardisation pre-processor.
    Stores per-feature mean and std (computed on training data) and applies
    (x - mean) / std at transform time.
    """

    def __init__(self):
        # Will be initialised during fit()
        self.mean_ = None      # torch.Tensor, shape (92,)
        self.std_ = None       # torch.Tensor, shape (92,)

    def fit(self, X, y=None):
        """
        Compute feature-wise mean and std on the provided tensor X (N, 92).
        """
        # Means / std are kept on CPU for picklability; transformation can
        # broadcast on any device afterwards.
        self.mean_ = X.mean(dim=0).cpu()
        self.std_ = X.std(dim=0).cpu()
        # Guard against zero std to avoid division by zero
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, X):
        """
        Apply the stored z-score transformation and return a tensor with the
        same dtype / device as the input.
        """
        # Ensure mean_/std_ are on the same device as incoming X
        mean = self.mean_.to(X.device, dtype=X.dtype)
        std  = self.std_.to(X.device,  dtype=X.dtype)
        X_std = (X - mean) / std          # (N, 92)
        return X_std

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def make_preprocessor():
    """
    Factory required by the harness.
    """
    return MyPreprocessor()


# 2. ---------- MODEL DEFINITION ----------
class MLPClassifier(nn.Module):
    """
    A moderately deep fully-connected network with BatchNorm and Dropout.
    Designed for tabular data with a fixed number of features.
    """

    def __init__(self, input_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(64, 1)        # Single logit output
        )

    def forward(self, x):
        """
        x : Tensor, shape (batch, input_dim)
        Returns raw logits, shape (batch,)
        """
        logits = self.net(x)               # (B, 1)
        return logits.squeeze(-1)          # (B,)


def make_model(input_shape, *, use_mask=False):
    """
    Construct an untrained binary classifier.
    input_shape is everything after the batch dimension; i.e. (92,)
    """
    # Flatten any remaining dims just in case
    input_dim = int(np.prod(input_shape))
    model = MLPClassifier(input_dim)
    return model


# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50                              # Upper bound; early-stopping will often halt earlier


def train_model(model, train_loader, val_loader, epochs):
    """
    Standard training loop with:
      - BCEWithLogitsLoss
      - AdamW optimiser
      - ReduceLROnPlateau scheduler monitoring validation AUC
      - Early stopping on validation AUC
    Returns the trained model and learning curves.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, threshold=1e-4
    )

    patience          = 7                # epochs with no AUC improvement before stopping
    best_auc          = -np.inf
    epochs_no_improve = 0
    best_state        = copy.deepcopy(model.state_dict())

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist,  val_acc_hist  = [], []

    sigmoid = nn.Sigmoid()

    for epoch in range(epochs):
        # -------- TRAIN --------
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for xb, yb in train_loader:            # (data, label)
            xb = xb.to(device)
            yb = yb.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            logits = model(xb)                 # (B,)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

            with torch.no_grad():
                preds = (sigmoid(logits) > 0.5).long()
                correct += (preds == yb.long()).sum().item()
                total   += xb.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc  = correct / total
        train_loss_hist.append(epoch_train_loss)
        train_acc_hist.append(epoch_train_acc)

        # -------- VALIDATION --------
        model.eval()
        val_running_loss = 0.0
        val_correct, val_total = 0, 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device, dtype=torch.float32)

                logits = model(xb)
                loss   = criterion(logits, yb)

                val_running_loss += loss.item() * xb.size(0)

                probs = sigmoid(logits)
                preds = (probs > 0.5).long()

                val_preds.append(probs.cpu())
                val_targets.append(yb.cpu())

                val_correct += (preds == yb.long()).sum().item()
                val_total   += xb.size(0)

        val_preds    = torch.cat(val_preds).numpy()
        val_targets  = torch.cat(val_targets).numpy()
        val_auc      = roc_auc_score(val_targets, val_preds)
        val_loss     = val_running_loss / val_total
        val_acc      = val_correct / val_total

        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        # Scheduler & Early stopping bookkeeping
        scheduler.step(val_auc)

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best val AUC: {best_auc:.4f}")
                break

    # Restore the best model weights before returning
    model.load_state_dict(best_state)
    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    pre = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train) # may be Tensor or Tuple
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

    # 4. *Dry-run safety check* – run a single toy forward pass
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

