
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
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch or sklearn (sub-)modules you actually use.
import math
from typing import Tuple, List

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    # Performs standard feature-wise z-score normalisation.
    # Statistics are computed on the training set and re-used for val / test.
    def __init__(self):
        self.mean = None  # Tensor[92]
        self.std  = None  # Tensor[92]

    def fit(self, X: torch.Tensor, y=None):
        # X : (N, 92)
        with torch.no_grad():
            self.mean = X.mean(dim=0)                       # (92,)
            self.std  = X.std (dim=0).clamp_min(1e-6)       # (92,)
        return self

    def transform(self, X):
        # Supports:
        #  - X : (N, 92) torch.Tensor
        # Returns the same shape, float32
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
        in_dim = 1
        for d in input_shape:                               # product of dims
            in_dim *= d

        hidden1 = 512
        hidden2 = 256
        hidden3 = 128

        self.net = nn.Sequential(                          # shapes in comments
            nn.Linear(in_dim, hidden1),                    # (B, 512)
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden1, hidden2),                   # (B, 256)
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden2, hidden3),                   # (B, 128)
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden3, 1)                          # (B, 1)
        )

    def forward(self, data: torch.Tensor, mask: torch.Tensor | None = None):
        # data expected shape : (B, F) or (B, L, F) flattened
        if data.dim() > 2:                                 # (B, L, F) -> (B, L*F)
            data = data.flatten(start_dim=1)
        logits = self.net(data)                            # (B,1)
        return logits.squeeze(1)                           # (B,)

def make_model(input_shape, *, use_mask=False):
    return BinaryClassifier(input_shape, use_mask=use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader:   torch.utils.data.DataLoader,
                epochs: int):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)

    train_loss: List[float] = []
    val_loss:   List[float] = []
    train_acc:  List[float] = []
    val_acc:    List[float] = []

    patience = 5
    best_val = math.inf
    epochs_no_improve = 0
    trained_model = model

    sigmoid = torch.nn.Sigmoid()

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0
        for batch in train_loader:
            # batch may be (data,label) or ((data,mask), label)
            if isinstance(batch[0], (tuple, list)):
                data, mask = batch[0]
                data = data.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
            else:
                data = batch[0].to(device, non_blocking=True)
                mask = None
            labels = batch[1].to(device, non_blocking=True).float()

            optimizer.zero_grad()
            outputs = model(data, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = (sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total   += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc  = correct / total
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # ---- Validation ----
        model.eval()
        v_running_loss = 0.0
        v_correct = 0
        v_total   = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch[0], (tuple, list)):
                    data, mask = batch[0]
                    data = data.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                else:
                    data = batch[0].to(device, non_blocking=True)
                    mask = None
                labels = batch[1].to(device, non_blocking=True).float()

                outputs = model(data, mask)
                loss = criterion(outputs, labels)

                v_running_loss += loss.item() * labels.size(0)
                preds = (sigmoid(outputs) > 0.5).long()
                v_correct += (preds == labels.long()).sum().item()
                v_total   += labels.size(0)

        epoch_val_loss = v_running_loss / v_total
        epoch_val_acc  = v_correct / v_total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val - 1e-4:
            best_val = epoch_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return trained_model, train_loss, val_loss, train_acc, val_acc

# ----------------  END OF LLM-CODE BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
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

