
import os, sys, pickle, torch, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
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

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))
                        
# ----------------  START OF LLM BLOCK  ----------------

import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X.numpy())  # fit on numpy data
        return self

    def transform(self, X):
        return torch.from_numpy(self.scaler.transform(X.numpy())).float()  # transform to tensors

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

def make_model(input_dim: int):
    model = SimpleNN(input_dim)
    return model

EPOCHS = 20  # reasonable epoch number to avoid overfitting

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCELoss()  # binary cross entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_x).squeeze()  # remove single-dimensional entries
            loss = criterion(y_pred, batch_y.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (y_pred > 0.5).float()
            correct += (preds == batch_y).sum().item()

        train_loss.append(total_loss / len(train_loader))
        train_acc.append(correct / len(train_loader.dataset))

        model.eval()
        total_val_loss, val_correct = 0, 0

        with torch.no_grad():  # no gradient needed for validation
de
total_val_loss += criterion(model(batch_x).squeeze(), batch_y.float()).item()
            val_preds = (model(batch_x).squeeze() > 0.5).float()
            val_correct += (val_preds == batch_y).sum().item()

        val_loss.append(total_val_loss / len(val_loader))
        val_acc.append(val_correct / len(val_loader.dataset))

    return model, train_loss, val_loss, train_acc, val_acc

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
    pre = make_preprocessor()
    pre.fit(X_train, Y_train)
    X_train = pre.transform(X_train)
    X_val = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    model = make_model(input_dim=X_train.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy = torch.zeros(8, X_train.shape[1])      # 8 fake events
        try:
            _ = trained_model(pre.transform(toy))
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return  # no files in dry-run

    # 4. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 6. Write JSON Summary
    summary = {
        "epochs": n_epochs,
        "train_loss": tr_loss,
        "val_loss":   va_loss,
        "train_acc":  tr_acc,
        "val_acc":    va_acc,
        "best_train_loss": min(tr_loss),
        "best_train_loss_epoch": tr_loss.argmin(),
        "best_train_acc":  max(tr_acc),
        "best_train_acc_epoch": tr_acc.argmax(),
        "best_val_loss": min(va_loss),
        "best_val_loss_epoch": va_loss.argmin(),
        "best_val_acc":  max(va_acc),
        "best_val_acc_epoch": va_acc.argmax(),
    }
    print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

