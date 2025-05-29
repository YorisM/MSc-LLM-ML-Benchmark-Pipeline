
import os, sys, pickle, torch, gc
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
from torch.nn import functional as F

class MyPreprocessor:
    def __init__(self):
        # No stateful components needed in this simple preprocessing
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Normalize E_T_miss (index 0) and phi_Et_miss (index 1)
        X_transformed = X.clone()
        X_transformed[:, 0] = (X_transformed[:, 0] - torch.mean(X_transformed[:, 0])) / torch.std(X_transformed[:, 0])  # Standardize E_T_miss
        X_transformed[:, 1] = (X_transformed[:, 1] - torch.mean(X_transformed[:, 1])) / torch.std(X_transformed[:, 1])  # Standardize phi_Et_miss
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, num_iters):
        super(SlotAttention, self).__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.num_iters = num_iters
        self.slots = nn.Parameter(torch.randn(num_slots, dim))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        b, n, d = x.shape
        slots = self.slots.unsqueeze(0).repeat(b, 1, 1)
        for _ in range(self.num_iters):
            slots_prev = slots.clone()
            attention = (x @ slots.transpose(1, 2)) / (d ** 0.5)  # Scaled dot-product attention
            attention = F.softmax(attention, dim=-1)
            slots = attention @ x  # Compute new slots
            slots = self.mlp(slots) + slots_prev  # Update slots with MLP
        return slots

def make_model(input_dim: int):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.slot_attention = SlotAttention(num_slots=10, dim=input_dim, num_iters=3)
            self.fc = nn.Linear(input_dim, 1)

        def forward(self, x):
            x = self.slot_attention(x.view(-1, 18, input_dim))  # Reshape to (batch_size, num_objects, features)
            x = x.mean(dim=1)  # Aggregate across the slots
            return torch.sigmoid(self.fc(x))  # Binary classification

    return Model()

EPOCHS = 10

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.float()).squeeze()
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == y_batch.float()).sum().item()
            total += y_batch.size(0)

        train_loss.append(epoch_loss / len(train_loader))
        train_acc.append(correct / total)

        # Validation step
        model.eval()
        val_epoch_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch.float()).squeeze()
                val_loss_item = criterion(outputs, y_batch.float())
                val_epoch_loss += val_loss_item.item()
                val_predictions = (outputs > 0.5).float()
                val_correct += (val_predictions == y_batch.float()).sum().item()
                val_total += y_batch.size(0)

        val_loss.append(val_epoch_loss / len(val_loader))
        val_acc.append(val_correct / val_total)

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

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

