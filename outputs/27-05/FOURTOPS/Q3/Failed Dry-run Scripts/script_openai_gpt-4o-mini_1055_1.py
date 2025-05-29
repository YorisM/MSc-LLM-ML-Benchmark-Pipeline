
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
class MyPreprocessor:
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Extract features relevant to physics processes
        # With E_T_miss and phi_E_t_miss
        features = X[:, :2]  # (N, 2)
        # Now process per-object data
        for i in range(18):
            obj_idx = 2 + i * 5
            E = X[:, obj_idx + 1].unsqueeze(1)  # (N, 1)
            pT = X[:, obj_idx + 2].unsqueeze(1)  # (N, 1)
            eta = X[:, obj_idx + 3].unsqueeze(1)  # (N, 1)
            phi = X[:, obj_idx + 4].unsqueeze(1)  # (N, 1)
            # Compute derived features
            features = torch.cat((features, E, pT, eta, phi), dim=1)  # (N, 2 + 5*18)
        return features
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim: int):
    class SlotAttention(nn.Module):
        def __init__(self, input_dim):
            super(SlotAttention, self).__init__()
            self.input_dim = input_dim
            self.slots = nn.Parameter(torch.randn(1, 18, input_dim))
            self.attention = nn.Linear(input_dim, input_dim)
            self.output_layer = nn.Linear(input_dim, 1)
        def forward(self, x):
            # Attention mechanism to focus on relevant slots
            weights = torch.softmax(self.attention(x), dim=1)
            attended = torch.bmm(weights.unsqueeze(1), self.slots)
            output = self.output_layer(attended)
            return output.squeeze()  # Output shape will be (N,)
    return SlotAttention(input_dim)

EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = (torch.sigmoid(output) >= 0.5).int()
            correct += (preds == y_batch).sum().item()
        train_loss.append(total_loss / len(train_loader))
        train_acc.append(correct / len(train_loader.dataset))
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                output_val = model(X_val)
                val_loss_value = criterion(output_val, y_val.float())
                total_val_loss += val_loss_value.item()
                preds_val = (torch.sigmoid(output_val) >= 0.5).int()
                correct_val += (preds_val == y_val).sum().item()
        val_loss.append(total_val_loss / len(val_loader))
        val_acc.append(correct_val / len(val_loader.dataset))
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

