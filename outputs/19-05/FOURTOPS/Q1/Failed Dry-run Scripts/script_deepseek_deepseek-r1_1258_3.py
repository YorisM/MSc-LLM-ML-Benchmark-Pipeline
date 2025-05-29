
import os, sys, json, pickle, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}

EPOCHS = 10
                        
def load_data():
    to_np = lambda path: pd.read_csv(path).values
    X_train = to_np(DATASET["X_train"])
    Y_train = to_np(DATASET["Y_train"]).ravel()
    X_val   = to_np(DATASET["X_val"])
    Y_val   = to_np(DATASET["Y_val"]).ravel()
    return X_train, Y_train, X_val, Y_val

def make_loaders(X_train, Y_train, X_val, Y_val, batch=1024):
    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train))
    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val))
    return (DataLoader(train, batch_size=batch, shuffle=True),
            DataLoader(val, batch_size=batch))
                        
# ----------------  START OF LLM BLOCK  ----------------
import os
import sys
import pickle
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class MyPreprocessor:
    def __init__(self):
        self.median_pT = 0.0

    def fit(self, X, y=None):
        all_pT = []
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        for event in X_np:
            for i in range(18):
                idx = 2 + i*5
                if event[idx] != 0:
                    all_pT.append(event[idx+2])
        self.median_pT = np.median(all_pT) if all_pT else 0.0
        return self

    def transform(self, X):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        processed = []
        for event in X_np:
            features = [
                event[0],  # ET_miss
                event[1],  # phi_Et_miss
                0, 0, 0, 0, 0, 0, 0  # Placeholders
            ]
            num_objects = sum_pT = max_pT = num_high_pT = num_central = num_forward = 0
            
            for i in range(18):
                idx = 2 + i*5
                if event[idx] != 0:
                    pT = event[idx+2]
                    eta = event[idx+3]
                    num_objects += 1
                    sum_pT += pT
                    max_pT = max(max_pT, pT)
                    num_high_pT += 1 if pT > self.median_pT else 0
                    if abs(eta) < 2.5:
                        num_central += 1
                    else:
                        num_forward += 1
            
            et_over_ht = event[0]/sum_pT if sum_pT > 0 else 0.0
            features[2:] = [
                num_objects,
                sum_pT,
                max_pT,
                num_high_pT,
                num_central,
                num_forward,
                et_over_ht
            ]
            processed.append(features)
        
        return torch.tensor(processed, dtype=torch.float32) if isinstance(X, torch.Tensor) \
               else np.array(processed, dtype=np.float32)

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X).squeeze()
            loss = criterion(outputs, y.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X.size(0)
            correct += ((outputs >= 0.5).float() == y).sum().item()
            total += y.size(0)
        train_loss.append(epoch_loss / total)
        train_acc.append(correct / total)
        
        model.eval()
        val_epoch_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X).squeeze()
                val_epoch_loss += criterion(outputs, y.float()).item() * X.size(0)
                val_correct += ((outputs >= 0.5).float() == y).sum().item()
                val_total += y.size(0)
        val_loss.append(val_epoch_loss / val_total)
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
    X_tr, y_tr, X_va, y_va = load_data()
    pre = make_preprocessor();  pre.fit(X_tr, y_tr)
    X_tr = pre.transform(X_tr); X_va = pre.transform(X_va)
    tr_loader, va_loader = make_loaders(X_tr, y_tr, X_va, y_va)

    # 2. Build model
    model = make_model(input_dim=X_tr.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    trained, tr_loss, va_loss, tr_acc, va_acc = train_model(
        model, tr_loader, va_loader, epochs=n_epochs
    )

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy = torch.zeros(8, X_tr.shape[1])      # 8 fake events
        try:
            _ = trained(pre.transform(toy))
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return  # no files in dry-run

    # 4. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
    torch.save(trained.state_dict(), f"{base}_state.pt")
    with open(f"{base}_model.pkl", "wb") as f: pickle.dump(trained, f)
    with open(f"{base}_preproc.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

