
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
import json
import pickle
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X, y=None):
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X
        
        valid = X_np[:, 0] != 0  # Filter valid events
        X_valid = X_np[valid]
        
        self.mean = np.nanmean(X_valid, axis=0)
        self.std = np.nanstd(X_valid, axis=0)
        self.std[self.std == 0] = 1  # Avoid division by zero
        return self
    
    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X
        
        X_processed = (X_np - self.mean) / self.std
        X_processed = np.nan_to_num(X_processed)
        return torch.tensor(X_processed, dtype=torch.float32)
    
def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim: int):
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    return model

EPOCHS = 15

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=False)
    
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss.append(epoch_loss / len(train_loader.dataset))
        train_acc.append(correct / total)
        
        model.eval()
        val_epoch_loss = 0.0
        val_correct = 0
        val_total = 0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                
                val_epoch_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss.append(val_epoch_loss / len(val_loader.dataset))
        val_acc.append(val_correct / val_total)
        
        val_auc = roc_auc_score(all_labels, all_outputs)
        scheduler.step(val_auc)
        
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

