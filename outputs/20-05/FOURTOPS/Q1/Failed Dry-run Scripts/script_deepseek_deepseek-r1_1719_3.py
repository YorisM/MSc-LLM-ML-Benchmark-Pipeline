
import os, sys, pickle, torch, gc
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
        self.obj_types = []
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        X_objects = X[:, 2:].view(-1, 18, 5)
        obj_n = X_objects[:, :, 0]
        obj_n_flat = obj_n.flatten()
        mask = obj_n_flat > 0.5
        unique_obj = torch.unique(obj_n_flat[mask])
        self.obj_types = sorted(unique_obj.tolist())

        # Compute features for normalization
        mask_train = (X_objects[:, :, 0] > 0.5)
        p_T = X_objects[:, :, 2]
        sum_pT = (p_T * mask_train).sum(dim=1)
        max_pT = torch.zeros(X.shape[0], dtype=torch.float32)
        for i in range(X.shape[0]):
            valid_pT = p_T[i][mask_train[i]]
            max_pT[i] = valid_pT.max() if valid_pT.numel() > 0 else 0.0
        num_objects = mask_train.sum(dim=1)
        
        counts = torch.zeros((X.shape[0], len(self.obj_types)), dtype=torch.float32)
        for i, obj in enumerate(self.obj_types):
            counts[:, i] = (X_objects[:, :, 0] == obj).sum(dim=1).float()
        
        E_T_miss = X[:, 0]
        phi = X[:, 1]
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        
        features = torch.stack([E_T_miss, sin_phi, cos_phi, sum_pT, max_pT, num_objects], dim=1)
        features = torch.cat([features, counts], dim=1)
        
        self.mean = features.mean(dim=0)
        self.std = features.std(dim=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        X_objects = X[:, 2:].view(-1, 18, 5)
        N = X.shape[0]
        mask = (X_objects[:, :, 0] > 0.5)
        p_T = X_objects[:, :, 2]
        
        sum_pT = (p_T * mask).sum(dim=1)
        max_pT = torch.zeros(N, dtype=torch.float32)
        for i in range(N):
            valid_pT = p_T[i][mask[i]]
            max_pT[i] = valid_pT.max() if valid_pT.numel() > 0 else 0.0
        num_objects = mask.sum(dim=1)
        
        counts = torch.zeros((N, len(self.obj_types)), dtype=torch.float32)
        for i, obj in enumerate(self.obj_types):
            counts[:, i] = (X_objects[:, :, 0] == obj).sum(dim=1).float()
        
        E_T_miss = X[:, 0]
        phi = X[:, 1]
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        
        features = torch.stack([E_T_miss, sin_phi, cos_phi, sum_pT, max_pT, num_objects], dim=1)
        features = torch.cat([features, counts], dim=1)
        
        features = (features - self.mean.to(features.device)) / self.std.to(features.device)
        return features

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim: int):
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )
    return model

EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct_train += (preds == labels.int()).sum().item()
            total_train += labels.size(0)

        train_loss.append(epoch_train_loss / total_train)
        train_acc.append(correct_train / total_train)

        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).int()
                correct_val += (preds == labels.int()).sum().item()
                total_val += labels.size(0)

        val_loss.append(epoch_val_loss / total_val)
        val_acc.append(correct_val / total_val)

        scheduler.step(val_loss[-1])

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
    torch.save(trained_model.state_dict(), f"{base}_state.pt")
    with open(f"{base}_model.pkl", "wb") as f: pickle.dump(trained_model, f)
    with open(f"{base}_preproc.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

