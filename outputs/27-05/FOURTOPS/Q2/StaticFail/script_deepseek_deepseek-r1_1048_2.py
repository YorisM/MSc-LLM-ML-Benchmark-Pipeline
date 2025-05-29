
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
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.feature_size = 10

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        N = X.size(0)
        objects = X.view(N, 18, 5)
        obj_type = objects[:, :, 0]
        mask = obj_type != 0

        E = objects[:, :, 1]
        pT = objects[:, :, 2]
        eta = objects[:, :, 3]
        phi = objects[:, :, 4]

        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)

        four_vectors = torch.stack([E, px, py, pz], dim=2)

        features = []
        for i in range(N):
            mask_i = mask[i]
            indices = torch.nonzero(mask_i).squeeze()
            fv = four_vectors[i, indices]
            num_real = len(indices) if indices.dim() > 0 else 0

            sum_dot = mean_dot = max_dot = min_dot = std_dot = 0.0
            if num_real >= 2:
                E_i = fv[:, 0]
                px_i = fv[:, 1]
                py_i = fv[:, 2]
                pz_i = fv[:, 3]

                E_ij = torch.outer(E_i, E_i)
                p_ij = torch.outer(px_i, px_i) + torch.outer(py_i, py_i) + torch.outer(pz_i, pz_i)
                dots = E_ij - p_ij
                triu = torch.triu_indices(num_real, num_real, 1)
                dots_flat = dots[triu[0], triu[1]]

                sum_dot = dots_flat.sum().item()
                mean_dot = dots_flat.mean().item()
                max_dot = dots_flat.max().item()
                min_dot = dots_flat.min().item()
                std_dot = dots_flat.std().item()

            count = num_real
            sum_e = E[i][mask_i].sum().item()
            sum_pt = pT[i][mask_i].sum().item()

            features.append([
                X[i,0].item(), X[i,1].item(),
                sum_dot, mean_dot, max_dot, min_dot, std_dot,
                count, sum_e, sum_pt
            ])

        return torch.tensor(features, dtype=torch.float32)

    def fit_transform(self, X, y=None):
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

class LorentzMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def make_model(input_dim: int):
    return LorentzMLP(input_dim)

EPOCHS = 15

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        y_true, y_score = [], []

        foratch,atch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            y_true.extend(y_batch.tolist())
            y_score.extend(torch.sigmoid(output).detach().tolist())

        train_loss.append(epoch_loss/len(train_loader))
        train_auc = roc_auc_score(y_true, y_score)
        train_acc.append(train_auc)

        model.eval()
        val_loss_epoch = 0
        y_true_val, y_score_val = [], []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                output = model(X_val_batch).squeeze()
                val_loss_epoch += criterion(output, y_val_batch.float()).item()
                y_true_val.extend(y_val_batch.tolist())
                y_score_val.extend(torch.sigmoid(output).tolist())

        val_loss.append(val_loss_epoch/len(val_loader))
        val_auc = roc_auc_score(y_true_val, y_score_val)
        val_acc.append(val_auc)
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

