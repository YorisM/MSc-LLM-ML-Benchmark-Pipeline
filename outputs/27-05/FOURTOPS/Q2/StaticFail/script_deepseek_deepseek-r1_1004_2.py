
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
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        processed = self._process(X)
        self.mean_ = processed.mean(dim=0)
        self.std_ = processed.std(dim=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        processed = self._process(X)
        if self.mean_ is not None and self.std_ is not None:
            processed = (processed - self.mean_) / self.std_
        return processed

    def _process(self, X):
        E_T_miss = X[:, 0]
        phi_Et_miss = X[:, 1]
        objects = X[:, 2:].view(-1, 18, 5)
        mask = objects[:, :, 0] != 0

        pT = objects[:, :, 2]
        eta = objects[:, :, 3]
        phi_objects = objects[:, :, 4]
        E_objects = objects[:, :, 1]

        px = pT * torch.cos(phi_objects)
        py = pT * torch.sin(phi_objects)
        pz = pT * torch.sinh(eta)

        delta_phi = phi_objects - phi_Et_miss.unsqueeze(1)
        delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
        mT = torch.sqrt(2 * pT * E_T_miss.unsqueeze(1) * (1 - torch.cos(delta_phi)))

        sum_pT = torch.sum(pT * mask.float(), dim=1)
        max_pT, _ = torch.max(pT * mask.float(), dim=1)
        num_objects = torch.sum(mask.float(), dim=1)
        sum_E = torch.sum(E_objects * mask.float(), dim=1)
        sum_px = torch.sum(px * mask.float(), dim=1)
        sum_py = torch.sum(py * mask.float(), dim=1)
        sum_pz = torch.sum(pz * mask.float(), dim=1)
        avg_delta_phi = torch.sum(delta_phi * mask.float(), dim=1) / (num_objects + 1e-8)
        sum_mT = torch.sum(mT * mask.float(), dim=1)

        E_T_miss_x = E_T_miss * torch.cos(phi_Et_miss)
        E_T_miss_y = E_T_miss * torch.sin(phi_Et_miss)
        sum_px_plus_E_T_x = sum_px + E_T_miss_x
        sum_py_plus_E_T_y = sum_py + E_T_miss_y
        vec_sum_mag = torch.sqrt(sum_px_plus_E_T_x**2 + sum_py_plus_E_T_y**2)

        four_vectors = torch.stack([E_objects, px, py, pz], dim=2)
        a = four_vectors.unsqueeze(2)
        b = four_vectors.unsqueeze(1)
        product = a * b
        signs = torch.tensor([1.0, -1.0, -1.0, -1.0], device=X.device)
        dot_products = (product * signs).sum(dim=-1)
        valid_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        dot_products_masked = dot_products * valid_mask.float()

        sum_dot = torch.sum(dot_products_masked.view(X.shape[0], -1), dim=1)
        max_dot = torch.max(dot_products_masked.view(X.shape[0], -1), dim=1)[0]
        min_dot = torch.min(dot_products_masked.view(X.shape[0], -1), dim=1)[0]
        mean_dot = sum_dot / (torch.sum(valid_mask.view(X.shape[0], -1), dim=1) + 1e-8)

        features = torch.stack([
            E_T_miss, phi_Et_miss, sum_pT, max_pT, num_objects, sum_E,
            sum_px, sum_py, sum_pz, avg_delta_phi, sum_mT,
            sum_px_plus_E_T_x, sum_py_plus_E_T_y, vec_sum_mag,
            sum_dot, max_dot, min_dot, mean_dot
        ], dim=1)

        return features

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1)
    )

EPOCHS = 50

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_val = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x).squeeze()
            loss = criterion(pred, y.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            correct += ((torch.sigmoid(pred) > 0.5).long() == y).sum().item()
            total += y.size(0)
        train_loss.append(epoch_loss / total)
        train_acc.append(correct / total)

        model.eval()
        val_epoch_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x).squeeze()
                loss = criterion(pred, y.float())
                val_epoch_loss += loss.item() * x.size(0)
                val_correct += ((torch.sigmoid(pred) > 0.5).long() == y).sum().item()
                val_total += y.size(0)
        val_loss.append(val_epoch_loss / val_total)
        val_acc.append(val_correct / val_total)
        scheduler.step(val_loss[-1])

        if val_loss[-1] < best_val:
            best_val = val_loss[-1]
            best_weights = model.state_dict()

    model.load_state_dict(best_weights)
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

