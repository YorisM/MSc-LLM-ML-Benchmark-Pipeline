
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
import torch.nn.functional as F

class MyPreprocessor:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Separate object features from other features
        other_features = X[:, :2]  # E_T_miss, phi_{E_t}_miss
        object_features = X[:, 2:].view(-1, 18, 5)  # obj_n, E_n, p_Tn, eta_n, phi_n
        
        # Create augmented particle features
        # Calculate p_x, p_y, p_z from p_T, eta, phi
        p_T = object_features[:, :, 2]  # p_T
        eta = object_features[:, :, 3]  # eta
        phi = object_features[:, :, 4]  # phi
        p_x = p_T * torch.cos(phi)  # (N, 18)
        p_y = p_T * torch.sin(phi)  # (N, 18)
        p_z = p_T * torch.sinh(eta)  # (N, 18)
        E = object_features[:, :, 1]  # E
        
        # Concatenate the new features with the original object features
        new_object_features = torch.stack([p_x, p_y, p_z, E], dim=-1)  # (N, 18, 4)
        object_features = torch.cat([object_features, new_object_features], dim=-1)  # (N, 18, 9)
        
        # Flatten the object features
        object_features = object_features.view(X.size(0), -1)  # (N, 18*9)
        
        # Concatenate other features with the object features
        X_transformed = torch.cat([other_features, object_features], dim=1)  # (N, 2 + 18*9)
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.scale = dim ** -0.5
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        slots_mu = self.slots_mu.expand(b, self.num_slots, -1)
        slots_sigma = self.slots_sigma.expand(b, self.num_slots, -1)
        slots = torch.normal(slots_mu, slots_sigma)
        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(slots)
            k = self.to_k(x)
            v = self.to_v(x)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1)
            updates = torch.einsum('bij,bjd->bid', attn, v)
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))
            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(slots)))
        return slots

def make_model(input_dim: int):
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.LayerNorm(128),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.LayerNorm(128),
        SlotAttention(num_slots=4, dim=128, iters=3),
        nn.Linear(128*4, 1),
        nn.Sigmoid()
    )
    return model

EPOCHS = 10
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.float().to(device)
            optimizer.zero_grad()
            output = model(X).squeeze()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += ((output > 0.5) == y).sum().item()
        train_loss.append(total_loss / len(train_loader))
        train_acc.append(total_correct / len(train_loader.dataset))
        model.eval()
        total_loss, total_correct = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.float().to(device)
                output = model(X).squeeze()
                loss = criterion(output, y)
                total_loss += loss.item()
                total_correct += ((output > 0.5) == y).sum().item()
        val_loss.append(total_loss / len(val_loader))
        val_acc.append(total_correct / len(val_loader.dataset))
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

