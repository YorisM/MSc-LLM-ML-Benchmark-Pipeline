
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
        object_features = X[:, 2:].view(-1, 18, 5)  # obj_n, E, p_T, eta, phi
        obj_ids = object_features[:, :, 0].long()
        energies = object_features[:, :, 1]
        pts = object_features[:, :, 2]
        etas = object_features[:, :, 3]
        phis = object_features[:, :, 4]
        
        # Create new features: p_T/E, eta, phi, and obj_id as category
        new_features = torch.stack([pts / (energies + 1e-8), etas, phis], dim=-1)
        new_features = torch.cat([new_features, obj_ids.unsqueeze(-1)], dim=-1)
        
        # Mask for valid objects (not zero-padded)
        mask = (obj_ids != 0).float().unsqueeze(-1)
        new_features = new_features * mask
        
        # Global features: E_T_miss, phi_{E_t}_miss, total visible energy, total visible p_T
        total_visible_energy = energies.sum(dim=1, keepdim=True)
        total_visible_pt = pts.sum(dim=1, keepdim=True)
        global_features = torch.cat([other_features, total_visible_energy, total_visible_pt], dim=1)
        
        return torch.cat([global_features, new_features.view(X.size(0), -1)], dim=1)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.dim = dim
        self.scale = dim ** -0.5
        
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        self.gru = nn.GRUCell(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.out_layer = nn.Linear(dim, dim)
        
    def forward(self, feat, mask=None):
        b, n, d = feat.shape
        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_sigma.expand(b, self.num_slots, -1)
        slots = torch.normal(mu, sigma)
        
        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(slots)
            k = self.to_k(feat)
            v = self.to_v(feat)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            if mask is not None:
                dots.masked_fill_(~mask[:, None, :], -float('inf'))
            attn = dots.softmax(dim=1)
            updates = torch.einsum('bij,bjd->bid', attn, v)
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d)).reshape(b, -1, d)
            slots = self.norm(slots)
        
        return self.out_layer(slots)

def make_model(input_dim: int):
    num_slots = 4  # Assuming 4 top quarks
    slot_dim = 8
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(input_dim, slot_dim)
            self.slot_attention = SlotAttention(num_slots, slot_dim)
            self.fc = nn.Sequential(
                nn.Linear(num_slots * slot_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
        
        def forward(self, x):
            x = self.embedding(x)
            x = x.view(-1, 18 + 4, x.size(-1))  # +4 for global features
            global_features = x[:, :4, :]
            object_features = x[:, 4:, :]
            mask = (object_features[:, :, 0] != 0)
            slots = self.slot_attention(object_features, mask)
            x = torch.cat([global_features.mean(dim=1), slots.view(slots.size(0), -1)], dim=1)
            return self.fc(x)
    
    return Model()

EPOCHS = 10

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(epochs):
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
        train_loss.append(epoch_loss / len(train_loader))
        train_acc.append(correct / total)
        
        model.eval()
        epoch_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        val_loss.append(epoch_loss / len(val_loader))
        val_acc.append(correct / total)
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

