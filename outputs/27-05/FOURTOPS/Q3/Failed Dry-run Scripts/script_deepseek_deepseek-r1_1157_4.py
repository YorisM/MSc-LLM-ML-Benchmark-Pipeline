
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
        batch_size = X.shape[0]
        et_miss = X[:, 0]
        phi_et_miss = X[:, 1]
        px_miss = et_miss * torch.cos(phi_et_miss)
        py_miss = et_miss * torch.sin(phi_et_miss)

        object_features = []
        for i in range(18):
            start_idx = 2 + 5 * i
            obj_n = X[:, start_idx]
            E = X[:, start_idx + 1]
            pT = X[:, start_idx + 2]
            eta = X[:, start_idx + 3]
            phi = X[:, start_idx + 4]
            px = pT * torch.cos(phi)
            py = pT * torch.sin(phi)
            pz = pT * torch.sinh(eta)
            obj_feat = torch.stack([obj_n, E, pT, eta, phi, px, py, pz, px_miss, py_miss], dim=1)
            object_features.append(obj_feat)
        all_features = torch.cat(object_features, dim=1)
        return all_features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.slots = nn.Parameter(torch.randn(1, num_slots, dim))
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, inputs, mask=None):
        b, n, d = inputs.shape
        slots = self.slots.expand(b, -1, -1)
        for _ in range(self.iters):
            q = self.to_q(slots)
            k = self.to_k(inputs)
            v = self.to_v(inputs)
            attn_logits = torch.einsum('bid,bjd->bij', q, k) * self.scale
            if mask is not None:
                attn_logits = attn_logits.masked_fill(mask.unsqueeze(1), -1e9)
            attn = F.softmax(attn_logits, dim=-1)
            updates = torch.einsum('bij,bjd->bid', attn, v)
            slots = self.norm(slots + updates)
        return slots

class MyModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, num_slots=4):
        super().__init__()
        self.embedding = nn.Embedding(100, embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim + 9,
                nhead=4,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=embedding_dim+9)
        self.classifier = nn.Sequential(
            nn.Linear(num_slots*(embedding_dim+9), 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), 18, 10)
        obj_n = x[:, :, 0].long()
        continuous = x[:, :, 1:]
        embedded = self.embedding(obj_n)
        combined = torch.cat([embedded, continuous], dim=2)
        mask = (obj_n == 0)
        encoded = self.transformer_encoder(combined, src_key_padding_mask=mask)
        slots = self.slot_attention(encoded, mask=mask)
        slots_flat = slots.flatten(1)
        return self.classifier(slots_flat).squeeze(-1)

def make_model(input_dim: int):
    return MyModel(input_dim)

EPOCHS = 10

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    
    for epoch in range(epochs):
        model.train()
        total_loss = correct = total = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            correct += ((torch.sigmoid(pred) > 0.5).long() == y).sum().item()
            total += y.size(0)
        scheduler.step()
        train_loss.append(total_loss / total)
        train_acc.append(correct / total)
        
        model.eval()
        val_total_loss = val_correct = val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X)
                loss = criterion(pred, y.float())
                val_total_loss += loss.item() * X.size(0)
                val_correct += ((torch.sigmoid(pred) > 0.5).long() == y).sum().item()
                val_total += y.size(0)
        val_loss.append(val_total_loss / val_total)
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

