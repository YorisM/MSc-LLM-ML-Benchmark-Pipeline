
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

# 0. ---------- IMPORTS ----------
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.obj_mask = None
        pass

    def fit(self, X, y=None):
        # Create a mask to identify object features
        obj_mask = torch.zeros(X.shape[1], dtype=torch.bool)
        obj_mask[2:] = True  # Assuming obj features start from index 2
        self.obj_mask = obj_mask
        return self

    def transform(self, X):
        # Separate E_T_miss, phi_Et_miss and object features
        E_T_miss = X[:, 0].unsqueeze(1)  # (N, 1)
        phi_Et_miss = X[:, 1].unsqueeze(1)  # (N, 1)
        obj_features = X[:, 2:]  # (N, 90)
        obj_features = obj_features.view(-1, 18, 5)  # (N, 18, 5)
        
        # Create augmented particle features
        pT = obj_features[:, :, 2].unsqueeze(-1)  # (N, 18, 1)
        eta = obj_features[:, :, 3].unsqueeze(-1)  # (N, 18, 1)
        phi = obj_features[:, :, 4].unsqueeze(-1)  # (N, 18, 1)
        E = obj_features[:, :, 1].unsqueeze(-1)  # (N, 18, 1)
        
        # Calculate new features
        new_features = torch.cat([pT, eta, phi, E, (pT ** 2 + (E * torch.sinh(eta)) ** 2) ** 0.5], dim=-1)  # (N, 18, 5)
        new_features = new_features.view(-1, 90)  # (N, 90)
        
        # Concatenate all features
        X_new = torch.cat([E_T_miss, phi_Et_miss, new_features], dim=1)  # (N, 92)
        return X_new

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.slot_norm = nn.LayerNorm(dim)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, dim))
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        x = self.norm(x)
        k, v = self.to_k(x), self.to_v(x)
        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_log_sigma.expand(b, self.num_slots, -1).exp()
        slots = mu + sigma * torch.randn(mu.shape).to(x.device)
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.slot_norm(slots)
            q = self.to_q(slots)
            dots = torch.einsum('bid,bjd->bij', q, k) * (d ** -0.5)
            attn = dots.softmax(dim=1) + 1e-5
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d)).reshape(b, -1, d)
            slots = slots + self.fc(F.relu(slots))
        return slots

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.preprocessor = make_preprocessor()
        self.embedding = nn.Linear(input_dim, 128)
        self.slot_attention = SlotAttention(num_slots=4, dim=128)
        self.transformer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=256, dropout=0.1)
        self.encoder = nn.TransformerEncoder(self.transformer, num_layers=2)
        self.fc = nn.Sequential(nn.Linear(128 * 4, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        x = self.preprocessor.transform(x)
        x = x[:, 2:].view(-1, 18, 5)  # (N, 18, 5)
        x = self.embedding(x)  # (N, 18, 128)
        slots = self.slot_attention(x)  # (N, 4, 128)
        x = slots.view(-1, 4, 128).permute(1, 0, 2)  # (4, N, 128)
        x = self.encoder(x)  # (4, N, 128)
        x = x.permute(1, 0, 2).reshape(-1, 128 * 4)  # (N, 512)
        x = self.fc(x)  # (N, 2)
        return x

def make_model(input_dim: int):
    return TransformerClassifier(input_dim)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (outputs.argmax(-1) == y).sum().item()
        train_loss.append(total_loss / len(train_loader))
        train_acc.append(total_correct / len(train_loader.dataset))
        model.eval()
        total_loss, total_correct = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                total_loss += loss.item()
                total_correct += (outputs.argmax(-1) == y).sum().item()
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

