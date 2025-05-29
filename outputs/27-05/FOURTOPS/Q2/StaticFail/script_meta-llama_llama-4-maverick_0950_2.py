
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
from torch_scatter import scatter_max, scatter_mean

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Separate objects and global features
        global_features = X[:, :2]  # E_T_miss, phi_{E_t}_miss
        object_features = X[:, 2:].view(-1, 18, 5)  # 18 objects, 5 features each
        obj_mask = (object_features[:, :, 0] != 0).float()  # Mask for non-zero-padded objects
        return global_features, object_features, obj_mask

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class LorentzEquivariantLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LorentzEquivariantLayer, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return F.relu(self.linear(x))

class ParticleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ParticleTransformer, self).__init__()
        self.query_linear = nn.Linear(input_dim, hidden_dim)
        self.key_linear = nn.Linear(input_dim, hidden_dim)
        self.value_linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, obj_mask):
        queries = self.query_linear(x)
        keys = self.key_linear(x)
        values = self.value_linear(x)
        attention_weights = torch.matmul(queries, keys.transpose(-1, -2)) / np.sqrt(x.size(-1))
        attention_weights = attention_weights.masked_fill(obj_mask.unsqueeze(-1) == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=-1)
        output = torch.matmul(attention_weights, values)
        return output

def make_model(input_dim: int):
    model = nn.Sequential(
        LorentzEquivariantLayer(input_dim, 128),
        ParticleTransformer(128, 128),
        LorentzEquivariantLayer(128, 64),
        nn.Flatten(),
        nn.Linear(64 * 18 + 128, 1)
    )
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for global_features, object_features, obj_mask, labels in train_loader:
            global_features, object_features, obj_mask, labels = global_features.to(device), object_features.to(device), obj_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            x = model[0](object_features)
            x = model[1](x, obj_mask)
            x = model[2](x)
            x = torch.cat([x.mean(dim=1), global_features], dim=1)
            x = model[4](x)
            loss = criterion(x.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += ((x.squeeze() > 0) == labels).sum().item()
        train_loss.append(total_loss / len(train_loader))
        train_acc.append(total_correct / len(train_loader.dataset))
        model.eval()
        total_loss, total_correct = 0, 0
        with torch.no_grad():
            for global_features, object_features, obj_mask, labels in val_loader:
                global_features, object_features, obj_mask, labels = global_features.to(device), object_features.to(device), obj_mask.to(device), labels.to(device)
                x = model[0](object_features)
                x = model[1](x, obj_mask)
                x = model[2](x)
                x = torch.cat([x.mean(dim=1), global_features], dim=1)
                x = model[4](x)
                loss = criterion(x.squeeze(), labels.float())
                total_loss += loss.item()
                total_correct += ((x.squeeze() > 0) == labels).sum().item()
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

