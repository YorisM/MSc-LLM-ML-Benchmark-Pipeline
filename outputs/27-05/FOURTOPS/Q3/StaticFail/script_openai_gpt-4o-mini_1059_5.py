
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Augment features by extracting additional meaningful information
        # For example, we can compute transverse mass or invariant mass from the objects
        obj_features = X[:, 2:].reshape(X.shape[0], -1, 5)  # Reshape to (N, num_objects, 5)
        E_T_miss = X[:, 0]
        phi_E_t_miss = X[:, 1]

        # Compute particle pair masses or angle differences as additional features
        # Here, we contextualize pairs in a way that could assist the transformer
        augmented_features = []
        for i in range(obj_features.shape[1]):
            for j in range(i + 1, obj_features.shape[1]):
                # Calculate mass and angle differences
                E1, pT1, eta1, phi1 = obj_features[:, i, 1:], obj_features[:, j, 1:], obj_features[:, j, 1:]
                m_squared = (E1 + E2)**2 - (pT1 * pT2 * torch.cos(phi2 - phi1)).sum(dim=1)**2 - (eta1 - eta2)**2
                m = torch.sqrt(torch.clamp(m_squared, min=0))
                augmented_features.append(m)

        augmented_tensor = torch.stack(augmented_features, dim=1)  # shape (N, num_pairs)
        final_features = torch.cat([X[:, :2], augmented_tensor], dim=1)  # Combine with E_T_miss and phi_E_t_miss
        return final_features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
def make_model(input_dim: int):
    # We define a transformer model based on slot-attention
    class SlotAttention(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SlotAttention, self).__init__()
            self.attention_layer = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4)
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, output_dim)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            attn_output, _ = self.attention_layer(x, x, x)
            x = attn_output.mean(dim=1)
            x = self.fc2(x)
            return x

    return SlotAttention(input_dim, 1)  # Binary classification

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            X_batch, y_batch = batch
            outputs = model(X_batch)  # Forward pass
            loss = criterion(outputs.squeeze(), y_batch.float())
            loss.backward()  # Backward pass
            optimizer.step()  # Optimization step
            total_loss += loss.item()
            preds = torch.sigmoid(outputs.squeeze()) > 0.5
            correct += (preds.int() == y_batch).sum().item()
        train_loss.append(total_loss / len(train_loader))
        train_acc.append(correct / len(train_loader.dataset))

        # Validation
        model.eval()
        total_val_loss, correct_val = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                X_val_batch, y_val_batch = batch
                val_outputs = model(X_val_batch)
                val_loss_value = criterion(val_outputs.squeeze(), y_val_batch.float())
                total_val_loss += val_loss_value.item()
                val_preds = torch.sigmoid(val_outputs.squeeze()) > 0.5
                correct_val += (val_preds.int() == y_val_batch).sum().item()
        val_loss.append(total_val_loss / len(val_loader))
        val_acc.append(correct_val / len(val_loader.dataset))

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

