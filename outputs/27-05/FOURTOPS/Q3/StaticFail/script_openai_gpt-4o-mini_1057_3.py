
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class MyPreprocessor:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Augment particle features to include pairwise interaction energies
data_shape = X.size(1)
        # Base features extraction: missing ET and azimuthal angle
        ET_miss = X[:, 0]  # shape: (N,)
        phi_ET_miss = X[:, 1]  # shape: (N,)

        # Initialize a list for augmented features
        augmented_features = [ET_miss.unsqueeze(1), phi_ET_miss.unsqueeze(1)]

        # Feature engineering for each particle object
dim_per_particle = 5
        max_particles = 18
        for i in range(max_particles):
            obj_idx = 2 + i * dim_per_particle
            if obj_idx + dim_per_particle <= data_shape:
                # Extract features for the i-th particle
                E = X[:, obj_idx + 1]  # E_i
                p_T = X[:, obj_idx + 2]  # p_T_i
                eta = X[:, obj_idx + 3]  # eta_i
                phi = X[:, obj_idx + 4]  # phi_i

                # Create additional features based on physics knowledge (example: transverse mass)
                transverse_mass = torch.sqrt((E**2 - p_T**2) + (p_T**2 * (torch.cosh(eta)**2 - 1)))
                augmented_features.append(transverse_mass.unsqueeze(1)) 

        # Concatenate all augmented features and return
        return torch.cat(augmented_features, dim=1)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, 128)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(128, embed_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = attn_output + x  # Residual connection
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x) + x  # Residual connection

def make_model(input_dim: int):
    class SlotAttentionModel(nn.Module):
        def __init__(self, input_dim):
            super(SlotAttentionModel, self).__init__()
            self.transformer_layer = TransformerLayer(input_dim, 4)
            self.fc = nn.Linear(input_dim, 1)  # Final layer for binary classification

        def forward(self, x):
            x = self.transformer_layer(x.unsqueeze(0))  # Add batch dimension
            return torch.sigmoid(self.fc(x.mean(dim=0)))  # Pooling & Sigmoid output

    return SlotAttentionModel(input_dim)

EPOCHS = 10

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.flatten(), target.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted = (output.flatten() > 0.5).long()
            correct += (predicted == target).sum().item()
            total += target.size(0)
        train_loss.append(epoch_loss / len(train_loader))
        train_acc.append(correct / total)

        # Validation phase
        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            correct = 0
            total = 0
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output.flatten(), target.float())
                epoch_loss += loss.item()
                predicted = (output.flatten() > 0.5).long()
                correct += (predicted == target).sum().item()
                total += target.size(0)
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

