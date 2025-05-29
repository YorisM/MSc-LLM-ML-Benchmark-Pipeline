
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
        # Separate the missing ET magnitude and azimuth from the object features
        E_T_miss = X[:, 0].unsqueeze(1)  # Shape: [N, 1]
        phi_E_T_miss = X[:, 1].unsqueeze(1)  # Shape: [N, 1]
        objects = X[:, 2:].view(-1, 18, 5)  # Shape: [N, 18, 5]
        # Calculate the total energy and momentum for each object
        E = objects[:, :, 1].unsqueeze(2)  # Shape: [N, 18, 1]
        p_T = objects[:, :, 2].unsqueeze(2)  # Shape: [N, 18, 1]
        eta = objects[:, :, 3].unsqueeze(2)  # Shape: [N, 18, 1]
        phi = objects[:, :, 4].unsqueeze(2)  # Shape: [N, 18, 1]
        p_x = p_T * torch.cos(phi)  # Shape: [N, 18, 1]
        p_y = p_T * torch.sin(phi)  # Shape: [N, 18, 1]
        p_z = p_T * torch.sinh(eta)  # Shape: [N, 18, 1]
        # Combine the features
        features = torch.cat((E_T_miss, phi_E_T_miss, E, p_x, p_y, p_z), dim=1)  # Shape: [N, 19, 1]
        # Mask out the padded objects
        mask = (objects[:, :, 0] != 0).unsqueeze(2)  # Shape: [N, 18, 1]
        features = torch.cat((features[:, :2, :], features[:, 2:, :] * mask), dim=1)  # Shape: [N, 19, 1]
        # Flatten the features
        features = features.view(features.size(0), -1)  # Shape: [N, 19]
        return features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

class LorentzEquivariantLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LorentzEquivariantLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x shape: [N, 19]
        E_T_miss = x[:, 0].unsqueeze(1)  # Shape: [N, 1]
        phi_E_T_miss = x[:, 1].unsqueeze(1)  # Shape: [N, 1]
        objects = x[:, 2:].view(-1, 17)  # Shape: [N, 17]
        E = objects[:, 0].unsqueeze(1)  # Shape: [N, 1]
        p_x = objects[:, 1].unsqueeze(1)  # Shape: [N, 1]
        p_y = objects[:, 2].unsqueeze(1)  # Shape: [N, 1]
        p_z = objects[:, 3].unsqueeze(1)  # Shape: [N, 1]
        # Calculate the Lorentz invariant dot product
        dot_product = E**2 - p_x**2 - p_y**2 - p_z**2  # Shape: [N, 1]
        # Concatenate the features
        features = torch.cat((E_T_miss, phi_E_T_miss, dot_product), dim=1)  # Shape: [N, 3]
        # Apply a fully connected layer
        output = self.fc(features)  # Shape: [N, output_dim]
        return output

def make_model(input_dim: int):
    model = nn.Sequential(
        LorentzEquivariantLayer(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    return model

EPOCHS = 10

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += ((output > 0.5) == y.unsqueeze(1)).sum().item()
            total_samples += y.size(0)
        train_loss.append(total_loss / len(train_loader))
        train_acc.append(total_correct / total_samples)
        model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for X, y in val_loader:
                output = model(X)
                loss = criterion(output, y.unsqueeze(1).float())
                total_loss += loss.item()
                total_correct += ((output > 0.5) == y.unsqueeze(1)).sum().item()
                total_samples += y.size(0)
        val_loss.append(total_loss / len(val_loader))
        val_acc.append(total_correct / total_samples)
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

