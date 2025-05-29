
import os, sys, pickle, torch, gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

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
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Reshape the data to separate objects and their features
        X = X.view(-1, 18, 5)  # (batch_size, num_objects, object_features)
        # Mask for non-zero padded objects
        mask = X[:, :, 0] != 0  # Assuming obj_n is at index 0 and is non-zero for valid objects
        # Calculate some basic features
        obj_energies = X[:, :, 1]  # Energies of objects
        obj_pts = X[:, :, 2]  # pT of objects
        obj_etas = X[:, :, 3]  # eta of objects
        obj_phis = X[:, :, 4]  # phi of objects
        # Sum of energies and pT for valid objects
        sum_energy = (obj_energies * mask).sum(dim=1)  # Sum of energies
        sum_pt = (obj_pts * mask).sum(dim=1)  # Sum of pT
        # Missing ET and its phi are directly available
        missing_et = X[:, 0, 0]  # E_T_miss is the first feature
        missing_et_phi = X[:, 0, 1]  # phi_{E_t}_miss is the second feature
        # Concatenate the features
        features = torch.stack([sum_energy, sum_pt, missing_et, missing_et_phi], dim=1)
        return features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
def make_model(input_dim: int):
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)
        epoch_train_loss /= len(train_loader)
        train_loss.append(epoch_train_loss)
        train_acc.append(correct_train / total_train)
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch.float())
                epoch_val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == y_batch).sum().item()
                total_val += y_batch.size(0)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        epoch_val_loss /= len(val_loader)
        val_loss.append(epoch_val_loss)
        val_acc.append(correct_val / total_val)
        val_auc = roc_auc_score(val_labels, val_preds)
        print(f'Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train Acc: {correct_train / total_train:.4f}, Val Acc: {correct_val / total_val:.4f}, Val AUC: {val_auc:.4f}')
    return model, train_loss, val_loss, train_acc, val_acc

# Preprocess the data
preprocessor = make_preprocessor()
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)

# Create data loaders
train_dataset = TensorDataset(X_train_preprocessed, Y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataset = TensorDataset(X_val_preprocessed, Y_val)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Initialize and train the model
model = make_model(X_train_preprocessed.shape[1])
trained_model, train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, EPOCHS)

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
    torch.save(trained_model.state_dict(), f"{base}_state.pt")
    with open(f"{base}_model.pkl", "wb") as f: pickle.dump(trained_model, f)
    with open(f"{base}_preproc.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

