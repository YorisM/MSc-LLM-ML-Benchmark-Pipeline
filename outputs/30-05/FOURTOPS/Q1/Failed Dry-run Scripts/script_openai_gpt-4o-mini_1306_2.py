
import os, sys, pickle, torch, gc, json
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

class MyPreprocessor:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Normalize the inputs using standard scaling
        mean = torch.mean(X, dim=0)
        std = torch.std(X, dim=0)
        X_normalized = (X - mean) / std  # Normalize (N, 92)
        return X_normalized

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim: int):
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
        nn.Sigmoid()  # output probability for binary classification
    )
    return model

EPOCHS = 50  # Set a moderate number of epochs for training.

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_train_loss = 0
        epoch_train_correct = 0
        total_train = 0

        for data, target in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights

            epoch_train_loss += loss.item() * data.size(0)
            predicted = (output > 0.5).int()
            epoch_train_correct += (predicted.squeeze() == target).sum().item()
            total_train += target.size(0)

        train_loss.append(epoch_train_loss / total_train)
        train_acc.append(epoch_train_correct / total_train)

        model.eval()  # Set the model to evaluation mode
        epoch_val_loss = 0
        epoch_val_correct = 0
        total_val = 0

        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output.squeeze(), target.float())

                epoch_val_loss += loss.item() * data.size(0)
                predicted = (output > 0.5).int()
                epoch_val_correct += (predicted.squeeze() == target).sum().item()
                total_val += target.size(0)

        val_loss.append(epoch_val_loss / total_val)
        val_acc.append(epoch_val_correct / total_val)

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

    # 6. Write JSON Summary
    summary = {
        "epochs": n_epochs,
        "train_loss": tr_loss,
        "val_loss":   va_loss,
        "train_acc":  tr_acc,
        "val_acc":    va_acc,
        "best_val_loss": min(va_loss),
        "best_val_acc":  max(va_acc),
    }
    print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

