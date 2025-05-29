
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
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.et_miss_mean = 0.0
        self.et_miss_std = 1.0
        self.phi_et_miss_mean = 0.0
        self.phi_et_miss_std = 1.0
        self.e_mean = 0.0
        self.e_std = 1.0
        self.pt_mean = 0.0
        self.pt_std = 1.0
        self.eta_mean = 0.0
        self.eta_std = 1.0
        self.phi_mean = 0.0
        self.phi_std = 1.0
        self.obj_types = []

    def fit(self, X, y=None):
        # Calculate missing ET stats
        self.et_miss_mean = X[:, 0].float().mean().item()
        self.et_miss_std = X[:, 0].float().std().item()
        self.phi_et_miss_mean = X[:, 1].float().mean().item()
        self.phi_et_miss_std = X[:, 1].float().std().item()

        # Collect object features
        obj_types = []
        e_list, pt_list, eta_list, phi_list = [], [], [], []
        
        for event in X:
            for i in range(18):
                start = 2 + 5*i
                E = event[start+1].item()
                if E > 0:
                    obj_types.append(event[start].item())
                    e_list.append(E)
                    pt_list.append(event[start+2].item())
                    eta_list.append(event[start+3].item())
                    phi_list.append(event[start+4].item())

        # Calculate feature stats
        self.e_mean = np.mean(e_list) if e_list else 0.0
        self.e_std = np.std(e_list) if e_list else 1.0
        self.pt_mean = np.mean(pt_list) if pt_list else 0.0
        self.pt_std = np.std(pt_list) if pt_list else 1.0
        self.eta_mean = np.mean(eta_list) if eta_list else 0.0
        self.eta_std = np.std(eta_list) if eta_list else 1.0
        self.phi_mean = np.mean(phi_list) if phi_list else 0.0
        self.phi_std = np.std(phi_list) if phi_list else 1.0
        
        # Get unique object types
        self.obj_types = sorted(list(set(obj_types)))
        return self

    def transform(self, X):
        processed = []
        for event in X:
            # Normalize missing ET
            et_norm = (event[0].item() - self.et_miss_mean) / self.et_miss_std
            phi_norm = (event[1].item() - self.phi_et_miss_mean) / self.phi_et_miss_std
            features = [et_norm, phi_norm]

            # Process objects
            for i in range(18):
                start = 2 + 5*i
                obj_type = event[start].item()
                E = event[start+1].item()
                
                if E > 0:
                    # One-hot encoding
                    one_hot = [0.0]*len(self.obj_types)
                    if obj_type in self.obj_types:
                        one_hot[self.obj_types.index(obj_type)] = 1.0
                    # Feature normalization
                    e_norm = (E - self.e_mean) / self.e_std
                    pt_norm = (event[start+2].item() - self.pt_mean) / self.pt_std
                    eta_norm = (event[start+3].item() - self.eta_mean) / self.eta_std
                    phi_norm = (event[start+4].item() - self.phi_mean) / self.phi_std
                else:
                    one_hot = [0.0]*len(self.obj_types)
                    e_norm = pt_norm = eta_norm = phi_norm = 0.0

                features.extend(one_hot + [e_norm, pt_norm, eta_norm, phi_norm])

            processed.append(torch.tensor(features, dtype=torch.float32))

        return torch.stack(processed)

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim: int):
    return nn.Sequential(
        nn.BatchNorm1d(input_dim),
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )

EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == labels.float()).sum().item()
            total += labels.size(0)

        train_loss.append(epoch_loss / total)
        train_acc.append(correct / total)

        # Validation
        model.eval()
        val_epoch_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                val_epoch_loss += loss.item() * inputs.size(0)
                preds = (outputs >= 0.5).float()
                val_correct += (preds == labels.float()).sum().item()
                val_total += labels.size(0)

        val_loss.append(val_epoch_loss / val_total)
        val_acc.append(val_correct / val_total)
        scheduler.step(val_loss[-1])

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

