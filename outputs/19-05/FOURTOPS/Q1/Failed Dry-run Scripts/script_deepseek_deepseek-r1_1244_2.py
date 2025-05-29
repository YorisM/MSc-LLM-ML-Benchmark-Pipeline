
import os, sys, json, pickle, torch
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

EPOCHS = 10
                        
def load_data():
    to_np = lambda path: pd.read_csv(path).values
    X_train = to_np(DATASET["X_train"])
    Y_train = to_np(DATASET["Y_train"]).ravel()
    X_val   = to_np(DATASET["X_val"])
    Y_val   = to_np(DATASET["Y_val"]).ravel()
    return X_train, Y_train, X_val, Y_val

def make_loaders(X_train, Y_train, X_val, Y_val, batch=1024):
    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train))
    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val))
    return (DataLoader(train, batch_size=batch, shuffle=True),
            DataLoader(val, batch_size=batch))
                        
# ----------------  START OF LLM BLOCK  ----------------
import os
import sys
import json
import pickle
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

class MyPreprocessor:
    def __init__(self):
        self.obj_categories_ = None
        self.etm_scaler = StandardScaler()
        self.etphi_scaler = StandardScaler()
        self.obj_e_scaler = StandardScaler()
        self.obj_pt_scaler = StandardScaler()
        self.obj_eta_scaler = StandardScaler()
        self.obj_phi_scaler = StandardScaler()

    def fit(self, X, y=None):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        
        etm = X_np[:, 0].reshape(-1, 1)
        etphi = X_np[:, 1].reshape(-1, 1)
        self.etm_scaler.partial_fit(etm)
        self.etphi_scaler.partial_fit(etphi)

        objects = []
        valid_mask = []
        for event in X_np:
            objs = event[2:].reshape(-1, 5)[:18]
            valid = (objs[:, 0] != 0).astype(int)
            valid_mask.extend(valid)
            objects.extend(objs[:, 1:][valid.astype(bool)])
        
        if len(objects) > 0:
            objects = np.array(objects)
            self.obj_e_scaler.partial_fit(objects[:, 0].reshape(-1, 1))
            self.obj_pt_scaler.partial_fit(objects[:, 1].reshape(-1, 1))
            self.obj_eta_scaler.partial_fit(objects[:, 2].reshape(-1, 1))
            self.obj_phi_scaler.partial_fit(objects[:, 3].reshape(-1, 1))

        unique_objs = np.unique(X_np[:, 2::5][X_np[:, 2::5] != 0])
        self.obj_categories_ = np.sort(unique_objs)
        return self

    def transform(self, X):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        transformed = []
        
        for event in X_np:
            features = []
            
            etm_norm = self.etm_scaler.transform([[event[0]]])[0][0]
            etphi_norm = self.etphi_scaler.transform([[event[1]]])[0][0]
            features.extend([etm_norm, etphi_norm])
            
            objs = event[2:].reshape(-1, 5)[:18]
            for obj in objs:
                if obj[0] != 0:
                    e_norm = self.obj_e_scaler.transform([[obj[1]]])[0][0]
                    pt_norm = self.obj_pt_scaler.transform([[obj[2]]])[0][0]
                    eta_norm = self.obj_eta_scaler.transform([[obj[3]]])[0][0]
                    phi_norm = self.obj_phi_scaler.transform([[obj[4]]])[0][0]
                    
                    obj_onehot = (self.obj_categories_ == obj[0]).astype(float)
                else:
                    e_norm = pt_norm = eta_norm = phi_norm = 0.0
                    obj_onehot = np.zeros_like(self.obj_categories_, dtype=float)
                
                features.extend(obj_onehot)
                features.extend([e_norm, pt_norm, eta_norm, phi_norm])
            
            transformed.append(features)
        
        return np.array(transformed, dtype=np.float32)

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def make_model(input_dim):
    return BinaryClassifier(input_dim)

EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
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
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_loss.append(epoch_loss / total)
        train_acc.append(correct / total)
        
        model.eval()
        val_epoch_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                
                val_epoch_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss.append(val_epoch_loss / val_total)
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
    X_tr, y_tr, X_va, y_va = load_data()
    pre = make_preprocessor();  pre.fit(X_tr, y_tr)
    X_tr = pre.transform(X_tr); X_va = pre.transform(X_va)
    tr_loader, va_loader = make_loaders(X_tr, y_tr, X_va, y_va)

    # 2. Build model
    model = make_model(input_dim=X_tr.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    trained, tr_loss, va_loss, tr_acc, va_acc = train_model(
        model, tr_loader, va_loader, epochs=n_epochs
    )

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy = torch.zeros(8, X_tr.shape[1])      # 8 fake events
        try:
            _ = trained(pre.transform(toy))
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return  # no files in dry-run

    # 4. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
    torch.save(trained.state_dict(), f"{base}_state.pt")
    with open(f"{base}_model.pkl", "wb") as f: pickle.dump(trained, f)
    with open(f"{base}_preproc.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

