
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

import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.obj_types = None
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        # Extract unique object types from non-padded objects
        objects = X[:, 2:].view(-1, 18, 5)
        obj_types = objects[:, :, 0].flatten()
        non_zero = obj_types[obj_types != 0]
        self.obj_types = torch.unique(non_zero).tolist()
        
        # Compute normalization parameters
        processed = self._transform_without_norm(X)
        self.mean = processed.mean(dim=0)
        self.std = processed.std(dim=0)
        self.std[self.std == 0] = 1.0
        return self

    def _transform_without_norm(self, X):
        # Process global features
        phi_et = X[:, 1]
        phi_sin = torch.sin(phi_et).unsqueeze(1)
        phi_cos = torch.cos(phi_et).unsqueeze(1)
        global_features = torch.cat([X[:, 0:1], phi_sin, phi_cos], dim=1)
        
        # Process object features
        objects = X[:, 2:].view(-1, 18, 5)
        obj_type_mask = objects[:, :, 0]
        mask = obj_type_mask != 0
        
        # Object type counts
        count_features = []
        if self.obj_types:
            for t in self.obj_types:
                count = (obj_type_mask == t).sum(dim=1, keepdim=True)
                count_features.append(count)
        count_features = torch.cat(count_features, dim=1) if count_features else torch.zeros((X.shape[0], 0))
        
        # Kinematic aggregations
        sum_pT = (objects[:, :, 2] * mask).sum(dim=1, keepdim=True)
        sum_E = (objects[:, :, 1] * mask).sum(dim=1, keepdim=True)
        count_objects = mask.sum(dim=1, keepdim=True)
        avg_eta = (objects[:, :, 3] * mask).sum(dim=1, keepdim=True) / (count_objects + 1e-8)
        max_pT, _ = (objects[:, :, 2] * mask).max(dim=1, keepdim=True)
        
        return torch.cat([
            global_features,
            count_features,
            sum_pT,
            sum_E,
            count_objects,
            avg_eta,
            max_pT
        ], dim=1)

    def transform(self, X):
        processed = self._transform_without_norm(X)
        return (processed - self.mean.to(X.device)) / self.std.to(X.device)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()


def make_model(input_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1)
    )

EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
    
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)
        
        train_loss.append(epoch_loss / total)
        train_acc.append(correct / total)
        
        model.eval()
        val_preds, val_labels = [], []
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                epoch_val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).int()
                val_correct += (preds == labels.int()).sum().item()
                val_total += labels.size(0)
                val_preds.append(outputs.cpu())
                val_labels.append(labels.cpu())
        
        val_loss.append(epoch_val_loss / val_total)
        val_acc.append(val_correct / val_total)
        
        # Calculate AUC
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        auc = roc_auc_score(val_labels, val_preds)
        scheduler.step(auc)
    
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
    torch.save(trained_model.state_dict(), f"{base}_state.pt")
    with open(f"{base}_model.pkl", "wb") as f: pickle.dump(trained_model, f)
    with open(f"{base}_preproc.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

