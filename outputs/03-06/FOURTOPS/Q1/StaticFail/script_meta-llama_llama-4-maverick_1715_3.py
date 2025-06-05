
import os, sys, pickle, torch, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
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

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        if isinstance(self.x, (tuple, list)):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])      

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = PairDataset(X_train, Y_train)
    val_ds   = PairDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))
                        
# ----------------  START OF LLM BLOCK  ----------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# 0. ---------- IMPORTS ----------
import os, sys, pickle, json
import pandas as pd
import matplotlib.pyplot as plt

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

def load_data():
    X_train = torch.from_numpy(pd.read_csv('./challenges/FOURTOPS/data/X_train.csv', dtype=np.float32).to_numpy(copy=False))
    Y_train = torch.from_numpy(pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv', dtype=np.int64).to_numpy(copy=False).ravel())
    X_val   = torch.from_numpy(pd.read_csv('./challenges/FOURTOPS/data/X_val.csv', dtype=np.float32).to_numpy(copy=False))
    Y_val   = torch.from_numpy(pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv', dtype=np.int64).to_numpy(copy=False).ravel())

    return X_train, Y_train, X_val, Y_val

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        if isinstance(self.x, (tuple, list)):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])      

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = PairDataset(X_train, Y_train)
    val_ds   = PairDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.per_object_features = 5
        self.max_objects = 18

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Reshape X to (B, L, F) where L is the sequence length (max_objects) and F is the feature size (per_object_features)
        X = X.view(-1, self.max_objects, self.per_object_features)  # (B, 18, 5)
        X = X[:, :, 1:]  # Remove obj_n identifier, (B, 18, 4)
        mask = (X[:, :, 0] != 0).float()  # (B, 18), mask based on p_T
        return X, mask

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], *, use_mask: bool):
        super().__init__()
        self.use_mask = use_mask
        self.feature_dim = input_shape[-1]
        self.hidden_dim = 128
        self.output_dim = 1

        self.embedding = nn.Linear(self.feature_dim, self.hidden_dim)
        self.encoder = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=self.hidden_dim, dropout=0.1)
        self.encoder = nn.TransformerEncoder(self.encoder, num_layers=3)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, data: torch.Tensor, mask: torch.Tensor | None = None):
        # data: (B, L, F), mask: (B, L)
        embedded = F.relu(self.embedding(data))  # (B, L, H)
        if mask is not None:
            mask = mask == 0  # Invert mask for TransformerEncoder
        encoded = self.encoder(embedded.transpose(0, 1), src_key_padding_mask=mask)  # (L, B, H)
        encoded = encoded.transpose(0, 1)[:, 0, :]  # Take the first token's representation, (B, H)
        output = self.classifier(encoded)  # (B, 1)
        return output.squeeze(-1)

def make_model(input_shape, *, use_mask=False):
    return BinaryClassifier(input_shape, use_mask=use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss, val_loss, train_acc, val_acc, train_auc, val_auc = [], [], [], [], [], []
    best_val_auc = 0.0
    patience = 3
    counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss, epoch_train_correct = 0.0, 0
        predictions, labels = [], []
        for data, label in train_loader:
            if isinstance(data, (tuple, list)):
                data, mask = data
                data, mask, label = data.to(device), mask.to(device), label.to(device).float()
            else:
                data, label = data.to(device), label.to(device).float()
                mask = None

            optimizer.zero_grad()
            output = model(data, mask)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * data.size(0)
            predictions.extend(torch.sigmoid(output).detach().cpu().numpy())
            labels.extend(label.detach().cpu().numpy())

        epoch_train_loss /= len(train_loader.dataset)
        train_loss.append(epoch_train_loss)
        train_auc.append(roc_auc_score(labels, predictions))

        model.eval()
        epoch_val_loss, val_predictions, val_labels = 0.0, [], []
        with torch.no_grad():
            for data, label in val_loader:
                if isinstance(data, (tuple, list)):
                    data, mask = data
                    data, mask, label = data.to(device), mask.to(device), label.to(device).float()
                else:
                    data, label = data.to(device), label.to(device).float()
                    mask = None

                output = model(data, mask)
                loss = criterion(output, label)
                epoch_val_loss += loss.item() * data.size(0)
                val_predictions.extend(torch.sigmoid(output).cpu().numpy())
                val_labels.extend(label.cpu().numpy())

        epoch_val_loss /= len(val_loader.dataset)
        val_loss.append(epoch_val_loss)
        val_auc.append(roc_auc_score(val_labels, val_predictions))

        if val_auc[-1] > best_val_auc:
            best_val_auc = val_auc[-1]
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    return model, train_loss, val_loss, train_auc, val_auc

def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    X_train, Y_train, X_val, Y_val = load_data()
    pre = make_preprocessor().fit(X_train, Y_train)
    X_train, mask_train = pre.transform(X_train)
    X_val, mask_val = pre.transform(X_val)
    train_loader, val_loader = make_loaders((X_train, mask_train), Y_train, (X_val, mask_val), Y_val)

    input_shape = X_train.shape[1:]
    use_mask = True
    model = make_model(input_shape, use_mask=use_mask)

    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_auc, va_auc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    if dryrun:
        toy_data = torch.zeros(8, *input_shape)
        toy_mask = torch.zeros(8, input_shape[0])
        toy_batch = (toy_data, toy_mask)
        toy_transformed = pre.transform(toy_batch)
        try:
            _ = trained_model(*toy_transformed)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_auc,  va_auc,  "AUC", os.path.join(SCRIPT_DIR, f"{base}_auc.png"))

    summary = {
        "epochs": n_epochs,
        "train_loss": tr_loss,
        "val_loss": va_loss,
        "train_auc": tr_auc,
        "val_auc": va_auc,
    }
    print("#TRAIN_METRICS#" + json.dumps(summary))

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END OF LLM-CODE BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    pre = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train) # may be Tensor or Tuple
    X_val   = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    if isinstance(X_train, torch.Tensor):               # single-tensor case
        temp_ref    = X_train
        input_shape = temp_ref.shape[1:]                # e.g. (F,)
        use_mask    = False
    else:                                               # tuple => (data, mask)
        temp_ref    = X_train
        input_shape = temp_ref[0].shape[1:]             # e.g. (L, F)
        use_mask    = True                              
    model = make_model(input_shape, use_mask=use_mask)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single toy forward pass
    if dryrun:
        toy_data = torch.zeros(8, *input_shape, dtype=torch.float32)
        if use_mask:
            toy_mask = torch.zeros(8, input_shape[0], dtype=torch.bool)
            toy_batch = (toy_data, toy_mask)
        else:
            toy_batch = toy_data

        toy_transformed = pre.transform(toy_batch)
        try:
            _ = trained_model(*toy_transformed) if isinstance(toy_transformed, (tuple, list)) \
                else trained_model(toy_transformed)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 6. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
            "train_loss": tr_loss   if tr_loss else None,
            "val_loss":   va_loss   if va_loss else None,
            "train_acc":  tr_acc    if tr_acc else None,
            "val_acc":    va_acc    if va_acc else None,
        }
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

