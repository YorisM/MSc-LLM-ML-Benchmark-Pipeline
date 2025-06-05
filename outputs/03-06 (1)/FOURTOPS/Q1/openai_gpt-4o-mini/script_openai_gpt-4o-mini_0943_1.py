
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

# 0. ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # No stateful components needed for this preprocessing
        pass

    def fit(self, X, y=None):
        # No specific statistics to extract for this dataset
        return self

    def transform(self, X):
        # Normalize the data: Here we simply replace NaNs and inf with zeros
        # Normalize magnitudes to bring values to a standard scale.
        X_cleaned = torch.where(torch.isnan(X), torch.zeros_like(X), X)  # Replace NaNs with 0
        X_cleaned = torch.where(torch.isinf(X_cleaned), torch.zeros_like(X_cleaned), X_cleaned)  # Replace inf with 0
        X_normalized = (X_cleaned - X_cleaned.mean(dim=0)) / (X_cleaned.std(dim=0) + 1e-6)  # Standardize
        return X_normalized

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class SimpleNN(nn.Module):
    def __init__(self, input_shape):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)               # Hidden layer
        self.fc3 = nn.Linear(64, 32)                # Hidden layer
        self.fc4 = nn.Linear(32, 1)                 # Output layer
        self.dropout = nn.Dropout(0.3)              # Dropout for regularization
        self.activation = nn.ReLU()                 # Activation function

    def forward(self, x):
        x = self.activation(self.fc1(x))  # First layer
        x = self.dropout(x)                # Dropout
        x = self.activation(self.fc2(x))  # Second layer
        x = self.dropout(x)                # Dropout
        x = self.activation(self.fc3(x))  # Third layer
        x = torch.sigmoid(self.fc4(x))    # Output layer with sigmoid
        return x

def make_model(input_shape, *, use_mask=False):
    return SimpleNN(input_shape)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30  # Number of training epochs

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_auc = -float('inf')
    patience = 5
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Accumulate training accuracy
            predictions = (outputs > 0.5).float()
            correct_train += (predictions == labels.float()).sum().item()
            total_train += labels.size(0)

        # Average training loss and accuracy
        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)
        train_acc = correct_train / total_train
        train_acc_history.append(train_acc)

        # Validate the model
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for val_data, val_labels in val_loader:
                val_outputs = model(val_data).squeeze()
                val_loss += criterion(val_outputs, val_labels.float()).item()

                # Accumulate validation accuracy
                val_predictions = (val_outputs > 0.5).float()
                correct_val += (val_predictions == val_labels.float()).sum().item()
                total_val += val_labels.size(0)

                all_val_labels.extend(val_labels.numpy())
                all_val_preds.extend(val_outputs.numpy())

        # Average validation loss and accuracy
        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)
        val_acc = correct_val / total_val
        val_acc_history.append(val_acc)

        # Calculate AUC metric
        val_auc = roc_auc_score(all_val_labels, all_val_preds)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')

        # Early stopping based on AUC
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

    # 4. *Dry-run safety check* – run a single toy forward pass
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

