
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
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()  # Initialize the scaler to standardize input features

    def fit(self, X, y=None):
        # Fit the scaler only on the training set
        self.scaler.fit(X.numpy())  # Convert to numpy for fitting
        return self

    def transform(self, X):
        # Transform the data by scaling
        X_scaled = self.scaler.transform(X.numpy())
        return torch.tensor(X_scaled, dtype=torch.float32)  # Convert back to tensor

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class SimpleNN(nn.Module):
    def __init__(self, input_shape):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], 128)  # Input layer to hidden layer
        self.fc2 = nn.Linear(128, 64)               # Hidden layer to hidden layer
        self.fc3 = nn.Linear(64, 1)                 # Hidden layer to output layer
        self.relu = nn.ReLU()                        # ReLU activation
        self.sigmoid = nn.Sigmoid()                  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Pass through first layer and activation
        x = self.relu(self.fc2(x))  # Pass through second layer and activation
        x = self.sigmoid(self.fc3(x))  # Output layer with sigmoid
        return x

def make_model(input_shape, *, use_mask=False):
    return SimpleNN(input_shape)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50  # Define the number of training epochs

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    best_val_auc = 0  # Track the best AUC for early stopping

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        for data, label in train_loader:
            optimizer.zero_grad()  # Zero gradients
            outputs = model(data).squeeze()  # Get model predictions
            loss = criterion(outputs, label.float())  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()  # Accumulate loss

        train_loss.append(total_loss / len(train_loader))  # Average train loss

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_targets, val_outputs = [], []
        with torch.no_grad():
            for data, label in val_loader:
                outputs = model(data).squeeze()
                val_outputs.append(outputs)
                val_targets.append(label)

        val_outputs = torch.cat(val_outputs).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_auc = roc_auc_score(val_targets, val_outputs)  # Calculate AUC

        val_loss.append(0)  # We can skip loss calculation since it's not needed for AUC maximization
        val_acc.append(np.mean((val_outputs > 0.5) == val_targets))  # Calculate accuracy

        # Check for improvement in validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            # Here we could implement model checkpointing if needed

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
    if isinstance(X_train, torch.Tensor):               # single-tensor case
        input_shape = X_train.shape[1:]
        use_mask    = False
    else:
        input_shape = X_train[0].shape[1:]              # tuple case
        use_mask    = True
    model = make_model(input_shape, use_mask=use_mask)

    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)

    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
              # 8 fake events
        if isinstance(X_train, torch.Tensor):           # single-tensor case
            toy = torch.zeros(8, *input_shape, dtype=torch.float32)
            toy_transformed = pre.transform(toy)
        else:                                           # tuple case
            toy = torch.zeros(8, *input_shape, dtype=torch.float32)
            mask = torch.zeros(8, input_shape[0], dtype=torch.bool)
            toy_transformed = (toy, mask)
        try: 
            _ = trained_model(*toy_transformed) if use_mask else trained_model(toy_transformed)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

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
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
            "train_loss": tr_loss,
            "val_loss":   va_loss,
            "train_acc":  tr_acc,
            "val_acc":    va_acc,
            "best_train_loss": min(tr_loss),
            "best_train_loss_epoch": tr_loss.index(min(tr_loss))+1,
            "best_train_acc":  max(tr_acc),
            "best_train_acc_epoch": tr_acc.index(max(tr_acc))+1,
            "best_val_loss": min(va_loss),
            "best_val_loss_epoch": va_loss.index(min(va_loss))+1,
            "best_val_acc":  max(va_acc),
            "best_val_acc_epoch": va_acc.index(max(va_acc))+1,
        }
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

