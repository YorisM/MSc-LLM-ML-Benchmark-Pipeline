
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
from sklearn.metrics import roc_auc_score
import torch.optim as optim

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # No stateful components are needed for this preprocessor
        pass

    def fit(self, X, y=None):
        # No fitting parameters are required in this case
        return self

    def transform(self, X):
        # Normalize the first two features (E_T_miss and phi_Et_miss)
        X_new = X.clone()
        X_new[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()  # E_T_miss normalization
        X_new[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()  # phi_Et_miss normalization
        return X_new 

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class SimpleFeedForwardModel(nn.Module):
    def __init__(self, input_shape):
        super(SimpleFeedForwardModel, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))  # Output layer
        return x

def make_model(input_shape, *, use_mask=False):
    model = SimpleFeedForwardModel(input_shape)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50  # Defined amount of training epochs

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_auc = 0  # Variable to store the best AUC
    patience = 5  # Early stopping patience
    trigger_times = 0  # To count the number of times validation AUC does not improve

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).int()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        # Validation step
        model.eval()
        val_loss = 0.0
        val_outputs = []
        val_labels = []

        with torch.no_grad():
            for val_data, val_labels_batch in val_loader:
                val_output = model(val_data).squeeze()
                val_loss += criterion(val_output, val_labels_batch.float()).item()
                val_outputs.append(val_output)
                val_labels.append(val_labels_batch)

        val_loss /= len(val_loader)
        val_outputs = torch.cat(val_outputs).cpu().numpy()
        val_labels = torch.cat(val_labels).cpu().numpy()

        val_auc = roc_auc_score(val_labels, val_outputs)
        val_loss_history.append(val_loss)

        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            trigger_times = 0  # Reset trigger times if the model improved
        else:
            trigger_times += 1

        if trigger_times >= patience:
            print(f'Early stopping on epoch {epoch+1}')
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

