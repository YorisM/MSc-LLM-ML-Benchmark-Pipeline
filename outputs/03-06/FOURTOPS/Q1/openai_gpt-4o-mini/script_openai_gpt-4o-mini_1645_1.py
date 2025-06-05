
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
import os, sys, pickle, torch, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler() # To standardize features

    def fit(self, X, y=None):
        self.scaler.fit(X.numpy())  # Fit scaler on the data
        return self

    def transform(self, X):
        X_normalized = self.scaler.transform(X.numpy())  # Transform the data
        return torch.tensor(X_normalized, dtype=torch.float32)  # (N, features)

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
        self.fc1 = nn.Linear(input_shape[0], 128)  # (F,) -> 128
        self.fc2 = nn.Linear(128, 64)               # 128 -> 64
        self.fc3 = nn.Linear(64, 32)                 # 64 -> 32
        self.output = nn.Linear(32, 1)              # 32 -> 1 (Logits)
        self.dropout = nn.Dropout(0.3)              # Dropout for regularization

    def forward(self, data: torch.Tensor, mask: torch.Tensor | None = None):
        x = torch.relu(self.fc1(data))  # apply first layer
        x = self.dropout(x)              # apply dropout
        x = torch.relu(self.fc2(x))      # apply second layer
        x = self.dropout(x)              # apply dropout
        x = torch.relu(self.fc3(x))      # apply third layer
        x = self.output(x)                # logits
        return torch.sigmoid(x)          # output probabilities

def make_model(input_shape, *, use_mask=False):
    return BinaryClassifier(input_shape, use_mask=use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50  # Set for enough training

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_auc = 0
    early_stopping_counter = 0
    early_stopping_limit = 5  # Early stopping patience

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data).squeeze()  # Get model output
            loss = criterion(outputs, labels.float())  # Calculate loss
            loss.backward()  # Backpropagate
            optimizer.step()  # Update weights

            epoch_loss += loss.item()
            preds = (outputs > 0.5).int()  # Binary prediction
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss.append(epoch_loss / len(train_loader))
        train_acc.append(correct / total)

        # Validation phase
        model.eval()
        val_predictions = []
        val_labels = []
        val_epoch_loss = 0

        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data).squeeze()
                val_loss_value = criterion(outputs, labels.float())
                val_epoch_loss += val_loss_value.item()
                val_predictions.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss.append(val_epoch_loss / len(val_loader))
        val_auc = roc_auc_score(val_labels, val_predictions)  # Calculate AUC

        val_acc.append(accuracy_score(val_labels, np.array(val_predictions) > 0.5))

        # Early stopping check
        if val_auc > best_auc:
            best_auc = val_auc
            early_stopping_counter = 0  # Reset counter if AUC improves
        else:
            early_stopping_counter += 1  # Increment counter if AUC does not improve

        if early_stopping_counter >= early_stopping_limit:
            print("Early stopping at epoch:", epoch + 1)
            break

    return model, train_loss, val_loss, train_acc, val_acc

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
# IMPORTANT: Strictly follow this code template, and do not deviate from it.

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

