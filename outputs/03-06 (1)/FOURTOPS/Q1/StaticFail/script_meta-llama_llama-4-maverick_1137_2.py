
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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # Initialize stateful components
        self.max_objects = 18  # Maximum number of objects encoded
        self.per_object_slice_size = 5  # Size of each object's slice

    def fit(self, X, y=None):
        # Extract statistics or fit transformers if needed
        return self

    def transform(self, X):
        # Reshape X to separate objects and their features
        # X shape: (batch_size, 92)
        batch_size = X.shape[0]
        X = X.view(batch_size, -1, self.per_object_slice_size)  # (N, 18, 5)

        # Extract object presence mask (obj_n != 0)
        mask = X[:, :, 0] != 0  # (N, 18)

        # Remove obj_n identifier from the features
        X = X[:, :, 1:]  # (N, 18, 4)

        return (X, mask)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, input_shape, use_mask=False):
        super(BinaryClassifier, self).__init__()
        self.use_mask = use_mask
        self.object_embedding = nn.Linear(4, 128)  # Embedding for each object's features
        self.encoder = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=256, dropout=0.1)
        self.encoder = nn.TransformerEncoder(self.encoder, num_layers=3)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        # x shape: (N, L, F) where L is the sequence length and F is the feature size
        x = torch.relu(self.object_embedding(x))  # (N, L, 128)
        if self.use_mask:
            mask = ~mask  # Invert mask for TransformerEncoder
            x = x.permute(1, 0, 2)  # (L, N, 128)
            x = self.encoder(x, src_key_padding_mask=mask)  # (L, N, 128)
            x = x.permute(1, 0, 2)  # (N, L, 128)
        else:
            x = x.permute(1, 0, 2)  # (L, N, 128)
            x = self.encoder(x)  # (L, N, 128)
            x = x.permute(1, 0, 2)  # (N, L, 128)

        # Global average pooling over the sequence length
        x = x.mean(dim=1)  # (N, 128)
        x = self.classifier(x)  # (N, 1)
        return x.squeeze(-1)  # (N)

def make_model(input_shape, *, use_mask=False):
    return BinaryClassifier(input_shape, use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20
BATCH_SIZE = 128

class MyDataset(Dataset):
    def __init__(self, X, y, preprocessor):
        self.X = X
        self.y = y
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        X, mask = self.preprocessor.transform(X.unsqueeze(0))
        return (X.squeeze(0), mask.squeeze(0)), y

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    val_auc = []

    best_val_auc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        train_preds = []
        train_labels = []
        for (X, mask), y in train_loader:
            X, mask, y = X.to(device), mask.to(device), y.to(device).float()
            optimizer.zero_grad()
            outputs = model(X, mask)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        epoch_train_loss /= len(train_loader)
        train_loss.append(epoch_train_loss)
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        train_auc = roc_auc_score(train_labels, train_preds)
        train_acc.append(np.mean((train_preds > 0.5) == train_labels))

        model.eval()
        epoch_val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for (X, mask), y in val_loader:
                X, mask, y = X.to(device), mask.to(device), y.to(device).float()
                outputs = model(X, mask)
                loss = criterion(outputs, y)
                epoch_val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        epoch_val_loss /= len(val_loader)
        val_loss.append(epoch_val_loss)
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        epoch_val_auc = roc_auc_score(val_labels, val_preds)
        val_auc.append(epoch_val_auc)
        val_acc.append(np.mean((val_preds > 0.5) == val_labels))

        scheduler.step(epoch_val_auc)

        print(f'Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {epoch_val_auc:.4f}')

        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
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

