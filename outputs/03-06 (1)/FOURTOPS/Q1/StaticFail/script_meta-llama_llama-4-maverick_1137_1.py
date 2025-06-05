
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

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # Define and initialize any stateful components here
        self.max_objects = 18  # Maximum number of objects encoded
        self.per_object_slice_size = 5  # Size of each object's slice

    def fit(self, X, y=None):
        # Extract statistics or fit transformers
        return self

    def transform(self, X):
        # Apply preprocessing logic, return torch.Tensor
        # Reshape X to separate objects and their features
        X = X.view(-1, self.max_objects, self.per_object_slice_size)

        # Extract relevant features: E, p_T, eta, phi for each object
        obj_features = X[:, :, 1:]  # Shape: (batch_size, max_objects, 4)

        # Mask for valid objects (obj_n != 0)
        mask = X[:, :, 0] != 0  # Shape: (batch_size, max_objects)

        # Return the features and the mask
        return obj_features, mask

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class ParticleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, dropout):
        super(ParticleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask):
        # x shape: (batch_size, max_objects, 4)
        # mask shape: (batch_size, max_objects)
        x = self.embedding(x)  # Shape: (batch_size, max_objects, hidden_dim)
        x = x.permute(1, 0, 2)  # Shape: (max_objects, batch_size, hidden_dim)
        mask = mask.permute(1, 0)  # Shape: (max_objects, batch_size)
        x = self.transformer_encoder(x, src_key_padding_mask=~mask)  # Shape: (max_objects, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)  # Shape: (batch_size, max_objects, hidden_dim)
        x = x[:, 0, :]  # Take the first token (or any aggregation method)  # Shape: (batch_size, hidden_dim)
        x = self.fc(x)  # Shape: (batch_size, output_dim)
        return x

def make_model(input_shape, *, use_mask=False):
    # Define a binary-classifier network
    input_dim = 4  # E, p_T, eta, phi
    hidden_dim = 128
    output_dim = 2  # Binary classification
    n_heads = 8
    dropout = 0.1

    model = ParticleTransformer(input_dim, hidden_dim, output_dim, n_heads, dropout)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10
BATCH_SIZE = 128

class ParticleDataset(Dataset):
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    train_auc = []
    val_auc = []

    best_val_auc = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        train_preds = []
        train_labels = []

        for (data, mask), labels in train_loader:
            data, mask, labels = data.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            train_preds.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        epoch_train_loss /= len(train_loader)
        train_loss.append(epoch_train_loss)
        train_acc.append(np.mean(np.argmax(torch.softmax(torch.tensor(train_preds), dim=0).numpy()) == np.array(train_labels)))
        train_auc.append(roc_auc_score(train_labels, train_preds))

        model.eval()
        epoch_val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for (data, mask), labels in val_loader:
                data, mask, labels = data.to(device), mask.to(device), labels.to(device)
                outputs = model(data, mask)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                val_preds.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())

        epoch_val_loss /= len(val_loader)
        val_loss.append(epoch_val_loss)
        val_acc.append(np.mean(np.argmax(torch.softmax(torch.tensor(val_preds), dim=0).numpy()) == np.array(val_labels)))
        val_auc.append(roc_auc_score(val_labels, val_preds))

        print(f'Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train AUC: {train_auc[-1]:.4f}, Val AUC: {val_auc[-1]:.4f}')

        if val_auc[-1] > best_val_auc:
            best_val_auc = val_auc[-1]
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step()

    model.load_state_dict(torch.load('best_model.pth'))
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

