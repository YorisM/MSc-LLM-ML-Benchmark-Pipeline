
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.per_object_slice_size = 5
        self.max_objects = 18

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Reshape X into (N, L, F) where L is the max number of objects and F is the feature size per object
        # (N, 92) -> (N, 18, 5) because 92 = 18*5 + 2 (for E_T_miss and phi_Et_miss)
        N = X.shape[0]
        X_new = torch.zeros((N, self.max_objects, self.per_object_slice_size), dtype=X.dtype)
        for i in range(self.max_objects):
            start_idx = 2 + i * self.per_object_slice_size
            end_idx = start_idx + self.per_object_slice_size
            X_new[:, i, :] = X[:, start_idx:end_idx]

        # Extract E_T_miss and phi_Et_miss
        E_T_miss = X[:, 0].unsqueeze(1)  # (N,) -> (N,1)
        phi_Et_miss = X[:, 1].unsqueeze(1)  # (N,) -> (N,1)

        # Concatenate E_T_miss and phi_Et_miss with the reshaped object features
        # (N, 1) + (N, 1) + (N, 18, 5) -> (N, 20, 5) but we actually want (N, 18+2, 5)
        # So, we concatenate E_T_miss and phi_Et_miss as two additional "objects"
        X_new = torch.cat((E_T_miss.unsqueeze(1), phi_Et_miss.unsqueeze(1), X_new), dim=1)  # (N, 20, 5)

        # Create a mask for the valid objects (including E_T_miss and phi_Et_miss)
        mask = torch.ones((N, self.max_objects+2), dtype=torch.bool)  # (N, 20)

        # Since the data is zero-padded, we need to mask out the padded objects
        # The first two "objects" are always valid (E_T_miss and phi_Et_miss)
        # For the rest, we check if the object ID is non-zero
        obj_ids = X_new[:, 2:, 0]  # (N, 18)
        mask[:, 2:] = (obj_ids != 0)  # (N, 18)

        return (X_new, mask)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], *, use_mask: bool):
        super().__init__()
        self.use_mask = use_mask
        self.feature_size = input_shape[-1]  # 5
        self.hidden_size = 128
        self.n_heads = 8
        self.dropout = 0.1
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_heads, dropout=self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.embedding = nn.Linear(self.feature_size, self.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size//2, 1)
        )

    def forward(self, data: torch.Tensor, mask: torch.Tensor | None = None):
        # data: (N, L, F) = (N, 20, 5)
        # mask: (N, L) = (N, 20)
        x = self.embedding(data)  # (N, 20, 5) -> (N, 20, 128)
        if mask is not None:
            # Transformer expects mask to be (L, N) or (N, L) and bool or float
            # Our mask is (N, L) and bool, so it's fine
            x = x.masked_fill(~mask.unsqueeze(-1), 0)  # (N, 20, 128)
        x = self.transformer_encoder(x, src_key_padding_mask=~mask)  # (N, 20, 128)
        # Global average pooling
        x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)  # (N, 128)
        x = self.classifier(x)  # (N, 128) -> (N, 1)
        return x.squeeze(1)  # (N,)

def make_model(input_shape, *, use_mask=False):
    return BinaryClassifier(input_shape, use_mask=use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    train_loss = []
    val_loss = []
    train_auc = []
    val_auc = []

    best_val_auc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        predictions = []
        labels = []
        for batch in train_loader:
            if isinstance(batch[0], (tuple, list)):
                data, mask = batch[0]
                data, mask = data.to(device), mask.to(device)
                label = batch[1].to(device)
                output = model(data, mask)
            else:
                data = batch[0].to(device)
                label = batch[1].to(device)
                output = model(data)

            loss = criterion(output, label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predictions.append(torch.sigmoid(output).detach().cpu().numpy())
            labels.append(label.cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        epoch_loss = total_loss / len(train_loader)
        epoch_auc = roc_auc_score(labels, predictions)
        train_loss.append(epoch_loss)
        train_auc.append(epoch_auc)

        model.eval()
        val_predictions = []
        val_labels = []
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch[0], (tuple, list)):
                    data, mask = batch[0]
                    data, mask = data.to(device), mask.to(device)
                    label = batch[1].to(device)
                    output = model(data, mask)
                else:
                    data = batch[0].to(device)
                    label = batch[1].to(device)
                    output = model(data)
                loss = criterion(output, label.float())
                total_val_loss += loss.item()
                val_predictions.append(torch.sigmoid(output).cpu().numpy())
                val_labels.append(label.cpu().numpy())

        val_predictions = np.concatenate(val_predictions)
        val_labels = np.concatenate(val_labels)
        epoch_val_loss = total_val_loss / len(val_loader)
        epoch_val_auc = roc_auc_score(val_labels, val_predictions)
        val_loss.append(epoch_val_loss)
        val_auc.append(epoch_val_auc)

        scheduler.step(epoch_val_auc)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train AUC: {epoch_auc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val AUC: {epoch_val_auc:.4f}")

        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            torch.save(model.state_dict(), 'best_model.pth')

    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_loss, val_loss, train_auc, val_auc

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

