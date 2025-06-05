
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

import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MyPreprocessor:
    def __init__(self):
        self.eps = 1e-8
        self.kinetic_means = None
        self.kinetic_stds = None
        self.met_means = None
        self.met_stds = None

    def fit(self, X, y=None):
        # Process MET features
        met = X[:, [0, 1]]
        self.met_means = met.mean(dim=0)
        self.met_stds = met.std(dim=0) + self.eps

        # Process object features
        objects = X[:, 2:].view(-1, 18, 5)
        mask = objects[:, :, 0] != 0
        valid_objects = objects[mask][:, 1:]
        self.kinetic_means = valid_objects.mean(dim=0)
        self.kinetic_stds = valid_objects.std(dim=0) + self.eps
        return self

    def transform(self, X):
        # Normalize MET
        met = X[:, [0, 1]]
        met_norm = (met - self.met_means) / self.met_stds

        # Process objects
        objects = X[:, 2:].view(-1, 18, 5)
        obj_type = objects[:, :, 0]
        kinetics = objects[:, :, 1:]
        kinetics_norm = (kinetics - self.kinetic_means) / self.kinetic_stds

        # Combine features
        met_expanded = met_norm.unsqueeze(1).repeat(1, 18, 1)
        combined = torch.cat([
            obj_type.unsqueeze(-1),
            kinetics_norm,
            met_expanded
        ], dim=-1)

        mask = obj_type != 0
        return (combined, mask)

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], *, use_mask: bool):
        super().__init__()
        self.use_mask = use_mask
        self.embedding = nn.Embedding(1000, 16)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=16+4+2,  # obj_embed + kinetic + MET
                nhead=4,
                dim_feedforward=128,
                dropout=0.2,
                batch_first=True
            ),
            num_layers=4
        )
        self.classifier = nn.Sequential(
            nn.Linear(16+4+2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, data, mask=None):
        obj_type = data[:, :, 0].long()
        features = data[:, :, 1:]

        embedded = self.embedding(obj_type)
        combined = torch.cat([embedded, features], dim=-1)

        if mask is not None:
            combined = combined * mask.unsqueeze(-1).float()

        encoded = self.transformer(combined, src_key_padding_mask=~mask if mask is not None else None)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)

def make_model(input_shape, use_mask=False):
    return BinaryClassifier(input_shape, use_mask=use_mask)

EPOCHS = 50

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    best_auc = 0
    best_weights = None
    patience = 5
    no_improve = 0

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            data, labels = batch
            if isinstance(data, (list, tuple)):
                inputs, mask = data[0].to(device), data[1].to(device)
            else:
                inputs, mask = data.to(device), None
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss.append(epoch_loss/len(train_loader))
        train_acc.append(correct/total)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        v_loss = 0
        v_correct = 0
        v_total = 0

        with torch.no_grad():
            for batch in val_loader:
                data, labels = batch
                if isinstance(data, (list, tuple)):
                    inputs, mask = data[0].to(device), data[1].to(device)
                else:
                    inputs, mask = data.to(device), None
                labels = labels.float().to(device)

                outputs = model(inputs, mask)
                loss = criterion(outputs, labels)
                v_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

                val_preds.append(torch.sigmoid(outputs).cpu())
                val_labels.append(labels.cpu())

        val_loss.append(v_loss/len(val_loader))
        val_acc.append(v_correct/v_total)

        # Calculate AUC
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        auc = roc_auc_score(val_labels, val_preds)

        # Update scheduler and early stopping
        scheduler.step(auc)
        if auc > best_auc:
            best_auc = auc
            best_weights = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    model.load_state_dict(best_weights)
    return model, train_loss, val_loss, train_acc, val_acc

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

