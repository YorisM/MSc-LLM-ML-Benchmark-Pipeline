
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
from torch import nn
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.per_object_features = 5
        self.max_objects = 18

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Reshape X to (N, max_objects, per_object_features) 
        # where N is the batch size, max_objects is 18, and per_object_features is 5
        X = X.view(-1, self.max_objects, self.per_object_features)  # (N, 18, 5)

        # Extract obj_id, E, p_T, eta, phi
        obj_id = X[:, :, 0].long()  # (N, 18)
        E = X[:, :, 1]  # (N, 18)
        p_T = X[:, :, 2]  # (N, 18)
        eta = X[:, :, 3]  # (N, 18)
        phi = X[:, :, 4]  # (N, 18)

        # Create a mask for valid objects (obj_id != 0)
        mask = obj_id != 0  # (N, 18)

        # Stack the features together
        features = torch.stack([E, p_T, eta, phi], dim=-1)  # (N, 18, 4)

        # Return the features and mask
        return features, mask

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], *, use_mask: bool):
        super().__init__()
        self.use_mask = use_mask
        self.embedding = nn.Linear(input_shape[-1], 128)  # (4) -> 128
        self.encoder = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=256, dropout=0.1)
        self.encoder = nn.TransformerEncoder(self.encoder, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data: torch.Tensor, mask: torch.Tensor | None = None):
        # data: (N, 18, 4)
        # mask: (N, 18)

        # Embed the features
        embedded = torch.relu(self.embedding(data))  # (N, 18, 128)

        # Create a mask for the TransformerEncoder
        if mask is not None:
            mask = ~mask  # (N, 18)
            mask = mask.unsqueeze(1).repeat(1, 18, 1)  # (N, 18, 18)
            mask = mask.masked_fill(mask, float('-inf'))  # (N, 18, 18)

        # Encode the sequence
        encoded = self.encoder(embedded.transpose(0, 1), src_key_padding_mask=mask[:, 0, :])  # (18, N, 128)
        encoded = encoded.mean(dim=0)  # (N, 128)

        # Classify
        output = self.classifier(encoded)  # (N, 1)

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

    train_loss = []
    val_loss = []
    train_auc = []
    val_auc = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        predictions = []
        labels = []

        for batch in train_loader:
            if isinstance(batch[0], (tuple, list)):
                data, mask = batch[0]
                data = data.to(device)
                mask = mask.to(device)
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
            predictions.extend(torch.sigmoid(output).detach().cpu().numpy())
            labels.extend(label.detach().cpu().numpy())

        train_loss.append(total_loss / len(train_loader))
        train_auc.append(roc_auc_score(labels, predictions))

        model.eval()
        total_loss = 0
        predictions = []
        labels = []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch[0], (tuple, list)):
                    data, mask = batch[0]
                    data = data.to(device)
                    mask = mask.to(device)
                    label = batch[1].to(device)
                    output = model(data, mask)
                else:
                    data = batch[0].to(device)
                    label = batch[1].to(device)
                    output = model(data)

                loss = criterion(output, label.float())
                total_loss += loss.item()
                predictions.extend(torch.sigmoid(output).detach().cpu().numpy())
                labels.extend(label.detach().cpu().numpy())

        val_loss.append(total_loss / len(val_loader))
        val_auc.append(roc_auc_score(labels, predictions))

        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.4f}, Train AUC: {train_auc[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, Val AUC: {val_auc[-1]:.4f}')

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

