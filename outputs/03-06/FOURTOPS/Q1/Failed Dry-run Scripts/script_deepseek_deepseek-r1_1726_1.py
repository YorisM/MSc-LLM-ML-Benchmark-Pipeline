
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
from sklearn.metrics import roc_auc_score
import copy

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.obj_E_mean = 0
        self.obj_E_std = 1
        self.obj_pT_mean = 0
        self.obj_pT_std = 1
        self.obj_eta_mean = 0
        self.obj_eta_std = 1
        self.obj_phi_mean = 0
        self.obj_phi_std = 1
        self.global_Etmean = 0
        self.global_Etstd = 1
        self.global_phimean = 0
        self.global_phistd = 1

    def fit(self, X, y=None):
        X_global = X[:, :2]
        X_objects = X[:, 2:].view(X.size(0), 18, 5)
        mask = X_objects[:, :, 0] != 0

        # Feature normalization params
        nonpad_objects = X_objects[mask][:, 1:5]
        self.obj_E_mean = nonpad_objects[:, 0].mean().item()
        self.obj_E_std = nonpad_objects[:, 0].std().item()
        self.obj_pT_mean = nonpad_objects[:, 1].mean().item()
        self.obj_pT_std = nonpad_objects[:, 1].std().item()
        self.obj_eta_mean = nonpad_objects[:, 2].mean().item()
        self.obj_eta_std = nonpad_objects[:, 2].std().item()
        self.obj_phi_mean = nonpad_objects[:, 3].mean().item()
        self.obj_phi_std = nonpad_objects[:, 3].std().item()

        # Global features
        self.global_Etmean = X_global[:, 0].mean().item()
        self.global_Etstd = X_global[:, 0].std().item()
        self.global_phimean = X_global[:, 1].mean().item()
        self.global_phistd = X_global[:, 1].std().item()

        return self

    def transform(self, X):
        X_global = X[:, :2]
        X_objects = X[:, 2:].view(X.size(0), 18, 5)
        mask = (X_objects[:, :, 0] != 0)

        # Normalize object features
        X_objects[:, :, 1] = (X_objects[:, :, 1] - self.obj_E_mean) / self.obj_E_std
        X_objects[:, :, 2] = (X_objects[:, :, 2] - self.obj_pT_mean) / self.obj_pT_std
        X_objects[:, :, 3] = (X_objects[:, :, 3] - self.obj_eta_mean) / self.obj_eta_std
        X_objects[:, :, 4] = (X_objects[:, :, 4] - self.obj_phi_mean) / self.obj_phi_std

        # Normalize and augument global features
        norm_global = torch.empty_like(X_global)
        norm_global[:, 0] = (X_global[:, 0] - self.global_Etmean) / self.global_Etstd
        norm_global[:, 1] = (X_global[:, 1] - self.global_phimean) / self.global_phistd
        global_features = norm_global.unsqueeze(1).expand(-1, 18, 2)

        combined = torch.cat([X_objects, global_features], dim=2)
        return (combined.float(), mask)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, input_shape, *, use_mask):
        super().__init__()
        self.use_mask = use_mask
        self.encoder = nn.Linear(7, 64)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256, dropout=0.2, batch_first=True), 
            num_layers=4
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, data, mask=None):
        x = self.encoder(data)
        padding_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        if padding_mask is not None:
            x.masked_fill_(padding_mask.unsqueeze(-1), 0.0)
            x = x.sum(dim=1) / (~padding_mask).sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)

        return self.classifier(x).squeeze(-1)

def make_model(input_shape, *, use_mask=False):
    return BinaryClassifier(input_shape, use_mask=use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50
def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_auc = 0
    best_weights = None
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            data, labels = batch
            inputs, mask = [d.to(device) for d in data] if isinstance(data, (list, tuple)) else (data.to(device), None)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

        train_loss.append(epoch_loss / total)
        train_acc.append(correct / total)

        # Validation
        model.eval()
        val_preds = []
        val_true = []
        v_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                data, labels = batch
                inputs, mask = [d.to(device) for d in data] if isinstance(data, (list, tuple)) else (data.to(device), None)
                labels = labels.to(device).float()

                outputs = model(inputs, mask)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * labels.size(0)

                val_preds.append(torch.sigmoid(outputs).cpu())
                val_true.append(labels.cpu())

        val_loss.append(v_loss / len(val_loader.dataset))
        val_preds = torch.cat(val_preds)
        val_true = torch.cat(val_true)
        val_auc = roc_auc_score(val_true, val_preds)
        val_acc.append((val_preds.round().numpy() == val_true.numpy()).mean())
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 5:
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

