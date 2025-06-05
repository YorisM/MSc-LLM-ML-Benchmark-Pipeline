
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
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from collections import defaultdict

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scalers = defaultdict(StandardScaler)
        self.obj_cols = []

    def fit(self, X, y=None):
        obj_features = X[:, 2:].view(-1, 18, 5)    # (N, 18, 5)
        valid_objects = obj_features[:, :, 0] != 0  # Object IDs

        flat_features = obj_features[valid_objects].numpy()
        self.scalers['objects'] = StandardScaler().fit(flat_features[:, 1:])  # Exclude obj_id

        et_features = X[:, :2].numpy()
        self.scalers['et'] = StandardScaler().fit(et_features)
        return self

    def transform(self, X):
        et_tensor = X[:, :2].float()
        obj_tensor = X[:, 2:].view(-1, 18, 5)  # (B, 18, 5)

        et_scale = torch.from_numpy(self.scalers['et'].transform(et_tensor))
        obj_features = obj_tensor.clone()
        valid_mask = obj_features[:, :, 0] != 0

        scaled_features = self.scalers['objects'].transform(
            obj_features[valid_mask][:, 1:].numpy()
        )
        obj_features[valid_mask][:, 1:] = torch.from_numpy(scaled_features)

        first_obj_idx = torch.argmax(valid_mask.float(), dim=1)
        seq_mask = torch.arange(18).unsqueeze(0) < first_obj_idx.unsqueeze(1)
        mask = valid_mask & ~seq_mask

        combined = torch.cat([et_scale, obj_features.view(-1, 90)], dim=1)
        return (combined, mask.bool())

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, input_shape: tuple, *, use_mask: bool):
        super().__init__()
        self.use_mask = use_mask
        self.emb_dim = 64

        self.et_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.LayerNorm(32)
        )

        self.obj_encoder = nn.Sequential(
            nn.Linear(5, self.emb_dim),
            nn.GELU(),
            nn.LayerNorm(self.emb_dim)
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=4)

        self.classifier = nn.Sequential(
            nn.Linear(32 + self.emb_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )

    def forward(self, data, mask=None):
        et_data, obj_data = data[:, :2], data[:, 2:].view(-1, 18, 5)
        batch_size = et_data.size(0)

        et_feat = self.et_encoder(et_data)  # (B, 32)

        obj_features = self.obj_encoder(obj_data)  # (B, 18, 64)
        if self.use_mask and mask is not None:
            pad_mask = ~mask
            obj_feat = self.transformer(obj_features, src_key_padding_mask=pad_mask)
        else:
            obj_feat = self.transformer(obj_features)

        global_feat = obj_feat.max(dim=1)[0]  # (B, 64)
        combined = torch.cat([et_feat, global_feat], dim=1)  # (B, 96)
        return self.classifier(combined).squeeze(-1)

def make_model(input_shape, *, use_mask=False):
    return BinaryClassifier(input_shape, use_mask=use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 35

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_val = float('inf')
    patience, wait = 5, 0
    history = defaultdict(list)

    for epoch in range(epochs):
        model.train()
        train_loss, train_pred, train_true = [], [], []

        for batch in train_loader:
            (data, mask), labels = (batch[0], batch[1]) if isinstance(batch[0], tuple) else (batch[0], None)
            data, labels = data.to(device), labels.to(device).float()
            if mask is not None: mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(data, mask)
            loss = criterion(outputs, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss.append(loss.item())
            train_pred.append(torch.sigmoid(outputs).detach().cpu())
            train_true.append(labels.detach().cpu())

        scheduler.step()
        model.eval()
        val_loss, val_pred, val_true = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                (data, mask), labels = (batch[0], batch[1]) if isinstance(batch[0], tuple) else (batch[0], None)
                data, labels = data.to(device), labels.to(device).float()
                if mask is not None: mask = mask.to(device)

                outputs = model(data, mask)
                loss = criterion(outputs, labels)
                val_loss.append(loss.item())
                val_pred.append(torch.sigmoid(outputs).detach().cpu())
                val_true.append(labels.detach().cpu())

        train_loss = np.mean(train_loss)
        val_loss = np.mean(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        train_pred = torch.cat(train_pred).numpy()
        train_true = torch.cat(train_true).numpy()
        val_pred = torch.cat(val_pred).numpy()
        val_true = torch.cat(val_true).numpy()

        train_acc = ((train_pred > 0.5) == train_true).mean()
        val_acc = ((val_pred > 0.5) == val_true).mean()
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    return (
        model,
        history['train_loss'],
        history['val_loss'],
        history['train_acc'],
        history['val_acc']
    )

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

