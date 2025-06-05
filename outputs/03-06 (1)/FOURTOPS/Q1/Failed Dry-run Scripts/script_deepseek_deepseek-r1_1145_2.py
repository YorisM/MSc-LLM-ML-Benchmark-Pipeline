
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

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.kinematic_mean = None
        self.kinematic_std = None

    def fit(self, X, y=None):
        # Compute global feature statistics
        global_features = X[:, :2]
        self.global_mean = global_features.mean(dim=0)
        self.global_std = global_features.std(dim=0)

        # Compute kinematic statistics for valid objects
        objects = X[:, 2:].view(-1, 5)
        mask = objects[:, 0] != 0
        valid_objects = objects[mask][:, 1:5]  # Exclude obj_type column
        self.kinematic_mean = valid_objects.mean(dim=0)
        self.kinematic_std = valid_objects.std(dim=0)
        return self

    def transform(self, X):
        # Process global features
        global_features = (X[:, :2] - self.global_mean) / self.global_std

        # Process object features
        objects = X[:, 2:].view(-1, 18, 5)
        obj_types = objects[:, :, 0]
        kinematic = objects[:, :, 1:5]

        # Normalize kinematic features and apply padding mask
        kinematic_norm = (kinematic - self.kinematic_mean) / self.kinematic_std
        mask = (obj_types != 0).unsqueeze(-1).float()
        kinematic_norm = kinematic_norm * mask

        # Combine features and add global context
        processed_objects = torch.cat([obj_types.unsqueeze(-1), kinematic_norm], dim=-1)
        global_tiled = global_features.unsqueeze(1).expand(-1, 18, -1)
        full_features = torch.cat([processed_objects, global_tiled], dim=-1)

        return (full_features, obj_types != 0)

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_shape, *, use_mask=False):
    class ParticleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 16)  # Embed object types
            self.pos_encoder = nn.Parameter(torch.randn(1, 18, 32))  # Learnable positional encoding

            encoder_layers = TransformerEncoderLayer(
                d_model=48,  # 16 (embed) + 4 (kin) + 2 (global) * 2 (pos)
                nhead=8,
                dim_feedforward=256,
                dropout=0.1
            )
            self.transformer = TransformerEncoder(encoder_layers, 3)

            self.classifier = nn.Sequential(
                nn.Linear(48*18, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )

        def forward(self, x, mask=None):
            x_seq, padding_mask = x
            obj_types = x_seq[:, :, 0].long()
            embedded = self.embedding(obj_types)
            features = torch.cat([embedded, x_seq[:, :, 1:]], dim=-1)
            features = features + self.pos_encoder

            if mask is not None:
                padding_mask = ~padding_mask  # Invert for PyTorch's mask convention
                features = features.permute(1, 0, 2)
                output = self.transformer(features, src_key_padding_mask=padding_mask)
                output = output.permute(1, 0, 2)
            else:
                output = self.transformer(features.permute(1, 0, 2)).permute(1, 0, 2)

            return self.classifier(output.flatten(start_dim=1)).squeeze()

    return ParticleTransformer()

EPOCHS = 100

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    best_auc = 0
    best_weights = None
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = (d.to(device) for d in data) if isinstance(data, tuple) else data.to(device)
            labels = target.float().to(device)

            output = model(*inputs) if isinstance(data, tuple) else model(inputs)
            loss = criterion(output, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # Validation phase
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for data, target in val_loader:
                inputs = (d.to(device) for d in data) if isinstance(data, tuple) else data.to(device)
                labels = target.float().to(device)
                output = model(*inputs) if isinstance(data, tuple) else model(inputs)
                val_preds.append(torch.sigmoid(output).cpu())
                val_true.append(labels.cpu())

        val_preds = torch.cat(val_preds)
        val_true = torch.cat(val_true)
        auc = roc_auc_score(val_true.numpy(), val_preds.numpy())
        scheduler.step(auc)

        if auc > best_auc:
            best_auc = auc
            best_weights = model.state_dict().copy()
            patience = 0
        else:
            patience += 1

        if patience >= 7:
            break

    model.load_state_dict(best_weights)
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

