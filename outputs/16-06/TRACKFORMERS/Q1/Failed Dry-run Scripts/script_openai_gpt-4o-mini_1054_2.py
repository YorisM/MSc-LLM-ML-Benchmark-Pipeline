
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: Python 3.12, PyTorch 2.6.0, NumPy 2.2.3, SciKit-Learn 1.6.1
import os, sys, pickle, gzip, json, torch, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data"
TAG      = "10_50_linear"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"REDVID_{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def split_X_y(evt):
    lay = evt["layer_id"].astype(np.float32)
    lay_norm = lay / lay.max()
    X = np.column_stack([evt["hit_r"],
                         evt["hit_theta"],
                         evt["hit_z"],
                         lay_norm])
    t_id = evt["track_id"].astype(np.int32)
    return (torch.from_numpy(X),
            torch.from_numpy(t_id))

class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, track_id = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, track_id)

def _ragged(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    # batch[i] = (hits_i, track_id_i)      ← shapes: (N_i, F), (N_i,)
    return batch

def make_loaders(batch_size=128, workers=0):
    tr = EventDataset(_load_events("train"), pre=None, train=True)
    va = EventDataset(_load_events("val"),   pre=None, train=False)

    train_ld = DataLoader(tr, batch_size=batch_size, shuffle=True,
                          collate_fn=_ragged, num_workers=workers)
    val_ld   = DataLoader(va, batch_size=batch_size, collate_fn=_ragged)
    return train_ld, val_ld

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# 0. ---------- IMPORTS ----------
from sklearn.preprocessing import StandardScaler

# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, events):
        all_hits = np.vstack([split_X_y(event)[0].numpy() for event in events])
        self.scaler.fit(all_hits)
        return self

    def transform(self, events):
        all_hits = np.vstack([split_X_y(event)[0].numpy() for event in events])
        scaled_hits = self.scaler.transform(all_hits)
        return torch.from_numpy(scaled_hits)

    def fit_transform(self, events):
        self.fit(events)
        return self.transform(events)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)  # Output: (N, 128)
        self.fc2 = nn.Linear(128, 64)            # Output: (N, 64)
        self.fc3 = nn.Linear(64, 32)             # Output: (N, 32)
        self.fc4 = nn.Linear(32, 1)              # Binary classification, Output: (N, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, batch):
        hits, _ = batch
        x = hits.view(-1, hits.size(-1))          # Flatten: (B, N, F) => (B*N, F)
        x = self.relu(self.fc1(x))                # (B*N, 128)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))                # (B*N, 64)
        x = self.dropout(x)
        x = self.relu(self.fc3(x))                # (B*N, 32)
        x = self.fc4(x)                           # (B*N, 1)
        return torch.sigmoid(x).view(hits.size(0), -1)  # Reshape back to (B, N)

def make_model(in_features):
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20   # Increasing epochs for better accuracy   
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for hits, track_ids in train_loader:
            hits = hits.to(device)
            track_ids = track_ids.to(device).float().view(-1, 1)  # (B, N) => (B*N, 1)

            optimizer.zero_grad()
            outputs = model((hits, track_ids))
            loss = criterion(outputs, track_ids)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == track_ids).sum().item()
            total += track_ids.numel()

        train_loss.append(epoch_loss / len(train_loader))
        train_acc.append(correct / total)

        model.eval()
        with torch.no_grad():
            val_epoch_loss = 0
            val_correct = 0
            val_total = 0

            for hits, track_ids in val_loader:
                hits = hits.to(device)
                track_ids = track_ids.to(device).float().view(-1, 1)

                val_outputs = model((hits, track_ids))
                val_loss_value = criterion(val_outputs, track_ids)

                val_epoch_loss += val_loss_value.item()
                val_preds = (val_outputs > 0.5).float()
                val_correct += (val_preds == track_ids).sum().item()
                val_total += track_ids.numel()

            val_loss.append(val_epoch_loss / len(val_loader))
            val_acc.append(val_correct / val_total)

    return model, train_loss, val_loss, train_acc, val_acc

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    pre = make_preprocessor().fit(raw_train)
    train_ds = EventDataset(raw_train, pre, train=True)
    val_ds   = EventDataset(raw_val , pre, train=False)
    train_ld = DataLoader(train_ds, batch_size=512,
                        shuffle=True, collate_fn=_ragged)
    val_ld   = DataLoader(val_ds,   batch_size=512,
                        collate_fn=_ragged)

    # 2. Build model
    in_features = train_ds[0][0].shape[-1]                   
    model = make_model(in_features)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_ld, val_ld, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single toy forward pass
    if dryrun:
        toy_event       = torch.zeros(10, in_features)
        toy_transformed = pre.transform(toy_event)
        toy_batch       = [toy_transformed]
        try:
            _ = trained_model(toy_batch)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    if not dryrun:
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
# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

