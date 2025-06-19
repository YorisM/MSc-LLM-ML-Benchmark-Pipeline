
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
from torch import nn, optim
import torch.nn.functional as F

# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, events):
        # Fit the scaler to the features
        X = np.vstack([np.column_stack([evt["hit_r"],
                                         evt["hit_theta"],
                                         evt["hit_z"],
                                         evt["layer_id"]]) for evt in events])
        self.scaler.fit(X)

    def transform(self, events):
        # Transform the features using the fitted scaler
        X = np.vstack([np.column_stack([evt["hit_r"],
                                         evt["hit_theta"],
                                         evt["hit_z"],
                                         evt["layer_id"]]) for evt in events])
        X_scaled = self.scaler.transform(X)
        return torch.from_numpy(X_scaled).float()

    def fit_transform(self, events):
        self.fit(events)
        return self.transform(events)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 128)  # (N_hits, 128)
        self.layer2 = nn.Linear(128, 64)           # (N_hits, 64)
        self.layer3 = nn.Linear(64, 32)            # (N_hits, 32)
        self.output_layer = nn.Linear(32, 1)       # (N_hits, 1)

    def forward(self, batch):
        # batch : list[torch.Tensor]  or  list[tuple[Tensor, Tensor]]
        X = batch[0]  # Get the hits
        x = F.relu(self.layer1(X))                # (N_hits, 128)
        x = F.relu(self.layer2(x))                 # (N_hits, 64)
        x = F.relu(self.layer3(x))                 # (N_hits, 32)
        output = torch.sigmoid(self.output_layer(x)) # (N_hits, 1)
        return output

def make_model(in_features):
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20  # Increased epochs for better training
def train_model(model, train_loader, val_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for hits, track_ids in train_loader:
            optimizer.zero_grad()
            outputs = model(hits)
            loss = criterion(outputs, track_ids.type_as(outputs).view(-1, 1).float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds.view(-1) == track_ids).sum().item()  
            total += track_ids.size(0)

        train_loss.append(total_loss / len(train_loader))
        train_acc.append(correct / total)

        # Validation step
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for hits, track_ids in val_loader:
                outputs = model(hits)
                loss = criterion(outputs, track_ids.type_as(outputs).view(-1, 1).float())

                total_val_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct_val += (preds.view(-1) == track_ids).sum().item()
                total_val += track_ids.size(0)

        val_loss.append(total_val_loss / len(val_loader))
        val_acc.append(correct_val / total_val)

        # Early stopping criteria could be applied here 

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

