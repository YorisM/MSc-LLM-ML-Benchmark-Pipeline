
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.llm_io import normalise_batch, assert_label_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data/train"
TAG      = "REDVID_10-50_linear_frac0.05"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def _split_X_y(evt):
    X = np.column_stack((evt["hit_r"].astype(np.float32),
                        evt["hit_theta"].astype(np.float32),
                        evt["hit_z"].astype(np.float32),
                        evt["layer_id"].astype(np.float32)))
    y = evt["track_id"].astype(np.int32)
    return (torch.from_numpy(X),torch.from_numpy(y))

def _make_dataset(events, pre, *, train: bool):
    custom = globals().get("make_dataset", None)
    if callable(custom):
        ds = custom(events, pre, train=train)
        if ds is not None:
            return ds
    return EventDataset(events, pre, train=train)

def make_loaders(raw_train, raw_val, pre, *, batch=512,
                 collate_fn=None, loader_cls=None, workers=0):
    train_ds  = _make_dataset(raw_train, pre, train=True)
    val_ds    = _make_dataset(raw_val,  pre, train=False)

    if loader_cls is None:
        loader_cls = DataLoader

    pin = (device.type == "cuda")
    train_ld = loader_cls(train_ds, batch_size=batch, shuffle=True,
                        num_workers=workers, collate_fn=collate_fn,
                        pin_memory=pin, persistent_workers=(workers > 0))
    val_ld   = loader_cls(val_ds,   batch_size=batch, shuffle=False,
                        num_workers=workers, collate_fn=collate_fn,
                        pin_memory=pin, persistent_workers=(workers > 0))
    return train_ld, val_ld
    
class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, track_id = _split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, track_id)

def _ragged(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    # batch[i] = (hits_i, track_id_i)      <- shapes: (N_i, F), (N_i)
    return batch

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# 0. ---------- IMPORTS ----------
from typing import List, Tuple
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# 1.2 ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def _raw_reshape(self, data):
        return data

    def make_loader_cfg(self):
        return {
            "loader_class": "torch.utils.data.DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True
        }

    def fit(self, data):
        all_features = []
        for evt in data:
            X, _ = _split_X_y(evt)
            all_features.append(X.numpy())
        self.scaler.fit(np.vstack(all_features))
        return self

    def transform(self, data):
        if isinstance(data, torch.Tensor):
            data_np = data.numpy()
        else:
            data_np = data
        scaled = self.scaler.transform(data_np)
        return torch.from_numpy(scaled).float()

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        in_features = example_batch_x[0].size(1) if isinstance(example_batch_x, list) else example_batch_x.size(1)

        self.conv1 = GATConv(in_features, 64, heads=4)
        self.conv2 = GATConv(64*4, 64, heads=4)
        self.conv3 = GATConv(64*4, 64, heads=4)

        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def build_graph(self, x):
        # Simple fully-connected graph for demonstration
        edge_index = []
        for i in range(x.size(0)):
            for j in range(i+1, min(i+5, x.size(0))):
                edge_index.append([i,j])
                edge_index.append([j,i])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(x.device)

    def forward(self, batch_x):
        if isinstance(batch_x, list):
            # Process each event separately
            all_preds = []
            for x in batch_x:
                x = x.to(device)
                edge_index = self.build_graph(x)

                x = F.elu(self.conv1(x, edge_index))
                x = F.elu(self.conv2(x, edge_index))
                x = F.elu(self.conv3(x, edge_index))

                pred = self.mlp(x).squeeze()
                all_preds.append(pred)
            return all_preds
        else:
            # Handle single tensor case
            x = batch_x.to(device)
            edge_index = self.build_graph(x)

            x = F.elu(self.conv1(x, edge_index))
            x = F.elu(self.conv2(x, edge_index))
            x = F.elu(self.conv3(x, edge_index))

            return self.mlp(x).squeeze()

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            optimizer.zero_grad()

            outputs = model(view.batch_x)
            if isinstance(outputs, list):
                loss = 0
                for out, y in zip(outputs, view.batch_y):
                    loss += criterion(out, y.to(device))
                loss /= len(outputs)
            else:
                loss = criterion(outputs, view.batch_y.to(device))

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            # Calculate accuracy
            if isinstance(outputs, list):
                for out, y in zip(outputs, view.batch_y):
                    _, predicted = torch.max(out.data, 0)
                    correct_train += (predicted == y.to(device)).sum().item()
                    total_train += y.size(0)
            else:
                _, predicted = torch.max(outputs.data, 1)
                correct_train += (predicted == view.batch_y.to(device)).sum().item()
                total_train += view.batch_y.size(0)

        train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / total_train if total_train > 0 else 0
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                outputs = model(view.batch_x)

                if isinstance(outputs, list):
                    batch_loss = 0
                    for out, y in zip(outputs, view.batch_y):
                        batch_loss += criterion(out, y.to(device))
                    batch_loss /= len(outputs)

                    for out, y in zip(outputs, view.batch_y):
                        _, predicted = torch.max(out.data, 0)
                        correct_val += (predicted == y.to(device)).sum().item()
                        total_val += y.size(0)
                else:
                    batch_loss = criterion(outputs, view.batch_y.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    correct_val += (predicted == view.batch_y.to(device)).sum().item()
                    total_val += view.batch_y.size(0)

                val_loss += batch_loss.item()

        val_loss /= len(val_loader)
        val_acc = correct_val / total_val if total_val > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

        if epoch > 10 and val_loss > best_val_loss * 1.1:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model)
    return model, train_losses, val_losses, train_accs, val_accs

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

def _import_dotted(path: str):
    mod, name = path.rsplit(".", 1)
    module = importlib.import_module(mod)
    return getattr(module, name)

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

    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None

    train_loader, val_loader = make_loaders(raw_train, raw_val, pre,
                                            batch = cfg.get("batch_size", 128),
                                            collate_fn = _ragged,
                                            loader_cls = loader_cls,
                                            workers    = cfg.get("num_workers", 0))

    # 2. Build model
    first_batch = next(iter(train_loader))
    view        = normalise_batch(first_batch, device=device)
    model       = make_model(view.batch_x).to(device)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single reduced forward pass
    if dryrun:
        try:
            batch = first_batch
            view  = normalise_batch(batch, device=device)
            with torch.no_grad():
                out = trained_model(view.batch_x)
            assert_label_output(view.batch_x, out) # check whether the LLM output labels
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

