
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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

# 1.1 -------- OPTIONAL: CUSTOM DATASET / DATA-CLASS --------
def make_dataset(events, pre, train: bool):
    return None  # Use default EventDataset

# 1.2 ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def _raw_reshape(self, data):
        return data  # Returns identity by default

    def make_loader_cfg(self):
        return {
            "loader_class": "torch.utils.data.DataLoader",
            "batch_size": 128,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True
        }

    def fit(self, data):
        # Concatenate all events' data for fitting the scaler
        all_hits = []
        for evt in data:
            hits = np.column_stack((evt["hit_r"].astype(np.float32),
                                    evt["hit_theta"].astype(np.float32),
                                    evt["hit_z"].astype(np.float32),
                                    evt["layer_id"].astype(np.float32)))
            all_hits.append(hits)
        all_hits = np.vstack(all_hits)
        self.scaler.fit(all_hits)
        return self

    def transform(self, data):
        # Apply scaling to the input data
        data_np = data.numpy()  # Shape: [N_hits, 4]
        data_scaled = self.scaler.transform(data_np)  # Shape: [N_hits, 4]
        return torch.from_numpy(data_scaled).float()  # Return as torch tensor

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Infer input features from example_batch_x
        in_features = example_batch_x[0].shape[-1]  # Should be 4 (r, theta, z, layer_id)

        # Graph Convolutional Layers for learning spatial relationships
        self.conv1 = GCNConv(in_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)

        # MLP for final track classification
        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 50)  # Output dimension set to maximum expected tracks per event
        )

    def forward(self, batch_x):
        # Handle ragged list of tensors (one per event)
        if isinstance(batch_x, list):
            batch_data = []
            for x in batch_x:
                # Build a simple fully connected graph for each event (or use proper geometry if available)
                num_hits = x.shape[0]
                edge_index = torch.combinations(torch.arange(num_hits), r=2).T.to(x.device)
                data = Data(x=x, edge_index=edge_index)
                batch_data.append(data)
            batch = Batch.from_data_list(batch_data).to(x.device)
        else:
            batch = batch_x  # Already in PyG Batch format

        # Graph Convolution passes
        x = batch.x
        edge_index = batch.edge_index

        x = self.conv1(x, edge_index)  # Shape: [N_hits_total, 64]
        x = torch.relu(x)
        x = self.conv2(x, edge_index)  # Shape: [N_hits_total, 128]
        x = torch.relu(x)
        x = self.conv3(x, edge_index)  # Shape: [N_hits_total, 64]
        x = torch.relu(x)

        # Global pooling per graph (event)
        x_pool = global_mean_pool(x, batch.batch)  # Shape: [N_events, 64]

        # Repeat pooled features for each hit in the batch to match dimensions
        x_expanded = x_pool[batch.batch]  # Shape: [N_hits_total, 64]

        # Final classification per hit
        logits = self.mlp(x_expanded)  # Shape: [N_hits_total, 50]
        preds = torch.argmax(logits, dim=-1)  # Shape: [N_hits_total,]
        return preds.long()  # Return integer labels for each hit

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20  # Adjusted for sufficient training iterations
def train_model(model, train_loader, val_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore noise labels if any

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    best_val_loss = float('inf')
    patience = 5
    no_improve = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            view = normalise_batch((batch_x, batch_y), device=device)
            outputs = model(view.batch_x)  # Shape: [N_hits_total,]
            loss = criterion(outputs, view.batch_y.long())
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            pred = outputs
            correct_train += (pred == view.batch_y).sum().item()
            total_train += view.batch_y.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = correct_train / total_train if total_train > 0 else 0

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                view = normalise_batch((batch_x, batch_y), device=device)
                outputs = model(view.batch_x)
                loss = criterion(outputs, view.batch_y.long())

                epoch_val_loss += loss.item()
                pred = outputs
                correct_val += (pred == view.batch_y).sum().item()
                total_val += view.batch_y.size(0)

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_acc = correct_val / total_val if total_val > 0 else 0

        # Log metrics
        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)
        train_acc.append(avg_train_acc)
        val_acc.append(avg_val_acc)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    return model, train_loss, val_loss, train_acc, val_acc

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

