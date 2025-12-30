
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 1.2 ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_scalers = {}

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
        for event in data:
            X, _ = _split_X_y(event)
            all_features.append(X.numpy())
        all_features = np.vstack(all_features)
        self.scaler.fit(all_features[:, :3])  # Only scale r, theta, z

        # Fit per-layer scalers for layer_id
        unique_layers = np.unique(all_features[:, 3])
        for layer in unique_layers:
            mask = all_features[:, 3] == layer
            if mask.sum() > 0:
                self.layer_scalers[int(layer)] = StandardScaler()
                self.layer_scalers[int(layer)].fit(all_features[mask, :3])
        return self

    def transform(self, data):
        if isinstance(data, torch.Tensor):
            data_np = data.numpy()
        else:
            data_np = data

        # Scale global features
        scaled_global = self.scaler.transform(data_np[:, :3])

        # Scale per-layer features
        scaled_features = np.zeros_like(data_np)
        scaled_features[:, :3] = scaled_global
        scaled_features[:, 3] = data_np[:, 3]

        # Apply per-layer scaling
        unique_layers = np.unique(data_np[:, 3])
        for layer in unique_layers:
            mask = data_np[:, 3] == layer
            if mask.sum() > 0 and int(layer) in self.layer_scalers:
                scaled_features[mask, :3] = self.layer_scalers[int(layer)].transform(data_np[mask, :3])

        # Add engineered features
        scaled_features = np.column_stack((
            scaled_features,
            np.sqrt(scaled_features[:, 0]**2 + scaled_features[:, 2]**2),  # cylindrical radius
            np.arctan2(scaled_features[:, 0], scaled_features[:, 2]),      # phi angle
        ))

        return torch.from_numpy(scaled_features).float()

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__()
        self.conv = GATConv(in_channels, out_channels, heads=heads)
        self.lin = nn.Linear(heads * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.elu(self.conv(x, edge_index))
        return self.lin(x)

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Extract feature dimension from example batch
        sample_features = example_batch_x[0]
        in_features = sample_features.shape[1] + 2  # +2 for engineered features

        # Graph network parameters
        hidden_dim = 128
        self.embedding = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Graph attention layers
        self.gat1 = GATLayer(hidden_dim, hidden_dim)
        self.gat2 = GATLayer(hidden_dim, hidden_dim)

        # Output layers
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Binary classification per hit
        )

    def build_edges(self, x):
        # Simple radius-based edge construction
        r = x[:, 0]
        theta = x[:, 1]
        z = x[:, 2]

        # Calculate distances in cylindrical coordinates
        dr = torch.cdist(r.unsqueeze(1), r.unsqueeze(1))
        dz = torch.cdist(z.unsqueeze(1), z.unsqueeze(1))
        dtheta = torch.cdist(theta.unsqueeze(1), theta.unsqueeze(1))

        # Threshold-based edges
        mask = (dr < 0.1) & (dz < 0.2) & (dtheta < 0.3)
        edge_index = torch.nonzero(mask).t()
        return edge_index

    def forward(self, batch_x):
        if isinstance(batch_x, list):
            # Process each event separately
            outputs = []
            for event in batch_x:
                # Build edges for this event
                edge_index = self.build_edges(event)

                # Embed features
                x = self.embedding(event)

                # Apply GAT layers
                x = self.gat1(x, edge_index)
                x = self.gat2(x, edge_index)

                # Classify hits
                out = self.output(x).squeeze(-1)
                outputs.append(out)

            return outputs
        else:
            # Assume it's a single event tensor
            edge_index = self.build_edges(batch_x)
            x = self.embedding(batch_x)
            x = self.gat1(x, edge_index)
            x = self.gat2(x, edge_index)
            return self.output(x).squeeze(-1)

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model, train_loader, val_loader, epochs):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=False)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            batch_x, batch_y = view.batch_x, view.batch_y

            optimizer.zero_grad()

            outputs = model(batch_x)
            if isinstance(outputs, list):
                loss = 0
                correct = 0
                total = 0
                for out, y in zip(outputs, batch_y):
                    if len(out) == 0:
                        continue
                    loss += criterion(out, y.to(device))
                    pred = out.argmax(dim=-1)
                    correct += (pred == y.to(device)).sum().item()
                    total += len(y)
                loss /= len(outputs)
                epoch_train_loss += loss.item()
                correct_train += correct
                total_train += total
            else:
                loss = criterion(outputs, batch_y.to(device))
                epoch_train_loss += loss.item()
                pred = outputs.argmax(dim=-1)
                correct_train += (pred == batch_y.to(device)).sum().item()
                total_train += len(batch_y)

            loss.backward()
            optimizer.step()

        train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / total_train if total_train > 0 else 0
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                batch_x, batch_y = view.batch_x, view.batch_y

                outputs = model(batch_x)
                if isinstance(outputs, list):
                    loss = 0
                    correct = 0
                    total = 0
                    for out, y in zip(outputs, batch_y):
                        if len(out) == 0:
                            continue
                        loss += criterion(out, y.to(device))
                        pred = out.argmax(dim=-1)
                        correct += (pred == y.to(device)).sum().item()
                        total += len(y)
                    loss /= len(outputs)
                    epoch_val_loss += loss.item()
                    correct_val += correct
                    total_val += total
                else:
                    loss = criterion(outputs, batch_y.to(device))
                    epoch_val_loss += loss.item()
                    pred = outputs.argmax(dim=-1)
                    correct_val += (pred == batch_y.to(device)).sum().item()
                    total_val += len(batch_y)

        val_loss = epoch_val_loss / len(val_loader)
        val_acc = correct_val / total_val if total_val > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

        if epoch > 10 and val_acc < best_val_acc * 0.95:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
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

