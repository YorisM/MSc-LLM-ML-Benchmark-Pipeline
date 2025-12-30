
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
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler
import numpy as np

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
        # Concatenate all features from all events for fitting the scaler
        all_features = []
        for evt in data:
            X, _ = _split_X_y(evt)
            all_features.append(X.numpy())
        all_features = np.concatenate(all_features, axis=0)
        self.scaler.fit(all_features)
        return self

    def transform(self, data):
        # Scale the features using the fitted scaler
        data_np = data.numpy()
        data_scaled = self.scaler.transform(data_np)
        return torch.from_numpy(data_scaled).float()  # Shape: [N_hits, 4]

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        self.in_features = example_batch_x[0].shape[1]  # Infer input dimension (4 features)
        self.hidden_dim = 64
        self.num_classes = 50  # Approximate maximum number of tracks per event

        # GCN layers for graph-based learning
        self.conv1 = GCNConv(self.in_features, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim)

        # Final classification layer
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, batch_x):
        # Handle ragged list of tensors (one per event)
        if isinstance(batch_x, list):
            graphs = []
            for x in batch_x:
                # Create a simple fully-connected graph for each event
                N = x.shape[0]
                edge_index = torch.ones(2, N * (N - 1) // 2, dtype=torch.long, device=x.device)
                idx = 0
                for i in range(N):
                    for j in range(i + 1, N):
                        edge_index[0, idx] = i
                        edge_index[1, idx] = j
                        idx += 1
                graph = Data(x=x, edge_index=edge_index)  # Shape: x=[N_hits, 4], edge_index=[2, N*(N-1)/2]
                graphs.append(graph)
            batch = Batch.from_data_list(graphs).to(device)
        else:
            batch = batch_x  # Already a batch (unlikely in this case)

        # GCN forward pass
        x = batch.x
        edge_index = batch.edge_index

        x = self.conv1(x, edge_index)  # Shape: [total_hits, hidden_dim]
        x = F.relu(x)
        x = self.conv2(x, edge_index)  # Shape: [total_hits, hidden_dim]
        x = F.relu(x)
        x = self.conv3(x, edge_index)  # Shape: [total_hits, hidden_dim]
        x = F.relu(x)

        # Classification per hit
        out = self.fc(x)  # Shape: [total_hits, num_classes]
        pred = torch.argmax(out, dim=-1)  # Shape: [total_hits]
        return pred.long()  # Return integer labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20   
def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore noise labels if any
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_acc = 0.0
    patience = 5
    trigger_times = 0

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):
        # Training loop
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            optimizer.zero_grad()
            out = model(view.batch_x)  # Shape: [total_hits]
            target = view.batch_y.view(-1).to(device)  # Shape: [total_hits]
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            tr_correct += (out == target).sum().item()
            tr_total += target.numel()

        tr_loss_avg = tr_loss / len(train_loader)
        tr_acc = tr_correct / tr_total if tr_total > 0 else 0.0

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                out = model(view.batch_x)  # Shape: [total_hits]
                target = view.batch_y.view(-1).to(device)  # Shape: [total_hits]
                loss = criterion(out, target)

                val_loss += loss.item()
                val_correct += (out == target).sum().item()
                val_total += target.numel()

        val_loss_avg = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        # Store metrics
        train_loss_list.append(tr_loss_avg)
        val_loss_list.append(val_loss_avg)
        train_acc_list.append(tr_acc)
        val_acc_list.append(val_acc)

        # Scheduler step based on validation loss
        scheduler.step(val_loss_avg)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

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

