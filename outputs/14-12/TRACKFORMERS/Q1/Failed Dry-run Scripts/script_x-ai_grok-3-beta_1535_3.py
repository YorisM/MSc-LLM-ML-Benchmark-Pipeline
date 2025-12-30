
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
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# 1.1 -------- OPTIONAL: CUSTOM DATASET / DATA-CLASS --------
# Not implemented, using default EventDataset

# 1.2 ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def _raw_reshape(self, data):
        return data

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
        all_data = []
        for evt in data:
            X, _ = _split_X_y(evt)
            all_data.append(X.numpy())
        all_data = np.concatenate(all_data, axis=0)
        self.scaler.fit(all_data)
        return self

    def transform(self, data):
        # Apply scaling to the data
        data_np = data.numpy()
        scaled_data = self.scaler.transform(data_np)
        return torch.from_numpy(scaled_data).float()  # Shape: [N_hits, 4]

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Infer input features from example_batch_x
        in_features = example_batch_x[0].shape[-1]  # 4 features (r, theta, z, layer_id)
        self.hidden_dim = 64
        self.conv1 = GCNConv(in_features, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.lin1 = nn.Linear(self.hidden_dim, 128)
        self.lin2 = nn.Linear(128, 50)  # Assuming max 50 tracks per event for classification

    def _build_graph(self, x):
        # Simple nearest neighbor graph construction for each event
        # This is a placeholder; in practice, more sophisticated edge construction based on detector geometry could be used
        num_nodes = x.size(0)
        if num_nodes < 2:
            edge_index = torch.tensor([[], []], dtype=torch.long, device=x.device)
        else:
            # Connect each node to its nearest neighbors based on spatial proximity
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(5, num_nodes-1)).fit(x.cpu().numpy())
            distances, indices = nbrs.kneighbors(x.cpu().numpy())
            edge_index = []
            for i in range(num_nodes):
                for j in indices[i]:
                    if i != j:
                        edge_index.append([i, j])
            if len(edge_index) == 0:
                edge_index = torch.tensor([[], []], dtype=torch.long, device=x.device)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=x.device).t().contiguous()
        return edge_index

    def forward(self, batch_x):
        batch_size = len(batch_x)
        preds = []

        for i in range(batch_size):
            x = batch_x[i]  # Shape: [N_hits, 4]
            edge_index = self._build_graph(x)  # Shape: [2, E]

            # Graph Convolution layers
            x = self.conv1(x, edge_index)  # Shape: [N_hits, hidden_dim]
            x = F.relu(x)
            x = self.conv2(x, edge_index)  # Shape: [N_hits, hidden_dim]
            x = F.relu(x)

            # MLP for classification
            x = self.lin1(x)  # Shape: [N_hits, 128]
            x = F.relu(x)
            x = self.lin2(x)  # Shape: [N_hits, 50]

            # Predict track ID (0 to 49, -1 for noise if needed)
            pred = torch.argmax(x, dim=-1)  # Shape: [N_hits,]
            preds.append(pred)

        return preds

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore noise labels if any
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_val_acc = 0.0
    patience = 5
    early_stop_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        correct, total = 0, 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            batch_x, batch_y = view.batch_x, view.batch_y

            optimizer.zero_grad()
            preds = model(batch_x)  # List of [N_hits,]

            loss = 0.0
            for i in range(len(preds)):
                if batch_y[i].max() >= 50:  # Adjust output dimension if needed
                    batch_y[i] = torch.clamp(batch_y[i], 0, 49)
                loss += criterion(preds[i].unsqueeze(0).float(), batch_y[i])

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Accuracy (simplified, per hit)
            for i in range(len(preds)):
                correct += (preds[i] == batch_y[i]).sum().item()
                total += batch_y[i].size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total if total > 0 else 0.0
        train_loss.append(avg_train_loss)
        train_acc.append(train_accuracy)

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct_val, total_val = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                batch_x, batch_y = view.batch_x, view.batch_y

                preds = model(batch_x)

                loss = 0.0
                for i in range(len(preds)):
                    if batch_y[i].max() >= 50:
                        batch_y[i] = torch.clamp(batch_y[i], 0, 49)
                    loss += criterion(preds[i].unsqueeze(0).float(), batch_y[i])

                total_val_loss += loss.item()

                for i in range(len(preds)):
                    correct_val += (preds[i] == batch_y[i]).sum().item()
                    total_val += batch_y[i].size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val if total_val > 0 else 0.0
        val_loss.append(avg_val_loss)
        val_acc.append(val_accuracy)

        scheduler.step(val_accuracy)

        # Early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

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

