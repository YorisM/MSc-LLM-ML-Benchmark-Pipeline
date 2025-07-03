
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: Python 3.12, PyTorch 2.6.0, Torch_Geometric 2.6.1, NumPy 2.2.3, SciPy v1.15.2, SciKit-Learn 1.6.1
import os, sys, pickle, torch, torch_geometric, gc, json, importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

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
    X_train = pd.read_csv(DATASET["X_train"], dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv(DATASET["Y_train"], dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv(DATASET["X_val"], dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv(DATASET['Y_val'], dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class PairDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
    
        if isinstance(self.x, (tuple, list)) and all(torch.is_tensor(t) for t in self.x):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])

def _make_dataset(x, y):
    custom = globals().get("make_dataset", None)
    if callable(custom):
        ds = custom(x, y)
        if ds is not None:
            return ds
    return PairDataset(x, y)

def make_loaders(X_train, Y_train, X_val, Y_val, *, batch=512, collate_fn=None, loader_cls=None):
    train_ds = _make_dataset(X_train, Y_train)
    val_ds   = _make_dataset(X_val , Y_val)

    if loader_cls is None: 
        loader_cls = DataLoader

    train_ld = loader_cls(train_ds, batch_size=batch, shuffle=True, num_workers=0, 
                        collate_fn=collate_fn)
    val_ld   = loader_cls(val_ds, batch_size=batch, shuffle=False, num_workers=0,
                        collate_fn=collate_fn)

    return train_ld, val_ld

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

# 0. ---------- IMPORTS ----------
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data, Batch
from math import sqrt
import itertools

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.max_objects = 18
        self.obj_feat_size = 5
        self.global_feat_size = 2

    def _raw_reshape(self, X):
        batch_size = X.shape[0]
        # Reshape to [batch_size, max_objects, obj_feat_size]
        obj_features = X[:, 2:].reshape(
            batch_size, self.max_objects, self.obj_feat_size
        )
        global_features = X[:, :2]
        return global_features, obj_features

    @staticmethod
    def _collate_fn(batch):
        x_list, y_list = zip(*batch)
        global_feats = []
        obj_feats = []
        edge_indices = []
        edge_attrs = []
        batch_indices = []

        for i, (global_feat, obj_feat) in enumerate(x_list):
            num_objects = (obj_feat[:, 0] != 0).sum().item()
            if num_objects < 2:
                continue

            obj_feat = obj_feat[:num_objects]
            global_feats.append(global_feat)
            obj_feats.append(obj_feat)

            # Create all pairwise combinations
            pairs = list(itertools.combinations(range(num_objects), 2))
            senders = [p[0] for p in pairs]
            receivers = [p[1] for p in pairs]
            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            edge_indices.append(edge_index)

            # Calculate delta R and invariant mass
            delta_eta = obj_feat[senders, 2] - obj_feat[receivers, 2]
            delta_phi = torch.remainder(
                obj_feat[senders, 3] - obj_feat[receivers, 3] + torch.pi, 
                2 * torch.pi
            ) - torch.pi
            delta_r = torch.sqrt(delta_eta**2 + delta_phi**2)

            # Calculate invariant mass (simplified approximation)
            px1 = obj_feat[senders, 1] * torch.cos(obj_feat[senders, 3])
            py1 = obj_feat[senders, 1] * torch.sin(obj_feat[senders, 3])
            pz1 = obj_feat[senders, 1] * torch.sinh(obj_feat[senders, 2])
            e1 = obj_feat[senders, 1] * torch.cosh(obj_feat[senders, 2])

            px2 = obj_feat[receivers, 1] * torch.cos(obj_feat[receivers, 3])
            py2 = obj_feat[receivers, 1] * torch.sin(obj_feat[receivers, 3])
            pz2 = obj_feat[receivers, 1] * torch.sinh(obj_feat[receivers, 2])
            e2 = obj_feat[receivers, 1] * torch.cosh(obj_feat[receivers, 2])

            inv_mass = torch.sqrt(
                (e1 + e2)**2 - 
                (px1 + px2)**2 - 
                (py1 + py2)**2 - 
                (pz1 + pz2)**2
            )
            edge_attrs.append(torch.stack([delta_r, inv_mass], dim=1))

            batch_indices.extend([i] * num_objects)

        if not global_feats:
            return None

        obj_feats = torch.cat(obj_feats, dim=0)
        global_feats = torch.stack(global_feats, dim=0)
        edge_indices = torch.cat(edge_indices, dim=1)
        edge_attrs = torch.cat(edge_attrs, dim=0)
        batch_indices = torch.tensor(batch_indices, dtype=torch.long)

        data = Batch.from_data_list([Data(
            x=obj_feats,
            edge_index=edge_indices,
            edge_attr=edge_attrs,
            batch=batch_indices,
            global_features=global_feats,
            num_nodes=obj_feats.size(0)
        )])

        y = torch.stack(y_list, dim=0)
        return data, y

    def make_loader_cfg(self):
        return {
            "loader_class": "torch.utils.data.DataLoader",
            "collate_fn": "self._collate_fn",
            "batch_size": 256,
            "shuffle": False,
            "num_workers": 0
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._raw_reshape(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        self.conv1 = TransformerConv(128, 256, edge_dim=128)
        self.conv2 = TransformerConv(256, 256, edge_dim=128)

        self.global_mlp = nn.Sequential(
            nn.Linear(128 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, data):
        x = self.encoder(data.x)  # [num_nodes, 128]
        edge_attr = self.edge_encoder(data.edge_attr)  # [num_edges, 128]

        x = self.conv1(x, data.edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        # Global pooling
        node_features = global_mean_pool(x, data.batch)  # [batch_size, 256]

        # Global features
        global_features = self.global_mlp(data.global_features)  # [batch_size, 256]

        # Concatenate and classify
        combined = torch.cat([node_features, global_features], dim=1)
        return self.classifier(combined).squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data, targets in train_loader:
            if data is None:
                continue

            data = data.to(device)
            targets = targets.float().to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == targets.long()).sum().item()
            total += len(targets)

        train_loss = total_loss / total if total > 0 else 0
        train_acc = correct / total if total > 0 else 0
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for data, targets in val_loader:
                if data is None:
                    continue

                data = data.to(device)
                targets = targets.float().to(device)

                outputs = model(data)
                loss = criterion(outputs, targets)

                total_val_loss += loss.item() * len(targets)
                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct_val += (preds == targets.long()).sum().item()
                total_val += len(targets)

        val_loss = total_val_loss / total_val if total_val > 0 else 0
        val_acc = correct_val / total_val if total_val > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
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
    epochs = range(1, len(series_train) + 1)
    plt.plot(epochs, series_train, label=f"Train {name}")
    plt.plot(epochs, series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        X_train, Y_train, X_val, Y_val = X_train[:200], Y_train[:200], X_val[:20], Y_val[:20]
    pre     = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train)
    X_val   = pre.transform(X_val)

    collate = getattr(pre, "_collate_fn", None)
    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val, 
                                            batch      = cfg.get("batch_size", 512), 
                                            collate_fn = collate,
                                            loader_cls = loader_cls)

    # 2. Build model
    first_batch    = next(iter(train_loader))
    example_sample = first_batch[0]
    model          = make_model(example_sample)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. Dry-run safety check
    if dryrun:
        sample, _ = first_batch
        try:
            _ = trained_model(*sample) if isinstance(sample, (tuple, list)) else trained_model(sample)
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

