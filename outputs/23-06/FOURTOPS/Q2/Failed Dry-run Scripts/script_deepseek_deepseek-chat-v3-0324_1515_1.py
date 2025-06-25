
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
from torch.nn import Linear, BatchNorm1d, Dropout
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from sklearn.metrics import roc_auc_score

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.max_objs = 18
        self.obj_feat_size = 5
        self.global_feats = 2

    def _raw_reshape(self, X):
        return X

    def _create_pairwise_features(self, objects):
        num_objs = objects.shape[0]
        pairwise_feats = []

        for i in range(num_objs):
            for j in range(i+1, num_objs):
                eta_i, phi_i = objects[i, 3], objects[i, 4]
                eta_j, phi_j = objects[j, 3], objects[j, 4]

                delta_eta = eta_i - eta_j
                delta_phi = phi_i - phi_j
                delta_phi = torch.remainder(delta_phi + math.pi, 2 * math.pi) - math.pi

                delta_r = torch.sqrt(delta_eta**2 + delta_phi**2)

                pti, ptj = objects[i, 2], objects[j, 2]
                m_inv = torch.sqrt(2 * pti * ptj * (torch.cosh(delta_eta) - torch.cos(delta_phi)))

                pairwise_feats.append(torch.stack([delta_r, m_inv]))

        if pairwise_feats:
            return torch.stack(pairwise_feats)
        return torch.zeros(0, 2)

    def _process_events(self, X):
        batch_global = []
        batch_objs = []
        batch_pairwise = []
        valid_objs = []

        for event in X:
            global_feats = event[:2]
            obj_data = event[2:].reshape(-1, self.obj_feat_size)

            valid_idx = obj_data[:, 0] != 0  # obj_id 0 means padding
            valid_obj = obj_data[valid_idx][:, 1:]  # remove obj_id column

            pairwise_feats = self._create_pairwise_features(valid_obj)

            batch_global.append(global_feats)
            batch_objs.append(valid_obj)
            batch_pairwise.append(pairwise_feats)
            valid_objs.append(valid_idx.sum().item())

        return batch_global, batch_objs, batch_pairwise, valid_objs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._process_events(X)

    @staticmethod
    def _collate_fn(batch):
        inputs, targets = zip(*batch)
        global_feats, objs, pairwise, valid_objs = zip(*inputs)

        # Pad objects and features to max in batch
        max_objs = max(len(o) for o in objs)
        max_pairwise = max(len(p) for p in pairwise)

        obj_padded = [
            F.pad(o, (0, 0, 0, max_objs - len(o)), value=0) 
            for o in objs
        ]
        pairwise_padded = [
            F.pad(p, (0, 0, 0, max_pairwise - len(p)), value=0)
            for p in pairwise
        ]

        global_feats = torch.stack(global_feats)
        obj_padded = torch.stack(obj_padded)
        pairwise_padded = torch.stack(pairwise_padded)
        valid_objs = torch.tensor(valid_objs)
        targets = torch.stack(targets)

        return (global_feats, obj_padded, pairwise_padded, valid_objs), targets

    def make_loader_cfg(self):
        return {
            "batch_size": 128,
            "collate_fn": self._collate_fn
        }

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        global_feats, obj_padded, pairwise_padded, _ = sample_object

        self.obj_feat_size = obj_padded.shape[-1]
        self.pairwise_feat_size = pairwise_padded.shape[-1]

        # Object features processing
        self.obj_encoder = nn.Sequential(
            Linear(self.obj_feat_size, 64),
            BatchNorm1d(64),
            nn.LeakyReLU(),
            Dropout(0.3),
            Linear(64, 128),
            BatchNorm1d(128),
            nn.LeakyReLU(),
            Dropout(0.3)
        )

        # Pairwise features processing
        self.pairwise_encoder = nn.Sequential(
            Linear(self.pairwise_feat_size, 32),
            BatchNorm1d(32),
            nn.LeakyReLU(),
            Dropout(0.2),
            Linear(32, 64),
            BatchNorm1d(64),
            nn.LeakyReLU(),
            Dropout(0.2)
        )

        # Global features processing
        self.global_encoder = nn.Sequential(
            Linear(2, 32),
            BatchNorm1d(32),
            nn.LeakyReLU(),
            Dropout(0.1),
            Linear(32, 64),
            BatchNorm1d(64),
            nn.LeakyReLU(),
            Dropout(0.1)
        )

        # Main classifier
        self.classifier = nn.Sequential(
            Linear(128 + 64 + 64, 256),
            BatchNorm1d(256),
            nn.LeakyReLU(),
            Dropout(0.4),
            Linear(256, 128),
            BatchNorm1d(128),
            nn.LeakyReLU(),
            Dropout(0.4),
            Linear(128, 1)
        )

    def forward(self, *data):
        global_feats, obj_padded, pairwise_padded, valid_objs = data

        # Process global features [batch_size, 2]
        global_encoded = self.global_encoder(global_feats)  # [batch_size, 64]

        # Process object features
        batch_size, max_objs, _ = obj_padded.shape
        obj_flat = obj_padded.view(batch_size * max_objs, -1)  # [batch*max_objs, feat_size]
        obj_encoded = self.obj_encoder(obj_flat).view(batch_size, max_objs, -1)  # [batch, max_objs, 128]

        # Mask out padded objects
        mask = torch.arange(max_objs).expand(batch_size, max_objs).to(obj_encoded.device) < valid_objs.unsqueeze(1)
        obj_encoded = obj_encoded * mask.unsqueeze(-1)  # [batch, max_objs, 128]

        # Average pooling over objects
        obj_pooled = obj_encoded.sum(dim=1) / valid_objs.unsqueeze(1).clamp(min=1)  # [batch, 128]

        # Process pairwise features
        batch_size, max_pairs, _ = pairwise_padded.shape
        pairwise_flat = pairwise_padded.view(batch_size * max_pairs, -1)  # [batch*max_pairs, 2]
        pairwise_encoded = self.pairwise_encoder(pairwise_flat).view(batch_size, max_pairs, -1)  # [batch, max_pairs, 64]

        # Average pooling over pairwise features
        valid_pairs = torch.tensor([max(1, (n*(n-1))//2) for n in valid_objs], 
                                 device=pairwise_encoded.device)
        pairwise_pooled = pairwise_encoded.sum(dim=1) / valid_pairs.unsqueeze(1)  # [batch, 64]

        # Concatenate all features
        combined = torch.cat([global_encoded, obj_pooled, pairwise_pooled], dim=1)  # [batch, 256]

        # Final prediction
        logits = self.classifier(combined)  # [batch, 1]
        return torch.sigmoid(logits.squeeze(1))  # [batch]

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=4, factor=0.5, verbose=False)
    criterion = nn.BCELoss()

    best_auc = 0
    best_model = None
    epochs_no_improve = 0
    patience = 7

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        all_targets = []
        all_preds = []

        for batch in train_loader:
            inputs, targets = batch
            inputs = tuple(t.to(device) for t in inputs)
            targets = targets.float().to(device)

            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            all_preds.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

        train_preds = torch.cat(all_preds)
        train_targets = torch.cat(all_targets)
        train_auc = roc_auc_score(train_targets.numpy(), train_preds.numpy())
        train_acc = ((train_preds > 0.5).float() == train_targets).float().mean().item()
        train_loss_history.append(epoch_train_loss / len(train_loader))
        train_acc_history.append(train_acc)

        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = tuple(t.to(device) for t in inputs)
                targets = targets.float().to(device)

                outputs = model(*inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                all_val_preds.append(outputs.cpu())
                all_val_targets.append(targets.cpu())

        val_preds = torch.cat(all_val_preds)
        val_targets = torch.cat(all_val_targets)
        val_auc = roc_auc_score(val_targets.numpy(), val_preds.numpy())
        val_acc = ((val_preds > 0.5).float() == val_targets).float().mean().item()

        val_loss_history.append(val_loss / len(val_loader))
        val_acc_history.append(val_acc)

        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    model.load_state_dict(best_model)
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

