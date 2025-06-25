
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
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.max_objects = 18
        self.obj_features = 4  # E, p_T, eta, phi
        self.global_features = 2  # E_T_miss, phi_Et_miss

    def _raw_reshape(self, X):
        return X  # Shape: [N, 92]

    def fit(self, X, y=None):
        # Reshape data to extract features for scaling
        N = X.shape[0]
        globals_data = X[:, :self.global_features]  # Shape: [N, 2]
        objects_data = X[:, self.global_features:].reshape(N, self.max_objects, self.obj_features + 1)  # Shape: [N, 18, 5]
        features = objects_data[:, :, 1:].reshape(N * self.max_objects, self.obj_features)  # Shape: [N*18, 4]
        valid_mask = (objects_data[:, :, 0] != 0).reshape(N * self.max_objects, 1)  # Shape: [N*18, 1]
        valid_features = features[valid_mask[:, 0]]  # Shape: [M, 4], M is number of valid objects

        if valid_features.shape[0] > 0:
            self.scaler.fit(valid_features)  # Fit scaler on valid object features
        return self

    def transform(self, X):
        N = X.shape[0]
        globals_data = X[:, :self.global_features]  # Shape: [N, 2]
        objects_data = X[:, self.global_features:].reshape(N, self.max_objects, self.obj_features + 1)  # Shape: [N, 18, 5]
        obj_ids = objects_data[:, :, 0:1]  # Shape: [N, 18, 1]
        features = objects_data[:, :, 1:].reshape(N, self.max_objects, self.obj_features)  # Shape: [N, 18, 4]

        # Scale features
        features_2d = features.reshape(N * self.max_objects, self.obj_features)  # Shape: [N*18, 4]
        scaled_features_2d = self.scaler.transform(features_2d)  # Shape: [N*18, 4]
        scaled_features = scaled_features_2d.reshape(N, self.max_objects, self.obj_features)  # Shape: [N, 18, 4]

        # Compute pairwise invariant mass and delta R
        pairwise_masses = []
        pairwise_deltaRs = []
        for i in range(self.max_objects):
            for j in range(i + 1, self.max_objects):
                E_i = scaled_features[:, i, 0]  # Shape: [N]
                pt_i = scaled_features[:, i, 1]  # Shape: [N]
                eta_i = scaled_features[:, i, 2]  # Shape: [N]
                phi_i = scaled_features[:, i, 3]  # Shape: [N]
                E_j = scaled_features[:, j, 0]  # Shape: [N]
                pt_j = scaled_features[:, j, 1]  # Shape: [N]
                eta_j = scaled_features[:, j, 2]  # Shape: [N]
                phi_j = scaled_features[:, j, 3]  # Shape: [N]

                # Compute invariant mass m_ij = sqrt((E_i + E_j)^2 - (p_i + p_j)^2)
                px_i = pt_i * torch.cos(phi_i)
                py_i = pt_i * torch.sin(phi_i)
                pz_i = pt_i * torch.sinh(eta_i)
                px_j = pt_j * torch.cos(phi_j)
                py_j = pt_j * torch.sin(phi_j)
                pz_j = pt_j * torch.sinh(eta_j)
                mass_ij = torch.sqrt(torch.clamp((E_i + E_j)**2 - (px_i + px_j)**2 - (py_i + py_j)**2 - (pz_i + pz_j)**2, min=0))  # Shape: [N]

                # Compute delta R = sqrt((eta_i - eta_j)^2 + (phi_i - phi_j)^2)
                delta_eta = eta_i - eta_j
                delta_phi = phi_i - phi_j
                delta_R_ij = torch.sqrt(delta_eta**2 + delta_phi**2)  # Shape: [N]

                pairwise_masses.append(mass_ij.unsqueeze(-1))  # Shape: [N, 1]
                pairwise_deltaRs.append(delta_R_ij.unsqueeze(-1))  # Shape: [N, 1]

        pairwise_masses = torch.cat(pairwise_masses, dim=-1) if pairwise_masses else torch.zeros(N, 0)  # Shape: [N, num_pairs]
        pairwise_deltaRs = torch.cat(pairwise_deltaRs, dim=-1) if pairwise_deltaRs else torch.zeros(N, 0)  # Shape: [N, num_pairs]

        return (globals_data, scaled_features, pairwise_masses, pairwise_deltaRs, obj_ids)  # Shapes: [N, 2], [N, 18, 4], [N, num_pairs], [N, num_pairs], [N, 18, 1]

    def make_loader_cfg(self):
        return None

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        _, obj_features, _, _, _ = sample_object
        self.obj_dim = obj_features.shape[-1]  # 4 (E, p_T, eta, phi)
        self.max_objects = obj_features.shape[-2]  # 18
        self.global_dim = sample_object[0].shape[-1]  # 2 (E_T_miss, phi_Et_miss)
        self.num_pairs = sample_object[2].shape[-1] if sample_object[2].shape[-1] > 0 else 0

        # Node feature embedding for objects
        self.node_emb = nn.Sequential(
            nn.Linear(self.obj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # GCN layers for object interactions
        self.conv1 = GCNConv(128, 128)
        self.conv2 = GCNConv(128, 64)

        # Global feature processing
        self.global_emb = nn.Sequential(
            nn.Linear(self.global_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Pairwise feature processing
        self.pairwise_emb = nn.Sequential(
            nn.Linear(2, 32),  # mass and delta_R
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * self.max_objects + 64 + 64 * (self.num_pairs if self.num_pairs > 0 else 1), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, *data):
        globals_data, obj_features, pairwise_masses, pairwise_deltaRs, obj_ids = data

        # Process object features
        batch_size = obj_features.shape[0]
        node_x = self.node_emb(obj_features)  # Shape: [batch_size, max_objects, 128]
        node_x = node_x.view(batch_size * self.max_objects, -1)  # Shape: [batch_size*max_objects, 128]

        # Create fully connected edge index for GCN
        edge_index = []
        for b in range(batch_size):
            offset = b * self.max_objects
            for i in range(self.max_objects):
                for j in range(self.max_objects):
                    if i != j:
                        edge_index.append([offset + i, offset + j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(obj_features.device)  # Shape: [2, batch_size*max_objects*(max_objects-1)]

        # Apply GCN
        node_x = F.relu(self.conv1(node_x, edge_index))  # Shape: [batch_size*max_objects, 128]
        node_x = F.relu(self.conv2(node_x, edge_index))  # Shape: [batch_size*max_objects, 64]
        node_x = node_x.view(batch_size, self.max_objects, -1)  # Shape: [batch_size, max_objects, 64]
        node_repr = node_x.view(batch_size, -1)  # Shape: [batch_size, max_objects*64]

        # Process global features
        global_repr = self.global_emb(globals_data)  # Shape: [batch_size, 64]

        # Process pairwise features
        if self.num_pairs > 0:
            pairwise_features = torch.stack([pairwise_masses, pairwise_deltaRs], dim=-1)  # Shape: [batch_size, num_pairs, 2]
            pairwise_repr = self.pairwise_emb(pairwise_features)  # Shape: [batch_size, num_pairs, 64]
            pairwise_repr = pairwise_repr.view(batch_size, -1)  # Shape: [batch_size, num_pairs*64]
        else:
            pairwise_repr = torch.zeros(batch_size, 64, device=obj_features.device)  # Shape: [batch_size, 64]

        # Concatenate all representations
        x = torch.cat([node_repr, global_repr, pairwise_repr], dim=-1)  # Shape: [batch_size, max_objects*64 + 64 + num_pairs*64]
        logits = self.classifier(x)  # Shape: [batch_size, 1]
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_auc = 0.0
    early_stopping_patience = 5
    early_stopping_counter = 0

    train_loss_history = []
    val_loss_history = []
    train_auc_history = []
    val_auc_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        for batch in train_loader:
            data, labels = batch
            data = [d.to(device) for d in data] if isinstance(data, (tuple, list)) else data.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(*data) if isinstance(data, (tuple, list)) else model(data)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs.squeeze()).detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)
        train_loss_history.append(avg_train_loss)
        train_auc_history.append(train_auc)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                data, labels = batch
                data = [d.to(device) for d in data] if isinstance(data, (tuple, list)) else data.to(device)
                labels = labels.to(device).float()
                outputs = model(*data) if isinstance(data, (tuple, list)) else model(data)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs.squeeze()).detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_loss_history.append(avg_val_loss)
        val_auc_history.append(val_auc)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            early_stopping_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("best_model.pt"))
    return model, train_loss_history, val_loss_history, train_auc_history, val_auc_history

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

