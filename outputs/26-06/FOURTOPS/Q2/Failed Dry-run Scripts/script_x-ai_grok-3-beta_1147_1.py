
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
from torch.nn import functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data, Batch

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler_global = StandardScaler()
        self.scaler_obj = StandardScaler()
        self.max_objects = 18
        self.obj_features = 4  # E, p_T, eta, phi
        self.global_features = 2  # E_T_miss, phi_Et_miss

    def _raw_reshape(self, X):
        return X  # Shape: [N, 92]

    def _compute_pairwise_features(self, X_obj):
        # Compute pairwise invariant mass and delta R for all object pairs
        n_events = X_obj.shape[0]
        n_pairs = (self.max_objects * (self.max_objects - 1)) // 2
        pairwise_features = np.zeros((n_events, n_pairs * 2))  # invariant mass + delta R for each pair

        idx = 0
        for i in range(self.max_objects):
            for j in range(i + 1, self.max_objects):
                # Extract E, p_T, eta, phi for objects i and j
                E_i = X_obj[:, i, 0]
                pT_i = X_obj[:, i, 1]
                eta_i = X_obj[:, i, 2]
                phi_i = X_obj[:, i, 3]

                E_j = X_obj[:, j, 0]
                pT_j = X_obj[:, j, 1]
                eta_j = X_obj[:, j, 2]
                phi_j = X_obj[:, j, 3]

                # Compute invariant mass (simplified, using energy and momentum components)
                px_i = pT_i * np.cos(phi_i)
                py_i = pT_i * np.sin(phi_i)
                pz_i = pT_i * np.sinh(eta_i)
                px_j = pT_j * np.cos(phi_j)
                py_j = pT_j * np.sin(phi_j)
                pz_j = pT_j * np.sinh(eta_j)

                invariant_mass = np.sqrt(np.clip((E_i + E_j)**2 - (px_i + px_j)**2 - (py_i + py_j)**2 - (pz_i + pz_j)**2, 0, None))

                # Compute delta R
                delta_eta = eta_i - eta_j
                delta_phi = phi_i - phi_j
                delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

                pairwise_features[:, idx * 2] = invariant_mass
                pairwise_features[:, idx * 2 + 1] = delta_R
                idx += 1

        return pairwise_features  # Shape: [N, n_pairs * 2]

    def make_loader_cfg(self):
        return {
            "loader_class": "torch_geometric.loader.DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 0
        }

    def fit(self, X, y=None):
        N = X.shape[0]
        X_global = X[:, :self.global_features]  # Shape: [N, 2]
        X_obj = X[:, self.global_features:].reshape(N, self.max_objects, self.obj_features + 1)[:, :, 1:]  # Shape: [N, 18, 4]

        # Fit scalers
        self.scaler_global.fit(X_global)
        X_obj_reshaped = X_obj.reshape(N * self.max_objects, self.obj_features)
        mask = X_obj_reshaped[:, 0] > 0  # Only scale valid objects (E > 0)
        if np.any(mask):
            self.scaler_obj.fit(X_obj_reshaped[mask])
        return self

    def transform(self, X):
        N = X.shape[0]
        X_global = X[:, :self.global_features]  # Shape: [N, 2]
        X_obj = X[:, self.global_features:].reshape(N, self.max_objects, self.obj_features + 1)[:, :, 1:]  # Shape: [N, 18, 4]

        # Scale features
        X_global = self.scaler_global.transform(X_global)  # Shape: [N, 2]
        X_obj_reshaped = X_obj.reshape(N * self.max_objects, self.obj_features)
        mask = X_obj_reshaped[:, 0] > 0
        if np.any(mask):
            X_obj_reshaped[mask] = self.scaler_obj.transform(X_obj_reshaped[mask])
        X_obj = X_obj_reshaped.reshape(N, self.max_objects, self.obj_features)  # Shape: [N, 18, 4]

        # Compute pairwise features
        pairwise_features = self._compute_pairwise_features(X_obj)  # Shape: [N, n_pairs * 2]

        # Create graph data for each event
        data_list = []
        for i in range(N):
            node_features = X_obj[i]  # Shape: [18, 4]
            global_feat = X_global[i]  # Shape: [2]
            pair_feat = pairwise_features[i]  # Shape: [n_pairs * 2]

            # Create fully connected graph (excluding self-loops)
            edge_index = []
            for j in range(self.max_objects):
                for k in range(j + 1, self.max_objects):
                    edge_index.append([j, k])
                    edge_index.append([k, j])  # Bidirectional edge
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Shape: [2, n_edges]

            # Edge features (invariant mass and delta R)
            edge_attr = []
            idx = 0
            for j in range(self.max_objects):
                for k in range(j + 1, self.max_objects):
                    edge_attr.append([pair_feat[idx * 2], pair_feat[idx * 2 + 1]])
                    edge_attr.append([pair_feat[idx * 2], pair_feat[idx * 2 + 1]])  # Bidirectional
                    idx += 1
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Shape: [n_edges, 2]

            x = torch.tensor(node_features, dtype=torch.float)  # Shape: [18, 4]
            u = torch.tensor(global_feat, dtype=torch.float)  # Shape: [2]
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)
            data_list.append(data)

        return data_list

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.node_dim = sample_object.x.shape[1]  # 4 (E, p_T, eta, phi)
        self.edge_dim = sample_object.edge_attr.shape[1]  # 2 (invariant mass, delta R)
        self.global_dim = sample_object.u.shape[0]  # 2 (E_T_miss, phi_Et_miss)

        # Transformer layers for graph processing
        self.conv1 = TransformerConv(self.node_dim, 64, edge_dim=self.edge_dim, heads=4, dropout=0.2)
        self.conv2 = TransformerConv(64 * 4, 128, edge_dim=self.edge_dim, heads=4, dropout=0.2)
        self.conv3 = TransformerConv(128 * 4, 256, edge_dim=self.edge_dim, heads=4, dropout=0.2)

        # MLP for global feature integration
        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        if isinstance(data, Batch):
            x, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u
            batch = data.batch
        else:
            x, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # Process node features with TransformerConv layers
        x = self.conv1(x, edge_index, edge_attr=edge_attr)  # Shape: [N_nodes, 64*4]
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)  # Shape: [N_nodes, 128*4]
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr=edge_attr)  # Shape: [N_nodes, 256*4]
        x = F.relu(x)

        # Global pooling to aggregate node features
        x_graph = global_mean_pool(x, batch)  # Shape: [N_batch, 256]

        # Process global features
        u_processed = self.global_mlp(u)  # Shape: [N_batch, 64]

        # Combine graph and global features
        combined = torch.cat([x_graph, u_processed], dim=-1)  # Shape: [N_batch, 256+64]

        # Classify
        out = self.classifier(combined)  # Shape: [N_batch, 1]
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience = 5
    counter = 0
    best_model_state = None

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            data, target = batch
            data = data.to(device)
            target = target.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = (torch.sigmoid(output) > 0.5).float()
            correct += (pred == target).sum().item()
            total += target.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        train_loss.append(avg_train_loss)
        train_acc.append(train_accuracy)

        # Validation
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                data, target = batch
                data = data.to(device)
                target = target.to(device).float().view(-1, 1)

                output = model(data)
                loss = criterion(output, target)
                total_val_loss += loss.item()

                pred = (torch.sigmoid(output) > 0.5).float()
                correct_val += (pred == target).sum().item()
                total_val += target.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_loss.append(avg_val_loss)
        val_acc.append(val_accuracy)

        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss, val_loss, train_acc, val_acc

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

