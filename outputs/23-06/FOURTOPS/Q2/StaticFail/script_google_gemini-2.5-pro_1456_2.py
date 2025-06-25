
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

# <start code template>
# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric.data
import torch_geometric.loader
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch.nn import functional as F
from collections import OrderedDict
import copy

# This custom dataset function is needed to correctly handle the output of the preprocessor
def make_dataset(x, y):
    """
    Combines the preprocessed features (a list of graph Data objects) 
    with their labels.
    """
    # x is a list of torch_geometric.data.Data objects from the preprocessor
    # y is a tensor of labels
    for i, data_obj in enumerate(x):
        data_obj.y = y[i].long()
    # The list of Data objects is a valid dataset for torch_geometric.loader.DataLoader
    return x

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    #    Must implement:
    #   - fit(...)               -> self
    #   - transform(X: ???)      -> ???

    # DATA SPECIFICS
    # Total flat length per event (X_train & X_val): 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    # ...
    # Indices 88-92 : object 18 ->  obj_18, E_18 , p_T_18 , eta_18 , phi_18
    # Global features       = 2
    # Per-object slice size = 5
    # Max objects encoded   = 18

    # TIPS
    # When modifying data features or feature engineering: annotate tensor size as comments after 
    # each tensor operation to reduce dimension mismatches.

    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module.
    # May allocate NumPy arrays or Torch tensors internally, but:
    # transform() must be deterministic.
    # Store only derived parameters needed for transform i.e. do not store the raw data
    # itself in the preprocessor object.

    GLOBAL_FEATS = 2
    OBJ_FEATS = 5
    MAX_OBJS = 18

    def __init__(self):
        # Define and initialize any stateful components here
        self.scalers = {}
        self.obj_id_map = {}
        self.fitted = False

    def _reshape_and_mask(self, X):
        # X shape: [N_events, 92]
        global_features = X[:, :self.GLOBAL_FEATS]  # Shape: [N_events, 2]
        # Particles shape: [N_events, 18, 5]
        particles = X[:, self.GLOBAL_FEATS:].reshape(-1, self.MAX_OBJS, self.OBJ_FEATS)

        # Mask for real particles (pT > 0). Using pT is a robust physical check for existence.
        mask = particles[:, :, 2] > 1e-6 # Shape: [N_events, 18]
        return global_features, particles, mask

    def make_loader_cfg(self):
        # Return dict or None.  If dict, evaluator uses it to rebuild loader:
        #{
        #   "loader_class": "torch.utils.data.DataLoader",
        #   "collate_fn": "self._collate_fn",
        #   "batch_size": 256,
        #   "shuffle": False,
        #   "num_workers": 0
        #}
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 256,
        }

    def fit(self, X, y=None):
        # Extract statistics for fit transformers
        global_features, particles, mask = self._reshape_and_mask(X)

        # Discover unique object IDs for one-hot encoding
        obj_ids_flat = particles[:, :, 0][mask].unique()
        self.obj_id_map = {int(v.item()): i for i, v in enumerate(obj_ids_flat)}

        # Container for features to be scaled
        feats_to_scale = {'global': [], 'node': [], 'edge': []}

        # Process each event to extract features for scaling
        for i in range(X.shape[0]):
            event_mask = mask[i]
            if not event_mask.any():
                continue

            valid_particles = particles[i][event_mask] # Shape: [n_valid, 5]

            # --- Global features ---
            g_feats = self._get_global_feats(global_features[i]) # Shape: [3]
            feats_to_scale['global'].append(g_feats)

            # --- Node features ---
            n_feats = self._get_node_feats(valid_particles) # Shape: [n_valid, 5]
            feats_to_scale['node'].append(n_feats)

            # --- Edge features ---
            if valid_particles.shape[0] > 1:
                e_feats = self._get_edge_feats(valid_particles) # Shape: [n_pairs, 4]
                feats_to_scale['edge'].append(e_feats)

        # Fit scalers on the collected features
        for key, feat_list in feats_to_scale.items():
            if not feat_list: continue
            scaler = StandardScaler()
            scaler.fit(torch.cat(feat_list, dim=0).numpy())
            self.scalers[key] = {
                "mean": torch.from_numpy(scaler.mean_).float(),
                "scale": torch.from_numpy(scaler.scale_).float()
            }

        self.fitted = True
        return self

    def transform(self, X):
        # Apply pre-processing logic
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        global_features, particles, mask = self._reshape_and_mask(X)
        data_list = []

        for i in range(X.shape[0]):
            event_mask = mask[i]
            num_valid_particles = event_mask.sum().item()

            if num_valid_particles == 0:
                continue

            valid_particles = particles[i][event_mask]

            # --- Global features (u) ---
            u = self._get_global_feats(global_features[i])
            u = (u - self.scalers['global']['mean']) / self.scalers['global']['scale']
            u = u.unsqueeze(0) # Shape: [1, n_global_feats]

            # --- Node features (x) ---
            node_feats_raw = self._get_node_feats(valid_particles)
            node_feats_scaled = (node_feats_raw - self.scalers['node']['mean']) / self.scalers['node']['scale']
            obj_ids = valid_particles[:, 0].long()
            obj_ids_mapped = torch.tensor([self.obj_id_map[id_val.item()] for id_val in obj_ids], dtype=torch.long)
            obj_ids_one_hot = F.one_hot(obj_ids_mapped, num_classes=len(self.obj_id_map)).float()
            x = torch.cat([node_feats_scaled, obj_ids_one_hot], dim=1) # Shape: [n_valid, n_node_feats]

            # --- Edge features (edge_attr, edge_index) ---
            if num_valid_particles > 1:
                adj = torch.ones(num_valid_particles, num_valid_particles) - torch.eye(num_valid_particles)
                edge_index = adj.to_sparse().indices()
                edge_feats_raw = self._get_edge_feats(valid_particles)
                edge_attr = (edge_feats_raw - self.scalers['edge']['mean']) / self.scalers['edge']['scale']
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 4)) # Must match n_edge_feats

            data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)
            data_list.append(data)

        # must return an indexable, picklable object
        return data_list

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _get_global_feats(self, global_vec):
        et_miss = global_vec[0].unsqueeze(0)
        phi_miss = global_vec[1]
        return torch.cat([et_miss, torch.cos(phi_miss).unsqueeze(0), torch.sin(phi_miss).unsqueeze(0)])

    def _get_node_feats(self, valid_particles):
        E, pT, eta, phi = valid_particles[:, 1], valid_particles[:, 2], valid_particles[:, 3], valid_particles[:, 4]
        return torch.stack([E, pT, eta, torch.cos(phi), torch.sin(phi)], dim=1)

    def _p4_from_kinematics(self, E, pT, eta, phi):
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)
        return torch.stack([E, px, py, pz], dim=1)

    def _get_edge_feats(self, valid_particles):
        n = valid_particles.shape[0]
        i, j = torch.triu_indices(n, n, 1)
        p1, p2 = valid_particles[i], valid_particles[j]

        d_eta = p1[:, 3] - p2[:, 3]
        d_phi = p1[:, 4] - p2[:, 4]
        d_phi = (d_phi + torch.pi) % (2 * torch.pi) - torch.pi
        delta_r = torch.sqrt(d_eta**2 + d_phi**2)

        p4_1 = self._p4_from_kinematics(p1[:, 1], p1[:, 2], p1[:, 3], p1[:, 4])
        p4_2 = self._p4_from_kinematics(p2[:, 1], p2[:, 2], p2[:, 3], p2[:, 4])
        p4_sum = p4_1 + p4_2
        m2 = p4_sum[:, 0]**2 - p4_sum[:, 1]**2 - p4_sum[:, 2]**2 - p4_sum[:, 3]**2

        sign_m2 = torch.sign(m2)
        sqrt_abs_m2 = torch.sqrt(torch.abs(m2))
        m_signed_sqrt = sign_m2 * sqrt_abs_m2
        log_m2 = torch.log(torch.abs(m2) + 1e-6)
        edge_feats = torch.stack([delta_r, m_signed_sqrt, log_m2, m2], dim=1)

        return torch.cat([edge_feats, edge_feats], dim=0)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Define and initialize any stateful components here
        node_dim = sample_object.x.shape[1]
        edge_dim = sample_object.edge_attr.shape[1]
        global_dim = sample_object.u.shape[1]

        hidden_dim = 128

        self.node_encoder = nn.Linear(node_dim + global_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=True, edge_dim=hidden_dim),
            GATv2Conv(hidden_dim * 4, hidden_dim, heads=4, concat=True, edge_dim=hidden_dim),
            GATv2Conv(hidden_dim * 4, hidden_dim, heads=1, concat=False, edge_dim=hidden_dim)
        ])

        self.bns = nn.ModuleList([
             nn.BatchNorm1d(hidden_dim * 4), 
             nn.BatchNorm1d(hidden_dim * 4), 
             nn.BatchNorm1d(hidden_dim)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    # optionally build extra layers here

    def forward(self, *data):
        # Define your model's forward pass here
        # The harness passes the sample as *data, so we unpack it.
        gdata = data[0]
        x, edge_index, edge_attr, u, batch = gdata.x, gdata.edge_index, gdata.edge_attr, gdata.u, gdata.batch

        u_expanded = u[batch] # Shape: [n_total_nodes, n_global_feats]
        x = torch.cat([x, u_expanded], dim=1) # Shape: [n_total_nodes, n_node_feats + n_global_feats]

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.gnn_layers[0](x, edge_index, edge_attr=edge_attr)
        x = F.relu(self.bns[0](x))
        x = self.gnn_layers[1](x, edge_index, edge_attr=edge_attr)
        x = F.relu(self.bns[1](x))
        x = self.gnn_layers[2](x, edge_index, edge_attr=edge_attr)
        x = F.relu(self.bns[2](x))

        x_pooled = global_add_pool(x, batch) # Shape: [batch_size, hidden_dim]

        final_vec = torch.cat([x_pooled, u], dim=1) # Shape: [batch_size, hidden_dim + global_dim]

        return self.classifier(final_vec)


def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30   # adjust if you wish
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS 
    # Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    # Must return trained_model, train_loss, val_loss, train_acc, val_acc
    # Implement early-stopping.
    # Forward signature must match.

    # Write code to define training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5
    best_model_state = None

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.num_graphs
            preds = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)
        avg_train_acc = correct_train / total_train
        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)

        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs.squeeze(), labels.float())

                running_val_loss += loss.item() * data.num_graphs
                preds = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        avg_val_acc = correct_val / total_val
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        scheduler.step(avg_val_loss)

        # Implement early stopping if possible
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    trained_model = model
    return trained_model, train_loss, val_loss, train_acc, val_acc

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
# <end code template>

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

