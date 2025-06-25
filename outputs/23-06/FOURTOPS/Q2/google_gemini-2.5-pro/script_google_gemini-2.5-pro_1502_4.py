
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
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
import math
import copy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GENConv, global_mean_pool
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.particle_mean = None
        self.particle_std = None
        self.obj_id_map = {}
        self.num_obj_ids = 0
        self._is_fit = False

    def _raw_reshape(self, X: torch.Tensor):
        # Globals: (N, 2)
        global_feats = X[:, :2]
        # Particles: (N, 18, 5)
        particle_feats = X[:, 2:].reshape(-1, 18, 5)
        # Mask for non-padded particles. pT is at index 2 of the 5 features.
        mask = particle_feats[:, :, 2] > 1e-6 # pT > 0
        return global_feats, particle_feats, mask

    def _get_cartesian(self, E, pT, eta, phi):
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)
        return torch.stack([E, px, py, pz], dim=-1)

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
        global_feats, particle_feats, mask = self._raw_reshape(X)

        # 1. Process object IDs
        all_obj_ids = particle_feats[:, :, 0][mask].unique()
        self.obj_id_map = {int(v.item()): i for i, v in enumerate(all_obj_ids)}
        self.num_obj_ids = len(self.obj_id_map)

        # 2. Process global features
        g_phi = global_feats[:, 1]
        trans_global_feats = torch.stack([
            torch.log1p(global_feats[:, 0]),
            torch.sin(g_phi),
            torch.cos(g_phi),
        ], dim=1) # Shape: (N, 3)
        self.global_mean = torch.mean(trans_global_feats, dim=0)
        self.global_std = torch.std(trans_global_feats, dim=0)
        self.global_std[self.global_std < 1e-8] = 1.0

        # 3. Process particle features
        p_E = particle_feats[:, :, 1]
        p_pT = particle_feats[:, :, 2]
        p_eta = particle_feats[:, :, 3]
        p_phi = particle_feats[:, :, 4]

        LOG_EPS = 1e-8
        trans_particle_feats = torch.stack([
            torch.log(p_E + LOG_EPS),
            torch.log(p_pT + LOG_EPS),
            p_eta,
            torch.sin(p_phi),
            torch.cos(p_phi)
        ], dim=-1) # Shape: (N, 18, 5)

        masked_particles = trans_particle_feats[mask]
        self.particle_mean = torch.mean(masked_particles, dim=0)
        self.particle_std = torch.std(masked_particles, dim=0)
        self.particle_std[self.particle_std < 1e-8] = 1.0

        self._is_fit = True
        return self

    def transform(self, X):
        if not self._is_fit:
            raise RuntimeError("Preprocessor is not fitted yet.")

        global_feats_raw, particle_feats_raw, mask = self._raw_reshape(X)

        data_list = []
        for i in range(X.shape[0]):
            # Select non-padded particles for this event
            event_mask = mask[i]
            if not torch.any(event_mask):
                # Handle events with no particles
                num_nodes = 0
                x = torch.empty((0, 5), dtype=torch.float32)
                obj_ids = torch.empty((0,), dtype=torch.int64)
                edge_index = torch.empty((2, 0), dtype=torch.int64)
                edge_attr = torch.empty((0, 2), dtype=torch.float32)
            else:
                num_nodes = event_mask.sum()

                # --- Particle features (x) ---
                p_E = particle_feats_raw[i, event_mask, 1]
                p_pT = particle_feats_raw[i, event_mask, 2]
                p_eta = particle_feats_raw[i, event_mask, 3]
                p_phi = particle_feats_raw[i, event_mask, 4]

                LOG_EPS = 1e-8
                p_trans = torch.stack([
                    torch.log(p_E + LOG_EPS), torch.log(p_pT + LOG_EPS), p_eta,
                    torch.sin(p_phi), torch.cos(p_phi)
                ], dim=-1)
                x = (p_trans - self.particle_mean) / self.particle_std

                # --- Object ID features ---
                raw_ids = particle_feats_raw[i, event_mask, 0]
                obj_ids = torch.tensor([self.obj_id_map.get(int(v.item()), -1) for v in raw_ids], dtype=torch.long)

                # --- Edge features (edge_index, edge_attr) ---
                if num_nodes > 1:
                    adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
                    edge_index = adj.nonzero().t().contiguous()

                    row, col = edge_index

                    # Delta R
                    d_eta = p_eta[row] - p_eta[col]
                    d_phi = torch.atan2(torch.sin(p_phi[row] - p_phi[col]), torch.cos(p_phi[row] - p_phi[col]))
                    delta_r = torch.sqrt(d_eta**2 + d_phi**2)
                    log_delta_r = torch.log(delta_r + LOG_EPS)

                    # Invariant Mass
                    p4_i = self._get_cartesian(p_E[row], p_pT[row], p_eta[row], p_phi[row])
                    p4_j = self._get_cartesian(p_E[col], p_pT[col], p_eta[col], p_phi[col])
                    p4_sum = p4_i + p4_j
                    m_sq = p4_sum[:, 0]**2 - torch.sum(p4_sum[:, 1:]**2, dim=-1)
                    signed_log_m_sq = torch.sign(m_sq) * torch.log(torch.abs(m_sq) + 1)

                    edge_attr = torch.stack([log_delta_r, signed_log_m_sq], dim=-1)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.int64)
                    edge_attr = torch.empty((0, 2), dtype=torch.float32)

            # --- Global features (u) ---
            g_phi = global_feats_raw[i, 1]
            g_trans = torch.tensor([torch.log1p(global_feats_raw[i, 0]), torch.sin(g_phi), torch.cos(g_phi)])
            u = (g_trans - self.global_mean) / self.global_std

            data = Data(x=x, obj_ids=obj_ids, edge_index=edge_index, edge_attr=edge_attr, u=u.unsqueeze(0))
            # Attach metadata needed by the model
            data.num_obj_classes = self.num_obj_ids
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

        # Infer dimensions from a sample batch from the dataloader
        num_obj_classes = sample_object.num_obj_classes[0] if hasattr(sample_object, 'num_obj_classes') else 10
        node_feat_dim = sample_object.x.shape[1]
        edge_feat_dim = sample_object.edge_attr.shape[1]
        global_feat_dim = sample_object.u.shape[1]

        hidden_dim = 128
        embedding_dim = 16

        self.obj_embedding = nn.Embedding(num_obj_classes, embedding_dim)

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.gnn_layers = nn.ModuleList()
        num_gnn_layers = 4
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                GENConv(hidden_dim, hidden_dim, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, *data):
        # A bit of a hack to handle both custom loader and default loader
        if len(data) == 1:
            data = data[0]
        else:
             # This case should ideally not be hit with the current setup
             raise ValueError("Unexpected input format to model forward pass")

        x, obj_ids, edge_index, edge_attr, u, batch = (
            data.x, data.obj_ids, data.edge_index, data.edge_attr, data.u, data.batch
        )

        # Encode nodes
        obj_embeds = self.obj_embedding(obj_ids)
        x_in = torch.cat([x, obj_embeds], dim=-1) # x_in: [num_nodes, node_feat_dim + embedding_dim]
        h = self.node_encoder(x_in) # h: [num_nodes, hidden_dim]

        # Encode edges
        edge_attr_encoded = self.edge_encoder(edge_attr) # edge_attr_encoded: [num_edges, hidden_dim]

        # GNN layers
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_index, edge_attr=edge_attr_encoded)

        # Readout
        graph_embedding = global_mean_pool(h, batch) # [batch_size, hidden_dim]

        # Classifier
        combined_features = torch.cat([graph_embedding, u], dim=-1) # [batch_size, hidden_dim + global_feat_dim]
        output = self.classifier(combined_features)

        return output.squeeze(-1) # [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS 
    # Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    # Must return trained_model, train_loss, val_loss, train_acc, val_acc
    # Implement early-stopping.
    # Forward signature must match.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

    best_val_loss = float('inf')
    early_stopping_patience = 5
    patience_counter = 0
    best_model_state = None

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device).float()

            optimizer.zero_grad()

            outputs = model(data)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.num_graphs
            preds = torch.sigmoid(outputs).round()
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_preds / total_preds
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        correct_val_preds = 0
        total_val_preds = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device).float()

                outputs = model(data)
                loss = loss_fn(outputs, labels)

                running_val_loss += loss.item() * data.num_graphs
                preds = torch.sigmoid(outputs).round()
                correct_val_preds += (preds == labels).sum().item()
                total_val_preds += labels.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val_preds / total_val_preds
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        # --- Early Stopping ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    if best_model_state:
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

