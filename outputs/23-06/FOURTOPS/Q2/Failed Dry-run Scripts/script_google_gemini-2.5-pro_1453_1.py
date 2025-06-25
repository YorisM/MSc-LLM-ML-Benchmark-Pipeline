
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
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Dataset as GeoDataset
from torch_geometric.nn import MetaLayer, global_mean_pool

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

    def __init__(self):
        self.g_mean, self.g_std = None, None
        self.n_mean, self.n_std = None, None
        self.e_mean, self.e_std = None, None
        self.num_obj_types = None
        self.CONT_NODE_FEATS = 8 
        self.PAIR_FEATS = 2

    def _get_phys_features(self, objects_raw):
        # objects_raw: [k, 5] -> [obj_id, E, pT, eta, phi]
        E = objects_raw[:, 1]
        pT = objects_raw[:, 2]
        eta = objects_raw[:, 3]
        phi = objects_raw[:, 4]

        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)

        mass_sq = E**2 - (px**2 + py**2 + pz**2)
        mass = torch.sqrt(torch.relu(mass_sq))

        # Returns [k, 8] tensor of continuous features
        return torch.stack([E, pT, eta, phi, px, py, pz, mass], dim=1)

    def _get_pair_features(self, phys_feats, p4_vectors):
        # phys_feats: [k, 8] -> [E, pT, eta, phi, px, py, pz, mass]
        # p4_vectors: [k, 4] -> [E, px, py, pz]
        k = phys_feats.shape[0]
        if k < 2:
            return torch.empty((0, self.PAIR_FEATS), dtype=torch.float32)

        p4_i = p4_vectors.unsqueeze(1)
        p4_j = p4_vectors.unsqueeze(0)
        p4_sum = p4_i + p4_j
        m_ij_sq = p4_sum[:, :, 0]**2 - torch.sum(p4_sum[:, :, 1:]**2, dim=-1)
        m_ij = torch.sqrt(torch.relu(m_ij_sq))

        eta_i = phys_feats[:, 2].unsqueeze(1)
        eta_j = phys_feats[:, 2].unsqueeze(0)
        phi_i = phys_feats[:, 3].unsqueeze(1)
        phi_j = phys_feats[:, 3].unsqueeze(0)

        d_eta = eta_i - eta_j
        d_phi = phi_i - phi_j
        d_phi = torch.remainder(d_phi + math.pi, 2 * math.pi) - math.pi
        dR_ij = torch.sqrt(d_eta**2 + d_phi**2)

        mask = ~torch.eye(k, dtype=torch.bool)
        m_ij_flat = m_ij[mask]
        dR_ij_flat = dR_ij[mask]

        return torch.stack([m_ij_flat, dR_ij_flat], dim=1)

    def fit(self, X, y=None):
        all_globals, all_nodes, all_edges = [], [], []
        max_obj_id = 0

        for i in range(X.shape[0]):
            event = X[i].clone()
            global_feats = event[:2]
            objects = event[2:].reshape(18, 5)
            is_particle = objects[:, 2] > 1e-6

            if not torch.any(is_particle): continue

            real_objects = objects[is_particle]
            obj_ids = real_objects[:, 0].long()
            if obj_ids.numel() > 0:
                max_obj_id = max(max_obj_id, obj_ids.max().item())

            phys_feats = self._get_phys_features(real_objects)
            p4_vectors = torch.stack([phys_feats[:,0], phys_feats[:,4], phys_feats[:,5], phys_feats[:,6]], dim=1)
            pair_feats = self._get_pair_features(phys_feats, p4_vectors)

            all_globals.append(global_feats)
            all_nodes.append(phys_feats)
            if pair_feats.shape[0] > 0:
                all_edges.append(pair_feats)

        g_tensor, n_tensor = torch.stack(all_globals), torch.cat(all_nodes, dim=0)
        self.g_mean, self.g_std = g_tensor.mean(dim=0), g_tensor.std(dim=0)
        self.n_mean, self.n_std = n_tensor.mean(dim=0), n_tensor.std(dim=0)

        if all_edges:
            e_tensor = torch.cat(all_edges, dim=0)
            self.e_mean, self.e_std = e_tensor.mean(dim=0), e_tensor.std(dim=0)
        else: # Handle case with no edges in dataset
            self.e_mean, self.e_std = torch.zeros(self.PAIR_FEATS), torch.ones(self.PAIR_FEATS)

        self.g_std[self.g_std < 1e-6] = 1.0
        self.n_std[self.n_std < 1e-6] = 1.0
        self.e_std[self.e_std < 1e-6] = 1.0
        self.num_obj_types = max_obj_id + 1
        return self

    def transform(self, X):
        data_list = []
        for i in range(X.shape[0]):
            event = X[i].clone()
            global_feats = event[:2]
            objects = event[2:].reshape(18, 5)
            is_particle = objects[:, 2] > 1e-6

            if not torch.any(is_particle):
                k = 1
                node_feats = torch.zeros((1, 1 + self.CONT_NODE_FEATS))
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, self.PAIR_FEATS))
            else:
                real_objects = objects[is_particle]
                k = real_objects.shape[0]
                obj_ids = real_objects[:, 0].long().unsqueeze(1)
                phys_feats = self._get_phys_features(real_objects)
                p4_vectors = torch.stack([phys_feats[:,0], phys_feats[:,4], phys_feats[:,5], phys_feats[:,6]], dim=1)
                pair_feats = self._get_pair_features(phys_feats, p4_vectors)

                norm_phys_feats = (phys_feats - self.n_mean) / self.n_std
                node_feats = torch.cat([obj_ids.float(), norm_phys_feats], dim=1)

                if pair_feats.shape[0] > 0:
                    edge_attr = (pair_feats - self.e_mean) / self.e_std
                else:
                    edge_attr = torch.empty((0, self.PAIR_FEATS))

                adj = torch.ones(k, k, dtype=torch.uint8); adj.fill_diagonal_(0)
                edge_index = adj.nonzero().t().contiguous()

            norm_global_feats = (global_feats - self.g_mean) / self.g_std
            data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr, u=norm_global_feats.unsqueeze(0))
            data_list.append(data)
        return data_list

    def make_loader_cfg(self):
        return {"loader_class": "torch_geometric.loader.DataLoader", "batch_size": 128}

def make_preprocessor():
    return MyPreprocessor()

class MyGraphDataset(GeoDataset):
    def __init__(self, data_list):
        super().__init__(None, None, None)
        self.data_list = data_list
    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]

def make_dataset(x, y):
    for i, data_obj in enumerate(x):
        data_obj.y = torch.tensor(y[i], dtype=torch.float32)
    return MyGraphDataset(x)

# 2. ---------- MODEL DEFINITION ----------
class EdgeModel(nn.Module):
    def __init__(self, node_in, edge_in, global_in, edge_out):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(node_in * 2 + edge_in + global_in, edge_out), nn.ReLU(), nn.LayerNorm(edge_out))
    def forward(self, src, dest, edge_attr, u, batch): return self.edge_mlp(torch.cat([src, dest, edge_attr, u[batch]], dim=1))

class NodeModel(nn.Module):
    def __init__(self, node_in, edge_in, global_in, node_out):
        super().__init__()
        self.node_mlp = nn.Sequential(nn.Linear(node_in + edge_in + global_in, node_out), nn.ReLU(), nn.LayerNorm(node_out))
    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index; edge_agg = global_mean_pool(edge_attr, col, size=x.size(0)); return self.node_mlp(torch.cat([x, edge_agg, u[batch]], dim=1))

class GlobalModel(nn.Module):
    def __init__(self, node_in, edge_in, global_in, global_out):
        super().__init__()
        self.global_mlp = nn.Sequential(nn.Linear(node_in + edge_in + global_in, global_out), nn.ReLU(), nn.LayerNorm(global_out))
    def forward(self, x, edge_index, edge_attr, u, batch):
        node_agg = global_mean_pool(x, batch); edge_agg = global_mean_pool(edge_attr, batch[edge_index[0]]); return self.global_mlp(torch.cat([node_agg, edge_agg, u], dim=1))

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        num_obj_types = int(sample_object.x[:, 0].max().item()) + 2
        id_embedding_dim = 8
        node_continuous_dim = sample_object.x.shape[1] - 1
        node_dim_initial = id_embedding_dim + node_continuous_dim
        edge_dim = max(1, sample_object.edge_attr.shape[1])
        global_dim = sample_object.u.shape[1]
        hidden_dim = 64

        self.id_embedding = nn.Embedding(num_embeddings=num_obj_types, embedding_dim=id_embedding_dim)
        self.meta_layer_1 = MetaLayer(EdgeModel(node_dim_initial, edge_dim, global_dim, hidden_dim), NodeModel(node_dim_initial, hidden_dim, global_dim, hidden_dim), GlobalModel(node_dim_initial, hidden_dim, global_dim, hidden_dim))
        self.meta_layer_2 = MetaLayer(EdgeModel(hidden_dim, hidden_dim, hidden_dim, hidden_dim), NodeModel(hidden_dim, hidden_dim, hidden_dim, hidden_dim), GlobalModel(hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim // 2, 1))

    def forward(self, *data):
        batch_data = data[0] if isinstance(data, tuple) else data
        x, edge_index, edge_attr, u, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.u, batch_data.batch
        obj_ids, cont_feats = x[:, 0].long(), x[:, 1:]
        id_embeds = self.id_embedding(obj_ids); x_initial = torch.cat([id_embeds, cont_feats], dim=1)
        x_1, edge_attr_1, u_1 = self.meta_layer_1(x_initial, edge_index, edge_attr, u, batch)
        x_2, edge_attr_2, u_2 = self.meta_layer_2(x_1, edge_index, edge_attr_1, u_1, batch)
        logits = self.classifier(u_2)
        return logits.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    best_val_loss, epochs_no_improve, best_model_state = float('inf'), 0, None
    patience = 5

    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data.y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.num_graphs
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct_train += (preds == data.y.long()).sum().item()
            total_train += data.num_graphs
        train_loss.append(running_loss / total_train); train_acc.append(correct_train / total_train)

        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs, data.y)
                running_val_loss += loss.item() * data.num_graphs
                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct_val += (preds == data.y.long()).sum().item()
                total_val += data.num_graphs
        epoch_val_loss = running_val_loss / total_val; val_loss.append(epoch_val_loss); val_acc.append(correct_val / total_val)

        scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss; epochs_no_improve = 0; best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience: break

    if best_model_state: model.load_state_dict(best_model_state)
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

