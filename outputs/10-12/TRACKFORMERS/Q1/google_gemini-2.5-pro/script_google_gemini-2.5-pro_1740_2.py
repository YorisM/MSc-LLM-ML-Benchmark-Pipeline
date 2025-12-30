
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data/train/"
TAG      = "10_50_linear_frac0.05"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"REDVID_{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def _split_X_y(evt):
    X = np.column_stack((evt["hit_r"],
                        evt["hit_theta"],
                        evt["hit_z"],
                        evt["layer_id"]))
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

# <start code template>
# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import Dataset as PyGDataset
import torch_geometric.loader
import torch_geometric.nn
import scipy.sparse

# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit()
    #   - transform()

    # REQUIREMENTS
    #   IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   May allocate NumPy arrays or Torch tensors internally, but:
    #   transform() must be deterministic.
    #   fit(events) receives the *raw* event dicts list, not a tensor batch.
    #   Store only derived parameters needed for transform i.e. do not store the raw data
    #    itself in the preprocessor object.

    # TIPS
    #   When modifying data features or feature engineering: annotate tensor size as comments after 
    #   each tensor operation to reduce dimension mismatches.

    def __init__(self):
        self.means = None
        self.stds = None

    def _raw_reshape(self, data):           
        return data # Returns identity by default

    def make_loader_cfg(self):
        return {
           "loader_class": "torch_geometric.loader.DataLoader",
           "batch_size": 16, # Batch multiple graphs together
           "shuffle": True,
        }

    def fit(self, data):
        all_hits = []
        for evt in data:
            r, th, z, l = evt["hit_r"], evt["hit_theta"], evt["hit_z"], evt["layer_id"]
            x = r * np.cos(th)
            y = r * np.sin(th)
            all_hits.append(np.column_stack((x, y, z, r, th, l)))
        all_hits_np = np.vstack(all_hits)

        self.means = torch.from_numpy(all_hits_np.mean(axis=0)).float()
        self.stds = torch.from_numpy(all_hits_np.std(axis=0)).float()
        self.stds[self.stds == 0] = 1.0 # Avoid division by zero
        return self

    def transform(self, data):
        r, th, z, l = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        x = r * torch.cos(th)
        y = r * torch.sin(th)

        features = torch.stack([x, y, z, r, th, l], dim=-1) # (N_hits, 6)

        # Standardize
        if self.means is not None and self.stds is not None:
             return (features - self.means.to(data.device)) / self.stds.to(data.device)
        return features

def make_preprocessor():
    return MyPreprocessor()

# Custom PyG Dataset to build graphs on the fly
class MyEventDataset(PyGDataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
        self.K = 16 # k for knn graph
        super().__init__()

    def len(self):
        return len(self.events)

    def get(self, idx):
        X_raw, track_id = _split_X_y(self.events[idx])

        node_features = self.pre.transform(X_raw) # (N, F)

        r, th, z = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2]
        pos = torch.stack([r * torch.cos(th), r * torch.sin(th), z], dim=-1) # (N, 3)

        # Using batch=None because we process one graph at a time
        edge_index = torch_geometric.nn.knn_graph(pos, k=self.K, loop=False, batch=None) # (2, E)

        row, col = edge_index
        y_edge = (track_id[row] == track_id[col]).to(torch.float)
        y_edge[(track_id[row] == 0) | (track_id[col] == 0)] = 0.

        # Return a torch_geometric.data.Data object
        return Data(x=node_features, edge_index=edge_index, y=track_id, y_edge=y_edge)

# This function overrides the default `_make_dataset` in the harness
def make_dataset(events, pre, *, train: bool):
    return MyEventDataset(events, pre, train=train)


# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features, k=16, threshold=0.5):
        super().__init__()
        self.k = k
        self.threshold = threshold

        # Node feature embedding network
        self.node_encoder = nn.Sequential(
            nn.Linear(in_features, 64), 
            nn.ReLU(), 
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # EdgeConv layers using a fixed graph from kNN on spatial coordinates
        self.conv1 = torch_geometric.nn.EdgeConv(
            nn.Sequential(nn.Linear(2 * 128, 128), nn.ReLU(), nn.LayerNorm(128)),
            aggr='mean'
        )
        self.conv2 = torch_geometric.nn.EdgeConv(
            nn.Sequential(nn.Linear(2 * 128, 128), nn.ReLU(), nn.LayerNorm(128)),
            aggr='mean'
        )
        self.conv3 = torch_geometric.nn.EdgeConv(
            nn.Sequential(nn.Linear(2 * 128, 128), nn.ReLU(), nn.LayerNorm(128)),
            aggr='mean'
        )

        # Edge feature classifier
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * 128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )

    def get_edge_logits(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.node_encoder(x) # [N_total, in_features] -> [N_total, 128]
        x = self.conv1(x, edge_index) # [N_total, 128] -> [N_total, 128]
        x = self.conv2(x, edge_index) # [N_total, 128] -> [N_total, 128]
        x = self.conv3(x, edge_index) # [N_total, 128] -> [N_total, 128]

        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1) # [E_total, 2 * 128]
        return self.edge_classifier(edge_features).squeeze(-1) # [E_total,]

    def forward(self, data):
        edge_logits = self.get_edge_logits(data)

        if self.training:
            return edge_logits
        else:
            # Inference mode: build tracks from edge scores
            pred_tracks = []
            num_graphs = data.num_graphs

            node_slices = torch.cat([torch.tensor([0], device=data.batch.device), torch.bincount(data.batch).cumsum(0)])
            edge_scores = torch.sigmoid(edge_logits)

            row, col = data.edge_index

            for i in range(num_graphs):
                start_node, end_node = node_slices[i], node_slices[i+1]
                num_nodes_in_graph = end_node - start_node

                mask = (row >= start_node) & (row < end_node)
                graph_edge_index = data.edge_index[:, mask] - start_node
                graph_edge_scores = edge_scores[mask]

                selected_edges_mask = graph_edge_scores > self.threshold
                selected_edges = graph_edge_index[:, selected_edges_mask]

                if selected_edges.shape[1] > 0:
                    adj = scipy.sparse.coo_matrix(
                        (np.ones(selected_edges.shape[1]), 
                         (selected_edges[0].cpu().numpy(), selected_edges[1].cpu().numpy())),
                        shape=(num_nodes_in_graph, num_nodes_in_graph)
                    )
                    n_components, labels = scipy.sparse.csgraph.connected_components(
                        csgraph=adj, directed=False, return_labels=True
                    )
                else:
                    labels = np.arange(num_nodes_in_graph)

                pred_tracks.append(torch.from_numpy(labels).to(data.x.device))

            return pred_tracks

def make_model(sample_data):
    if isinstance(sample_data, int):
        in_features = sample_data
    elif hasattr(sample_data, 'num_node_features'): # PyG Data object
        in_features = sample_data.num_node_features
    elif hasattr(sample_data, 'shape'): # Tensor
        in_features = sample_data.shape[1]
    else:
        raise TypeError(f"Unknown type for make_model argument: {type(sample_data)}")

    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30
def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    total_steps = len(train_loader) * epochs if len(train_loader) > 0 else 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = model.state_dict()

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct_edges, total_edges = 0.0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            edge_logits = model(batch)
            edge_labels = batch.y_edge

            num_pos = edge_labels.sum()
            pos_weight = (len(edge_labels) - num_pos) / num_pos if num_pos > 0 else torch.tensor(1.0).to(device)

            loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * batch.num_graphs
            preds = (edge_logits > 0.0).float()
            correct_edges += (preds == edge_labels).sum().item()
            total_edges += len(edge_labels)

        epoch_train_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        epoch_train_acc = correct_edges / total_edges if total_edges > 0 else 0
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        model.eval()
        running_val_loss, correct_val_edges, total_val_edges = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                edge_logits = model.get_edge_logits(batch)
                edge_labels = batch.y_edge

                num_pos = edge_labels.sum()
                pos_weight = (len(edge_labels) - num_pos) / num_pos if num_pos > 0 else torch.tensor(1.0).to(device)

                val_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels, pos_weight=pos_weight)
                running_val_loss += val_loss.item() * batch.num_graphs

                preds = (edge_logits > 0.0).float()
                correct_val_edges += (preds == edge_labels).sum().item()
                total_val_edges += len(edge_labels)

        epoch_val_loss = running_val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        epoch_val_acc = correct_val_edges / total_val_edges if total_val_edges > 0 else 0
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, train_accs, val_accs
# <end code template>

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
    collate = getattr(pre, "_collate_fn", None)
    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None

    train_loader, val_loader = make_loaders(raw_train, raw_val, pre,
                                            batch = cfg.get("batch_size", 128),
                                            collate_fn = collate or _ragged,
                                            loader_cls = loader_cls,
                                            workers    = cfg.get("num_workers", 0))

    # 2. Build model
    first_batch    = next(iter(train_loader))
    example_sample = first_batch[0]
    model          = make_model(example_sample)
    model          = model.to(device)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single toy forward pass
    if dryrun:
        try:
            sample_batch = first_batch.to(device)
            _ = trained_model(sample_batch)
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


