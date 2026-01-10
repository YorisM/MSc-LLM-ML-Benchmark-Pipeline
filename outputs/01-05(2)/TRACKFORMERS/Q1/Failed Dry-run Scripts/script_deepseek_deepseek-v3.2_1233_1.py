
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import detect_and_assert_lane, assert_label_output_by_lane, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, build_trackformers_model

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

def split_X_y(evt):
    X = np.column_stack([
        evt["hit_r"].astype(np.float32),
        evt["hit_theta"].astype(np.float32),
        evt["hit_z"].astype(np.float32),
        evt["layer_id"].astype(np.float32)
    ])
    y = evt["track_id"].astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, labels = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, labels)

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_cluster
from torch_scatter import scatter_mean, scatter_max
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        self.events = events
        self.pre = pre
        self.train = train

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        X, y = split_X_y(self.events[idx])
        if self.pre is not None:
            X = self.pre.transform(X)
        # Build k-NN graph
        x_np = X.numpy()
        edge_index = torch_cluster.knn_graph(
            torch.from_numpy(x_np[:, :3]), k=16, loop=False
        )
        # Compute edge features (relative coordinates)
        pos = X[:, :3]  # r, theta, z
        row, col = edge_index
        edge_attr = torch.cat([
            pos[row] - pos[col],
            torch.norm(pos[row] - pos[col], dim=1, keepdim=True)
        ], dim=1)

        return torch_geometric.data.Data(
            x=X, y=y, edge_index=edge_index, edge_attr=edge_attr, num_nodes=X.shape[0]
        )

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 16,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {"follow_batch": []},
            "eval_overrides": {"shuffle": False, "batch_size": 16}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, F_raw]
        all_features = []
        for X in Xs:
            X_np = X.numpy()
            # Convert to Cartesian for better scaling
            r = X_np[:, 0]
            theta = X_np[:, 1]
            z = X_np[:, 2]
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            features = np.column_stack([x, y, z, X_np[:, 3]])
            all_features.append(features)
        all_features = np.vstack(all_features)
        self.scaler.fit(all_features[:, :3])  # Fit on x,y,z
        self.layer_scaler.fit(all_features[:, 3:4])  # Fit on layer_id
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, F_raw]
        X_np = X.numpy()
        r = X_np[:, 0]
        theta = X_np[:, 1]
        z = X_np[:, 2]
        layer_id = X_np[:, 3]

        # Convert to Cartesian
        x = r * np.cos(theta)  # [N_hits]
        y = r * np.sin(theta)  # [N_hits]

        # Scale spatial coordinates
        spatial = np.column_stack([x, y, z])  # [N_hits, 3]
        spatial_scaled = self.scaler.transform(spatial)

        # Scale layer_id
        layer_scaled = self.layer_scaler.transform(layer_id.reshape(-1, 1))

        # Additional features
        r_scaled = (r - self.scaler.mean_[0]) / np.sqrt(self.scaler.var_[0])  # Approximate
        phi = theta  # Use theta as azimuthal angle

        # Combine features: spatial + layer + cylindrical coordinates
        features = np.column_stack([
            spatial_scaled,              # 3 features: x, y, z
            layer_scaled.flatten(),      # 1 feature: layer_id
            r_scaled,                    # 1 feature: r
            np.cos(phi),                 # 1 feature: cos(theta)
            np.sin(phi),                 # 1 feature: sin(theta)
            np.arctan2(y, x),            # 1 feature: atan2(y,x)
        ])  # [N_hits, 8]

        return torch.FloatTensor(features)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.gat = torch_geometric.nn.GATConv(
            in_dim, out_dim, heads=heads, dropout=dropout, concat=True
        )
        self.bn = nn.BatchNorm1d(out_dim * heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.gat(x, edge_index)
        x = self.bn(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        return self.dropout(x)

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x=None):
        super().__init__()
        # Example_batch_x is a Data object in PyG lane
        input_dim = 8  # From preprocessor

        # Graph attention layers
        self.gat1 = GATLayer(input_dim, 128, heads=4)
        self.gat2 = GATLayer(128 * 4, 128, heads=4)
        self.gat3 = GATLayer(128 * 4, 128, heads=4)
        self.gat4 = GATLayer(128 * 4, 128, heads=1)  # Last layer, single head

        # Edge prediction head
        self.edge_mlp = nn.Sequential(
            nn.Linear(128 * 2 + 5, 256),  # 2*node_feat + edge_attr
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        # Node embedding projection (for clustering)
        self.node_proj = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Graph attention layers
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)
        x = self.gat3(x, edge_index)
        x = self.gat4(x, edge_index)  # [N_nodes, 128]

        # Get edge features
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=1)  # [N_edges, 128*2+5]
        edge_scores = self.edge_mlp(edge_features).squeeze(-1)  # [N_edges]

        # Node embeddings for clustering
        node_emb = self.node_proj(x)  # [N_nodes, 32]

        return edge_scores, node_emb

    def predict_labels(self, batch):
        # Predict cluster assignments using learned edge scores
        if isinstance(batch, list):
            batch = batch[0]  # In PyG loader, batch is already a Batch object

        edge_scores, node_emb = self.forward(batch)

        # Get batch assignment
        if hasattr(batch, 'batch'):
            batch_idx = batch.batch
        else:
            batch_idx = torch.zeros(batch.x.shape[0], dtype=torch.long, device=batch.x.device)

        # Apply sigmoid to get probabilities
        edge_probs = torch.sigmoid(edge_scores)

        # Build adjacency matrix for each batch separately
        all_labels = []
        for b in range(batch_idx.max().item() + 1):
            # Get nodes in this batch
            mask = (batch_idx == b)
            node_indices = torch.where(mask)[0]

            # Get edges within this batch
            edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
            edges = batch.edge_index[:, edge_mask]
            probs = edge_probs[edge_mask]

            # Remap node indices to 0..N_b
            local_nodes = {int(node): i for i, node in enumerate(node_indices)}
            edges = torch.tensor([[local_nodes[int(edges[0,i])], 
                                 local_nodes[int(edges[1,i])]] for i in range(edges.shape[1])], 
                               device=edges.device).T

            # Build sparse adjacency matrix with probabilities
            N = len(node_indices)
            adj = torch.zeros((N, N), device=batch.x.device)
            if edges.shape[1] > 0:
                adj[edges[0], edges[1]] = probs
                adj[edges[1], edges[0]] = probs  # Make symmetric

            # Spectral clustering with learned similarities
            from scipy.sparse.linalg import eigsh
            import scipy.sparse as sp

            # Convert to CPU numpy for spectral clustering
            adj_np = adj.cpu().numpy()
            eps = 1e-8
            D = np.diag(1.0 / np.sqrt(np.sum(adj_np, axis=1) + eps))
            L = np.eye(N) - D @ adj_np @ D

            # Estimate number of clusters using eigenvalues
            try:
                eigvals = eigsh(L, k=min(20, N-1), which='SM', return_eigenvectors=False)
                eigvals = np.sort(eigvals)
                # Find gap in eigenvalues (Gap statistic)
                gaps = eigvals[1:] - eigvals[:-1]
                if len(gaps) > 0:
                    n_clusters = np.argmax(gaps) + 1
                    n_clusters = max(2, min(n_clusters, 50))  # Limit to reasonable range
                else:
                    n_clusters = 2
            except:
                n_clusters = 2

            # Perform spectral clustering
            try:
                from sklearn.cluster import SpectralClustering
                sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                                       random_state=42, n_init=10)
                labels_b = sc.fit_predict(adj_np + eps)
                # Remap labels to original indices
                labels_full = -torch.ones(mask.sum().item(), dtype=torch.long, device=batch.x.device)
                labels_full[:len(labels_b)] = torch.from_numpy(labels_b).to(batch.x.device)
            except:
                # Fallback to connected components
                from torch_geometric.utils import to_networkx
                import networkx as nx
                threshold = 0.5
                strong_edges = edges[:, probs > threshold]
                if strong_edges.shape[1] > 0:
                    G = nx.Graph()
                    G.add_nodes_from(range(N))
                    G.add_edges_from(strong_edges.cpu().T.numpy())
                    labels_b = np.zeros(N, dtype=int)
                    for i, comp in enumerate(nx.connected_components(G)):
                        for node in comp:
                            labels_b[node] = i
                    labels_full = torch.from_numpy(labels_b).to(batch.x.device)
                else:
                    labels_full = torch.zeros(N, dtype=torch.long, device=batch.x.device)

            all_labels.append(labels_full)

        # Concatenate labels
        labels = torch.cat(all_labels, dim=0)

        # Filter small clusters
        unique_labels = torch.unique(labels)
        for lab in unique_labels:
            mask = labels == lab
            if mask.sum() < 4:
                labels[mask] = -1  # Mark as noise

        return labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    best_val_acc = 0.0
    best_model_state = None
    patience = 10
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            edge_scores, _ = model(batch)

            # Create edge labels (1 if same track, 0 otherwise)
            row, col = batch.edge_index
            edge_labels = (batch.y[row] == batch.y[col]).float()
            edge_labels *= (batch.y[row] != 0).float()  # Ignore noise-to-noise edges
            edge_labels *= (batch.y[col] != 0).float()

            # Weight positive edges more (since they are rare)
            pos_weight = torch.tensor([5.0], device=device)
            loss = F.binary_cross_entropy_with_logits(
                edge_scores, edge_labels, pos_weight=pos_weight
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs
            train_total += 1

            # Monitor accuracy
            with torch.no_grad():
                pred_probs = torch.sigmoid(edge_scores)
                pred_labels = (pred_probs > 0.5).float()
                correct = ((pred_labels == edge_labels) & (edge_labels != 0.5)).sum().item()
                total = (edge_labels != 0.5).sum().item()
                if total > 0:
                    train_correct += correct

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / max(train_total, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                edge_scores, _ = model(batch)

                # Edge labels
                row, col = batch.edge_index
                edge_labels = (batch.y[row] == batch.y[col]).float()
                edge_labels *= (batch.y[row] != 0).float()
                edge_labels *= (batch.y[col] != 0).float()

                pos_weight = torch.tensor([5.0], device=device)
                loss = F.binary_cross_entropy_with_logits(
                    edge_scores, edge_labels, pos_weight=pos_weight
                )

                val_loss += loss.item() * batch.num_graphs
                val_total += 1

                # Compute edge accuracy
                pred_probs = torch.sigmoid(edge_scores)
                pred_labels = (pred_probs > 0.5).float()
                correct = ((pred_labels == edge_labels) & (edge_labels != 0.5)).sum().item()
                total = (edge_labels != 0.5).sum().item()
                if total > 0:
                    val_correct += correct

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / max(val_total, 1)

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    Xs = [split_X_y(evt)[0] for evt in raw_train]
    pre = make_preprocessor().fit(Xs)

    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, raw_train, pre, train=True)
    val_ds       = build_dataset(spec, raw_val,   pre, train=False)
    train_loader = build_dataloader(spec, train_ds, is_eval=False)
    val_loader   = build_dataloader(spec, val_ds,   is_eval=True)

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane(spec, first_batch)

    # Build model
    model = build_trackformers_model(mode, first_batch, make_model, device)

    # Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        if not hasattr(trained_model, "predict_labels") or not callable(getattr(trained_model, "predict_labels")):
            raise TypeError("Contract error: trained model must implement predict_labels(batch_x).")

        trained_model.eval()
        try:
            with torch.no_grad():
                mode = None
                for i, batch in enumerate(val_loader):
                    if mode is None:
                        mode = detect_and_assert_lane(spec, batch)

                    if mode == "torch_ragged_xy":
                        Xs, _ys = batch
                        Xs = [x.to(device) for x in Xs]
                        out = trained_model.predict_labels(Xs)
                    elif mode == "pyg_batch":
                        G = batch.to(device)
                        out = trained_model.predict_labels(G)
                    else:
                        raise RuntimeError(f"Unknown lane mode: {mode}")

                    assert_label_output_by_lane(mode, batch, out, allow_noise_label=True)
                    if i >= 3:  # 4 batches
                        break
        except Exception as e:
            raise RuntimeError("Sanity-check predict_labels() failed") from e
        return


    if not dryrun:
        # Persist artefacts
        base = base_from_argv0()
        persist_artefacts(base, SCRIPT_DIR, trained_model, pre, spec)

        # Save plots
        plot_train_val(tr_loss, va_loss, f"{base} Loss", os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        plot_train_val(tr_acc, va_acc, f"{base} Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))
        
        # Write JSON Summary
        summary = {
            "epochs": n_epochs      if n_epochs else None,
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

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

